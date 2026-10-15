"""Approved 2026-10-15 postprocessing of frozen SHAP/transfer evidence.

No model is loaded, fitted, scored or explained. Original arrays, metric JSONs
and manifests are read only. New summaries live in a separate dated directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from generate_revision_interpretability import group_contributions, original_feature_name

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "results_revision/20261005"
OUTPUT = ROOT / "results_revision/20261015_supervisor/interpretability_transfer"
PLANNED_STRATEGIES = ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")
EFFECTIVE_STRATEGIES = tuple(s for s in PLANNED_STRATEGIES if s != "smote_tomek")
MODELS = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")
SHAP_MODELS = ("logreg", "lgbm", "catboost")
LABELS = dict(zip(MODELS, ("LR", "RF", "LGBM", "CatBoost", "FT-Trans.", "OCSVM")))
COHORTS = ("baf_base", "baf_var1", "baf_var2", "baf_var3", "baf_var4", "baf_var5")
COHORT_LABELS = dict(zip(COHORTS, ("Base", "I", "II", "III", "IV", "V")))
MAIN_POLICY = "mean_i(abs(sum_j_in_G(phi_ij)))"
SENSITIVITY_POLICY = "mean_i(sum_j_in_G(abs(phi_ij)))"
FOCUS = ("customer_age", "date_of_birth_distinct_emails_4w")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def source(path, sources, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("A scientific source escapes the author archive")
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Pinned source changed: {path}")
    sources[path.relative_to(ROOT).as_posix()] = actual
    return path


def effective_strategy_summary(metrics):
    """Summarise the six audited conditions, not six independent experiments.

    All seven labels remain required. SMOTE+Tomek is excluded specifically,
    not arbitrary metric duplicates. Input AP/F2 must already be full precision.
    """
    if set(metrics) != set(PLANNED_STRATEGIES):
        raise ValueError("Expected all seven planned strategy labels")
    values = np.asarray([metrics[s] for s in EFFECTIVE_STRATEGIES], dtype=float)
    historical = np.asarray([metrics[s] for s in PLANNED_STRATEGIES], dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(historical).all():
        raise ValueError("Strategy summary requires finite full-precision metrics")
    return {"mean": float(values.mean()), "population_sd": float(values.std(ddof=0)),
            "historical_seven_mean": float(historical.mean()),
            "historical_seven_population_sd": float(historical.std(ddof=0)),
            "effective_conditions": list(EFFECTIVE_STRATEGIES), "ddof": 0}


def summarise_shap(values, names, chunk_rows=8192):
    """Compute both aggregations, checking grouping/additivity without inference."""
    if (values.ndim != 2 or values.shape[1] != len(names) or values.shape[0] == 0
            or len(set(names)) != len(names) or chunk_rows <= 0):
        raise ValueError("Expected a nonempty aligned matrix with unique encoded names")
    parents = [original_feature_name(name) for name in names]
    features = list(dict.fromkeys(parents))
    net_sum = np.zeros(len(features), dtype=np.float64)
    mass_sum = np.zeros(len(features), dtype=np.float64)
    max_additivity = 0.0
    for start in range(0, len(values), chunk_rows):
        chunk = np.asarray(values[start:start + chunk_rows], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("Non-finite frozen attribution")
        signed, mapped = group_contributions(chunk, names, absolute=False)
        mass, mass_names = group_contributions(chunk, names, absolute=True)
        if mapped != features or mapped != mass_names:
            raise ValueError("Inconsistent original-feature map")
        error = float(np.max(np.abs(signed.sum(axis=1) - chunk.sum(axis=1))))
        max_additivity = max(max_additivity, error)
        if error > 1e-10:
            raise ValueError("Signed grouping does not preserve additive output")
        if np.any(np.abs(signed) > mass + 1e-12):
            raise ValueError("Net contribution exceeds encoded absolute mass")
        for index, name in enumerate(features):
            members = [j for j, parent in enumerate(parents) if parent == name]
            if len(members) == 1 and not np.array_equal(np.abs(signed[:, index]), mass[:, index]):
                raise ValueError("A singleton feature changed under aggregation")
        net_sum += np.abs(signed).sum(axis=0)
        mass_sum += mass.sum(axis=0)
    main = dict(zip(features, (net_sum / len(values)).tolist()))
    sensitivity = dict(zip(features, (mass_sum / len(values)).tolist()))
    return {"rows": len(values), "encoded_columns": len(names),
            "original_features": features, "complete_global_importance": main,
            "encoded_absolute_mass_sensitivity": sensitivity,
            "ranking": ranked(main), "top10": ranked(main)[:10],
            "sensitivity_ranking": ranked(sensitivity),
            "maximum_signed_grouping_additivity_error": max_additivity}


def ranked(importance):
    """Deterministic descending order; alphabetical exact-tie convention."""
    return sorted(importance, key=lambda name: (-importance[name], name))


def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def magnitude_contrasts(cohorts):
    base = cohorts["baf_base"]["complete_global_importance"]
    features = cohorts["baf_base"]["original_features"]
    rows = []
    for cohort in COHORTS:
        current = cohorts[cohort]
        if set(current["complete_global_importance"]) != set(base):
            raise ValueError("SHAP contrasts require all original features in every cohort")
        for feature in features:
            importance = current["complete_global_importance"][feature]
            delta = importance - base[feature]
            rows.append({"cohort": cohort, "feature": feature, "importance": importance,
                         "base_importance": base[feature], "absolute_difference": delta,
                         "relative_difference_percent": 100 * delta / base[feature] if base[feature] else None,
                         "rank": current["ranking"].index(feature) + 1,
                         "base_rank": cohorts["baf_base"]["ranking"].index(feature) + 1})
    return rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_current_shap_summary(output=OUTPUT, verify_sources=True):
    """Read the explicitly dated main policy, with no historical fallback."""
    output = Path(output)
    path = output / "shap_summary.json"
    if not path.is_file() or not (output / "manifest.json").is_file():
        raise FileNotFoundError("Generate the approved dated supervisor SHAP postprocessing first")
    manifest = read_json(output / "manifest.json")
    if digest(path) != manifest["output_sha256"]["shap_summary.json"]:
        raise ValueError("The current SHAP summary differs from its derivative manifest")
    if verify_sources:
        from export_public_evidence import resolve_pinned_author_source
        for name, expected in manifest["source_sha256"].items():
            resolve_pinned_author_source(ROOT, name, expected)
    report = read_json(path)
    if report["policy"] != MAIN_POLICY or any(len(model["complete_global_importance"]) != 30 for model in report["models"].values()):
        raise ValueError("The current main SHAP policy requires complete net-contribution vectors")
    return report


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def table(path, headers, rows, alignment=None, long=False, caption_macro=False):
    environment = "longtable" if long else "tabular"
    align = alignment or "l" + "r" * (len(headers) - 1)
    head = " & ".join(headers) + r" \\"
    lines = [rf"\begin{{{environment}}}{{@{{}}{align}@{{}}}}"]
    if caption_macro:
        if not long:
            raise ValueError("The optional in-table caption requires longtable")
        # The manuscript wrapper defines both macros locally. longtable owns
        # the table counter; an external captionof would increment it twice.
        lines.append(r"\ifdefined\SupervisorLongTableCaption\caption{\SupervisorLongTableCaption}\label{\SupervisorLongTableLabel}\\\fi")
    lines += [r"\toprule", head, r"\midrule"]
    if long:
        lines += [r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead"]
    lines += [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", rf"\end{{{environment}}}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_shap(output, sources, qa):
    shap_qa_path = source(REVISION / "derived/interpretability/interpretability_qa.json", sources)
    shap_qa = read_json(shap_qa_path)
    stability_path = source(REVISION / "derived/interpretability/variant_stability/shap_variant_stability.json", sources)
    stability = read_json(stability_path)
    names = stability["encoded_feature_names"]
    if len(names) != 51 or len(set(original_feature_name(n) for n in names)) != 30:
        raise ValueError("The approved BAF schema requires 51 encoded and 30 original features")
    models = {}
    common_labels = None
    for model in SHAP_MODELS:
        record = shap_qa["models"][model]
        run = Path(record["run_directory"])
        for filename, expected in record["source_sha256"].items():
            source(run / filename, sources, expected)
        if run != Path(qa["compatible_shap_sources"][model]["run_dir"]):
            raise ValueError("SHAP source does not match selected compatible function")
        labels = np.load(run / "y_test.npy", allow_pickle=False)
        scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
        values = np.load(run / "shap_values.npy", mmap_mode="r", allow_pickle=False)
        if (values.shape != (200000, 51) or labels.shape != (200000,) or scores.shape != labels.shape
                or not np.isin(labels, (0, 1)).all() or int(labels.sum()) != 2206
                or not np.isfinite(scores).all() or not np.all((scores > 0) & (scores < 1))):
            raise ValueError("Wrong Base SHAP population")
        if common_labels is None:
            common_labels = labels
        elif not np.array_equal(common_labels, labels):
            raise ValueError("Base SHAP models do not share ordered TEST labels")
        summary = summarise_shap(values, names)
        old = record["complete_global_importance"]
        if set(old) != set(summary["original_features"]):
            raise ValueError("Original-feature names do not match archived model summary")
        old_error = max(abs(old[f] - summary["encoded_absolute_mass_sensitivity"][f]) for f in old)
        if old_error > 1e-10:
            raise ValueError("Encoded feature order is incompatible with archived Base summary")
        raw = np.log(scores) - np.log1p(-scores)
        expected = record["expected_logit"]
        additivity_error = float(np.max(np.abs(raw - values.sum(axis=1) - expected)))
        if additivity_error > 1e-7:
            raise ValueError("Frozen Base SHAP no longer reconstructs saved logits")
        summary.update(source_run=run.relative_to(ROOT).as_posix(),
                       historical_mass_reproduction_error=old_error,
                       raw_logit_additivity_error=additivity_error)
        models[model] = summary
    cohorts = {"baf_base": models["lgbm"]}
    for cohort in COHORTS[1:]:
        folder = stability_path.parent / cohort
        record = stability["variants"][cohort]
        for filename, expected in record["artefacts_sha256"].items():
            source(folder / filename, sources, expected)
        labels = np.load(folder / "y_test.npy", allow_pickle=False)
        rows = np.load(folder / "test_row_indices.npy", allow_pickle=False)
        values = np.load(folder / "shap_values.npy", mmap_mode="r", allow_pickle=False)
        population = stability["populations"][cohort]
        if (values.shape != (population["rows"], 51) or labels.shape != (len(values),)
                or rows.shape != labels.shape or not np.isin(labels, (0, 1)).all()
                or len(np.unique(rows)) != len(rows) or int(labels.sum()) != population["positive_rows"]):
            raise ValueError("Wrong filtered Variant SHAP population")
        pins = qa["transfer_sources"]["lgbm"]["variants"][cohort]["artefacts"]
        for filename, array in (("y_test.npy", labels), ("test_row_indices.npy", rows)):
            pinned_path = source(pins[filename]["path"], sources, pins[filename]["sha256"])
            if not np.array_equal(array, np.load(pinned_path, allow_pickle=False)):
                raise ValueError("Variant SHAP rows differ from retained transfer cohort")
        summary = summarise_shap(values, names)
        # Variant matrices were stored float32; archived summaries accumulated
        # original float64 outputs. Quantify, do not disguise storage rounding.
        prior = stability["global_importance"][cohort]
        summary["archived_float64_mass_max_difference"] = max(
            abs(prior[f] - summary["encoded_absolute_mass_sensitivity"][f]) for f in prior)
        cohorts[cohort] = summary
    model_j = {a: {b: jaccard(models[a]["top10"], models[b]["top10"]) for b in SHAP_MODELS} for a in SHAP_MODELS}
    cohort_j = {a: {b: jaccard(cohorts[a]["top10"], cohorts[b]["top10"]) for b in COHORTS} for a in COHORTS}
    contrasts = magnitude_contrasts(cohorts)
    report = {"policy": MAIN_POLICY, "sensitivity_policy": SENSITIVITY_POLICY,
              "units": "raw log-odds", "grouping_scope": "post-hoc sum of encoded attributions, not a new group-Shapley game",
              "ranking_ties": "alphabetical order for exact equal importance",
              "models": models, "cohorts": cohorts,
              "cross_model_top10_jaccard": model_j, "cohort_top10_jaccard": cohort_j,
              "magnitude_contrasts": contrasts,
              "storage_note": "Base matrices retain original precision; Variant matrices are archived float32, accumulated here in float64. Differences from older float64 summaries are explicitly recorded.",
              "new_shap_computation": False, "model_inference": False, "model_fitting": False}
    write_json(output / "shap_summary.json", report)
    # Prepare, but do not publish, a portable replacement for the existing
    # Article 1 summary. Local signed waterfalls remain exactly unchanged.
    portable = read_json(source(ROOT / "results/interpretability/shap_summary.json", sources))
    portable["policy"] = MAIN_POLICY
    portable["global_grouping"] = MAIN_POLICY
    portable["sensitivity_policy"] = SENSITIVITY_POLICY
    portable["revision_id"] = "20261015_supervisor"
    portable["grouping_scope"] = report["grouping_scope"]
    for model in SHAP_MODELS:
        portable["models"][model].update(models[model])
    portable["consistency"] = {"top_k": 10, "rankings": {m: models[m]["top10"] for m in SHAP_MODELS},
                               "jaccard_matrix": model_j}
    if "outputs" in portable:
        portable["historical_outputs"] = portable.pop("outputs")
    write_json(output / "portable_shap_summary.json", portable)
    write_csv(output / "shap_magnitude_contrasts.csv", contrasts)
    for model in SHAP_MODELS:
        rows = [{"feature": f, "mean_absolute_net_contribution": models[model]["complete_global_importance"][f],
                 "mean_encoded_absolute_mass": models[model]["encoded_absolute_mass_sensitivity"][f],
                 "net_rank": models[model]["ranking"].index(f) + 1,
                 "mass_rank": models[model]["sensitivity_ranking"].index(f) + 1} for f in models[model]["original_features"]]
        write_csv(output / f"shap_aggregation_{model}.csv", rows)
    return report


def _population(metrics):
    raw = [metrics[key] for key in ("TP", "FP", "FN", "TN")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value != int(value) for value in raw):
        raise ValueError("Saved confusion counts must be finite integers")
    counts = [int(value) for value in raw]
    if any(count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("Invalid saved confusion counts")
    return {"rows": sum(counts), "frauds": counts[0] + counts[2],
            "alerts": counts[0] + counts[1], "alert_rate": (counts[0] + counts[1]) / sum(counts)}


def generate_transfer(output, sources, qa, selected):
    pairs, workload = [], []
    cohort_populations = {}
    for model in MODELS:
        audit_pin = qa["transfer_sources"][model]
        audit_path = source(audit_pin["path"], sources, audit_pin["sha256"])
        audit = read_json(audit_path)
        if model == "fttransformer":
            if "run_20261006_073640_773309" not in audit["source_run"]:
                raise ValueError("Transfer must use final B, not validation A or historical FT")
        baseline_pin = selected[f"baf_base/{model}/none"]
        baseline_path = source(ROOT / baseline_pin["public_metrics"], sources)
        baseline = read_json(baseline_path)
        baseline_pop = _population(baseline)
        if baseline_pop["rows"] != 200000 or baseline_pop["frauds"] != 2206:
            raise ValueError("Wrong Base TEST population")
        workload.append({"model": model, "cohort": "baf_base", **baseline_pop,
                         "relative_rate_difference_percent": 0.0, "threshold": baseline["threshold"]})
        for cohort in COHORTS[1:]:
            pin = audit_pin["variants"][cohort]
            path = source(pin["path"], sources, pin["sha256"])
            record = read_json(path)
            original = record["original_population_same_frozen_model"]
            if original.get("role") != "newly_scored_original_population_not_a_retuned_or_historical_run":
                raise ValueError("Original population is not the same frozen function")
            retained_pop = _population(record["metrics"])
            original_pop = _population(original["metrics"])
            if original_pop["rows"] != 200000 or retained_pop["rows"] != record["population"]["rows"]:
                raise ValueError("Wrong original or filtered population")
            identity = tuple(record["artefacts"][name]["file_sha256"] for name in ("y_test.npy", "test_row_indices.npy"))
            if cohort in cohort_populations and cohort_populations[cohort] != identity:
                raise ValueError("Models use different retained cohort identities")
            cohort_populations[cohort] = identity
            threshold = record["metrics_full_precision"]["threshold"]
            if threshold != original["metrics_full_precision"]["threshold"] or threshold != baseline["threshold"]:
                raise ValueError("A transfer comparison changed its frozen threshold")
            unfiltered_ap = original["metrics_full_precision"]["average_precision"]
            filtered_ap = record["metrics_full_precision"]["average_precision"]
            excluded_frauds = original_pop["frauds"] - retained_pop["frauds"]
            excluded_tp = int(original["metrics"]["TP"]) - int(record["metrics"]["TP"])
            if excluded_frauds != record["excluded_population"]["positive_rows"]:
                raise ValueError("Excluded fraud counts do not add up")
            pairs.append({"model": model, "cohort": cohort, "original_rows": original_pop["rows"],
                          "filtered_rows": retained_pop["rows"],
                          "excluded_rows": original_pop["rows"] - retained_pop["rows"],
                          "original_ap": unfiltered_ap, "filtered_ap": filtered_ap,
                          "original_ap_relative_to_filtered_percent": 100 * (unfiltered_ap / filtered_ap - 1),
                          "excluded_frauds": excluded_frauds, "excluded_tp": excluded_tp,
                          "threshold": threshold, "source": path.relative_to(ROOT).as_posix()})
            workload.append({"model": model, "cohort": cohort, **retained_pop,
                             "relative_rate_difference_percent": 100 * (retained_pop["alert_rate"] / baseline_pop["alert_rate"] - 1),
                             "threshold": threshold})
    for cohort in COHORTS[1:]:
        current = [r for r in pairs if r["cohort"] == cohort]
        before = sorted(current, key=lambda r: (-r["original_ap"], r["model"]))
        after = sorted(current, key=lambda r: (-r["filtered_ap"], r["model"]))
        for row in current:
            row["original_ap_rank"] = before.index(row) + 1
            row["filtered_ap_rank"] = after.index(row) + 1
    report = {"comparison_scope": "Same frozen function, original versus filtered Variant population",
              "filter_scope": "Exact common-30 predictor matches to Base DEV, independent of labels/scores",
              "pairs": pairs, "workload": workload,
              "workload_scope": "Same frozen final model and Base threshold, Base TEST versus retained Variants",
              "model_fitting": False, "model_inference": False,
              "historical_ft_excluded": True, "validation_checkpoint_A_excluded": True}
    write_json(output / "transfer_summary.json", report)
    write_csv(output / "transfer_overlap_pairs.csv", pairs)
    write_csv(output / "transfer_alert_workload.csv", workload)
    return report


def generate_strategy_summaries(output, sources, selected):
    summaries, metrics_by_dataset = {}, {}
    identity_path = source(ROOT / "results/provenance/sampler_identity.json", sources)
    identity = read_json(identity_path)
    if identity["status"] != "complete":
        raise ValueError("Six-condition reporting requires the completed sampler identity audit")
    for dataset in ("ulb_2013", "baf_base"):
        metrics_by_dataset[dataset] = {}
        for model in ("fttransformer", "catboost"):
            audit_key = f"{dataset}/{model}"
            audit = (identity["new_training_observations"].get(audit_key)
                     or (identity["historical_saved_scores"].get(model) if dataset == "baf_base" else None))
            if audit is None or not audit["TEST_scores_bitwise_identical"]:
                raise ValueError("The selected FT/CatBoost pair lacks the specific audited identity")
            for strategy, field in (("smote", "smote_run"), ("smote_tomek", "tomek_run")):
                if selected[f"{dataset}/{model}/{strategy}"]["archive_run_id"] != audit[field]:
                    raise ValueError("The strategy summary does not use the audited sampler pair")
            cells = {}
            for strategy in PLANNED_STRATEGIES:
                pin = selected[f"{dataset}/{model}/{strategy}"]
                path = source(ROOT / pin["public_metrics"], sources)
                metrics = read_json(path)
                denominator = 5 * metrics["TP"] + 4 * metrics["FN"] + metrics["FP"]
                cells[strategy] = {"PR-AUC": metrics["PR-AUC"], "F2": 5 * metrics["TP"] / denominator if denominator else 0.0}
            if cells["smote"] != cells["smote_tomek"]:
                raise ValueError("Audited strategy metrics no longer agree")
            summaries[f"{dataset}/{model}"] = {metric: effective_strategy_summary({s: cells[s][metric] for s in cells}) for metric in ("PR-AUC", "F2")}
            metrics_by_dataset[dataset][model] = cells
    report = {"planned_strategies": list(PLANNED_STRATEGIES), "effective_conditions": list(EFFECTIVE_STRATEGIES),
              "representative": {"smote_tomek": "smote"}, "primary_configurations_retained": 72,
              "identity_audit": identity_path.relative_to(ROOT).as_posix(),
              "policy": "Exclude only audited SMOTE+Tomek from between-strategy summaries; retain all seven displayed results",
              "summaries": summaries}
    write_json(output / "strategy_summary.json", report)
    return report, metrics_by_dataset


def generate_tables(output, shap, transfer, strategies, metrics_by_dataset):
    tables = output / "tables"
    table(tables / "rf_overlap.tex", ["Variant", "Excluded", "AP original", "AP filtered", r"Relative (\%)", "Rank"],
          [[COHORT_LABELS[r["cohort"]], f'{r["excluded_rows"]:,}', f'{r["original_ap"]:.4f}', f'{r["filtered_ap"]:.4f}',
            f'+{r["original_ap_relative_to_filtered_percent"]:.1f}', f'{r["original_ap_rank"]} to {r["filtered_ap_rank"]}']
           for r in transfer["pairs"] if r["model"] == "rf"])
    table(tables / "transfer_overlap_full.tex", ["Model", "Variant", "AP original", "AP filtered", r"Relative (\%)", "Rank"],
          [[LABELS[r["model"]], COHORT_LABELS[r["cohort"]], f'{r["original_ap"]:.4f}', f'{r["filtered_ap"]:.4f}',
            f'{r["original_ap_relative_to_filtered_percent"]:+.2f}', f'{r["original_ap_rank"]} to {r["filtered_ap_rank"]}'] for r in transfer["pairs"]],
          alignment="llrrrl", long=True, caption_macro=True)
    table(tables / "transfer_alert_rates.tex", ["Model", "Base", "I", "II", "III", "IV", "V"],
          [[LABELS[model]] + [f'{100 * next(r["alert_rate"] for r in transfer["workload"] if r["model"] == model and r["cohort"] == cohort):.2f}' for cohort in COHORTS] for model in MODELS])
    table(tables / "transfer_alert_counts.tex", ["Model", "Cohort", "$N$", "Alerts", r"Rate (\%)", r"Relative (\%)"],
          [[LABELS[r["model"]], COHORT_LABELS[r["cohort"]], f'{r["rows"]:,}', f'{r["alerts"]:,}', f'{100*r["alert_rate"]:.3f}',
            f'{r["relative_rate_difference_percent"]:+.2f}'] for r in transfer["workload"]], alignment="llrrrr", long=True, caption_macro=True)
    rows = []
    for feature in FOCUS:
        rows.append([rf"\multicolumn{{5}}{{l}}{{\textit{{{'Customer age' if feature == FOCUS[0] else 'DOB distinct emails (4 weeks)'}}}}}"])
        rows.extend([[COHORT_LABELS[r["cohort"]], f'{r["importance"]:.4f}', f'{r["absolute_difference"]:+.4f}',
                      "---" if r["relative_difference_percent"] is None else f'{r["relative_difference_percent"]:+.2f}', str(r["rank"])]
                     for r in shap["magnitude_contrasts"] if r["feature"] == feature])
    table(tables / "shap_magnitude_compact.tex", ["Cohort", "Mean magnitude", r"$\Delta$ magnitude", r"Relative (\%)", "Rank"], rows)
    fullrows = [[tex_escape(r["feature"]), COHORT_LABELS[r["cohort"]], f'{r["importance"]:.5f}',
                 f'{r["absolute_difference"]:+.5f}', "---" if r["relative_difference_percent"] is None else f'{r["relative_difference_percent"]:+.2f}', str(r["rank"])]
                for r in shap["magnitude_contrasts"]]
    table(tables / "shap_magnitude_full.tex", ["Feature", "Cohort", "Magnitude", r"$\Delta$", r"Relative (\%)", "Rank"], fullrows, alignment="llrrrr", long=True)
    sensitivity_rows = [[LABELS[model], tex_escape(feature), f'{shap["models"][model]["complete_global_importance"][feature]:.5f}',
                         f'{shap["models"][model]["encoded_absolute_mass_sensitivity"][feature]:.5f}']
                        for model in SHAP_MODELS for feature in shap["models"][model]["original_features"]]
    table(tables / "shap_aggregation_sensitivity.tex", ["Model", "Feature", "Net magnitude", "Encoded mass"], sensitivity_rows, alignment="llrr", long=True)
    table(tables / "shap_consistency.tex", ["Model"] + [LABELS[m] for m in SHAP_MODELS],
          [[LABELS[a]] + [f'{shap["cross_model_top10_jaccard"][a][b]:.2f}' for b in SHAP_MODELS] for a in SHAP_MODELS])
    table(tables / "shap_stability.tex", ["Cohort"] + [COHORT_LABELS[c] for c in COHORTS],
          [[COHORT_LABELS[a]] + [f'{shap["cohort_top10_jaccard"][a][b]:.2f}' for b in COHORTS] for a in COHORTS])
    strategy_labels = dict(zip(PLANNED_STRATEGIES, ("None", "RUS", "ROS", "SMOTE", r"SM+Tomek$^{\dagger}$", "SMOTEENN", "Weights")))
    for dataset, short in (("ulb_2013", "ulb"), ("baf_base", "baf")):
        rows = [[strategy_labels[s]] + [f'{metrics_by_dataset[dataset][m][s][metric]:.3f}' for m in ("fttransformer", "catboost") for metric in ("PR-AUC", "F2")] for s in PLANNED_STRATEGIES]
        rows.append([r"\textit{Pop.\ SD (six)}"] + [rf'\textit{{{strategies["summaries"][f"{dataset}/{m}"][metric]["population_sd"]:.3f}}}' for m in ("fttransformer", "catboost") for metric in ("PR-AUC", "F2")])
        table(tables / f"robustness_{short}.tex", ["Strategy", "FT AP", r"FT $F_2$", "CB AP", r"CB $F_2$"], rows)


def generate_figure(output, summary):
    from article_interpretability import _plot_global, _pyplot
    destination = output / "shap_global.pdf"
    _plot_global(summary, destination, _pyplot())
    return destination


def install_private_assets(output):
    """Copy generated assets only; frozen results/publications are never targets."""
    destinations = (ROOT / "Overleaf/tables/supervisor", ROOT / "Article 1/tables/supervisor")
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for path in (output / "tables").glob("*.tex"):
            shutil.copy2(path, destination / path.name)
    for short in ("ulb", "baf"):
        shutil.copy2(output / f"tables/robustness_{short}.tex", ROOT / f"Article 1/tables/robustness_{short}.tex")
    for name in ("shap_consistency", "shap_stability"):
        shutil.copy2(output / f"tables/{name}.tex", ROOT / f"Article 1/tables/{name}.tex")
    for destination in (ROOT / "Overleaf/figures/baf/shap_global.pdf", ROOT / "Article 1/figs/shap_global.pdf"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "shap_global.pdf", destination)


def main(output=OUTPUT, install=False, figures=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "results_revision/20261015_supervisor"):
        raise ValueError("Use the new dated derivative directory, never the frozen archive")
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    qa_path = source(REVISION / "derived/reporting/generation_qa.json", sources)
    qa = read_json(qa_path)
    source(qa["manifest_path"], sources, qa["manifest_sha256"])
    selected = read_json(source(ROOT / "results/provenance/selected_runs.json", sources))
    shap = generate_shap(output, sources, qa)
    transfer = generate_transfer(output, sources, qa, selected)
    strategies, metrics_by_dataset = generate_strategy_summaries(output, sources, selected)
    generate_tables(output, shap, transfer, strategies, metrics_by_dataset)
    if figures:
        generate_figure(output, shap)
    # Sources are read again after generation to detect accidental mutation.
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"A source changed during postprocessing: {name}")
    write_json(output / "manifest.json", {"revision_id": "20261015_supervisor", "scope": "P7/P9/P10/P11/S11/S12/S13",
              "source_sha256": sources, "code_sha256": digest(Path(__file__)),
              "model_training": False, "model_inference": False, "new_shap_computation": False,
              "source_files_unchanged_after_generation": True,
              "output_sha256": {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    if install:
        if not figures:
            raise ValueError("Private asset installation requires --figures")
        install_private_assets(output)
    print(json.dumps({"output": str(output), "leaders": {m: shap["models"][m]["ranking"][:3] for m in SHAP_MODELS},
                      "jaccard": shap["cross_model_top10_jaccard"],
                      "focused_contrasts": [r for r in shap["magnitude_contrasts"] if r["feature"] in FOCUS]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--install-private-assets", action="store_true")
    args = parser.parse_args()
    main(args.output_dir, args.install_private_assets, args.figures)
