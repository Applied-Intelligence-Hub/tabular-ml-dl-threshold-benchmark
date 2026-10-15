"""Export the frozen thesis evidence without modifying training archives.

The author-side export reads the explicitly selected October 2026 archive. The
public release contains metrics, configurations, aggregate explanations, plots,
tables and hashes, not model weights or per-row predictions. ``--verify`` only
needs the exported release and the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRECTORIES = {
    "metrics", "figures", "tables", "provenance", "thresholds", "transfer",
    "controls", "interpretability",
}
SCHEMA_VERSION = 1
SUPERVISOR_REVISION_ID = "20261015_supervisor"
SUPERVISOR_NET_SHAP_POLICY = "mean_i(abs(sum_j_in_G(phi_ij)))"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def portable(value, root: Path):
    """Remove machine-specific prefixes while retaining logical archive IDs.

    Relative archive paths describe evidence retained by the author; they do not
    assert that omitted model/score files are distributed in the public release.
    """
    if isinstance(value, dict):
        return {portable(str(key), root): portable(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item, root) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if not isinstance(value, str):
        return value
    normal = value.replace("\\", "/")
    root_text = str(root.resolve()).replace("\\", "/").rstrip("/") + "/"
    normal = re.sub(re.escape(root_text), "", normal, flags=re.IGNORECASE)
    # External runtime locations are not useful reproducibility information.
    if re.search(r"(?i)(?:^|[\s\"'(])[a-z]:/", normal):
        return "[external local path omitted]"
    if normal.startswith("/") and not normal.startswith("//"):
        return "[external local path omitted]"
    return normal


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def source_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source is outside the project archive")
    return path


def checked_source(path: Path, expected: str | None = None) -> str:
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Pinned source hash mismatch: {path.name}")
    return actual


def _historical_public_export_name(logical: str) -> str | None:
    """Identify only public summaries referenced by the immutable derivative."""
    if logical == "results/interpretability/shap_summary.json":
        return logical.removeprefix("results/")
    if logical in {
        "results/thresholds/ulb_2013/fttransformer.json",
        "results/thresholds/baf_base/fttransformer.json",
    }:
        return logical.removeprefix("results/")
    if re.fullmatch(
        r"results/metrics/(?:ulb_2013|baf_base)/(?:logreg|rf|lgbm|catboost|fttransformer|ocsvm)/"
        r"(?:none|rus|ros|smote|smote_tomek|smoteenn|weights)/metrics_full_precision\.json", logical
    ):
        return logical.removeprefix("results/")
    return None


def resolve_pinned_author_source(root: Path, value: str | Path, expected: str) -> Path:
    """Resolve a logical immutable pin across its one approved public overlay.

    Only the former public SHAP summary, exact full-precision metric paths and
    the two named FT threshold studies have historical aliases. Their replacement
    and preserved bytes must both be owned by the current corrected export
    manifest, with matching supersession provenance. No numeric equivalence or
    regenerated JSON substitutes for pinned bytes; every other change fails.
    """
    root = root.resolve()
    current = source_path(str(value), root)
    actual = digest(current)
    if actual == expected:
        return current
    logical = current.relative_to(root).as_posix()
    logical_export = _historical_public_export_name(logical)
    if logical_export is None:
        raise ValueError(f"Pinned source hash mismatch: {current.name}")
    manifest = read_json(source_path("results/provenance/release_manifest.json", root))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("exporter") != "export_public_evidence.py"
            or manifest.get("supervisor_revision_id") != SUPERVISOR_REVISION_ID):
        raise ValueError("Historical public source alias requires the approved corrected public manifest")
    historical_export = "provenance/historical_pre_supervisor/" + logical_export
    files = manifest.get("files", {})
    current_record = files.get(logical_export, {})
    old_record = files.get(historical_export, {})
    prior = current_record.get("supersedes_export_provenance", {})
    original = old_record.get("original_export_provenance", {})
    if (current_record.get("public_sha256") != actual
            or current_record.get("bytes") != current.stat().st_size
            or old_record.get("public_sha256") != expected
            or prior.get("public_sha256") != expected
            or original != prior or not prior
            or old_record.get("bytes") != prior.get("bytes")):
        raise ValueError("Historical public source alias lacks consistent replacement/original provenance")
    historical = source_path("results/" + historical_export, root)
    checked_source(historical, expected)
    if historical.stat().st_size != old_record["bytes"]:
        raise ValueError("Historical public source bytes differ from their public pin")
    return historical


def recover_metrics(run: Path):
    """Recover precision from saved scores and authoritative confusion counts.

    No model is fitted, no threshold is selected on TEST, and historical rounded
    thresholds are never used to replace their original confusion counts.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from evaluation.metrics import TOP_K_POLICY, precision_recall_at_k

    config = read_json(run / "config.json")
    metrics = read_json(run / "metrics_test.json")
    labels = np.load(run / "y_test.npy", allow_pickle=False)
    scores = np.load(run / "y_test_scores.npy", allow_pickle=False)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid saved TEST arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TEST labels must contain both binary classes")
    tp, fp, tn, fn = (int(metrics[name]) for name in ("TP", "FP", "TN", "FN"))
    if tp + fp + tn + fn != len(labels) or tp + fn != int(labels.sum()):
        raise ValueError("Confusion counts and TEST population differ")
    threshold = float(config.get("threshold_exact", metrics["threshold"]))
    if "threshold_exact" in config:
        prediction = scores >= threshold
        counts = (int(np.sum(prediction & (labels == 1))), int(np.sum(prediction & (labels == 0))),
                  int(np.sum(~prediction & (labels == 0))), int(np.sum(~prediction & (labels == 1))))
        if counts != (tp, fp, tn, fn):
            raise ValueError("Frozen exact threshold does not reproduce saved counts")
    full = dict(metrics)
    for name, value in (("PR-AUC", average_precision_score(labels, scores)),
                        ("ROC-AUC", roc_auc_score(labels, scores))):
        if not np.isclose(value, metrics[name], rtol=0, atol=1e-6):
            raise ValueError(f"Saved {name} differs from score evidence")
        full[name] = float(value)
    full["F1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    full["F2"] = 5 * tp / (5 * tp + fp + 4 * fn) if tp else 0.0
    full["precision"] = tp / (tp + fp) if tp + fp else 0.0
    full["recall"] = tp / (tp + fn)
    full["alert_rate"] = (tp + fp) / len(labels)
    full["FP/TP"] = fp / tp if tp else float("inf")
    full["threshold"] = threshold
    k = int(metrics["k_used"])
    if not 1 <= k <= len(labels):
        raise ValueError("Invalid saved workload budget")
    full["precision_at_k"], full["recall_at_k"], _ = precision_recall_at_k(labels, scores, k=k)
    full["top_k_policy"] = TOP_K_POLICY
    full["brier_score"] = None
    if config["model"] != "ocsvm":
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Supervised scores are outside [0, 1]")
        full["brier_score"] = float(brier_score_loss(labels, scores))
    full["test_samples"] = int(len(labels))
    full["test_fraud_samples"] = int(labels.sum())
    return full


def compact_curve(curve, max_points=3000):
    """Bound plot data size; reported AP is always computed from full scores."""
    if max_points < 2:
        raise ValueError("Plotting subset needs at least two endpoint slots")
    precision = curve.get("precision", curve.get("precisions"))
    recall = curve.get("recall", curve.get("recalls"))
    if precision is None or recall is None or len(precision) != len(recall) or not precision:
        raise ValueError("Invalid precision--recall coordinates")
    count = len(precision)
    indices = sorted({round(i * (count - 1) / (min(max_points, count) - 1))
                      for i in range(min(max_points, count))}) if count > 1 else [0]
    return {"precision": [precision[i] for i in indices],
            "recall": [recall[i] for i in indices],
            "original_points": count, "retained_points": len(indices),
            "representation": "uniform-index subset for plotting only; do not recompute AP from this subset"}


class ExportPlan:
    """Collect and validate every destination before writing any export file."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files: dict[str, bytes] = {}
        self.provenance: dict[str, dict] = {}

    def add(self, target: str, data: bytes, sources=(), transformation="byte-identical copy"):
        path = Path(target)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                or path.parts[0] not in ALLOWED_DIRECTORIES):
            raise ValueError("Export target is outside the public evidence directories")
        target = path.as_posix()
        if target in self.files:
            raise ValueError(f"Duplicate export target: {target}")
        self.files[target] = data
        self.provenance[target] = {
            "public_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "transformation": transformation,
            "sources": [{"archive_path": portable(str(p), self.root), "sha256": digest(p)}
                        for p in sources],
        }

    def add_json(self, target, data, sources=(), transformation="JSON: machine-specific paths normalised; numeric values preserved"):
        self.add(target, json_bytes(portable(data, self.root)), sources, transformation)

    def copy(self, target, source: Path, expected=None):
        checked_source(source, expected)
        self.add(target, source.read_bytes(), (source,))

    def reviewed_overlay(self, target, data, sources, transformation):
        """Replace one planned asset, preserving the old planned bytes explicitly."""
        prior_provenance = None
        if target in self.files:
            historical = "provenance/historical_pre_supervisor/" + target
            prior_provenance = self.provenance[target]
            self.add(historical, self.files[target], (), "Historical pre-supervisor export asset; not current reporting")
            self.provenance[historical]["original_export_provenance"] = self.provenance[target]
            del self.files[target]
            del self.provenance[target]
        self.add(target, data, sources, transformation)
        if prior_provenance is not None:
            self.provenance[target]["supersedes_export_provenance"] = prior_provenance

    def write(self, output: Path):
        output = output.resolve()
        manifest_file = output / "provenance" / "release_manifest.json"
        previous = read_json(manifest_file) if manifest_file.is_file() else None
        old_files = previous.get("files", {}) if previous else {}
        # Never overwrite an unrelated file, including a previous release altered
        # by a human. Directories not owned by this plan remain untouched.
        for target in self.files:
            destination = output / target
            if target == "provenance/release_manifest.json":
                continue  # Its exporter identity is checked separately below.
            if destination.is_file():
                record = old_files.get(target)
                if not record or digest(destination) != record["public_sha256"]:
                    raise FileExistsError(f"Unowned or modified export destination: {target}")
        if manifest_file.exists() and (not previous or previous.get("exporter") != "export_public_evidence.py"):
            raise FileExistsError("Unowned release manifest")
        for target, data in self.files.items():
            destination = output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return manifest_file


def validated_supervisor_inputs(root: Path, supervisor_root: Path | None):
    """Fail closed unless every approved dated derivative is explicitly selected.

    This only reads/hash-checks files. It neither generates missing evidence nor
    exports anything. No raw model, attribution or prediction array is selected
    for publication, even when its hash is verified as an author-side input.
    """
    root = root.resolve()
    if supervisor_root is None:
        raise ValueError("Corrected export requires an explicit supervisor root; old-only releases are disabled")
    folder = supervisor_root.resolve()
    if not folder.is_relative_to(root / "results_revision"):
        raise ValueError("Supervisor inputs must lie within this project's dated results_revision archive")
    files = {}

    def selected(relative, expected=None):
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Supervisor manifest output escapes its selected root")
        checked_source(path, expected)
        files[relative] = path
        return path

    def author_sources(records):
        if not isinstance(records, dict) or not records:
            raise ValueError("Supervisor evidence lacks source hash provenance")
        for name, expected in records.items():
            resolve_pinned_author_source(root, name, expected)

    interp_manifest = read_json(selected("interpretability_transfer/manifest.json"))
    if (interp_manifest.get("revision_id") != SUPERVISOR_REVISION_ID
            or interp_manifest.get("model_training") is not False
            or interp_manifest.get("model_inference") is not False
            or interp_manifest.get("new_shap_computation") is not False
            or interp_manifest.get("source_files_unchanged_after_generation") is not True):
        raise ValueError("The selected SHAP/transfer manifest is not the approved postprocessing revision")
    author_sources(interp_manifest.get("source_sha256"))
    required_interp = (
        "portable_shap_summary.json", "shap_summary.json", "shap_global.pdf",
        "strategy_summary.json", "transfer_summary.json", "transfer_overlap_pairs.csv",
        "transfer_alert_workload.csv", "shap_magnitude_contrasts.csv",
        "shap_aggregation_logreg.csv", "shap_aggregation_lgbm.csv", "shap_aggregation_catboost.csv",
        "tables/robustness_ulb.tex", "tables/robustness_baf.tex", "tables/shap_consistency.tex",
        "tables/shap_stability.tex", "tables/shap_aggregation_sensitivity.tex",
        "tables/shap_magnitude_compact.tex", "tables/shap_magnitude_full.tex",
        "tables/rf_overlap.tex", "tables/transfer_overlap_full.tex",
        "tables/transfer_alert_counts.tex", "tables/transfer_alert_rates.tex")
    for name in required_interp:
        expected = interp_manifest.get("output_sha256", {}).get(name)
        if not expected:
            raise ValueError(f"Supervisor manifest omits the corrected reporting asset: {name}")
        selected("interpretability_transfer/" + name, expected)
    shap = read_json(files["interpretability_transfer/shap_summary.json"])
    if (shap.get("policy") != SUPERVISOR_NET_SHAP_POLICY
            or set(shap.get("models", {})) != {"logreg", "lgbm", "catboost"}
            or any(len(value.get("complete_global_importance", {})) != 30 for value in shap["models"].values())):
        raise ValueError("Corrected SHAP must contain complete net-group original-feature vectors")
    strategies = read_json(files["interpretability_transfer/strategy_summary.json"])
    if (strategies.get("planned_strategies") != ["none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights"]
            or strategies.get("effective_conditions") != ["none", "rus", "ros", "smote", "smoteenn", "weights"]
            or strategies.get("primary_configurations_retained") != 72):
        raise ValueError("Corrected strategy summaries must preserve seven labels and summarise six conditions")
    threshold_qa = read_json(selected("threshold_diagnostics/qa/diagnostics.json"))
    if (threshold_qa.get("status") != "complete"
            or threshold_qa.get("all_original_input_hashes_unchanged") is not True
            or threshold_qa.get("shared_tables_identical_across_documents") is not True
            or threshold_qa.get("analytic_transformations") != 15
            or threshold_qa.get("paired_bootstrap_requested") != 1000):
        raise ValueError("Approved threshold diagnostics have not passed their complete QA")
    inventory = {record["path"]: record["sha256"] for record in threshold_qa["canonical_output_inventory"]}
    required_threshold = (
        "basic_verified/diagnostics.json", "basic_verified/precision.csv",
        "basic_verified/lgbm_ulb_cut_sensitivity.csv", "paired_ap/diagnostics.json",
        "ft_transport/diagnostics.json", "ft_transport/sensitivity.csv")
    reports = {}
    for name in required_threshold:
        path = source_path(str(folder / "threshold_diagnostics" / name), root)
        expected = inventory.get(path.relative_to(root).as_posix())
        if not expected:
            raise ValueError(f"Canonical threshold inventory omits {name}")
        selected("threshold_diagnostics/" + name, expected)
        if name.endswith("diagnostics.json"):
            report = read_json(path)
            if report.get("status") != "complete":
                raise ValueError("A required diagnostic is incomplete")
            author_sources({record["path"]: record["sha256"] for record in report["sources"]})
            reports[name.split("/")[0]] = report
    if (len(reports["basic_verified"].get("precision", [])) != 48
            or len(reports["basic_verified"].get("prior_brier", [])) != 20
            or reports["paired_ap"].get("requested") != 1000
            or reports["paired_ap"].get("valid", 0) < 1
            or reports["ft_transport"].get("A_validation_reproduction", {}).get("decisions_changed") != 0):
        raise ValueError("The approved diagnostic coverage or A/VAL fidelity gate is missing")
    table_names = ("supervisor_ft_transport.tex", "supervisor_precision_ulb.tex", "supervisor_precision_baf.tex",
                   "supervisor_threshold_folds.tex", "supervisor_threshold_folds_full.tex",
                   "supervisor_lgbm_cut_sensitivity.tex", "supervisor_prior_brier.tex", "supervisor_paired_ap.tex")
    private_tables = {}
    for name in table_names:
        path = source_path("Overleaf/tables/" + name, root)
        expected = inventory.get(path.relative_to(root).as_posix())
        if not expected:
            raise ValueError(f"Threshold QA omits its generated reporting table: {name}")
        checked_source(path, expected)
        private_tables[name] = path
    # This file has aggregate rows only; no per-observation arrays are exported.
    prior_csv = selected("threshold_diagnostics/qa/prior_brier_full.csv")
    with prior_csv.open(encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    expected_rows = []
    for row in reports["basic_verified"]["prior_brier"]:
        expected_rows.append({"model": row["model"], "strategy": row["strategy"], "constant_c": row["constant_c"],
                              "Brier_raw": row["raw"]["brier"], "Brier_adjusted": row["transformed"]["brier"],
                              "AP_raw": row["raw"]["average_precision"], "AP_adjusted": row["transformed"]["average_precision"],
                              "ROC_raw": row["raw"]["roc_auc"], "ROC_adjusted": row["transformed"]["roc_auc"],
                              "threshold_raw": row["threshold_raw_available"], "threshold_adjusted": row["threshold_transformed"],
                              "count_replay": row["available_cut_reproduces_historical_counts"],
                              "coalescences_float64": row["distinct_raw_scores_coalesced_float64"],
                              "AP_difference_float64": row["AP_numerical_difference"],
                              "ROC_difference_float64": row["ROC_AUC_numerical_difference"]})
    if prior_rows != [{key: str(value) for key, value in row.items()} for row in expected_rows]:
        raise ValueError("Aggregate prior CSV differs from its pinned canonical diagnostic")
    topk = read_json(selected("topk_claims/topk_claims_audit.json"))
    if (topk.get("policy") != "expected_uniform_boundary_v1" or topk.get("cell_count") != 110
            or topk.get("original_inputs_unchanged") is not True
            or topk.get("training_inference_or_test_selection") is not False):
        raise ValueError("Corrected top-k evidence is incomplete or uses another policy")
    author_sources(topk.get("sources_sha256"))
    return {"root": folder, "files": files, "tables": private_tables,
            "reports": reports, "shap": shap, "strategies": strategies}


def apply_supervisor_overlay(plan: ExportPlan, bundle):
    """Overlay only the allowlisted aggregate assets, preserving old planned bytes."""
    files = bundle["files"]
    mapping = {
        "interpretability_transfer/portable_shap_summary.json": "interpretability/shap_summary.json",
        "interpretability_transfer/shap_summary.json": "interpretability/supervisor_net_shap_complete.json",
        "interpretability_transfer/shap_global.pdf": "figures/baf/shap_global.pdf",
        "interpretability_transfer/strategy_summary.json": "provenance/strategy_summary.json",
        "interpretability_transfer/transfer_summary.json": "transfer/supervisor_summary.json",
        "interpretability_transfer/manifest.json": "provenance/supervisor_interpretability_manifest.json",
        "topk_claims/topk_claims_audit.json": "provenance/supervisor_topk_claims_audit.json",
        "threshold_diagnostics/qa/diagnostics.json": "provenance/supervisor_threshold_qa.json",
        "threshold_diagnostics/basic_verified/diagnostics.json": "controls/supervisor_threshold_diagnostics.json",
        "threshold_diagnostics/basic_verified/precision.csv": "thresholds/supervisor_precision.csv",
        "threshold_diagnostics/basic_verified/lgbm_ulb_cut_sensitivity.csv": "thresholds/supervisor_lgbm_ulb_cut_sensitivity.csv",
        "threshold_diagnostics/paired_ap/diagnostics.json": "controls/supervisor_paired_ap.json",
        "threshold_diagnostics/ft_transport/diagnostics.json": "thresholds/supervisor_ft_transport.json",
        "threshold_diagnostics/ft_transport/sensitivity.csv": "thresholds/supervisor_ft_transport.csv",
        "threshold_diagnostics/qa/prior_brier_full.csv": "controls/supervisor_prior_brier_full.csv",
    }
    for name in ("transfer_overlap_pairs", "transfer_alert_workload"):
        mapping[f"interpretability_transfer/{name}.csv"] = f"transfer/{name}.csv"
    for name in ("shap_magnitude_contrasts", "shap_aggregation_logreg", "shap_aggregation_lgbm", "shap_aggregation_catboost"):
        mapping[f"interpretability_transfer/{name}.csv"] = f"interpretability/{name}.csv"
    for name in ("rf_overlap", "shap_consistency", "shap_stability", "shap_aggregation_sensitivity",
                 "shap_magnitude_compact", "shap_magnitude_full", "transfer_overlap_full",
                 "transfer_alert_counts", "transfer_alert_rates"):
        mapping[f"interpretability_transfer/tables/{name}.tex"] = f"tables/baf/{name}.tex"
    for short in ("ulb", "baf"):
        mapping[f"interpretability_transfer/tables/robustness_{short}.tex"] = f"tables/{short}/transformer_robustness.tex"
    for relative, target in mapping.items():
        path = files[relative]
        data = json_bytes(portable(read_json(path), plan.root)) if path.suffix == ".json" else path.read_bytes()
        plan.reviewed_overlay(target, data, (path,), "Approved dated supervisor derivative; current reporting; originals preserved separately")
    for name, path in bundle["tables"].items():
        plan.reviewed_overlay("tables/supervisor/" + name, path.read_bytes(), (path,), "Generated aggregate table body; no private manuscript prose")
    # Factorial cell values are unchanged, but the duplicate treatment must
    # also be identified in the exported presentation, not only in prose.
    for dataset in ("ulb", "baf"):
        for metric in ("f1", "f2", "prauc", "rocauc"):
            target = f"tables/{dataset}/factorial_{metric}.tex"
            content = plan.files[target].decode("utf-8")
            lines = content.splitlines()
            captions = [index for index, line in enumerate(lines) if line.startswith(r"\caption{")]
            if len(captions) != 1 or not lines[captions[0]].endswith("}"):
                raise ValueError("Factorial caption layout changed; explicitly regenerate the reviewed table before export")
            caption_index = captions[0]
            lines[caption_index] = (lines[caption_index][:-1]
                                    + r" $^{*}$SMOTE+Tomek duplicates SMOTE in the audited runs; all seven strategy results remain visible.}")
            content = "\n".join(lines).replace("SM+T &", "SM+T$^{*}$ &") + "\n"
            plan.reviewed_overlay(target, content.encode("utf-8"),
                                  (files["interpretability_transfer/strategy_summary.json"],),
                                  "Presentation-only duplicate-treatment marking; factorial cell values unchanged")
    shap = bundle["shap"]
    stability = {"policy": SUPERVISOR_NET_SHAP_POLICY, "variant_keys": list(shap["cohorts"]),
                 "jaccard_matrix": shap["cohort_top10_jaccard"],
                 "global_importance": {key: value["complete_global_importance"] for key, value in shap["cohorts"].items()},
                 "rankings": {key: value["ranking"] for key, value in shap["cohorts"].items()},
                 "magnitude_contrasts": shap["magnitude_contrasts"],
                 "scope": "Frozen LGBM; filtered Variants; membership is not magnitude or rank invariance"}
    plan.reviewed_overlay("interpretability/shap_variant_stability.json", json_bytes(stability),
                          (files["interpretability_transfer/shap_summary.json"],), "Current net-group stability and complete magnitude contrasts")
    # Correct metadata only; do not recompute or replace any threshold/count.
    from revision_thresholds import precision_constraint_scope
    for dataset in ("ulb_2013", "baf_base"):
        target = f"thresholds/{dataset}/fttransformer.json"
        study = json.loads(plan.files[target])
        study["historical_precision_constraint_scope"] = study.get("precision_constraint_scope")
        study["precision_constraint_scope"] = precision_constraint_scope("fttransformer")
        plan.reviewed_overlay(target, json_bytes(study), (), "Metadata-only correction: FT single holdout and transport; thresholds/counts unchanged")


def preserve_pinned_public_sources(plan: ExportPlan, bundle):
    """Retain exact former public inputs still pinned by the dated derivative.

    The immutable derivative records the pre-overlay public summaries among its
    sources. Re-exporting those public paths changes presentation metadata and
    net SHAP reporting, not the source identities. Carry their original bytes and
    explicit supersession bridge forward in every later public export so a
    clean rerun can still verify all of the original derivative's input hashes.
    """
    source_manifest = read_json(bundle["files"]["interpretability_transfer/manifest.json"])
    source_pins = dict(source_manifest["source_sha256"])
    # These are the canonical reports accepted by validated_supervisor_inputs,
    # not arbitrary directories or a newest-run scan. Their provenance retains
    # all source paths/hashes, including former public baseline metric summaries.
    for phase in ("basic_verified", "paired_ap", "ft_transport"):
        for record in bundle.get("reports", {}).get(phase, {}).get("sources", []):
            logical = source_path(record["path"], plan.root).relative_to(plan.root).as_posix()
            expected = record["sha256"]
            if logical in source_pins and source_pins[logical] != expected:
                raise ValueError(f"Immutable diagnostic source identities disagree: {logical}")
            source_pins[logical] = expected
    prior_release = None
    for logical, expected in source_pins.items():
        public_name = _historical_public_export_name(logical)
        if public_name is None:
            continue
        if public_name not in plan.files:
            raise ValueError(f"A pinned public input is absent from the new export plan: {logical}")
        if hashlib.sha256(plan.files[public_name]).hexdigest() == expected:
            continue
        source = resolve_pinned_author_source(plan.root, logical, expected)
        historical = "provenance/historical_pre_supervisor/" + public_name
        if historical in plan.files:
            if hashlib.sha256(plan.files[historical]).hexdigest() != expected:
                raise ValueError(f"Planned historical public source differs from its immutable pin: {logical}")
            # reviewed_overlay already retained the original SHAP provenance.
            original = plan.provenance[historical].get("original_export_provenance", {})
        else:
            if prior_release is None:
                prior_release = read_json(plan.root / "results/provenance/release_manifest.json")
            prior_files = prior_release.get("files", {})
            current_record = prior_files.get(public_name, {})
            original = (current_record if current_record.get("public_sha256") == expected
                        else prior_files.get(historical, {}).get("original_export_provenance", {}))
            if original.get("public_sha256") != expected or original.get("bytes") != source.stat().st_size:
                raise ValueError(f"Original public source provenance is not pinned: {logical}")
            plan.copy(historical, source, expected)
            plan.provenance[historical]["transformation"] = "Byte-identical historical input pinned by the immutable supervisor derivative; not current reporting"
            plan.provenance[historical]["original_export_provenance"] = original
        if original.get("public_sha256") != expected:
            raise ValueError(f"Historical public source lost its original provenance: {logical}")
        plan.provenance[public_name]["supersedes_export_provenance"] = original


def build_export(root: Path, revision: Path, thesis: Path, *, release_id: str,
                 supervisor_root: Path | None = None):
    if not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_id):
        raise ValueError("An explicit non-empty release ID is required")
    thesis = thesis.resolve()
    if not thesis.is_file() or thesis.read_bytes()[:5] != b"%PDF-":
        raise ValueError("An explicit existing thesis PDF is required")
    supervisor = validated_supervisor_inputs(root, supervisor_root)
    qa_file = revision / "derived" / "reporting" / "generation_qa.json"
    manifest_file = revision / "revision_manifest.json"
    qa, original = read_json(qa_file), read_json(manifest_file)
    checked_source(manifest_file, qa["manifest_sha256"])
    if qa["required_runs"] != 72 or len(qa["source_runs"]) != 72:
        raise ValueError("The complete 72-cell primary evidence grid is required")
    plan = ExportPlan(root)
    metrics, selections = {}, {}
    for key, record in sorted(qa["source_runs"].items()):
        run = source_path(record["run_dir"], root)
        selection = original.get("runs", {}).get(key)
        if selection is None:
            selection = original.get("historical_runs", {}).get(key)
        if (selection is None or source_path(selection["run_dir"], root) != run
                or selection["config_sha256"] != record["config_sha256"]):
            raise ValueError(f"Reporting source is not the manifest-pinned primary selection: {key}")
        for name, field in (("config.json", "config_sha256"), ("metrics_test.json", "metrics_sha256"),
                            ("metrics_cv.json", "metrics_cv_sha256"), ("y_test.npy", "y_test_sha256"),
                            ("y_test_scores.npy", "y_test_scores_sha256"),
                            ("pr_curve_data.json", "pr_curve_data_sha256")):
            checked_source(run / name, record[field])
        full = recover_metrics(run)
        if full["test_samples"] != record["test_samples"] or full["test_fraud_samples"] != record["test_fraud_samples"]:
            raise ValueError("Reporting QA and source TEST population differ")
        metrics[key] = full
        prefix = "metrics/" + key
        plan.add_json(prefix + "/config.json", read_json(run / "config.json"), (run / "config.json",))
        plan.add_json(prefix + "/metrics_saved.json", read_json(run / "metrics_test.json"), (run / "metrics_test.json",))
        plan.add_json(prefix + "/metrics_full_precision.json", full,
                      tuple(run / name for name in ("config.json", "metrics_test.json", "y_test.npy", "y_test_scores.npy")),
                      "AP/ROC-AUC/Brier/top-k recomputed from full frozen TEST arrays; operating metrics from original integer counts; no fitting/TEST threshold selection")
        plan.add_json(prefix + "/metrics_cv.json", read_json(run / "metrics_cv.json"), (run / "metrics_cv.json",))
        if key.endswith("/none"):
            curve = compact_curve(read_json(run / "pr_curve_data.json"))
            curve["average_precision_full_scores"] = full["PR-AUC"]
            plan.add_json("metrics/curves/" + key.replace("/none", "") + ".json", curve,
                          (run / "pr_curve_data.json", run / "y_test.npy", run / "y_test_scores.npy"),
                          "At most 3000 original PR coordinates retained for plotting; AP comes from complete TEST scores")
        selections[key] = {**record, "archive_run_id": portable(str(run), root),
                           "public_config": "results/" + prefix + "/config.json",
                           "public_metrics": "results/" + prefix + "/metrics_full_precision.json",
                           "model_weights_and_row_predictions_distributed": False}
        selections[key].pop("run_dir", None)
    precision_basis = {**qa["metric_precision_basis"],
                       "top_k_policy": "expected_uniform_boundary_v1",
                       "strategy_population_sd": "Six effective conditions (exclude audited SMOTE+Tomek), ddof=0, full precision; seven executed labels retained"}
    plan.add_json("metrics/primary_metrics.json", {"cells": metrics, "precision_basis": precision_basis}, (qa_file,))
    fields = ["cell", "PR-AUC", "ROC-AUC", "F1", "F2", "precision", "recall", "threshold", "brier_score",
              "TP", "FP", "TN", "FN", "alert_rate", "FP/TP", "precision_at_k", "recall_at_k", "k_used",
              "test_samples", "test_fraud_samples"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator="\n")
    writer.writeheader()
    for key, cell in sorted(metrics.items()):
        writer.writerow({field: portable(key if field == "cell" else cell.get(field), root) for field in fields})
    plan.add("metrics/primary_metrics.csv", buffer.getvalue().encode(), (qa_file,), "Full-precision exported metric grid rendered as CSV")
    plan.add_json("provenance/selected_runs.json", selections, (qa_file, manifest_file))
    # Preserve original selection identities, including historical BAF final
    # models reused by validation-only recovery, without copying obsolete ULB.
    historical = {k: v for k, v in original.get("historical_runs", {}).items()
                  if k.startswith("baf_base/") and k.split("/")[1] != "fttransformer"}
    plan.add_json("provenance/historical_baf_selection.json", historical, (manifest_file,))
    for key, record in historical.items():
        run = source_path(record["run_dir"], root)
        checked_source(run / "config.json", record["config_sha256"])
        plan.add_json("provenance/historical_baf/" + key.replace("baf_base/", "") + "/config.json",
                      read_json(run / "config.json"), (run / "config.json",))
    plan.add_json("provenance/dataset_protocol.json", original["datasets"], (manifest_file,))
    plan.add_json("provenance/reporting_source_audit.json", qa, (qa_file,))
    for relative, expected in qa["output_sha256"].items():
        path = Path(relative.replace("\\", "/"))
        plan.copy(path.as_posix(), qa_file.parent / path, expected)
    for key, record in qa["threshold_studies"].items():
        path = source_path(record["path"], root)
        checked_source(path, record["sha256"])
        plan.add_json("thresholds/" + key + ".json", read_json(path), (path,))
    for model, record in qa["transfer_sources"].items():
        path = source_path(record["path"], root)
        checked_source(path, record["sha256"])
        plan.add_json("transfer/" + model + "/source_audit.json", read_json(path), (path,))
        for variant, variant_record in record["variants"].items():
            path = source_path(variant_record["path"], root)
            checked_source(path, variant_record["sha256"])
            plan.add_json("transfer/" + model + "/" + variant + "/metrics_test.json", read_json(path), (path,))
    partition = source_path(original["transfer_partitions"]["path"], root)
    checked_source(partition, original["transfer_partitions"]["sha256"])
    plan.add_json("transfer/partition_manifest.json", read_json(partition), (partition,))
    paired = revision / "derived" / "paired" / "paired_analysis.json"
    checked_source(paired, qa["paired_analysis"]["sha256"])
    for path in sorted(paired.parent.glob("*.json")):
        plan.add_json("controls/" + path.name, read_json(path), (path,))
    sensitivity_manifest = revision / "sensitivity" / "revision_manifest.json"
    checked_source(sensitivity_manifest, qa["paired_analysis"]["sensitivity_manifest"]["sha256"])
    plan.add_json("controls/source_manifest.json", read_json(sensitivity_manifest), (sensitivity_manifest,))
    plan.copy("tables/baf/sensitivity_controls.tex", revision / "derived" / "sensitivity_text" / "sensitivity_tables.tex")
    interp = revision / "derived" / "interpretability"
    explanation = read_json(interp / "interpretability_qa.json")
    # The complete attribution matrices remain local. Public aggregates preserve
    # all 30 original-feature importances, log-odds units, grouping and cases.
    explanation.pop("outputs", None)
    plan.add_json("interpretability/shap_summary.json", explanation, (interp / "interpretability_qa.json",))
    stability = interp / "variant_stability" / "shap_variant_stability.json"
    checked_source(stability, original["interpretability"]["variant_stability_sha256"])
    plan.add_json("interpretability/shap_variant_stability.json", read_json(stability), (stability,))
    attention_dir = interp / "attention"
    attention_file = attention_dir / "attention_summary.json"
    attention = read_json(attention_file)
    plan.add_json("interpretability/attention_summary.json", attention, (attention_file,))
    for path in sorted(attention_dir.glob("*.png")):
        plan.copy("figures/baf/" + path.name, path)
    for name in ("baf_absence_codes.json", "sampler_identity.json", "lgbm_score_diagnostics.json",
                 "execution_provenance_notes.json"):
        path = revision / "audit" / name
        plan.add_json("provenance/" + name, read_json(path), (path,))
    apply_supervisor_overlay(plan, supervisor)
    preserve_pinned_public_sources(plan, supervisor)
    limits = (
        "# Evidence release\n\n"
        "This release accompanies the final October 2026 thesis. It contains the complete 72-cell primary metric grid, "
        "frozen configurations, validation summaries, 12 threshold studies, corrected BAF transfer metrics, predefined "
        "sensitivity controls, aggregate interpretability evidence, and thesis tables/figures. The explicitly selected "
        "supervisor revision adds threshold transport/precision/fold/prior and paired-AP diagnostics, net-group SHAP "
        "summaries and six-condition strategy dispersion. Replaced pre-supervisor assets remain marked historical; "
        "the current results are not uniformly numerically identical to the preceding release.\n\n"
        "The original manifest and reporting QA are immutable author-side records; their SHA-256 identities are pinned "
        "in release_manifest.json. The original manifest's administrative completion flags were not rewritten. The release "
        "inventory instead verifies the selected completed evidence. Machine-specific path prefixes are normalised; both "
        "the original and public file hashes are recorded. Archive-relative paths refer to author-retained material, not "
        "files promised in this public release. Historical BAF final-model selection is explicit.\n\n"
        "Per-row TEST/validation predictions, model weights, raw attribution matrices, local logs, checkpoints and "
        "editorial records are not distributed here. Hashes identify that evidence but are not a substitute for its "
        "contents. Accordingly this bundle supports inspection and verification of reported metrics, tables, plots and "
        "provenance, not independent exact recomputation of historical fitted models from this bundle alone. Compact PR "
        "coordinates are for plotting only; reported average precision is recovered from complete frozen scores.\n\n"
        "Verify public file integrity locally with `python src/export_public_evidence.py --verify`. This is not a "
        "cross-computer reproducibility claim. The author-side export requires the retained local result archives.\n"
    )
    plan.add("provenance/README.md", limits.encode(), (), "Release scope and limitations")
    release = {
        "schema_version": SCHEMA_VERSION, "exporter": "export_public_evidence.py", "release_id": release_id,
        "evidence_status": "selected frozen evidence plus explicitly validated approved supervisor-derivative overlay; final PDF identity separately selected",
        "original_manifest_sha256": digest(manifest_file), "original_reporting_qa_sha256": digest(qa_file),
        "thesis_pdf_sha256": digest(thesis), "thesis_public_path": "publications/thesis.pdf",
        "thesis_source_filename": thesis.name,
        "reference_commit": original["reference_commit"], "primary_cell_count": len(metrics),
        "threshold_study_count": len(qa["threshold_studies"]), "transfer_model_count": len(qa["transfer_sources"]),
        "original_reporting_output_count": len(qa["output_sha256"]),
        "reporting_output_count": sum(name.startswith(("tables/", "figures/")) for name in plan.files),
        "supervisor_revision_id": SUPERVISOR_REVISION_ID,
        "supervisor_source_root": portable(str(supervisor["root"]), root),
        "supervisor_inputs_sha256": {name: digest(path) for name, path in supervisor["files"].items()},
        "omitted": ["raw datasets", "per-row scores/labels/indices", "weights/checkpoints", "raw SHAP matrices",
                    "editorial QA", "private writing sources", "temporary files", "locks/logs"],
        "normalisation": "Archive-relative source identifiers; original and public hashes differ for normalised JSON",
        "files": dict(sorted(plan.provenance.items())),
    }
    # The manifest deliberately does not hash itself.
    plan.files["provenance/release_manifest.json"] = json_bytes(release)
    return plan, release


def verify_primary_metrics(document, expected_cells: int):
    """Check the public aggregate independently of private score archives."""
    cells = document.get("cells", {})
    if expected_cells != 72 or len(cells) != expected_cells:
        raise ValueError("Public primary metric grid must contain all 72 cells")
    for key, cell in cells.items():
        names = ("TP", "FP", "TN", "FN", "test_samples", "test_fraud_samples")
        if any(type(cell.get(name)) is not int or cell[name] < 0 for name in names):
            raise ValueError(f"Invalid integer population/count evidence: {key}")
        tp, fp, tn, fn = (cell[name] for name in ("TP", "FP", "TN", "FN"))
        total, positives = cell["test_samples"], cell["test_fraud_samples"]
        if (total != tp + fp + tn + fn or positives != tp + fn
                or not 0 < positives < total):
            raise ValueError(f"Public confusion counts and TEST population differ: {key}")
        expected = {
            "F1": 2 * tp / (2 * tp + fp + fn),
            "F2": 5 * tp / (5 * tp + fp + 4 * fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / positives,
            "alert_rate": (tp + fp) / total,
        }
        for name, value in expected.items():
            actual = cell.get(name)
            if (type(actual) not in (int, float) or not math.isfinite(actual)
                    or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Public count-derived {name} is inconsistent: {key}")
    return len(cells)


def verify_release(output: Path, *, project_root: Path | None = None):
    output = output.resolve()
    manifest = read_json(output / "provenance" / "release_manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["exporter"] != "export_public_evidence.py":
        raise ValueError("Unsupported public release schema")
    for relative, record in manifest["files"].items():
        destination = (output / relative).resolve()
        if not destination.is_relative_to(output):
            raise ValueError("Release inventory path escapes output root")
        checked_source(destination, record["public_sha256"])
        if destination.stat().st_size != record["bytes"]:
            raise ValueError("Public file size changed")
    primary_path = output / "metrics" / "primary_metrics.json"
    arithmetic_cells = 0
    if "metrics/primary_metrics.json" in manifest["files"]:
        arithmetic_cells = verify_primary_metrics(read_json(primary_path), manifest["primary_cell_count"])
    thesis_verified = False
    if "thesis_public_path" in manifest or "thesis_pdf_sha256" in manifest:
        if project_root is None:
            raise ValueError("An explicit project root is required to verify the thesis PDF")
        project_root = project_root.resolve()
        thesis_reference = Path(manifest["thesis_public_path"])
        if thesis_reference.is_absolute():
            raise ValueError("The public thesis path must be project-relative")
        thesis = (project_root / thesis_reference).resolve()
        if not thesis.is_relative_to(project_root):
            raise ValueError("Public thesis path escapes the project root")
        checked_source(thesis, manifest["thesis_pdf_sha256"])
        thesis_verified = True
    return {"verified_files": len(manifest["files"]), "primary_cells": manifest["primary_cell_count"],
            "count_derived_cells_verified": arithmetic_cells, "thesis_pdf_verified": thesis_verified,
            "release_id": manifest["release_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thesis", type=Path)
    parser.add_argument("--release-id", help="Explicit identity of the reviewed PDF/evidence release; no implicit V7 fallback")
    parser.add_argument("--supervisor-root", type=Path, help="Explicit dated approved supervisor-derivative root; mandatory for a new corrected export")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "results"
    if args.verify:
        print(json.dumps(verify_release(output, project_root=root), indent=2))
        return
    if args.thesis is None or args.release_id is None or args.supervisor_root is None:
        parser.error("Export requires --thesis, --release-id and --supervisor-root; validate the corrected PDF before publishing")
    revision = args.revision_root or root / "results_revision" / "20261005"
    thesis = args.thesis
    plan, release = build_export(root, revision, thesis, release_id=args.release_id, supervisor_root=args.supervisor_root)
    plan.write(output)
    verified = verify_release(output, project_root=root)
    print(json.dumps({**verified, "bytes": sum(len(data) for data in plan.files.values()),
                      "threshold_studies": release["threshold_study_count"],
                      "transfer_models": release["transfer_model_count"]}, indent=2))


if __name__ == "__main__":
    main()
