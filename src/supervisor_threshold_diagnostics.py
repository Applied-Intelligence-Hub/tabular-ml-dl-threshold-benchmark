"""Approved October supervisor diagnostics over explicitly frozen evidence.

No training, calibration fitting, TEST threshold selection or original-file
mutation occurs. ``transport`` alone permits the approved validation-checkpoint
inference, and reproduces A/VAL before evaluating A/TEST. Other phases only
post-process stored scores/counts. New outputs are exclusive, versioned sidecars.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from experiment_protocol import PROJECT_ROOT as ROOT, file_sha256
from revision_thresholds import precision_constraint_scope

MODELS = ("logreg", "rf", "lgbm", "catboost", "fttransformer", "ocsvm")
SUPERVISED = MODELS[:-1]
DATASETS = ("ulb_2013", "baf_base")
RULES = ("fixed_05", "max_f1", "max_f2", "prec_ge_05")
DISPLAY = {"logreg": "LR", "rf": "RF", "lgbm": "LGBM", "catboost": "CatBoost",
           "fttransformer": "FT-Trans.", "ocsvm": "OCSVM"}
DEFAULT_OUTPUT = ROOT / "results_revision/20261015_supervisor/threshold_diagnostics"
PINS_PATH = ROOT / "results/provenance/selected_runs.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def portable(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def source_record(path):
    return {"path": portable(path), "sha256": file_sha256(path)}


def new_phase(output, phase):
    output = Path(output).resolve()
    allowed = (ROOT / "results_revision/20261015_supervisor").resolve()
    if allowed not in output.parents:
        raise ValueError("Diagnostic outputs must be below results_revision/20261015_supervisor")
    destination = output / phase
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_arrays(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not labels.size:
        raise ValueError("Labels and scores must be non-empty aligned vectors")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Both binary classes and finite scores are required")
    return labels, scores


def counts_metrics(labels, scores, threshold):
    labels, scores = validate_arrays(labels, scores)
    if np.isnan(threshold):
        raise ValueError("Threshold must not be NaN")
    predicted = scores >= threshold
    positive = labels == 1
    tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
    fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
    return metrics_from_counts({"TP": tp, "FP": fp, "FN": fn, "TN": tn})


def metrics_from_counts(record):
    counts = {key: int(record[key]) for key in ("TP", "FP", "TN", "FN")}
    if any(value < 0 or value != record[key] for key, value in counts.items()):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, tn, fn = (counts[key] for key in ("TP", "FP", "TN", "FN"))
    n, alerts = tp + fp + tn + fn, tp + fp
    if not n or not tp + fn:
        raise ValueError("Population and fraud count must be positive")
    return {**counts, "rows": n, "fraud_rows": tp + fn, "alerts": alerts,
            "precision": tp / alerts if alerts else None, "recall": tp / (tp + fn),
            "F2": 5 * tp / (5 * tp + 4 * fn + fp), "alert_rate": alerts / n,
            "precision_target_0_5_met": bool(alerts and tp >= fp)}


def ranking_metrics(labels, scores):
    labels, scores = validate_arrays(labels, scores)
    return {"average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "brier": float(brier_score_loss(labels, scores))}


def load_pinned(pins, dataset, model, strategy="none", *, require_indices=False):
    key = f"{dataset}/{model}/{strategy}"
    pin = pins[key]
    run = (ROOT / pin["archive_run_id"]).resolve()
    config_path = run / "config.json"
    if file_sha256(config_path) != pin["config_sha256"]:
        raise ValueError(f"Changed configuration pin: {key}")
    config = read_json(config_path)
    if (config.get("dataset"), config.get("model")) != (dataset, model):
        raise ValueError(f"Wrong pinned source identity: {key}")
    if config.get("strategy") != strategy and not (model == "ocsvm" and config.get("strategy") == "n/a"):
        raise ValueError(f"Wrong strategy: {key}")
    arrays, sources = {}, {"config": source_record(config_path)}
    for name in ("y_test", "y_test_scores", "test_row_indices"):
        path = run / f"{name}.npy"
        expected = pin.get(f"{name}_sha256")
        if not path.is_file():
            if name == "test_row_indices" and not require_indices and expected is None:
                continue
            raise FileNotFoundError(path)
        if expected is None or file_sha256(path) != expected:
            raise ValueError(f"Missing or changed array pin: {key}/{name}")
        arrays[name] = np.load(path, allow_pickle=False)
        sources[name] = source_record(path)
    validate_arrays(arrays["y_test"], arrays["y_test_scores"])
    if "test_row_indices" in arrays:
        rows = arrays["test_row_indices"]
        if rows.shape != arrays["y_test"].shape or not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != len(rows):
            raise ValueError("TEST row identities must be unique aligned integers")
    metrics_path = ROOT / pin["public_metrics"]
    sources["full_precision_metrics"] = source_record(metrics_path)
    return run, config, arrays, read_json(metrics_path), sources


def verify_sources(records):
    for record in records:
        if file_sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"Original source changed during diagnostic: {record['path']}")


def basic(output):
    """P2 precision, P3 folds/cuts, P5 prior/Brier; no inference."""
    pins = read_json(PINS_PATH)
    sources = [source_record(PINS_PATH)]
    precision, folds, studies = [], [], {}
    for dataset in DATASETS:
        for model in MODELS:
            path = ROOT / f"results/thresholds/{dataset}/{model}.json"
            study = read_json(path)
            sources.append(source_record(path))
            run, config, arrays, original, run_sources = load_pinned(pins, dataset, model)
            sources.extend(run_sources.values())
            if (ROOT / study["source_run"]).resolve() != run:
                raise ValueError("Threshold study is not the pinned baseline")
            for name, digest in study["source_sha256"].items():
                if file_sha256(run / name) != digest:
                    raise ValueError("Threshold study source hash mismatch")
                sources.append(source_record(run / name))
            for rule in RULES:
                result = metrics_from_counts(study["test_results"][rule])
                threshold = study["thresholds_median"][rule]
                actual = counts_metrics(arrays["y_test"], arrays["y_test_scores"], np.inf if threshold is None else threshold)
                if result != actual:
                    raise ValueError("Saved four-rule counts do not reproduce")
                precision.append({"dataset": dataset, "model": model, "rule": rule,
                                  "threshold": threshold, **result})
            cuts = study["thresholds_per_fold"]["max_f2"]
            expected = 1 if model == "fttransformer" else 5
            if len(cuts) != expected or float(np.median(cuts)) != study["thresholds_median"]["max_f2"]:
                raise ValueError("Invalid fold aggregation")
            folds.append({"dataset": dataset, "model": model, "aggregation": study["aggregation"],
                          "fraud_per_fold": [row["fraud_samples"] for row in study["validation_folds"]],
                          "thresholds_per_fold": cuts,
                          "minimum": min(cuts) if expected == 5 else None,
                          "median_or_holdout": float(np.median(cuts)),
                          "maximum": max(cuts) if expected == 5 else None,
                          "precision_constraint_scope": precision_constraint_scope(model)})
            studies[dataset, model] = study
    _, _, arrays, _, _ = load_pinned(pins, "ulb_2013", "lgbm", require_indices=True)
    study = studies["ulb_2013", "lgbm"]
    median = study["thresholds_median"]["max_f2"]
    cut_sensitivity = [{"fold": i, "threshold": cut, "principal_median": cut == median,
                        **counts_metrics(arrays["y_test"], arrays["y_test_scores"], cut)}
                       for i, cut in enumerate(study["thresholds_per_fold"]["max_f2"], 1)]
    prior = []
    baseline_labels = None
    for model in SUPERVISED:
        for strategy in ("none", "rus", "ros", "weights"):
            run, config, arrays, original, run_sources = load_pinned(pins, "baf_base", model, strategy)
            sources.extend(run_sources.values())
            labels, scores = arrays["y_test"], arrays["y_test_scores"]
            if baseline_labels is None:
                baseline_labels = labels.copy()
            if not np.array_equal(labels, baseline_labels):
                raise ValueError("BAF prior sources do not have identical ordered TEST labels")
            c, parameter_evidence = prior_constant(config, strategy, run)
            extra_sources = [source_record(run / name) for name in parameter_evidence.pop("source_files", [])]
            sources.extend(extra_sources)
            parameter_evidence["source_artefacts"] = extra_sources
            threshold = float(config.get("threshold_exact", original["threshold"]))
            raw_cut = counts_metrics(labels, scores, threshold)
            transformed = prior_transform(scores, c)
            transformed_threshold = float(prior_transform(np.asarray([threshold]), c)[0])
            transformed_cut = counts_metrics(labels, transformed, transformed_threshold)
            raw_metrics, transformed_metrics = ranking_metrics(labels, scores), ranking_metrics(labels, transformed)
            changes = int(np.count_nonzero((scores >= threshold) != (transformed >= transformed_threshold)))
            unique_scores = np.unique(scores)
            unique_transformed = prior_transform(unique_scores, c)
            strict_numerical_order = bool(np.all(np.diff(unique_transformed) > 0))
            coalescences = int(np.count_nonzero(np.diff(unique_transformed) == 0))
            reversals = int(np.count_nonzero(np.diff(unique_transformed) < 0))
            ap_equal = raw_metrics["average_precision"] == transformed_metrics["average_precision"]
            roc_equal = raw_metrics["roc_auc"] == transformed_metrics["roc_auc"]
            if changes or reversals or raw_cut != transformed_cut:
                raise ValueError(f"Analytic transformation failed {model}/{strategy}: decisions={changes}, strict_order={strict_numerical_order}, AP_equal={ap_equal}, ROC_equal={roc_equal}")
            historical = {key: int(original[key]) for key in ("TP", "FP", "TN", "FN")}
            replay = all(raw_cut[key] == historical[key] for key in historical)
            prior.append({"model": model, "strategy": strategy, "constant_c": c,
                          "parameter_evidence": parameter_evidence, "raw": raw_metrics,
                          "transformed": transformed_metrics, "threshold_raw_available": threshold,
                          "threshold_transformed": transformed_threshold,
                          "threshold_precision": "exact stored threshold" if "threshold_exact" in config else "historical six-decimal threshold only",
                          "available_cut_reproduces_historical_counts": replay,
                          "historical_counts": historical, "available_cut_counts": raw_cut,
                          "decisions_changed_by_transformation": changes,
                          "strict_order_and_ties_preserved_numerically": strict_numerical_order,
                          "distinct_raw_scores_coalesced_float64": coalescences,
                          "score_order_reversals_float64": reversals,
                          "AP_bitwise_equal": ap_equal, "ROC_AUC_bitwise_equal": roc_equal,
                          "AP_numerical_difference": transformed_metrics["average_precision"] - raw_metrics["average_precision"],
                          "ROC_AUC_numerical_difference": transformed_metrics["roc_auc"] - raw_metrics["roc_auc"],
                          "source_run": portable(run), "sources": run_sources})
            print(f"Prior {model}/{strategy}: Brier {raw_metrics['brier']:.8f} -> {transformed_metrics['brier']:.8f}; cut replay={replay}", flush=True)
    verify_sources(sources)
    dest = new_phase(output, "basic_verified")
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "new_fits": 0, "new_inference": 0, "TEST_parameter_selection": False,
              "precision": precision, "folds": folds, "lgbm_ulb_cut_sensitivity": cut_sensitivity,
              "lgbm_ulb_unchanged_AP": float(read_json(ROOT / pins["ulb_2013/lgbm/none"]["public_metrics"])["PR-AUC"]),
              "prior_brier": prior,
              "prior_scope": "None references plus 15 BAF analytic transformations; parameters use DEV/training only; no fitted calibrator.",
              "historical_replay_limitation": "Six-decimal classical cuts are replayed as stored, without TEST reconstruction; historical counts are preserved even when replay differs.",
              "sources": list({row["path"]: row for row in sources}.values())}
    write_json(dest / "diagnostics.json", report)
    write_csv(dest / "precision.csv", precision)
    write_csv(dest / "lgbm_ulb_cut_sensitivity.csv", cut_sensitivity)
    return report


def prior_transform(scores, constant):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(constant) or constant <= 0 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Prior transformation requires probabilities and finite c > 0")
    if constant == 1:
        return scores.copy()
    # Parenthesise (1-s): subtracting s after adding 1 to c*s loses
    # endpoint accuracy and can introduce avoidable round-off near s=1.
    return constant * scores / (constant * scores + (1 - scores))


def prior_constant(config, strategy, run):
    positive, total = int(config["train_fraud"]), int(config["train_samples"])
    if (positive, total) != (8823, 800000):
        raise ValueError("Prior parameters must use the original full-DEV BAF counts")
    pi = positive / total
    evidence = {"DEV_positive": positive, "DEV_total": total, "pi_DEV": pi, "source_files": []}
    if strategy == "none":
        return 1.0, {**evidence, "basis": "untransformed reference"}
    if strategy in ("rus", "ros"):
        diagnostics = run / "sampler_diagnostics.json"
        if diagnostics.exists():
            records = [row for row in read_json(diagnostics) if row["stage"] == "full_dev"]
            if len(records) != 1:
                raise ValueError("A single full-DEV sampler record is required")
            row = records[0]
            before, after = row["class_counts_before"], row["class_counts_after"]
            if before != {"0": total - positive, "1": positive} or after["0"] != after["1"]:
                raise ValueError("Sampler evidence is not full-DEV balanced 1:1")
            q = after["1"] / (after["0"] + after["1"])
            evidence.update(basis="observed full-DEV sampler log", q=q, class_counts_after=after,
                            source_files=["sampler_diagnostics.json"])
        else:
            # The implementation's automatic binary balancing is evidence of
            # the policy, not a claim to have recovered historical fold logs.
            q = 0.5
            evidence.update(basis="documented automatic 1:1 balancing policy; original full-DEV/fold sampler logs unavailable",
                            q=q, observed_historical_sampler_counts=False)
        return (pi / (1 - pi)) * ((1 - q) / q), evidence
    if strategy != "weights":
        raise ValueError("Only None/RUS/ROS/Weights are approved")
    if config["model"] == "fttransformer":
        # The stored execution snapshot records positive weight = n_neg/n_pos
        # for the fresh full-DEV loss (not its internal-holdout counterpart).
        snapshot = config.get("source_snapshot_path")
        if not snapshot or not config.get("source_snapshot_sha256"):
            raise ValueError("FT weighted source lacks its execution source snapshot")
        snapshot = Path(snapshot)
        snapshot = snapshot if snapshot.is_absolute() else ROOT / snapshot
        if file_sha256(snapshot) != config["source_snapshot_sha256"]:
            raise ValueError("FT weighted source snapshot changed")
        import zipfile
        with zipfile.ZipFile(snapshot) as archive:
            candidates = [name for name in archive.namelist() if name.replace('\\', '/').endswith("src/main_transformer.py")]
            if len(candidates) != 1:
                raise ValueError("Source snapshot lacks the weighted full-DEV implementation")
            code = archive.read(candidates[0]).decode("utf-8")
        if "n_pos = int(y_train.sum())" not in code or "n_neg / n_pos" not in code or "final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)" not in code:
            raise ValueError("Execution snapshot does not verify full-DEV positive weighting")
        # Ratio actually supplied to PyTorch is float32, as in the execution.
        positive_weight = float(np.float32((total - positive) / positive))
        evidence.update(basis="full-DEV pos_weight, float32 as supplied to BCEWithLogitsLoss; execution snapshot verified",
                        weight_0=1.0, weight_1=positive_weight,
                        snapshot=source_record(snapshot))
        return 1 / positive_weight, evidence
    import joblib
    fitted = joblib.load(run / "model.joblib")
    classifier = fitted.named_steps["classifier"]
    params = classifier.get_params()
    if config["model"] == "catboost":
        if params.get("auto_class_weights") != "Balanced":
            raise ValueError("Frozen CatBoost does not identify Balanced class weights")
        full = classifier.get_all_params()
        weights = full.get("class_weights")
        if not weights or len(weights) != 2 or weights[0] <= 0 or weights[1] <= 0:
            raise ValueError("Frozen CatBoost lacks realised class weights")
        w0, w1 = map(float, weights)
        evidence.update(basis="frozen CatBoost fitted class_weights", auto_class_weights="Balanced")
    else:
        if params.get("class_weight") != "balanced":
            raise ValueError("Frozen classifier does not identify balanced class weights")
        w0, w1 = total / (2 * (total - positive)), total / (2 * positive)
        evidence.update(basis="frozen classifier class_weight=balanced and full-DEV counts")
    evidence.update(weight_0=w0, weight_1=w1, source_files=["model.joblib"])
    del fitted, classifier
    gc.collect()
    return w0 / w1, evidence


def paired_ap_difference(labels, first_scores, second_scores, *, iterations=1000, seed=42):
    """Independent cross-model wrapper; leaves representation-control guards intact."""
    labels, first_scores = validate_arrays(labels, first_scores)
    _, second_scores = validate_arrays(labels, second_scores)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("Positive integer bootstrap iterations required")
    rng = np.random.default_rng(seed)
    differences = []
    for iteration in range(iterations):
        sample = rng.integers(0, len(labels), size=len(labels))
        sampled_y = labels[sample]
        if np.unique(sampled_y).size != 2:
            continue
        differences.append(float(average_precision_score(sampled_y, second_scores[sample])
                                 - average_precision_score(sampled_y, first_scores[sample])))
        if (iteration + 1) % 100 == 0:
            print(f"Paired AP: {iteration + 1}/{iterations}", flush=True)
    if not differences:
        raise ValueError("No valid two-class bootstrap draw")
    return {"difference_second_minus_first": float(average_precision_score(labels, second_scores)
                                                   - average_precision_score(labels, first_scores)),
            "first_AP": float(average_precision_score(labels, first_scores)),
            "second_AP": float(average_precision_score(labels, second_scores)),
            "percentile_95_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
            "requested": iterations, "valid": len(differences), "discarded_monoclass": iterations - len(differences),
            "seed": seed, "generator": "numpy.random.default_rng / PCG64",
            "resampling": "paired TEST rows with replacement; same indices for labels and both score arrays",
            "quantile_method": "numpy.quantile default linear interpolation",
            "bootstrap_differences": differences}


def paired(output):
    pins = read_json(PINS_PATH)
    first_run, first_cfg, first, _, first_sources = load_pinned(pins, "baf_base", "lgbm", require_indices=True)
    second_run, second_cfg, second, _, second_sources = load_pinned(pins, "baf_base", "catboost", require_indices=True)
    for name in ("y_test", "test_row_indices"):
        if not np.array_equal(first[name], second[name]):
            raise ValueError("Cross-model AP requires identical TEST identities and label order")
    if first_cfg["dataset_hash_sha256"] != second_cfg["dataset_hash_sha256"] or (len(first["y_test"]), int(first["y_test"].sum())) != (200000, 2206):
        raise ValueError("The predefined BAF baseline population does not match")
    sources = [source_record(PINS_PATH), *first_sources.values(), *second_sources.values()]
    result = paired_ap_difference(first["y_test"], first["y_test_scores"], second["y_test_scores"])
    verify_sources(sources)
    dest = new_phase(output, "paired_ap")
    report = {"status": "complete", "first_model": "lgbm", "second_model": "catboost", "dataset": "baf_base",
              "strategy": "none", "rows": 200000, "fraud_rows": 2206,
              "first_run": portable(first_run), "second_run": portable(second_run),
              "new_fits": 0, "new_inference": 0, "TEST_model_selection": False,
              "uncertainty_scope": "Conditional on fitted models and split; excludes training/seed/entity-cluster variability; containing zero does not establish equivalence.",
              "sources": sources, **result}
    write_json(dest / "diagnostics.json", report)
    return report


def transport(output, *, device="cuda", batch_size=2048):
    """Approved A/VAL reproduction gate followed by A/TEST inference only."""
    import joblib
    import torch
    from torch.utils.data import DataLoader
    from data import load_dataset
    from models.fttransformer import TabularDataset, build_model, evaluate

    pins = read_json(PINS_PATH)
    run, config, arrays, _, run_sources = load_pinned(pins, "baf_base", "fttransformer", require_indices=True)
    required = ("validation_model.pt", "validation_preprocessors.joblib", "validation_row_indices.npy",
                "y_val.npy", "y_val_scores.npy", "model.pt", "preprocessors.joblib", "completed.json")
    completion = read_json(run / "completed.json")
    for name in required:
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    sources = [source_record(PINS_PATH), *run_sources.values(), *[source_record(run / name) for name in required]]
    checkpoint = torch.load(run / "validation_model.pt", map_location="cpu", weights_only=False)
    preprocessors = joblib.load(run / "validation_preprocessors.joblib")
    for key in ("num_cols", "cat_cols"):
        if checkpoint[key] != preprocessors[key]:
            raise ValueError("A checkpoint and A preprocessor schema differ")
    if checkpoint["hyperparams"] != config["best_params"]:
        raise ValueError("A checkpoint hyperparameters do not match the pinned baseline")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA unavailable; choose device explicitly")
    model = build_model(checkpoint["hyperparams"], checkpoint["d_numerical"], checkpoint["cat_cardinalities"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(config["threshold_exact"])
    if threshold != 0.05843411013484001:
        raise ValueError("Unexpected approved BAF baseline max-F2 threshold")
    X_dev, X_test, y_dev, y_test, metadata = load_dataset("baf_base", return_metadata=True)
    if metadata["raw_file_sha256"] != config["dataset_hash_sha256"]:
        raise ValueError("BAF raw-file identity differs from the pinned source")
    if not np.array_equal(metadata["test_indices"], arrays["test_row_indices"]) or not np.array_equal(y_test, arrays["y_test"]):
        raise ValueError("BAF TEST differs from the saved population")
    val_indices = np.load(run / "validation_row_indices.npy", allow_pickle=False)
    if np.intersect1d(val_indices, arrays["test_row_indices"]).size or not np.isin(val_indices, metadata["dev_indices"]).all():
        raise ValueError("A validation indices must lie in DEV and outside TEST")
    X_val = X_dev.loc[val_indices]
    labels_val = np.load(run / "y_val.npy", allow_pickle=False)
    saved_val = np.load(run / "y_val_scores.npy", allow_pickle=False)
    if not np.array_equal(y_dev.loc[val_indices].to_numpy(), labels_val):
        raise ValueError("Loaded validation labels differ from the frozen A evidence")

    def predict(frame):
        numeric = np.asarray(preprocessors["num_preprocessor"].transform(frame[preprocessors["num_cols"]]), dtype=np.float32)
        categorical = np.asarray(preprocessors["cat_encoder"].transform(frame[preprocessors["cat_cols"]]), dtype=np.int64)
        loader = DataLoader(TabularDataset(numeric, categorical), batch_size=batch_size, shuffle=False, num_workers=0)
        return evaluate(model, loader, torch.device(device))[1]

    print("Reproducing frozen A/VAL before any A/TEST inference", flush=True)
    reproduced_val = predict(X_val)
    max_difference = float(np.max(np.abs(reproduced_val - saved_val)))
    if not np.allclose(reproduced_val, saved_val, rtol=1e-5, atol=1e-6):
        raise ValueError(f"A/VAL numerical fidelity failed: max difference {max_difference}")
    decisions_changed = int(np.count_nonzero((reproduced_val >= threshold) != (saved_val >= threshold)))
    if decisions_changed:
        raise ValueError("A/VAL frozen-threshold decisions do not reproduce")
    print("A/VAL fidelity passed; evaluating approved A/TEST", flush=True)
    scores_test_A = predict(X_test)
    records = []
    for role, labels, scores in (("A/VAL", labels_val, saved_val),
                                ("A/TEST", arrays["y_test"], scores_test_A),
                                ("B/TEST", arrays["y_test"], arrays["y_test_scores"])):
        records.append({"role": role, "threshold": threshold, **counts_metrics(labels, scores, threshold),
                        "average_precision": float(average_precision_score(labels, scores))})
    verify_sources(sources)
    dest = new_phase(output, "ft_transport")
    with (dest / "A_test_scores.npy").open("xb") as stream:
        np.save(stream, scores_test_A, allow_pickle=False)
    report = {"status": "complete", "source_run": portable(run), "rows": records,
              "A_validation_reproduction": {"rows": len(saved_val), "scores_bitwise_equal": bool(np.array_equal(reproduced_val, saved_val)),
                                            "maximum_absolute_score_difference": max_difference, "atol": 1e-6, "rtol": 1e-5,
                                            "decisions_changed": decisions_changed, "precedes_A_TEST_inference": True},
              "new_fits": 0, "new_TEST_score_arrays": 1, "TEST_checkpoint_selection": False,
              "principal_model": "B full-DEV", "supplementary_model": "A validation checkpoint with A fitted preprocessors",
              "interpretation": "Post-hoc sensitivity of two frozen pipelines; A/VAL versus B/TEST changes fitted pipeline and observations; A/TEST versus B/TEST does not isolate initialization, training-size or preprocessing causes.",
              "device": device, "batch_size": batch_size, "python": platform.python_version(), "torch": torch.__version__,
              "sources": sources, "new_artefact": source_record(dest / "A_test_scores.npy")}
    write_json(dest / "diagnostics.json", report)
    write_csv(dest / "sensitivity.csv", records)
    return report


def tabular(headers, rows, alignment=None):
    alignment = "@{}" + (alignment or ("l " + "r " * (len(headers) - 1)).strip()) + "@{}"
    return ("% Generated by supervisor_threshold_diagnostics.py; frozen evidence only.\n"
            + "\\begin{tabular}{" + alignment + "}\n\\toprule\n"
            + " & ".join(headers) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(map(str, row)) + " \\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def tables(output):
    output = Path(output)
    report = read_json(output / "basic_verified/diagnostics.json")
    paired_report = read_json(output / "paired_ap/diagnostics.json")
    transport_report = read_json(output / "ft_transport/diagnostics.json")
    generated = {}
    for dataset, suffix in (("ulb_2013", "ulb"), ("baf_base", "baf")):
        rows = []
        for model in MODELS:
            cells = []
            for rule in RULES:
                record = next(row for row in report["precision"] if (row["dataset"], row["model"], row["rule"]) == (dataset, model, rule))
                cells.append("---" if record["precision"] is None else f"{record['precision']:.3f}")
            rows.append([DISPLAY[model], *cells])
        generated[f"supervisor_precision_{suffix}.tex"] = tabular(
            ["Model", "Fixed 0.5", "max-$F_1$", "max-$F_2$", "VAL precision target"], rows)
    rows = []
    for row in report["folds"]:
        dataset = "ULB" if row["dataset"] == "ulb_2013" else "BAF"
        if row["minimum"] is None:
            rows.append([dataset, DISPLAY[row["model"]], str(row["fraud_per_fold"][0]) + " (holdout)", "N/A", f"{row['median_or_holdout']:.6f}", "N/A"])
        else:
            fraud = row["fraud_per_fold"]
            fraud_text = str(min(fraud)) if min(fraud) == max(fraud) else f"{min(fraud)}--{max(fraud)}"
            rows.append([dataset, DISPLAY[row["model"]], fraud_text, f"{row['minimum']:.6f}", f"{row['median_or_holdout']:.6f}", f"{row['maximum']:.6f}"])
    generated["supervisor_threshold_folds.tex"] = tabular(["Dataset", "Model", "\\shortstack{Frauds/\\\\fold}", "Min.", "\\shortstack{Median/\\\\holdout}", "Max."], rows, "l l r r r r")
    generated["supervisor_threshold_folds_full.tex"] = tabular(
        ["Dataset", "Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        [["ULB" if row["dataset"] == "ulb_2013" else "BAF", DISPLAY[row["model"]],
          *([f"{cut:.10f}" for cut in row["thresholds_per_fold"]] if row["minimum"] is not None else
            [f"{row['median_or_holdout']:.10f} (holdout)", "N/A", "N/A", "N/A", "N/A"])]
         for row in report["folds"]], "l l r r r r r")
    generated["supervisor_lgbm_cut_sensitivity.tex"] = tabular(
        ["DEV fold cut", "$\\tau$", "$P$", "$R$", "$F_2$", "Alerts", "Rate (\\%)"],
        [[str(row["fold"]) + (" (median)" if row["principal_median"] else ""), f"{row['threshold']:.6f}", f"{row['precision']:.3f}",
          f"{row['recall']:.3f}", f"{row['F2']:.3f}", str(row["alerts"]), f"{100 * row['alert_rate']:.3f}\\%"]
         for row in report["lgbm_ulb_cut_sensitivity"]])
    generated["supervisor_prior_brier.tex"] = tabular(
        ["Model", "Strategy", "\\shortstack{Raw\\\\Brier}", "\\shortstack{Adjusted\\\\Brier}", "$\\tau$ raw", "$g(\\tau)$", "\\shortstack{CM\\\\replay}"],
        [[DISPLAY[row["model"]], row["strategy"].capitalize(), f"{row['raw']['brier']:.6f}", f"{row['transformed']['brier']:.6f}",
          f"{row['threshold_raw_available']:.6f}", f"{row['threshold_transformed']:.6f}", "Yes" if row["available_cut_reproduces_historical_counts"] else "No$^{\\dagger}$"]
         for row in report["prior_brier"]])
    generated["supervisor_ft_transport.tex"] = tabular(
        ["Pipeline/cohort", "AP", "$P$", "$R$", "$F_2$", "Alerts", "Rate (\\%)"],
        [[row["role"], f"{row['average_precision']:.4f}", f"{row['precision']:.3f}", f"{row['recall']:.3f}",
          f"{row['F2']:.3f}", str(row["alerts"]), f"{100 * row['alert_rate']:.3f}\\%"] for row in transport_report["rows"]])
    interval = paired_report["percentile_95_interval"]
    generated["supervisor_paired_ap.tex"] = tabular(
        ["Contrast (BAF None)", "$\\Delta$ AP", "Paired 95\\% interval"],
        [["CatBoost $-$ LGBM", f"{paired_report['difference_second_minus_first']:+.4f}", f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"]], "l r l")
    destinations = (ROOT / "Overleaf/tables", ROOT / "Article 1/tables", ROOT / "Article 2/tables")
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in generated.items():
            if destination == ROOT / "Article 1/tables" and ("precision" in name or "lgbm_cut" in name):
                continue
            path = destination / name
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Refusing to overwrite a changed diagnostic table: {path}")
            if not path.exists():
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)
    dest = new_phase(output, "tables")
    write_json(dest / "manifest.json", {"generated_tables": list(generated),
                                        "sources": [source_record(output / phase / "diagnostics.json") for phase in ("basic_verified", "paired_ap", "ft_transport")]})
    return {"generated_tables": list(generated)}


def qa(output):
    """Read-only fidelity checks, then a new sidecar inventory; no inference."""
    output = Path(output)
    reports = {phase: read_json(output / phase / "diagnostics.json")
               for phase in ("basic_verified", "paired_ap", "ft_transport")}
    for report in reports.values():
        verify_sources(report["sources"])
    basic_report = reports["basic_verified"]
    if len(basic_report["precision"]) != 48 or len(basic_report["folds"]) != 12 or len(basic_report["prior_brier"]) != 20:
        raise ValueError("Approved diagnostics have incomplete coverage")
    if sum(row["strategy"] != "none" for row in basic_report["prior_brier"]) != 15:
        raise ValueError("Exactly 15 analytic transformations are required")
    sources = [source_record(output / phase / "diagnostics.json") for phase in reports]
    files = []
    for phase in (*reports, "tables"):
        for path in sorted((output / phase).iterdir()):
            if path.is_file():
                files.append(source_record(path))
    generated_tables = read_json(output / "tables/manifest.json")["generated_tables"]
    for name in generated_tables:
        content = (ROOT / "Overleaf/tables" / name).read_bytes()
        for folder in ("Overleaf/tables", "Article 1/tables", "Article 2/tables"):
            path = ROOT / folder / name
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError("Shared diagnostic table differs across documents")
                files.append(source_record(path))
    transport_report = reports["ft_transport"]
    verify_sources([transport_report["new_artefact"]])
    records = basic_report["prior_brier"]
    if any(row["decisions_changed_by_transformation"] or row["score_order_reversals_float64"] for row in records):
        raise ValueError("Numerical transformation reverses order or changes decisions")
    destination = new_phase(output, "qa")
    flat_prior = [{"model": row["model"], "strategy": row["strategy"], "constant_c": row["constant_c"],
                   "Brier_raw": row["raw"]["brier"], "Brier_adjusted": row["transformed"]["brier"],
                   "AP_raw": row["raw"]["average_precision"], "AP_adjusted": row["transformed"]["average_precision"],
                   "ROC_raw": row["raw"]["roc_auc"], "ROC_adjusted": row["transformed"]["roc_auc"],
                   "threshold_raw": row["threshold_raw_available"], "threshold_adjusted": row["threshold_transformed"],
                   "count_replay": row["available_cut_reproduces_historical_counts"],
                   "coalescences_float64": row["distinct_raw_scores_coalesced_float64"],
                   "AP_difference_float64": row["AP_numerical_difference"],
                   "ROC_difference_float64": row["ROC_AUC_numerical_difference"]} for row in records]
    write_csv(destination / "prior_brier_full.csv", flat_prior)
    result = {"status": "complete", "canonical_phases": list(reports),
              "excluded_noncanonical_candidate": "basic/diagnostics.json: exploratory pre-QA arithmetic candidate, superseded explicitly by basic_verified; never use for reporting/export",
              "all_original_input_hashes_unchanged": True,
              "shared_tables_identical_across_documents": True,
              "new_fits": 0, "new_TEST_score_arrays": 1,
              "analytic_transformations": 15, "paired_bootstrap_requested": 1000,
              "paired_bootstrap_valid": reports["paired_ap"]["valid"],
              "all_available_historical_classical_cuts_replay": all(row["available_cut_reproduces_historical_counts"] for row in records),
              "all_15_adjusted_Brier_values_lower": all(row["transformed"]["brier"] < row["raw"]["brier"] for row in records if row["strategy"] != "none"),
              "numerical_limitations": [{key: row[key] for key in ("model", "strategy", "distinct_raw_scores_coalesced_float64", "AP_numerical_difference", "ROC_AUC_numerical_difference")}
                                        for row in records if not row["strict_order_and_ties_preserved_numerically"]],
              "code": [source_record(Path(__file__)), source_record(ROOT / "src/revision_thresholds.py"),
                       source_record(ROOT / "tests/test_supervisor_threshold_diagnostics.py")],
              "canonical_output_inventory": files, "canonical_report_sources": sources}
    write_json(destination / "diagnostics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("basic", "paired", "transport", "tables", "qa"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    action = {"basic": basic, "paired": paired, "transport": transport, "tables": tables, "qa": qa}[args.phase]
    result = action(args.output, device=args.device) if args.phase == "transport" else action(args.output)
    print(json.dumps({"phase": args.phase, "status": result.get("status", "complete")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
