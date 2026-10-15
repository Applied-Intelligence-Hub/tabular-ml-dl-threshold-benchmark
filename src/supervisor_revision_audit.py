"""Read-only evidence audit for approved top-k and numeric reporting corrections.

Only new, explicitly selected output directories can be written. No model is
loaded, fitted or scored, and the October 5 archive is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from evaluation.metrics import (HISTORICAL_TOP_K_POLICY, TOP_K_POLICY,
                                precision_recall_at_k)

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "results_revision/20261005"
OUTPUT = ROOT / "results_revision/20261015_supervisor/topk_claims"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pinned(path, sources, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Source escapes the author archive")
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Pinned input changed: {path}")
    sources[path.relative_to(ROOT).as_posix()] = actual
    return path


def audit_cell(labels, scores, metrics):
    k = int(metrics["k_used"])
    expected = precision_recall_at_k(labels, scores, k=k)
    historical = precision_recall_at_k(labels, scores, k=k, policy=HISTORICAL_TOP_K_POLICY)
    for name, replay in zip(("precision_at_k", "recall_at_k"), historical[:2]):
        if not math.isclose(replay, float(metrics[name]), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Historical {name} does not reproduce its frozen record")
    ap = float(average_precision_score(labels, scores))
    if not math.isclose(ap, float(metrics["PR-AUC"]), rel_tol=0, abs_tol=1e-6):
        raise ValueError("Grouped AP differs from its frozen record")
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > boundary
    tied = scores == boundary
    return {
        "rows": len(labels), "frauds": int(labels.sum()), "k": k,
        "saved_policy": HISTORICAL_TOP_K_POLICY, "current_policy": TOP_K_POLICY,
        "historical_precision_at_k": historical[0], "historical_recall_at_k": historical[1],
        "precision_at_k": expected[0], "recall_at_k": expected[1],
        "changed": any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                       for a, b in zip(expected[:2], historical[:2])),
        "boundary_score": float(boundary), "boundary_tied_rows": int(tied.sum()),
        "boundary_tied_frauds": int(labels[tied].sum()), "rows_strictly_above": int(above.sum()),
        "slots_from_boundary": k - int(above.sum()),
        "grouped_average_precision": ap, "scores_exactly_zero": int((scores == 0).sum()),
        "scores_exactly_one": int((scores == 1).sum()),
        "frauds_with_score_one": int(labels[scores == 1].sum()),
        "operating_counts_preserved": {name: int(metrics[name]) for name in ("TP", "FP", "TN", "FN")}}


def collect_audit(qa, paired, sensitivity, sources):
    records = {}

    def add(key, run, hashes=None, metric_document=None):
        run = Path(run)
        hashes = hashes or {}
        metric_path = run / "metrics_test.json"
        pinned(metric_path, sources, hashes.get("metrics_test.json"))
        metrics = read_json(metric_path) if metric_document is None else metric_document
        arrays = []
        for name in ("y_test.npy", "y_test_scores.npy"):
            arrays.append(np.load(pinned(run / name, sources, hashes.get(name)), allow_pickle=False))
        records[key] = audit_cell(*arrays, metrics)
        records[key]["source_directory"] = run.relative_to(ROOT).as_posix()

    for key, record in qa["source_runs"].items():
        add("primary/" + key, record["run_dir"], {
            "metrics_test.json": record["metrics_sha256"],
            "y_test.npy": record["y_test_sha256"],
            "y_test_scores.npy": record["y_test_scores_sha256"]})
    for model, pair in paired["missingness"].items():
        selected = sensitivity["baseline_runs"]["baf_base/" + model]
        if Path(selected["run_dir"]).resolve() != Path(pair["second_run"]).resolve():
            raise ValueError("Missingness control pin differs from paired evidence")
        add("control/missingness/" + model, pair["second_run"], pair["second_artefacts_sha256"])
    ft = paired["categorical_controls"]["fttransformer"]
    add("control/categorical/fttransformer", ft["second_run"], ft["second_artefacts_sha256"])
    cb = paired["catboost_control_source"]
    add("control/categorical/catboost", cb["run_dir"], cb["artefacts_sha256"])
    for model, transfer in qa["transfer_sources"].items():
        for cohort, record in transfer["variants"].items():
            path = pinned(record["path"], sources, record["sha256"])
            hashes = {name: entry["sha256"] for name, entry in record["artefacts"].items()}
            hashes["metrics_test.json"] = record["sha256"]
            add("transfer/" + model + "/" + cohort, path.parent, hashes, read_json(path)["metrics"])
    if len(records) != 110:
        raise ValueError("Audit must cover 72 primary cells, eight controls and 30 retained-cohort evaluations")
    return records


def f2_from_counts(counts):
    tp, fp, fn = (int(counts[key]) for key in ("TP", "FP", "FN"))
    denominator = 5 * tp + fp + 4 * fn
    return 5 * tp / denominator if denominator else 0.0


def numeric_claims(qa, sources):
    claims, max_f2_values = {}, []
    for model in ("lgbm", "catboost", "fttransformer"):
        record = qa["threshold_studies"]["baf_base/" + model]
        path = pinned(record["path"], sources, record["sha256"])
        study = read_json(path)
        fixed = f2_from_counts(study["test_results"]["fixed_05"])
        selected = f2_from_counts(study["test_results"]["max_f2"])
        ratio = selected / fixed
        max_f2_values.append(selected)
        claims["baf_" + model + "_fixed_to_max_f2"] = {
            "dataset": "baf_base", "strategy": "none", "model": model,
            "source": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
            "fixed_f2": fixed, "selected_f2": selected,
            "ratio": ratio, "ratio_display": f"{ratio:.1f}", "absolute_difference": selected - fixed,
            "formula": "F2=5TP/(5TP+4FN+FP); ratio=selected/fixed",
            "scope": "same final model and saved TEST scores; DEV-selected max-F2 versus predefined 0.5"}
    claims["baf_three_baseline_max_f2_range"] = {
        "models": ["lgbm", "catboost", "fttransformer"], "strategy": "none",
        "minimum": min(max_f2_values), "maximum": max(max_f2_values),
        "range": max(max_f2_values) - min(max_f2_values),
        "range_display": f"{max(max_f2_values) - min(max_f2_values):.4f}"}
    ft_ap = {}
    for strategy in ("none", "smote"):
        record = qa["source_runs"]["baf_base/fttransformer/" + strategy]
        run = Path(record["run_dir"])
        labels = np.load(pinned(run / "y_test.npy", sources, record["y_test_sha256"]), allow_pickle=False)
        scores = np.load(pinned(run / "y_test_scores.npy", sources, record["y_test_scores_sha256"]), allow_pickle=False)
        ft_ap[strategy] = float(average_precision_score(labels, scores))
    claims["baf_ft_smote_ap_reduction"] = {
        "model": "fttransformer", "dataset": "baf_base", "average_precision": ft_ap,
        "absolute_difference_smote_minus_none": ft_ap["smote"] - ft_ap["none"],
        "relative_reduction_percent": 100 * (1 - ft_ap["smote"] / ft_ap["none"]),
        "formula": "100*(1-AP_smote/AP_none)", "display": "approximately 35%"}
    return claims


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    allowed = (ROOT / "results_revision/20261015_supervisor").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("Use a new child output directory under results_revision/20261015_supervisor")
    sources = {}
    qa_path = pinned(REVISION / "derived/reporting/generation_qa.json", sources)
    qa = read_json(qa_path)
    manifest = pinned(REVISION / "revision_manifest.json", sources, qa["manifest_sha256"])
    paired_path = pinned(REVISION / "derived/paired/paired_analysis.json", sources, qa["paired_analysis"]["sha256"])
    paired = read_json(paired_path)
    sensitivity_path = pinned(REVISION / "sensitivity/revision_manifest.json", sources,
                              paired["sensitivity_manifest_sha256"])
    records = collect_audit(qa, paired, read_json(sensitivity_path), sources)
    claims = numeric_claims(qa, sources)
    # Prove that all files read above remain unchanged before saving derivatives.
    for relative, expected in sources.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Input modified during audit: {relative}")
    report = {"schema_version": 1, "policy": TOP_K_POLICY, "historical_policy": HISTORICAL_TOP_K_POLICY,
              "source_manifest": manifest.relative_to(ROOT).as_posix(), "cells": records,
              "cell_count": len(records), "changed_cells": [key for key, record in records.items() if record["changed"]],
              "numeric_claims": claims, "sources_sha256": sources,
              "original_inputs_unchanged": True, "training_inference_or_test_selection": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "topk_claims_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(records), "changed_cells": report["changed_cells"], "output": str(output)}))


if __name__ == "__main__":
    main()
