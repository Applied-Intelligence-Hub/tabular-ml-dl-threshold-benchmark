"""Local safety/integrity checks for the selective public evidence export."""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_public_evidence import (ExportPlan, compact_curve, json_bytes, portable,
                                    recover_metrics, verify_primary_metrics, verify_release,
                                    build_export, main, validated_supervisor_inputs,
                                    apply_supervisor_overlay)


class PublicEvidenceTests(unittest.TestCase):
    def test_export_requires_explicit_pdf_and_release_without_old_version_fallback(self):
        for arguments in ([], ["--release-id", "reviewed"], ["--thesis", "corrected.pdf"],
                          ["--thesis", "corrected.pdf", "--release-id", "reviewed"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                main(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_corrected_export_requires_supervisor_bundle_before_archive_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            thesis = root / "reviewed.pdf"
            thesis.write_bytes(b"%PDF-fixture, no actual PDF export")
            with self.assertRaisesRegex(ValueError, "supervisor root"):
                build_export(root, root / "absent_archive", thesis, release_id="reviewed")
            with self.assertRaisesRegex(ValueError, "within this project"):
                validated_supervisor_inputs(root, root / "outside_dated_archive")

    def test_supervisor_manifest_rejects_old_revision_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            supervisor = root / "results_revision" / "selected"
            interp = supervisor / "interpretability_transfer"
            interp.mkdir(parents=True)
            manifest = {"revision_id": "old", "model_training": False, "model_inference": False,
                        "new_shap_computation": False, "source_files_unchanged_after_generation": True}
            path = interp / "manifest.json"
            path.write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "approved postprocessing revision"):
                validated_supervisor_inputs(root, supervisor)
            manifest["revision_id"] = "20261015_supervisor"
            path.write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "source hash provenance"):
                validated_supervisor_inputs(root, supervisor)
            manifest["source_sha256"] = {"../escapes_archive": "fake"}
            path.write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "outside the project archive"):
                validated_supervisor_inputs(root, supervisor)

    def test_reviewed_overlay_preserves_previous_planned_bytes_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = ExportPlan(root)
            plan.add("interpretability/shap_summary.json", b"old-mass-summary")
            plan.reviewed_overlay("interpretability/shap_summary.json", b"new-net-summary", (), "approved policy")
            self.assertEqual(plan.files["interpretability/shap_summary.json"], b"new-net-summary")
            historical = "provenance/historical_pre_supervisor/interpretability/shap_summary.json"
            self.assertEqual(plan.files[historical], b"old-mass-summary")
            self.assertIn("original_export_provenance", plan.provenance[historical])
            self.assertEqual(list(root.iterdir()), [])

    def test_supervisor_overlay_is_aggregate_only_and_preserves_FT_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = root / "aggregate_fixture.json"
            fixture.write_bytes(json_bytes({"fixture": "aggregate only"}))
            plan = ExportPlan(root)
            for dataset in ("ulb", "baf"):
                for metric in ("f1", "f2", "prauc", "rocauc"):
                    plan.add(f"tables/{dataset}/factorial_{metric}.tex",
                             b"\\caption{Frozen factorial}\nSM+T & 0.123 \\\\\n")
            original = {"precision_constraint_scope": "old incorrect fold description",
                        "thresholds_median": {"max_f2": 0.05843411013484001},
                        "test_results": {"max_f2": {"TP": 1231, "FP": 10183}}}
            for dataset in ("ulb_2013", "baf_base"):
                plan.add_json(f"thresholds/{dataset}/fttransformer.json", original)
            shap = {"cohorts": {"baf_base": {"complete_global_importance": {"age": 0.1}, "ranking": ["age"]}},
                    "cohort_top10_jaccard": {"baf_base": {"baf_base": 1.0}}, "magnitude_contrasts": []}
            bundle = {"files": defaultdict(lambda: fixture), "tables": {}, "shap": shap}
            apply_supervisor_overlay(plan, bundle)
            updated = json.loads(plan.files["thresholds/baf_base/fttransformer.json"])
            self.assertEqual(updated["thresholds_median"], original["thresholds_median"])
            self.assertEqual(updated["test_results"], original["test_results"])
            self.assertIn("single DEV validation holdout", updated["precision_constraint_scope"])
            self.assertEqual(updated["historical_precision_constraint_scope"], original["precision_constraint_scope"])
            self.assertIn("SM+T$^{*}$", plan.files["tables/baf/factorial_prauc.tex"].decode())
            self.assertIn("duplicates SMOTE", plan.files["tables/baf/factorial_prauc.tex"].decode())
            self.assertIn("provenance/historical_pre_supervisor/thresholds/baf_base/fttransformer.json", plan.files)
            self.assertFalse(any(name.endswith((".npy", ".pt", ".joblib")) for name in plan.files))
            self.assertEqual(list(root.iterdir()), [fixture])

    def test_release_identity_is_checked_before_reading_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for identity in ("", "../old-release", "two words", None):
                with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "release ID"):
                    build_export(root, root, root / "absent.pdf", release_id=identity)
            with self.assertRaisesRegex(ValueError, "existing thesis PDF"):
                build_export(root, root, root / "absent.pdf", release_id="reviewed_version")

    def test_paths_are_portable_and_nonfinite_values_explicit(self):
        root = Path.cwd()
        data = {"source": str(root / "results_revision" / "x"), "value": float("inf"),
                "runtime": "C:/Users/secret/python.exe", "name": "catboost"}
        public = portable(data, root)
        self.assertEqual(public["source"], "results_revision/x")
        self.assertEqual(public["runtime"], "[external local path omitted]")
        self.assertEqual(public["value"], "+Infinity")
        self.assertEqual(public["name"], "catboost")
        json.loads(json_bytes(public))

    def test_curve_subset_keeps_endpoints_and_does_not_claim_exact_ap(self):
        result = compact_curve({"precision": list(range(100)), "recall": list(range(100))}, 10)
        self.assertEqual(len(result["recall"]), 10)
        self.assertEqual(result["recall"][0], 0)
        self.assertEqual(result["recall"][-1], 99)
        self.assertIn("do not recompute AP", result["representation"])

    def test_export_cannot_write_to_historical_runs(self):
        plan = ExportPlan(Path.cwd())
        for target in ("ulb_2013/lgbm/none/config.json", "../config.json", "metrics/../../config.json"):
            with self.assertRaises(ValueError):
                plan.add(target, b"not allowed")

    def test_existing_unowned_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            destination = output / "metrics" / "mine.json"
            destination.parent.mkdir()
            destination.write_bytes(b"original")
            plan = ExportPlan(output)
            plan.add("metrics/mine.json", b"replacement")
            with self.assertRaises(FileExistsError):
                plan.write(output)
            self.assertEqual(destination.read_bytes(), b"original")

    def test_verify_detects_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            plan = ExportPlan(output)
            plan.add("metrics/grid.json", b"{}\n")
            release = {"schema_version": 1, "exporter": "export_public_evidence.py", "release_id": "test",
                       "primary_cell_count": 72, "files": plan.provenance}
            plan.files["provenance/release_manifest.json"] = json_bytes(release)
            plan.write(output)
            self.assertEqual(verify_release(output)["verified_files"], 1)
            (output / "metrics" / "grid.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_release(output)

    def test_existing_owned_export_can_be_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            plan = ExportPlan(output)
            plan.add("metrics/grid.json", b"{}\n")
            release = {"schema_version": 1, "exporter": "export_public_evidence.py", "release_id": "test",
                       "primary_cell_count": 72, "files": plan.provenance}
            plan.files["provenance/release_manifest.json"] = json_bytes(release)
            plan.write(output)
            plan.write(output)
            self.assertEqual(hashlib.sha256((output / "metrics" / "grid.json").read_bytes()).hexdigest(),
                             release["files"]["metrics/grid.json"]["public_sha256"])

    def test_metric_recovery_uses_scores_and_original_integer_counts(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            np.save(run / "y_test.npy", np.array([0, 1, 0, 1]))
            np.save(run / "y_test_scores.npy", np.array([0.1, 0.4, 0.3, 0.8]))
            (run / "config.json").write_bytes(json_bytes({"model": "logreg", "threshold_exact": 0.5}))
            (run / "metrics_test.json").write_bytes(json_bytes({"PR-AUC": 1.0, "ROC-AUC": 1.0,
                "TP": 1, "FP": 0, "TN": 2, "FN": 1, "threshold": 0.5, "k_used": 2}))
            full = recover_metrics(run)
            self.assertEqual(full["PR-AUC"], 1.0)
            self.assertAlmostEqual(full["F2"], 5 / 9)
            self.assertEqual(full["recall"], 0.5)
            self.assertEqual(full["precision_at_k"], 1.0)
            (run / "config.json").write_bytes(json_bytes({"model": "logreg", "threshold_exact": 0.35}))
            with self.assertRaisesRegex(ValueError, "Frozen exact threshold"):
                recover_metrics(run)

    def test_public_metric_arithmetic_rejects_bad_counts_and_f2(self):
        valid = {"TP": 1, "FP": 0, "TN": 2, "FN": 1, "test_samples": 4,
                 "test_fraud_samples": 2, "F1": 2 / 3, "F2": 5 / 9,
                 "precision": 1.0, "recall": 0.5, "alert_rate": 0.25}
        document = {"cells": {str(index): dict(valid) for index in range(72)}}
        self.assertEqual(verify_primary_metrics(document, 72), 72)
        document["cells"]["0"]["test_fraud_samples"] = 3
        with self.assertRaisesRegex(ValueError, "counts and TEST population"):
            verify_primary_metrics(document, 72)
        document["cells"]["0"] = dict(valid, F2=0.9)
        with self.assertRaisesRegex(ValueError, "count-derived F2"):
            verify_primary_metrics(document, 72)
        document["cells"].pop("0")
        with self.assertRaisesRegex(ValueError, "all 72 cells"):
            verify_primary_metrics(document, 72)

    def test_public_thesis_verification_uses_explicit_project_root(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            output = project / "results"
            thesis = project / "publications" / "thesis.pdf"
            thesis.parent.mkdir()
            thesis.write_bytes(b"frozen thesis fixture")
            plan = ExportPlan(project)
            plan.add("metrics/grid.json", b"{}\n")
            release = {"schema_version": 1, "exporter": "export_public_evidence.py", "release_id": "test",
                       "primary_cell_count": 72, "files": plan.provenance,
                       "thesis_public_path": "publications/thesis.pdf",
                       "thesis_pdf_sha256": hashlib.sha256(thesis.read_bytes()).hexdigest()}
            plan.files["provenance/release_manifest.json"] = json_bytes(release)
            plan.write(output)
            self.assertEqual(verify_release(output, project_root=project)["verified_files"], 1)
            with self.assertRaisesRegex(ValueError, "explicit project root"):
                verify_release(output)
            thesis.write_bytes(b"changed thesis")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_release(output, project_root=project)


if __name__ == "__main__":
    unittest.main()
