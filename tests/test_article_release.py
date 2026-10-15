"""Portable article evidence guards; no training or private archive required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import article_release as release


class ArticleReleaseTests(unittest.TestCase):
    def reviewed_fixture(self, temporary, article=1):
        root, destination = Path(temporary) / "author", Path(temporary) / "article"
        root.mkdir()
        destination.mkdir()
        (destination / ".git").mkdir()

        def write(folder, relative, data):
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path

        payload = write(destination, "results/metrics/example.json", b'{"version":"old"}\n')
        frozen = write(destination, "results/frozen/example.npz", b"unchanged frozen array fixture")
        pdf_path = f"publications/article_{article}.pdf"
        old_pdf = write(destination, pdf_path, b"%PDF-1.4 old")
        keys = [f"{dataset}/{model}/{strategy}" for dataset in ("ulb_2013", "baf_base")
                for model in ("logreg", "rf", "lgbm", "catboost", "fttransformer")
                for strategy in ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")]
        keys += ["ulb_2013/ocsvm/none", "baf_base/ocsvm/none"]
        run = {"archive": "results/frozen/example.npz", "metrics": "results/metrics/example.json",
               "config": "results/metrics/example.json"}
        manifest = {"schema": "dedicated_article_evidence_v1", "article": article,
                    "primary_runs": {key: dict(run) for key in keys},
                    "control_runs": {str(index): dict(run) for index in range(8)},
                    "transfer_runs": {str(index): dict(run) for index in range(30)} if article == 1 else {},
                    "threshold_studies": {str(index): "results/metrics/example.json" for index in range(12)} if article == 2 else {},
                    "files": {str(path.relative_to(destination).as_posix()): {"sha256": release.digest(path), "bytes": path.stat().st_size}
                              for path in (payload, frozen, old_pdf)}}
        write(destination, release.MANIFEST, release.json_bytes(manifest))
        pdf = write(root, "deliverables/reviewed.pdf", b"%PDF-1.4 reviewed fixture")
        validation = {"status": "passed_in_reviewed_scope", "pdfs": {
            f"article_{article}": {"path": "deliverables/reviewed.pdf", "sha256": release.digest(pdf), "pages": 21}}}
        validation_path = write(root, "reports/final_pdf_validation_20261015_r2.json", release.json_bytes(validation))
        parent_files = {}
        for relative, data in {
            "metrics/example.json": b'{"version":"reviewed"}\n',
            "controls/supervisor_paired_ap.json": b'{"requested":1000}\n',
            "interpretability/shap_summary.json": b'{"policy":"net"}\n',
            "transfer/supervisor_summary.json": b'{"filtered":true}\n',
            "thresholds/baf_base/fttransformer.json": b'{"metadata":"corrected"}\n',
            "provenance/dataset_protocol.json": b'{"ulb_2013":{},"baf_base":{},"baf_var1":{}}\n',
        }.items():
            path = write(root, "results/" + relative, data)
            parent_files[relative] = {"public_sha256": release.digest(path), "bytes": path.stat().st_size}
        parent = {"exporter": "export_public_evidence.py", "release_id": "reviewed_r2",
                  "supervisor_revision_id": "20261015_supervisor", "files": parent_files}
        write(root, "results/provenance/release_manifest.json", release.json_bytes(parent))
        article_text = "\\documentclass{cas-sc}\n"
        if article == 2:
            article_text += "\\begin{tabular}{c}value\\end{tabular}\n" * 14
        write(root, f"Article {article}/main.tex", article_text.encode("utf-8"))
        return root, destination, pdf, frozen, validation_path

    def sync_fixture(self, root, destination, pdf, article=1, **overrides):
        kwargs = {"release_id": "reviewed_r2", "pdf": pdf, "expected_pdf_sha256": release.digest(pdf)}
        kwargs.update(overrides)
        with mock.patch.object(release.subprocess, "check_output", return_value="a" * 40 + "\n"):
            return release.sync_reviewed_release(root, destination, article, **kwargs)

    def test_reviewed_sync_preserves_frozen_arrays_and_archives_replaced_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, frozen, _ = self.reviewed_fixture(temporary)
            initial_frozen_hash = release.digest(frozen)
            result = self.sync_fixture(root, destination, pdf)
            self.assertEqual(result["release_id"], "reviewed_r2")
            self.assertEqual(release.digest(frozen), initial_frozen_hash)
            history = destination / "results/provenance/historical_releases/before_reviewed_r2"
            self.assertEqual((history / "results/metrics/example.json").read_bytes(), b'{"version":"old"}\n')
            self.assertEqual((history / "publications/article_1.pdf").read_bytes(), b"%PDF-1.4 old")
            self.assertEqual((destination / "publications/article_1.pdf").read_bytes(), pdf.read_bytes())
            self.assertEqual(result["reviewed_pdf"]["sha256"], release.digest(pdf))
            self.assertTrue((destination / "results/interpretability/shap_summary.json").is_file())
            # Repeating an already-completed update is safe and does not recurse
            # a new manifest into the original historical release.
            before = (history / "article_manifest.json").read_bytes()
            self.sync_fixture(root, destination, pdf)
            self.assertEqual((history / "article_manifest.json").read_bytes(), before)

    def test_article_two_scope_excludes_shap_and_variant_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary, article=2)
            self.sync_fixture(root, destination, pdf, article=2)
            self.assertFalse((destination / "results/interpretability/shap_summary.json").exists())
            self.assertFalse((destination / "results/transfer/supervisor_summary.json").exists())
            protocol = release.read_json(destination / "results/provenance/dataset_protocol.json")
            self.assertEqual(set(protocol), {"ulb_2013", "baf_base"})
            self.assertTrue((destination / "results/thresholds/baf_base/fttransformer.json").is_file())
            self.assertEqual(len(list((destination / "results/tables").glob("table_*.tex"))), 14)

    def test_article_two_no_pdf_repairs_only_the_absent_old_pdf_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, frozen, _ = self.reviewed_fixture(temporary, article=2)
            target = destination / "publications/article_2.pdf"
            target.unlink()  # Synthetic fixture mirrors the collaborator's commit.
            with self.assertRaisesRegex(ValueError, "article_2.pdf"):
                release.verify(destination)
            frozen_hash = release.digest(frozen)
            result = self.sync_fixture(root, destination, pdf, article=2, publish_pdf=False)
            self.assertFalse(target.exists())
            self.assertNotIn("publications/article_2.pdf", result["files"])
            self.assertFalse(result["reviewed_pdf"]["distributed"])
            self.assertNotIn("path", result["reviewed_pdf"])
            self.assertEqual(result["reviewed_pdf"]["sha256"], release.digest(pdf))
            self.assertEqual(result["reviewed_pdf"]["external_reference"], release.PARENT + "/blob/main/publications/article_2.pdf")
            self.assertEqual(release.digest(frozen), frozen_hash)
            self.assertFalse(list((destination / "results").rglob("article_2.pdf")))
            self.assertEqual(release.verify(destination)["release_id"], "reviewed_r2")
            self.sync_fixture(root, destination, pdf, article=2, publish_pdf=False)

    def test_no_pdf_migration_cannot_ignore_another_missing_or_changed_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, frozen, _ = self.reviewed_fixture(temporary, article=2)
            (destination / "publications/article_2.pdf").unlink()
            frozen.unlink()
            with self.assertRaisesRegex(ValueError, "frozen/example.npz"):
                self.sync_fixture(root, destination, pdf, article=2, publish_pdf=False)
            self.assertEqual((destination / "results/metrics/example.json").read_bytes(), b'{"version":"old"}\n')
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary, article=2)
            (destination / "publications/article_2.pdf").write_bytes(b"%PDF-1.4 changed")
            with self.assertRaisesRegex(ValueError, "article_2.pdf"):
                release.verify(destination, allow_missing_replaced_pdf=True)

    def test_no_pdf_policy_does_not_delete_a_present_pdf_or_apply_to_article_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary, article=2)
            target = destination / "publications/article_2.pdf"
            with self.assertRaisesRegex(FileExistsError, "still contains a manuscript"):
                self.sync_fixture(root, destination, pdf, article=2, publish_pdf=False)
            self.assertEqual(target.read_bytes(), b"%PDF-1.4 old")
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary, article=1)
            with self.assertRaisesRegex(ValueError, "only for Article 2"):
                self.sync_fixture(root, destination, pdf, article=1, publish_pdf=False)

    def test_reviewed_sync_rejects_missing_pdf_review_or_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, validation = self.reviewed_fixture(temporary)
            with self.assertRaisesRegex(ValueError, "explicitly reviewed PDF"):
                self.sync_fixture(root, destination, pdf, expected_pdf_sha256="0" * 64)
            record = release.read_json(validation)
            record["status"] = "not_validated"
            validation.write_bytes(release.json_bytes(record))
            with self.assertRaisesRegex(ValueError, "passed review"):
                self.sync_fixture(root, destination, pdf)
            self.assertEqual((destination / "publications/article_1.pdf").read_bytes(), b"%PDF-1.4 old")

    def test_reviewed_sync_preflights_unowned_targets_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary)
            target = destination / "results/controls/supervisor_paired_ap.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"unrelated author modification")
            with self.assertRaisesRegex(FileExistsError, "Unowned or modified"):
                self.sync_fixture(root, destination, pdf)
            self.assertEqual((destination / "results/metrics/example.json").read_bytes(), b'{"version":"old"}\n')
            self.assertFalse((destination / "results/interpretability/shap_summary.json").exists())

    def test_reviewed_sync_requires_corrected_parent_release_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary)
            with self.assertRaisesRegex(ValueError, "corrected parent release"):
                self.sync_fixture(root, destination, pdf, release_id="unreviewed")

    def test_reviewed_manifest_cannot_pin_a_different_pdf_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary)
            result = self.sync_fixture(root, destination, pdf)
            result["reviewed_pdf"]["sha256"] = "0" * 64
            (destination / release.MANIFEST).write_bytes(release.json_bytes(result))
            with self.assertRaisesRegex(ValueError, "Reviewed PDF identity"):
                release.verify(destination)

    def test_reviewed_sync_checks_parent_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, destination, pdf, _, _ = self.reviewed_fixture(temporary)
            (root / "results/metrics/example.json").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "parent aggregate differs"):
                self.sync_fixture(root, destination, pdf)

    def test_reviewed_allowlist_cannot_publish_arrays_private_or_irrelevant_files(self):
        for relative in ("frozen/array.npz", "interpretability/raw.npy", "../secret.json",
                         "provenance/release_manifest.json", "provenance/private_review.json"):
            self.assertFalse(release._reviewed_result_path(relative, 1))
        self.assertFalse(release._reviewed_result_path("interpretability/shap_summary.json", 2))
        self.assertFalse(release._reviewed_result_path("transfer/supervisor_summary.json", 2))
        self.assertTrue(release._reviewed_result_path("provenance/strategy_summary.json", 2))

    def test_relative_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            for unsafe in ("../secret", "a/../../secret", str(Path(temporary).resolve())):
                with self.assertRaises(ValueError):
                    release.safe_path(temporary, unsafe)

    def test_changed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "results/example.json"
            payload.parent.mkdir()
            payload.write_text("{}", encoding="utf-8")
            record = {"sha256": release.digest(payload), "bytes": payload.stat().st_size}
            keys = [f"{dataset}/{model}/{strategy}" for dataset in ("ulb_2013", "baf_base")
                    for model in ("logreg", "rf", "lgbm", "catboost", "fttransformer")
                    for strategy in ("none", "rus", "ros", "smote", "smote_tomek", "smoteenn", "weights")]
            keys += ["ulb_2013/ocsvm/none", "baf_base/ocsvm/none"]
            primary = {key: {"archive": "results/example.json", "metrics": "results/example.json", "config": "results/example.json"} for key in keys}
            derived = {str(index): {"archive": "results/example.json", "metrics": "results/example.json"} for index in range(30)}
            manifest = {"schema": "dedicated_article_evidence_v1", "article": 1, "primary_runs": primary,
                        "control_runs": dict(list(derived.items())[:8]), "transfer_runs": derived, "threshold_studies": {},
                        "files": {"results/example.json": record}}
            target = root / release.MANIFEST
            target.parent.mkdir()
            target.write_bytes(release.json_bytes(manifest))
            self.assertEqual(release.verify(root)["article"], 1)
            payload.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                release.verify(root)

    def test_frozen_metrics_and_infeasible_threshold(self):
        import numpy as np
        labels, scores = np.array([0, 1, 1, 0]), np.array([.1, .8, .6, .3])
        values = release.frozen_metrics(labels, scores, .5)
        self.assertEqual((values["TP"], values["FP"], values["FN"], values["TN"]), (2, 0, 0, 2))
        self.assertEqual(values["F2"], 1.)
        none = release.frozen_metrics(labels, scores, float("inf"))
        self.assertIsNone(none["threshold"])
        self.assertEqual(none["TP"] + none["FP"], 0)
        self.assertIsNotNone(json.loads(release.json_bytes(none)))


if __name__ == "__main__":
    unittest.main()
