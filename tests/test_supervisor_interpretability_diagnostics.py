"""Tests of postprocessing policies, never fitting, inference or new SHAP."""
import json
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervisor_interpretability_diagnostics import (
    COHORTS, EFFECTIVE_STRATEGIES, MAIN_POLICY, OUTPUT, PLANNED_STRATEGIES,
    _population, effective_strategy_summary, jaccard, magnitude_contrasts,
    load_current_shap_summary, ranked, summarise_shap,
)
from export_public_evidence import (
    ROOT, SUPERVISOR_REVISION_ID, ExportPlan, digest, json_bytes,
    preserve_pinned_public_sources, resolve_pinned_author_source,
)


class ImmutablePublicSourceTests(unittest.TestCase):
    def historical_fixture(self, temporary, public="interpretability/shap_summary.json"):
        root = Path(temporary)
        current = root / "results" / public
        historical = root / "results/provenance/historical_pre_supervisor" / public
        current.parent.mkdir(parents=True)
        historical.parent.mkdir(parents=True)
        current.write_bytes(b'{"policy":"net"}\n')
        historical.write_bytes(b'{"policy":"historical_mass"}\n')
        expected = digest(historical)
        prior = {"public_sha256": expected, "bytes": historical.stat().st_size,
                 "sources": [{"archive_path": "immutable/summary.json", "sha256": "a" * 64}],
                 "transformation": "Original portable summary"}
        manifest = {"schema_version": 1, "exporter": "export_public_evidence.py",
                    "supervisor_revision_id": SUPERVISOR_REVISION_ID,
                    "files": {
                        public: {
                            "public_sha256": digest(current), "bytes": current.stat().st_size,
                            "supersedes_export_provenance": prior},
                        "provenance/historical_pre_supervisor/" + public: {
                            "public_sha256": expected, "bytes": historical.stat().st_size,
                            "original_export_provenance": dict(prior)},
                    }}
        path = root / "results/provenance/release_manifest.json"
        path.write_bytes(json_bytes(manifest))
        return root, current, historical, expected, path

    def test_exact_source_stays_direct_without_a_public_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "immutable.json"
            source.write_bytes(b"{}")
            self.assertEqual(resolve_pinned_author_source(root, "immutable.json", digest(source)), source)

    def test_only_the_owned_and_hash_verified_historical_shap_source_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, current, historical, expected, _ = self.historical_fixture(temporary)
            before = current.read_bytes(), historical.read_bytes()
            self.assertEqual(resolve_pinned_author_source(root, "results/interpretability/shap_summary.json", expected), historical)
            self.assertEqual((current.read_bytes(), historical.read_bytes()), before)

    def test_exact_allowlisted_metric_source_uses_only_its_pinned_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = "metrics/baf_base/fttransformer/none/metrics_full_precision.json"
            root, current, historical, expected, _ = self.historical_fixture(temporary, public)
            self.assertEqual(resolve_pinned_author_source(root, "results/" + public, expected), historical)
            current.write_bytes(b"changed current metric")
            with self.assertRaises(ValueError):
                resolve_pinned_author_source(root, "results/" + public, expected)

    def test_two_explicit_ft_threshold_aliases_require_the_same_exact_provenance(self):
        for dataset in ("ulb_2013", "baf_base"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                public = f"thresholds/{dataset}/fttransformer.json"
                root, _, historical, expected, _ = self.historical_fixture(temporary, public)
                self.assertEqual(resolve_pinned_author_source(root, "results/" + public, expected), historical)

    def test_future_export_retains_exact_historical_metric_and_supersession_bridge(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = "metrics/baf_base/fttransformer/none/metrics_full_precision.json"
            root, current, historical, expected, _ = self.historical_fixture(temporary, public)
            immutable_manifest = root / "immutable_derivative.json"
            immutable_manifest.write_bytes(json_bytes({"source_sha256": {"results/" + public: expected}}))
            plan = ExportPlan(root)
            plan.add(public, current.read_bytes())
            before = current.read_bytes(), historical.read_bytes(), immutable_manifest.read_bytes()
            preserve_pinned_public_sources(plan, {"files": {"interpretability_transfer/manifest.json": immutable_manifest}})
            alias = "provenance/historical_pre_supervisor/" + public
            self.assertEqual(plan.files[alias], historical.read_bytes())
            self.assertEqual(plan.provenance[public]["supersedes_export_provenance"],
                             plan.provenance[alias]["original_export_provenance"])
            self.assertEqual(plan.provenance[alias]["public_sha256"], expected)
            self.assertEqual((current.read_bytes(), historical.read_bytes(), immutable_manifest.read_bytes()), before)

    def test_future_export_rejects_a_conflicting_historical_asset_in_its_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, current, _, expected, _ = self.historical_fixture(temporary)
            immutable_manifest = root / "immutable_derivative.json"
            immutable_manifest.write_bytes(json_bytes({"source_sha256": {"results/interpretability/shap_summary.json": expected}}))
            plan = ExportPlan(root)
            plan.add("interpretability/shap_summary.json", current.read_bytes())
            plan.add("provenance/historical_pre_supervisor/interpretability/shap_summary.json", b"not the pinned original")
            with self.assertRaisesRegex(ValueError, "immutable pin"):
                preserve_pinned_public_sources(plan, {"files": {"interpretability_transfer/manifest.json": immutable_manifest}})

    def test_future_export_uses_union_of_interpretability_and_canonical_threshold_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = "metrics/ulb_2013/logreg/none/metrics_full_precision.json"
            root, current, historical, expected, _ = self.historical_fixture(temporary, public)
            immutable_manifest = root / "immutable_derivative.json"
            immutable_manifest.write_bytes(json_bytes({"source_sha256": {}}))
            plan = ExportPlan(root)
            plan.add(public, current.read_bytes())
            bundle = {"files": {"interpretability_transfer/manifest.json": immutable_manifest},
                      "reports": {"basic_verified": {"sources": [{"path": "results/" + public, "sha256": expected}]}}}
            preserve_pinned_public_sources(plan, bundle)
            alias = "provenance/historical_pre_supervisor/" + public
            self.assertEqual(plan.files[alias], historical.read_bytes())
            # Repeated preparation preserves the same bytes and provenance.
            preserve_pinned_public_sources(plan, bundle)
            self.assertEqual(plan.provenance[alias]["public_sha256"], expected)

    def test_canonical_source_union_rejects_disagreeing_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, current, _, expected, _ = self.historical_fixture(temporary)
            logical = "results/interpretability/shap_summary.json"
            immutable_manifest = root / "immutable_derivative.json"
            immutable_manifest.write_bytes(json_bytes({"source_sha256": {logical: expected}}))
            plan = ExportPlan(root)
            plan.add("interpretability/shap_summary.json", current.read_bytes())
            bundle = {"files": {"interpretability_transfer/manifest.json": immutable_manifest},
                      "reports": {"basic_verified": {"sources": [{"path": logical, "sha256": "0" * 64}]}}}
            with self.assertRaisesRegex(ValueError, "identities disagree"):
                preserve_pinned_public_sources(plan, bundle)

    def test_current_or_historical_tampering_fails_closed(self):
        for which in ("current", "historical"):
            with self.subTest(which=which), tempfile.TemporaryDirectory() as temporary:
                root, current, historical, expected, _ = self.historical_fixture(temporary)
                (current if which == "current" else historical).write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    resolve_pinned_author_source(root, "results/interpretability/shap_summary.json", expected)

    def test_missing_manifest_or_changed_supersession_proof_is_not_an_alias(self):
        for change in ("missing", "different_proof", "different_exporter"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root, _, _, expected, path = self.historical_fixture(temporary)
                manifest = json.loads(path.read_text())
                if change == "missing":
                    path.unlink()
                elif change == "different_proof":
                    manifest["files"]["interpretability/shap_summary.json"]["supersedes_export_provenance"]["bytes"] += 1
                    path.write_bytes(json_bytes(manifest))
                else:
                    manifest["exporter"] = "unowned.py"
                    path.write_bytes(json_bytes(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    resolve_pinned_author_source(root, "results/interpretability/shap_summary.json", expected)

    def test_a_matching_historical_file_for_any_other_source_is_not_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, historical, expected, _ = self.historical_fixture(temporary)
            other = root / "results/metrics/changed.json"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"changed")
            alias = root / "results/provenance/historical_pre_supervisor/metrics/changed.json"
            alias.parent.mkdir(parents=True)
            alias.write_bytes(historical.read_bytes())
            with self.assertRaises(ValueError):
                resolve_pinned_author_source(root, "results/metrics/changed.json", expected)


class GroupedShapTests(unittest.TestCase):
    def test_signed_before_absolute_not_absolute_before_signed(self):
        values = np.array([[.4, -.3, .2], [-.4, .3, -.2]])
        names = ["device_os_linux", "device_os_windows", "age"]
        result = summarise_shap(values, names, chunk_rows=1)
        self.assertAlmostEqual(result["complete_global_importance"]["device_os"], .1)
        self.assertAlmostEqual(result["encoded_absolute_mass_sensitivity"]["device_os"], .7)
        # abs(mean(signed_group)) would have given zero, which is not our policy.
        self.assertAlmostEqual(result["complete_global_importance"]["age"], .2)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(len(result["original_features"]), 2)
        self.assertLess(result["maximum_signed_grouping_additivity_error"], 1e-15)

    def test_complete_vector_before_top10(self):
        names = [f"numeric{i:02}" for i in range(30)]
        values = np.tile(np.arange(30, dtype=float), (3, 1))
        result = summarise_shap(values, names)
        self.assertEqual(len(result["complete_global_importance"]), 30)
        self.assertEqual(result["ranking"][0], "numeric29")
        self.assertEqual(len(result["top10"]), 10)
        self.assertEqual(result["top10"][-1], "numeric20")

    def test_chunk_and_row_order_invariance(self):
        values = np.array([[.4, -.3, .2], [-.4, .3, -.2], [.2, .1, .9]])
        names = ["device_os_linux", "device_os_windows", "age"]
        a = summarise_shap(values, names, 1)
        b = summarise_shap(values[::-1], names, 7)
        for key in ("complete_global_importance", "encoded_absolute_mass_sensitivity"):
            for feature in a[key]:
                self.assertAlmostEqual(a[key][feature], b[key][feature], places=15)

    def test_invalid_inputs(self):
        for values, names in ((np.empty((0, 2)), ["a", "b"]),
                              (np.ones((2, 2)), ["a"]),
                              (np.ones((2, 2)), ["a", "a"]),
                              (np.array([[np.nan]]), ["a"])):
            with self.subTest(names=names), self.assertRaises(ValueError):
                summarise_shap(values, names)

    def test_jaccard_does_not_compare_order_or_magnitude(self):
        self.assertEqual(jaccard(["a", "b"], ["b", "a"]), 1.0)
        self.assertAlmostEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(ranked({"b": 1., "a": 1.}), ["a", "b"])

    def test_zero_reference_is_undefined(self):
        cohorts = {}
        for cohort in COHORTS:
            importance = {"zero": 0.0, "age": 2.0 if cohort == "baf_base" else 3.0}
            cohorts[cohort] = {"complete_global_importance": importance,
                               "original_features": list(importance), "ranking": ranked(importance)}
        result = magnitude_contrasts(cohorts)
        self.assertTrue(all(r["relative_difference_percent"] is None for r in result if r["feature"] == "zero"))
        self.assertTrue(all(r["relative_difference_percent"] == 50 for r in result if r["feature"] == "age" and r["cohort"] != "baf_base"))


class StrategyAndTransferTests(unittest.TestCase):
    def test_six_conditions_not_generic_deduplication(self):
        metrics = dict(zip(PLANNED_STRATEGIES, (1., 2., 2., 4., 4., 6., 7.)))
        result = effective_strategy_summary(metrics)
        expected = [1., 2., 2., 4., 6., 7.]
        self.assertEqual(len(PLANNED_STRATEGIES), 7)
        self.assertEqual(len(EFFECTIVE_STRATEGIES), 6)
        self.assertEqual(result["effective_conditions"], list(EFFECTIVE_STRATEGIES))
        self.assertEqual(result["mean"], np.mean(expected))
        self.assertEqual(result["population_sd"], np.std(expected, ddof=0))
        self.assertNotEqual(result["population_sd"], np.std(expected, ddof=1))
        self.assertNotEqual(result["population_sd"], result["historical_seven_population_sd"])
        self.assertEqual(metrics["smote_tomek"], 4.)

    def test_seven_labels_and_finite_values_required(self):
        with self.assertRaises(ValueError):
            effective_strategy_summary(dict.fromkeys(EFFECTIVE_STRATEGIES, 0.))
        with self.assertRaises(ValueError):
            effective_strategy_summary(dict.fromkeys(PLANNED_STRATEGIES, np.nan))

    def test_population_uses_integer_counts_and_distinct_rate_denominator(self):
        result = _population({"TP": 7, "FP": 3, "TN": 88, "FN": 2})
        self.assertEqual(result, {"rows": 100, "frauds": 9, "alerts": 10, "alert_rate": .1})
        with self.assertRaises(ValueError):
            _population({"TP": -1, "FP": 3, "TN": 88, "FN": 2})
        with self.assertRaises(ValueError):
            _population({"TP": 1.5, "FP": 3, "TN": 88, "FN": 2})


@unittest.skipUnless((OUTPUT / "manifest.json").is_file(), "Dated derived evidence not generated")
class FrozenEvidenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shap = json.loads((OUTPUT / "shap_summary.json").read_text(encoding="utf-8"))
        cls.transfer = json.loads((OUTPUT / "transfer_summary.json").read_text(encoding="utf-8"))
        cls.strategy = json.loads((OUTPUT / "strategy_summary.json").read_text(encoding="utf-8"))

    def test_complete_vectors_and_confirmed_membership(self):
        self.assertEqual(self.shap["policy"], MAIN_POLICY)
        for model in self.shap["models"].values():
            self.assertEqual(len(model["complete_global_importance"]), 30)
            self.assertEqual(len(model["top10"]), 10)
        self.assertEqual(self.shap["cross_model_top10_jaccard"]["lgbm"]["catboost"], 1.)
        self.assertTrue(all(value == 1 for row in self.shap["cohort_top10_jaccard"].values() for value in row.values()))
        self.assertFalse(self.shap["new_shap_computation"])
        self.assertFalse(self.shap["model_inference"])
        self.assertFalse(self.shap["model_fitting"])

    def test_main_reporting_consumer_requires_current_policy_without_model_load(self):
        import generate_results
        manifest_path = Path(__file__).resolve().parents[1] / "results_revision/20261005/revision_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with patch.object(generate_results, "RUN_MANIFEST", manifest), patch("joblib.load", side_effect=AssertionError("No model load in grouping consumer")):
            global_data, consistency, stability, _ = generate_results.collect_shap_data()
        self.assertEqual(list(global_data["logreg"])[:3], self.shap["models"]["logreg"]["ranking"][:3])
        self.assertEqual(consistency["grouping_policy"], MAIN_POLICY)
        self.assertEqual(stability["grouping_policy"], MAIN_POLICY)

    def test_portable_summary_preserves_local_cases(self):
        manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        logical = "results/interpretability/shap_summary.json"
        original_path = resolve_pinned_author_source(ROOT, logical, manifest["source_sha256"][logical])
        original = json.loads(original_path.read_text(encoding="utf-8"))
        current = json.loads((OUTPUT / "portable_shap_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(original["local_cases"], current["local_cases"])
        self.assertEqual(current["policy"], MAIN_POLICY)
        self.assertNotEqual(original["models"]["logreg"]["complete_global_importance"], current["models"]["logreg"]["complete_global_importance"])

    def test_no_fallback_to_historical_when_new_derivative_missing(self):
        with self.assertRaises(FileNotFoundError):
            load_current_shap_summary(OUTPUT / "does_not_exist")

    def test_longtable_captions_are_inside_counter_owner(self):
        for filename in ("transfer_overlap_full.tex", "transfer_alert_counts.tex"):
            content = (OUTPUT / "tables" / filename).read_text(encoding="utf-8")
            self.assertTrue(content.startswith(r"\begin{longtable}"))
            self.assertEqual(content.count(r"\caption{\SupervisorLongTableCaption}"), 1)
            self.assertEqual(content.count(r"\label{\SupervisorLongTableLabel}"), 1)
            self.assertLess(content.index(r"\caption{"), content.index(r"\toprule"))
            self.assertNotIn(r"\captionof", content)

    def test_regrouping_changes_leaders_and_age_ranks_not_its_magnitude(self):
        self.assertEqual(self.shap["models"]["logreg"]["ranking"][:3], ["phone_home_valid", "device_os", "has_other_cards"])
        for row in self.shap["magnitude_contrasts"]:
            if row["feature"] == "customer_age" and row["cohort"] in COHORTS[2:]:
                self.assertEqual(row["rank"], 11)
                self.assertTrue(71.2 < row["relative_difference_percent"] < 71.8)
        self.assertEqual(len(self.shap["magnitude_contrasts"]), 180)

    def test_all_transfer_pairs_same_cohorts_and_rf_ranks(self):
        self.assertEqual(len(self.transfer["pairs"]), 30)
        self.assertEqual(len(self.transfer["workload"]), 36)
        for row in self.transfer["pairs"]:
            self.assertEqual(row["original_rows"], 200000)
            if row["model"] == "rf":
                self.assertEqual((row["original_ap_rank"], row["filtered_ap_rank"]), (1, 4))
                self.assertEqual(row["excluded_tp"], row["excluded_frauds"])
                self.assertTrue(39 <= row["original_ap_relative_to_filtered_percent"] <= 54)
        self.assertTrue(self.transfer["historical_ft_excluded"])
        self.assertTrue(self.transfer["validation_checkpoint_A_excluded"])

    def test_confirmed_six_condition_sd_and_all_runs_retained(self):
        self.assertEqual(self.strategy["primary_configurations_retained"], 72)
        for dataset, model, ap_sd in (("ulb_2013", "fttransformer", .05541387125402923),
                                      ("ulb_2013", "catboost", .06325479156104857),
                                      ("baf_base", "fttransformer", .024744153260653254),
                                      ("baf_base", "catboost", .011089194142685494)):
            self.assertAlmostEqual(self.strategy["summaries"][f"{dataset}/{model}"]["PR-AUC"]["population_sd"], ap_sd, places=13)


if __name__ == "__main__":
    unittest.main()
