"""Expected boundary ties are evaluated once, without changing frozen scores."""

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.metrics import (HISTORICAL_TOP_K_POLICY, TOP_K_POLICY,
                                compute_all_metrics, precision_recall_at_k)
from export_public_evidence import recover_metrics
from generate_results import FULL_PRECISION_CACHE, full_precision_metrics
import generate_results
from supervisor_revision_audit import f2_from_counts


class TopKPolicyTests(unittest.TestCase):
    def test_mixed_boundary_has_analytical_expectation(self):
        labels = np.array([1, 0, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
        precision, recall, budget = precision_recall_at_k(labels, scores, k=2)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertEqual(budget, 2)
        np.testing.assert_array_equal(scores, [0.9, 0.8, 0.8, 0.8, 0.1])

    def test_exact_boundary_is_invariant_to_all_row_permutations(self):
        labels = np.array([1, 0, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
        expected = precision_recall_at_k(labels, scores, k=2)
        original_ap = average_precision_score(labels, scores)
        for order in itertools.permutations(range(5)):
            indices = np.array(order)
            self.assertEqual(precision_recall_at_k(labels[indices], scores[indices], k=2), expected)
            self.assertEqual(average_precision_score(labels[indices], scores[indices]), original_ap)

    def test_all_tied_no_ties_and_no_positives(self):
        self.assertEqual(precision_recall_at_k([1, 0, 1, 0], [1.0] * 4, k=2), (0.5, 0.5, 2))
        self.assertEqual(precision_recall_at_k([1, 0, 1], [0.9, 0.7, 0.8], k=2), (1.0, 1.0, 2))
        self.assertEqual(precision_recall_at_k([0, 0], [0.3, 0.2], k=1), (0.0, 0.0, 1))
        self.assertEqual(precision_recall_at_k([1, 0], [0.3, 0.2], k=2), (0.5, 1.0, 2))

    def test_no_epsilon_can_merge_adjacent_float_scores(self):
        lower = 0.5
        upper = np.nextafter(lower, np.inf)
        self.assertEqual(precision_recall_at_k([0, 1], [lower, upper], k=1), (1.0, 1.0, 1))

    def test_default_budget_uses_floor_with_one_case_minimum(self):
        labels = np.zeros(56746, dtype=int)
        labels[0] = 1
        self.assertEqual(precision_recall_at_k(labels, np.arange(56746))[2], 283)
        self.assertEqual(precision_recall_at_k([1], [0.4]), (1.0, 1.0, 1))

    def test_invalid_input_is_rejected(self):
        for labels, scores in (([], []), ([0], [0.2, 0.3]), ([[0]], [[0.2]]),
                               ([2], [0.2]), ([1], [np.nan]), ([0], [np.inf]),
                               ([1], [1 + 2j]), ([0], ["bad"])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                precision_recall_at_k(labels, scores)
        for budget in (0, -1, 3, 1.5, True):
            with self.subTest(k=budget), self.assertRaises(ValueError):
                precision_recall_at_k([0, 1], [0.2, 0.3], k=budget)
        for fraction in (0, -1, 1.1, np.nan):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                precision_recall_at_k([0, 1], [0.2, 0.3], fraction)

    def test_historical_replay_is_explicit_and_preserves_other_metrics(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.5] * 4)
        current = compute_all_metrics(labels, scores, 0.4)
        historical = compute_all_metrics(labels, scores, 0.4, top_k_policy=HISTORICAL_TOP_K_POLICY)
        self.assertEqual(current["top_k_policy"], TOP_K_POLICY)
        for name in ("PR-AUC", "ROC-AUC", "F1", "F2", "TP", "FP", "TN", "FN", "alert_rate"):
            self.assertEqual(current[name], historical[name])
        self.assertNotEqual(current["precision_at_k"], historical["precision_at_k"])
        with self.assertRaises(ValueError):
            precision_recall_at_k(labels, scores, policy="unknown")

    def test_reporting_and_export_share_the_same_unrounded_policy(self):
        labels = np.array([1, 0, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            np.save(run / "y_test.npy", labels)
            np.save(run / "y_test_scores.npy", scores)
            (run / "config.json").write_text(json.dumps({"model": "logreg", "threshold_exact": 0.85}), encoding="utf-8")
            (run / "metrics_test.json").write_text(json.dumps({
                "PR-AUC": average_precision_score(labels, scores),
                "ROC-AUC": roc_auc_score(labels, scores), "TP": 1, "FP": 0, "TN": 3,
                "FN": 1, "threshold": 0.85, "k_used": 2}), encoding="utf-8")
            expected = precision_recall_at_k(labels, scores, k=2)
            try:
                for consumer in (full_precision_metrics, recover_metrics):
                    result = consumer(run)
                    self.assertEqual(result["top_k_policy"], TOP_K_POLICY)
                    self.assertEqual((result["precision_at_k"], result["recall_at_k"]), expected[:2])
            finally:
                FULL_PRECISION_CACHE.pop(run.resolve(), None)

    def test_robustness_table_keeps_seven_rows_but_summarises_six(self):
        values = dict(zip(generate_results.BALANCE_ORDER, [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6]))
        data = {model: {strategy: {"PR-AUC": value, "F2": value / 2}
                        for strategy, value in values.items()}
                for model in ("fttransformer", "catboost")}
        with tempfile.TemporaryDirectory() as folder:
            original = generate_results.TABLES_BASE
            try:
                generate_results.TABLES_BASE = Path(folder)
                generate_results.generate_transformer_robustness_table(data, "Fixture", "fixture")
                table = (Path(folder) / "fixture/transformer_robustness.tex").read_text(encoding="utf-8")
            finally:
                generate_results.TABLES_BASE = original
        self.assertIn("six effective conditions", table)
        self.assertIn(r"SM+Tomek$^{*}$", table)
        six = [value for key, value in values.items() if key != "smote_tomek"]
        self.assertIn(r"\textit{" + f"{np.std(six, ddof=0):.3f}" + "}", table)
        for strategy in ("None", "RUS", "ROS", "SMOTE", "SMOTEENN", "Weights"):
            self.assertIn(strategy, table)

    def test_f2_numeric_claim_uses_counts_before_rounding(self):
        self.assertEqual(f2_from_counts({"TP": 0, "FP": 0, "FN": 1}), 0.0)
        self.assertEqual(f2_from_counts({"TP": 1, "FP": 0, "FN": 1}), 5 / 9)


if __name__ == "__main__":
    unittest.main()
