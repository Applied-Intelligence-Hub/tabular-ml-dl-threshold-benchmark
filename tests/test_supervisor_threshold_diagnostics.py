"""Small deterministic tests; no empirical fitting, inference or archive writes."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervisor_threshold_diagnostics import (
    counts_metrics, metrics_from_counts, new_phase, paired_ap_difference,
    prior_constant, prior_transform, tabular,
)
from revision_thresholds import precision_constraint_scope


def test_precision_undefined_and_target_exact_counts():
    result = metrics_from_counts({"TP": 0, "FP": 0, "TN": 9, "FN": 1})
    assert result["precision"] is None
    assert not result["precision_target_0_5_met"]
    almost = metrics_from_counts({"TP": 4999, "FP": 5000, "TN": 0, "FN": 1})
    assert round(almost["precision"], 3) == 0.5
    assert not almost["precision_target_0_5_met"]
    assert metrics_from_counts({"TP": 1, "FP": 1, "TN": 0, "FN": 0})["precision_target_0_5_met"]


def test_counts_metrics_and_F2_formula():
    result = counts_metrics(np.array([1, 1, 0, 0]), np.array([0.6, 0.1, 0.5, 0.2]), 0.5)
    assert (result["TP"], result["FP"], result["TN"], result["FN"]) == (1, 1, 1, 1)
    assert result["F2"] == 0.5
    with unittest.TestCase().assertRaises(ValueError):
        metrics_from_counts({"TP": 0.5, "FP": 1, "TN": 3, "FN": 1})


def test_prior_preserves_extremes_identity_order_ties_and_cuts():
    scores = np.array([0, 0.02, 0.02, 0.1, 0.5, 0.9, 1.0])
    assert np.array_equal(prior_transform(scores, 1), scores)
    for constant in (0.01, 0.1, 2.0, 50.0):
        transformed = prior_transform(scores, constant)
        assert (transformed[0], transformed[-1]) == (0, 1)
        assert transformed[1] == transformed[2]
        assert np.all(np.diff(prior_transform(np.unique(scores), constant)) > 0)
        for cut in scores:
            assert np.array_equal(scores >= cut, transformed >= prior_transform([cut], constant)[0])
    for bad in (0, -1, np.inf, np.nan):
        with unittest.TestCase().assertRaises(ValueError):
            prior_transform(scores, bad)
    with unittest.TestCase().assertRaises(ValueError):
        prior_transform([1.01], 0.1)


def test_prior_constant_excludes_other_strategies_and_wrong_population():
    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    cfg = {"train_fraud": 8823, "train_samples": 800000, "model": "rf"}
    c, evidence = prior_constant(cfg, "ros", tmp_path)
    assert c == (8823 / 800000) / (1 - 8823 / 800000)
    assert evidence["observed_historical_sampler_counts"] is False
    with unittest.TestCase().assertRaises(ValueError):
        prior_constant(cfg, "smote", tmp_path)
    with unittest.TestCase().assertRaises(ValueError):
        prior_constant({**cfg, "train_fraud": 2206}, "ros", tmp_path)
    temporary.cleanup()


def test_paired_AP_reproducible_sign_and_monoclass_skips():
    y = np.array([0, 0, 0, 1, 1])
    first = np.array([0.7, 0.6, 0.4, 0.2, 0.1])
    second = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    a = paired_ap_difference(y, first, second, iterations=40, seed=42)
    b = paired_ap_difference(y, first, second, iterations=40, seed=42)
    assert a == b
    assert a["difference_second_minus_first"] > 0
    assert a["valid"] + a["discarded_monoclass"] == 40
    assert a["valid"] < 40
    swapped = paired_ap_difference(y, second, first, iterations=40, seed=42)
    assert swapped["difference_second_minus_first"] == -a["difference_second_minus_first"]
    assert np.array_equal(-np.array(a["bootstrap_differences"]), swapped["bootstrap_differences"])
    same = paired_ap_difference(y, first, first, iterations=40)
    assert same["percentile_95_interval"] == [0, 0]


def test_metadata_single_holdout_vs_fold_median():
    assert "single DEV validation holdout" in precision_constraint_scope("fttransformer")
    assert "full-DEV final network" in precision_constraint_scope("fttransformer")
    assert "median" not in precision_constraint_scope("fttransformer")
    assert "median" in precision_constraint_scope("rf")


def test_protected_output_and_tabular():
    with unittest.TestCase().assertRaises(ValueError):
        new_phase(Path(tempfile.gettempdir()) / "invalid_threshold_diagnostic", "basic")
    content = tabular(["A", "B"], [["row", "1"]])
    assert "\\begin{tabular}" in content and "row & 1 \\\\" in content


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
