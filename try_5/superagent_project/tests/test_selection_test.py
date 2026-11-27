"""Selection must report whether the winner is distinguishable from its rivals.

The tests here fall into three groups:

1. the statistics (exact McNemar, bootstrap CI, MDE) are correct;
2. the annotation is ADDITIVE -- adding labels or predictions must never change
   which model is selected;
3. label loading covers the three data formats and degrades safely.

Group 2 is the load-bearing one: the whole point of this change is that it
records uncertainty without silently altering behaviour.
"""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.base import ExperimentResult
from superagent.model_selector import select_best_model
from superagent.selection_labels import LabelLoadError, load_validation_labels
from superagent.selection_test import (
    accuracy_interval_halfwidth,
    evaluate_selection,
    mcnemar_exact,
    minimum_detectable_difference,
    paired_bootstrap_delta,
)


def make_result(exp_id, accuracy, predictions_path=None, seconds=1.0,
                model="model"):
    return ExperimentResult(
        experiment_id=exp_id,
        model_name=model,
        status="success",
        validation_accuracy=accuracy,
        training_time_seconds=seconds,
        artifact_path="model.joblib",
        val_predictions_path=predictions_path,
    )


class McNemarTest(unittest.TestCase):
    def test_identical_predictions_are_not_significant(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        n_a, n_b, p = mcnemar_exact(y, y, y)
        self.assertEqual((n_a, n_b), (0, 0))
        self.assertEqual(p, 1.0)

    def test_large_asymmetric_discordance_is_significant(self):
        # a is right on 40 rows b gets wrong, and b is right on 0 that a misses
        y = np.array([1] * 40 + [0] * 40)
        a = np.array([1] * 40 + [0] * 40)          # perfect
        b = np.array([0] * 40 + [0] * 40)          # wrong on the 1s
        n_a_only, n_b_only, p = mcnemar_exact(y, a, b)
        self.assertEqual(n_a_only, 40)
        self.assertEqual(n_b_only, 0)
        self.assertLess(p, 0.001)

    def test_small_symmetric_discordance_is_not_significant(self):
        y = np.array([1, 0, 1, 0])
        a = np.array([1, 0, 0, 1])
        b = np.array([0, 1, 1, 0])
        _, _, p = mcnemar_exact(y, a, b)
        self.assertGreater(p, 0.5)

    def test_p_is_symmetric_in_its_arguments(self):
        y = np.array([1] * 30 + [0] * 30)
        a = np.array([1] * 30 + [0] * 30)
        b = np.array([0] * 30 + [0] * 30)
        self.assertEqual(mcnemar_exact(y, a, b)[2], mcnemar_exact(y, b, a)[2])


class BootstrapTest(unittest.TestCase):
    def test_identical_predictions_give_zero_width_interval(self):
        y = np.array([0, 1, 1, 0, 1])
        d, lo, hi = paired_bootstrap_delta(y, y, y, n_boot=200)
        self.assertEqual(d, 0.0)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        a = y.copy()
        b = y.copy()
        # b is wrong on 20% of rows, a on 10%
        flip_b = rng.choice(500, 100, replace=False)
        flip_a = rng.choice(500, 50, replace=False)
        b[flip_b] = 1 - b[flip_b]
        a[flip_a] = 1 - a[flip_a]
        d, lo, hi = paired_bootstrap_delta(y, a, b, n_boot=2000)
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)
        self.assertGreater(d, 0)


class MetricHelpersTest(unittest.TestCase):
    def test_halfwidth_shrinks_with_more_rows(self):
        small = accuracy_interval_halfwidth(0.8, 100)
        large = accuracy_interval_halfwidth(0.8, 10_000)
        self.assertGreater(small, large)

    def test_mde_shrinks_with_more_rows(self):
        small = minimum_detectable_difference(940, 0.126)
        large = minimum_detectable_difference(15_000, 0.126)
        self.assertGreater(small, large)
        # at n=2030 and psi=0.126 the project measured ~2.2pp
        self.assertAlmostEqual(
            minimum_detectable_difference(2030, 0.126) * 100, 2.2, delta=0.2)


class EvaluateSelectionTest(unittest.TestCase):
    def test_flags_a_tie_when_difference_is_noise(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        sel = y.copy()
        chal = y.copy()
        # tiny perturbation: 3 rows differ out of 400
        idx = rng.choice(400, 3, replace=False)
        chal[idx] = 1 - chal[idx]
        test = evaluate_selection(
            "exp_002", sel, [("exp_001", chal, 0.99)], y, 0.995)
        self.assertTrue(test.is_tied)
        self.assertEqual(test.tied_with[0]["challenger"], "exp_001")
        self.assertEqual(test.significant_over, [])

    def test_flags_clear_win_when_difference_is_large(self):
        y = np.array([1] * 100 + [0] * 100)
        sel = np.array([1] * 100 + [0] * 100)   # perfect
        chal = np.array([0] * 100 + [0] * 100)  # wrong on half
        test = evaluate_selection(
            "exp_002", sel, [("exp_001", chal, 0.5)], y, 1.0)
        self.assertFalse(test.is_tied)
        self.assertEqual(test.significant_over[0]["challenger"], "exp_001")

    def test_length_mismatch_is_reported_not_raised(self):
        y = np.array([0, 1, 0])
        test = evaluate_selection(
            "exp_001", np.array([0, 1]), [], y, 0.9)
        self.assertIsNotNone(test.skipped_reason)
        self.assertEqual(test.to_dict()["verdict"], "no_comparison")

    def test_verdict_strings_are_stable(self):
        y = np.array([1] * 60 + [0] * 60)
        sel = np.array([1] * 60 + [0] * 60)
        chal = np.array([0] * 60 + [0] * 60)
        clear = evaluate_selection("a", sel, [("b", chal, 0.5)], y, 1.0)
        self.assertEqual(clear.to_dict()["verdict"], "selected_clearly")


class SelectionIsUnchangedTest(unittest.TestCase):
    """The annotation must never move the choice."""

    def test_labels_do_not_change_the_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            y = rng.integers(0, 2, 200)
            a = y.copy()
            b = y.copy()
            b[rng.choice(200, 40, replace=False)] ^= 1
            np.save(tmp / "a.npy", a)
            np.save(tmp / "b.npy", b)

            results = [
                make_result("exp_001", 1.0, str(tmp / "a.npy"), seconds=5.0),
                make_result("exp_002", 0.8, str(tmp / "b.npy"), seconds=1.0),
            ]
            without = select_best_model(list(results), metric="accuracy")
            self.assertEqual(without.experiment_id, "exp_001")

            # fresh objects: select_best_model mutates selection_test on the best
            results2 = [
                make_result("exp_001", 1.0, str(tmp / "a.npy"), seconds=5.0),
                make_result("exp_002", 0.8, str(tmp / "b.npy"), seconds=1.0),
            ]
            with_labels = select_best_model(
                results2, metric="accuracy", validation_labels=y)
            self.assertEqual(with_labels.experiment_id, "exp_001")
            self.assertIsNotNone(with_labels.selection_test)

    def test_missing_prediction_files_do_not_break_selection(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 50)
        results = [
            make_result("exp_001", 0.9, None),
            make_result("exp_002", 0.8, "/nonexistent/path.npy"),
        ]
        best = select_best_model(results, metric="accuracy",
                                 validation_labels=y)
        self.assertEqual(best.experiment_id, "exp_001")
        self.assertEqual(best.selection_test["verdict"], "no_comparison")

    def test_training_time_still_breaks_exact_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(3)
            y = rng.integers(0, 2, 100)
            np.save(tmp / "a.npy", y)
            np.save(tmp / "b.npy", y)
            results = [
                make_result("exp_001", 0.9, str(tmp / "a.npy"), seconds=9.0),
                make_result("exp_002", 0.9, str(tmp / "b.npy"), seconds=1.0),
            ]
            best = select_best_model(results, metric="accuracy",
                                     validation_labels=y)
            self.assertEqual(best.experiment_id, "exp_002")


class LabelLoadingTest(unittest.TestCase):
    def test_npy_split_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "y_val.npy", np.array([0, 1, 1, 0]))

            class Config:
                data = {
                    "format": "npy_split",
                    "validation_labels": str(tmp / "y_val.npy"),
                }

            labels = load_validation_labels(Config(), {})
            np.testing.assert_array_equal(labels, [0, 1, 1, 0])

    def test_csv_split_reads_target_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "val.csv"
            with path.open("w", encoding="utf-8", newline="") as h:
                w = csv.writer(h)
                w.writerow(["text", "label"])
                w.writerow(["a", "yes"])
                w.writerow(["b", "no"])

            class Config:
                data = {
                    "format": "csv_split",
                    "validation": str(path),
                    "target_column": "label",
                }

            labels = load_validation_labels(
                Config(), {"label_mapping": {"no": 0, "yes": 1}})
            np.testing.assert_array_equal(labels, [1, 0])

    def test_missing_file_raises_label_load_error(self):
        class Config:
            data = {"format": "npy_split",
                    "validation_labels": "/nonexistent/y.npy"}

        with self.assertRaises(LabelLoadError):
            load_validation_labels(Config(), {})

    def test_unsupported_format_raises(self):
        class Config:
            data = {"format": "parquet"}

        with self.assertRaises(LabelLoadError):
            load_validation_labels(Config(), {})

    def test_config_without_data_raises(self):
        class Config:
            data = None

        with self.assertRaises(LabelLoadError):
            load_validation_labels(Config(), {})


if __name__ == "__main__":
    unittest.main()
