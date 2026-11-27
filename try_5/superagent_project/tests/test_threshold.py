"""Tests for superagent/threshold.py.

The important behaviours to lock down:
  1. unbalanced data can genuinely improve the objective
  2. multiclass is refused rather than silently argmax-ed
  3. small validation splits are refused (the try_2 bootstrap finding)
  4. ties keep 0.5, so behaviour is unchanged when tuning buys nothing
  5. monotonicity: raising the threshold never predicts MORE positives
  6. a threshold chosen on one split transfers without refitting anything
"""
import unittest

import numpy as np

from superagent.threshold import (
    MIN_SAMPLES_FOR_THRESHOLD,
    SUPPORTED_OBJECTIVES,
    apply_threshold,
    binary_positive_probabilities,
    positive_class_shares,
    select_threshold,
    threshold_grid,
)


def imbalanced_case(n_major=800, n_minor=200, seed=0):
    """Majority class easy, minority class separable but under-predicted."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_major + [1] * n_minor)
    # p1 is informative but shifted low for the minority class, so 0.5 misses
    # most of them and a lower threshold recovers them.
    p1 = np.where(y == 1,
                  rng.beta(5, 3, size=len(y)),   # minority: mean ~0.62
                  rng.beta(2, 8, size=len(y)))   # majority: mean ~0.20
    return y, p1


class ThresholdGridTest(unittest.TestCase):
    def test_grid_is_inclusive_and_sorted(self):
        grid = threshold_grid(0.05, 0.95, 0.1)
        self.assertAlmostEqual(grid[0], 0.05)
        self.assertAlmostEqual(grid[-1], 0.95)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_grid_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            threshold_grid(0.0, 0.9, 0.1)
        with self.assertRaises(ValueError):
            threshold_grid(0.1, 1.0, 0.1)

    def test_grid_rejects_nonpositive_step(self):
        with self.assertRaises(ValueError):
            threshold_grid(0.1, 0.9, 0.0)


class BinaryProbabilityTest(unittest.TestCase):
    def test_two_column_matrix_returns_positive_column(self):
        proba = np.array([[0.9, 0.1], [0.2, 0.8]])
        np.testing.assert_allclose(
            binary_positive_probabilities(proba), [0.1, 0.8])

    def test_single_column_is_rejected(self):
        self.assertIsNone(binary_positive_probabilities(np.array([[0.4], [0.6]])))

    def test_three_columns_is_rejected(self):
        proba = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        self.assertIsNone(binary_positive_probabilities(proba))

    def test_out_of_range_values_are_rejected(self):
        self.assertIsNone(
            binary_positive_probabilities(np.array([[1.4, -0.4], [0.5, 0.5]])))

    def test_none_passes_through(self):
        self.assertIsNone(binary_positive_probabilities(None))


class SelectThresholdTest(unittest.TestCase):
    def test_improves_macro_f1_on_imbalanced_data(self):
        y, p1 = imbalanced_case()
        choice = select_threshold(y, p1, objective="macro_f1")
        self.assertTrue(choice.applied, choice.reason)
        self.assertLess(choice.threshold, 0.5)
        self.assertGreater(choice.tuned_value, choice.baseline_value)

    def test_balanced_easy_data_may_keep_default(self):
        # perfectly separated, threshold 0.5 already optimal for accuracy
        y = np.array([0] * 300 + [1] * 300)
        p1 = np.concatenate([np.full(300, 0.05), np.full(300, 0.95)])
        choice = select_threshold(y, p1, objective="accuracy")
        self.assertFalse(choice.applied)
        self.assertIn("no threshold beat", choice.reason)

    def test_multiclass_is_refused(self):
        y = np.array([0, 1, 2] * 100)
        p1 = np.linspace(0.01, 0.99, 300)
        choice = select_threshold(y, p1, objective="macro_f1")
        self.assertFalse(choice.applied)
        self.assertIn("binary", choice.reason)

    def test_small_validation_is_refused(self):
        y, p1 = imbalanced_case(n_major=60, n_minor=20)
        self.assertLess(len(y), MIN_SAMPLES_FOR_THRESHOLD)
        choice = select_threshold(y, p1, objective="macro_f1")
        self.assertFalse(choice.applied)
        self.assertIn("below the", choice.reason)

    def test_unknown_objective_raises(self):
        y, p1 = imbalanced_case()
        with self.assertRaises(ValueError):
            select_threshold(y, p1, objective="auc_pr")

    def test_missing_probabilities_is_refused(self):
        y, _ = imbalanced_case()
        choice = select_threshold(y, None, objective="macro_f1")
        self.assertFalse(choice.applied)
        self.assertIn("no binary probabilities", choice.reason)

    def test_length_mismatch_is_refused(self):
        y, p1 = imbalanced_case()
        choice = select_threshold(y, p1[:-5], objective="macro_f1")
        self.assertFalse(choice.applied)
        self.assertIn("length mismatch", choice.reason)

    def test_all_supported_objectives_run(self):
        y, p1 = imbalanced_case()
        for objective in SUPPORTED_OBJECTIVES:
            choice = select_threshold(y, p1, objective=objective)
            self.assertEqual(choice.objective, objective)


class ApplyThresholdTest(unittest.TestCase):
    def test_none_threshold_reproduces_argmax(self):
        p1 = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
        np.testing.assert_array_equal(apply_threshold(p1, None),
                                      [0, 0, 1, 1, 1])

    def test_explicit_threshold_inclusive(self):
        p1 = np.array([0.19, 0.2, 0.21])
        np.testing.assert_array_equal(apply_threshold(p1, 0.20), [0, 1, 1])

    def test_lower_threshold_predicts_more_positives(self):
        _, p1 = imbalanced_case()
        grid = threshold_grid(0.1, 0.9, 0.1)
        shares = positive_class_shares(p1, grid)
        self.assertTrue(all(a >= b for a, b in zip(shares, shares[1:])),
                        "raising the threshold must not increase positives")


class SelectionDisciplineTest(unittest.TestCase):
    def test_threshold_from_one_split_applies_to_another(self):
        """Mirrors the real protocol: pick on validation, apply to held-out."""
        y_val, p_val = imbalanced_case(seed=1)
        y_test, p_test = imbalanced_case(seed=2)

        choice = select_threshold(y_val, p_val, objective="macro_f1")
        self.assertTrue(choice.applied)

        from superagent.threshold import _score
        base = _score(y_test, (p_test >= 0.5).astype(int), "macro_f1")
        tuned = _score(y_test, apply_threshold(p_test, choice.threshold),
                       "macro_f1")
        self.assertGreater(tuned, base,
                           "a threshold chosen on validation should transfer")

    def test_selection_does_not_mutate_inputs(self):
        y, p1 = imbalanced_case()
        y_before, p_before = y.copy(), p1.copy()
        select_threshold(y, p1, objective="macro_f1")
        np.testing.assert_array_equal(y, y_before)
        np.testing.assert_array_equal(p1, p_before)


if __name__ == "__main__":
    unittest.main()
