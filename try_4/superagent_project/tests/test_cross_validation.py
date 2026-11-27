"""Cross-validation as a second estimator for selection.

The important properties:

1. CV is opt-in; without the flag an adapter behaves exactly as before.
2. CV folds the TRAINING split only, so the validation split stays clean.
3. Small classes are refused rather than producing a one-row fold.
4. The shipped ``validation_accuracy`` never changes when CV is enabled.
"""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from adapters.base import ExperimentResult
from adapters.cv_support import (
    attach_cross_validation,
    cv_folds,
    cv_requested,
    run_cross_validation,
)
from superagent.cross_validation import (
    CrossValidationResult,
    compare_to_selection,
    conservative_score,
    cross_validate_classifier,
    cv_lock_blocks,
    cv_summary,
)


def separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.2, size=n) > 0).astype(int)
    return X, y


def build_lr():
    return LogisticRegression(max_iter=500)


class CrossValidateTest(unittest.TestCase):
    def test_returns_one_score_per_fold(self):
        X, y = separable()
        cv = cross_validate_classifier(build_lr, X, y, folds=5)
        self.assertTrue(cv.available)
        self.assertEqual(len(cv.scores), 5)
        self.assertEqual(cv.folds, 5)
        self.assertEqual(cv.n_train, len(y))

    def test_mean_and_std_match_the_scores(self):
        X, y = separable()
        cv = cross_validate_classifier(build_lr, X, y, folds=4)
        self.assertAlmostEqual(cv.mean, float(np.mean(cv.scores)), places=10)
        self.assertAlmostEqual(cv.std, float(np.std(cv.scores)), places=10)

    def test_small_class_is_skipped_not_crashed(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 2))
        y = np.zeros(60, dtype=int)
        y[:3] = 1          # 3 positives, fewer than 5 folds
        cv = cross_validate_classifier(build_lr, X, y, folds=5)
        self.assertFalse(cv.available)
        self.assertIn("smallest class", cv.skipped_reason)
        self.assertEqual(cv.scores, [])

    def test_too_few_rows_is_skipped(self):
        X = np.zeros((4, 2))
        y = np.array([0, 1, 0, 1])
        cv = cross_validate_classifier(build_lr, X, y, folds=5)
        self.assertFalse(cv.available)

    def test_scores_are_deterministic_for_a_fixed_seed(self):
        X, y = separable()
        a = cross_validate_classifier(build_lr, X, y, folds=3, seed=7)
        b = cross_validate_classifier(build_lr, X, y, folds=3, seed=7)
        self.assertEqual(a.scores, b.scores)

    def test_each_fold_gets_a_fresh_estimator(self):
        """A reused estimator would carry fold-specific state into the next."""
        seen = []

        def build():
            seen.append(object())
            return LogisticRegression(max_iter=200)

        X, y = separable()
        cross_validate_classifier(build, X, y, folds=4)
        self.assertEqual(len(seen), 4)

    def test_conservative_score_penalises_variance(self):
        stable = conservative_score(0.80, 0.01, weight=0.5)
        volatile = conservative_score(0.80, 0.10, weight=0.5)
        self.assertGreater(stable, volatile)
        self.assertAlmostEqual(conservative_score(0.80, 0.10, weight=0.0), 0.80)

    def test_nan_mean_survives_conservative_score(self):
        self.assertTrue(np.isnan(conservative_score(float("nan"), 0.1)))


class HyperparameterGateTest(unittest.TestCase):
    def test_cv_is_off_by_default(self):
        self.assertFalse(cv_requested({}))
        self.assertFalse(cv_requested(None))
        self.assertFalse(cv_requested({"cross_validation": False}))

    def test_flag_turns_it_on(self):
        self.assertTrue(cv_requested({"cross_validation": True}))

    def test_folds_are_clamped(self):
        self.assertEqual(cv_folds({"cv_folds": 1}), 2)
        self.assertEqual(cv_folds({"cv_folds": 99}), 10)
        self.assertEqual(cv_folds({"cv_folds": 4}), 4)
        self.assertEqual(cv_folds({"cv_folds": "bad"}), 5)
        self.assertEqual(cv_folds({}), 5)

    def test_run_helper_returns_none_when_disabled(self):
        X, y = separable()
        self.assertIsNone(run_cross_validation(build_lr, X, y, {}))

    def test_run_helper_runs_when_enabled(self):
        X, y = separable()
        cv = run_cross_validation(build_lr, X, y, {"cross_validation": True,
                                                   "cv_folds": 3})
        self.assertEqual(len(cv.scores), 3)


class AttachTest(unittest.TestCase):
    def make_result(self):
        return ExperimentResult(
            experiment_id="exp_001", model_name="m", status="success",
            validation_accuracy=0.8, training_time_seconds=1.0,
            artifact_path="a.joblib")

    def test_attach_copies_numbers(self):
        cv = CrossValidationResult([0.8, 0.82], 0.81, 0.01, 2, 100)
        result = attach_cross_validation(self.make_result(), cv)
        self.assertEqual(result.cv_scores, [0.8, 0.82])
        self.assertAlmostEqual(result.cv_mean, 0.81)
        self.assertAlmostEqual(result.cv_std, 0.01)
        self.assertEqual(result.cv_folds, 2)

    def test_attach_leaves_validation_accuracy_untouched(self):
        cv = CrossValidationResult([0.9], 0.9, 0.0, 1, 100)
        result = attach_cross_validation(self.make_result(), cv)
        self.assertEqual(result.validation_accuracy, 0.8)

    def test_attach_none_is_noop(self):
        result = attach_cross_validation(self.make_result(), None)
        self.assertIsNone(result.cv_mean)

    def test_skipped_result_still_carries_the_reason(self):
        cv = CrossValidationResult([], float("nan"), float("nan"), 5, 10,
                                   skipped_reason="too small")
        result = attach_cross_validation(self.make_result(), cv)
        self.assertEqual(result.cv_skipped_reason, "too small")
        self.assertIsNone(result.cv_mean)


class LockAnnotationTest(unittest.TestCase):
    """The lock must gain CV numbers without losing what it had before."""

    def make_result(self, experiment_id, accuracy, cv_mean, cv_std=0.01):
        result = ExperimentResult(
            experiment_id=experiment_id, model_name="m", status="success",
            validation_accuracy=accuracy, training_time_seconds=1.0,
            artifact_path="a.joblib")
        result.cv_scores = [cv_mean] * 5
        result.cv_mean = cv_mean
        result.cv_std = cv_std
        result.cv_folds = 5
        return result

    def test_experiment_without_cv_yields_none(self):
        plain = ExperimentResult(
            experiment_id="e", model_name="m", status="success",
            validation_accuracy=0.8, training_time_seconds=1.0,
            artifact_path="a.joblib")
        self.assertIsNone(cv_summary(plain))

    def test_a_run_without_cv_adds_nothing_to_the_lock(self):
        """The whole point of opt-in: no CV means not one new key."""
        plain = [
            ExperimentResult(
                experiment_id=f"e{i}", model_name="m", status="success",
                validation_accuracy=0.8 - i * 0.01, training_time_seconds=1.0,
                artifact_path="a.joblib")
            for i in range(3)
        ]
        self.assertEqual(cv_lock_blocks(plain, plain[0]), {})

    def test_a_run_with_cv_adds_both_blocks(self):
        results = [self.make_result("e1", 0.80, 0.79),
                   self.make_result("e2", 0.79, 0.75)]
        blocks = cv_lock_blocks(results, results[0])
        self.assertIn("cross_validation", blocks)
        self.assertIn("cv_comparison", blocks)
        self.assertTrue(blocks["cv_comparison"]["agrees"])

    def test_summary_reads_scores_and_carries_the_bias_note(self):
        block = cv_summary(self.make_result("e1", 0.80, 0.78))
        self.assertEqual(block["folds"], 5)
        self.assertAlmostEqual(block["mean"], 0.78)
        self.assertEqual(len(block["scores"]), 5)
        self.assertIn("never in place of it", block["note"])

    def test_comparison_reports_agreement(self):
        results = [self.make_result("e1", 0.80, 0.79),
                   self.make_result("e2", 0.79, 0.75)]
        block = compare_to_selection(results, results[0])
        self.assertTrue(block["available"])
        self.assertTrue(block["agrees"])
        self.assertEqual(block["cv_best"], "e1")

    def test_comparison_reports_disagreement_without_changing_anything(self):
        results = [self.make_result("e1", 0.80, 0.75),
                   self.make_result("e2", 0.79, 0.79)]
        block = compare_to_selection(results, results[0])
        self.assertFalse(block["agrees"])
        self.assertEqual(block["cv_best"], "e2")
        # the caller's selection is untouched
        self.assertEqual(block["selection"], "e1")
        self.assertAlmostEqual(block["gap_to_cv_best"], 0.04)

    def test_comparison_needs_two_candidates(self):
        results = [self.make_result("e1", 0.80, 0.75)]
        block = compare_to_selection(results, results[0])
        self.assertFalse(block["available"])
        self.assertIn("nothing to compare", block["reason"])

    def test_comparison_ignores_failed_experiments(self):
        failed = self.make_result("e2", 0.99, 0.99)
        failed.status = "failed"
        results = [self.make_result("e1", 0.80, 0.75), failed]
        block = compare_to_selection(results, results[0])
        self.assertFalse(block["available"])


if __name__ == "__main__":
    unittest.main()
