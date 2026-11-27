"""Integration tests: the locked threshold must travel end to end.

unit tests in tests/test_threshold.py cover the maths. These cover the wiring:
  1. evaluate_classifier records a decision_rule
  2. solution_lock.json round-trips the threshold
  3. locked_threshold() reads it back from the exported directory
  4. a run with no tuning keeps pre-threshold behaviour exactly
  5. applying at test time never re-searches (the threshold is an input)
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.predictor import apply_locked_threshold, locked_threshold
from superagent.threshold import apply_threshold, select_threshold
from sklearn.linear_model import LogisticRegression


def imbalanced(seed=0, n_major=900, n_minor=300):
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_major + [1] * n_minor)
    # x carries real signal; minority shift makes 0.5 a poor operating point
    x = np.where(y == 1, rng.normal(1.0, 1.0, len(y)),
                 rng.normal(-0.3, 1.0, len(y))).reshape(-1, 1)
    return x, y


class DecisionRuleRecordingTest(unittest.TestCase):
    def test_evaluate_classifier_records_decision_rule(self):
        X, y = imbalanced()
        model = LogisticRegression(max_iter=1000).fit(X[:1000], y[:1000])
        metrics = evaluate_classifier(model, X[:1000], y[:1000], X[1000:], y[1000:])

        self.assertIn("decision_rule", metrics)
        rule = metrics["decision_rule"]
        for key in ("threshold", "applied", "objective", "reason",
                    "baseline_value", "tuned_value", "validation_samples"):
            self.assertIn(key, rule, f"decision_rule missing {key}")
        self.assertEqual(rule["objective"], "macro_f1")
        self.assertEqual(rule["validation_samples"], len(y[1000:]))

    def test_decision_rule_survives_result_metric_fields(self):
        X, y = imbalanced()
        model = LogisticRegression(max_iter=1000).fit(X[:1000], y[:1000])
        metrics = evaluate_classifier(model, X[:1000], y[:1000], X[1000:], y[1000:])
        carried = result_metric_fields(metrics)
        self.assertIn("decision_rule", carried)

    def test_objective_is_configurable(self):
        X, y = imbalanced()
        model = LogisticRegression(max_iter=1000).fit(X[:1000], y[:1000])
        metrics = evaluate_classifier(
            model, X[:1000], y[:1000], X[1000:], y[1000:],
            threshold_objective="balanced_accuracy")
        self.assertEqual(metrics["decision_rule"]["objective"],
                         "balanced_accuracy")


class LockRoundTripTest(unittest.TestCase):
    def test_threshold_reads_back_from_final_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            final.mkdir()
            (final / "final_model.joblib").write_bytes(b"stub")
            (final / "final_solution.json").write_text(json.dumps({
                "decision_rule": {"applied": True, "threshold": 0.23},
            }), encoding="utf-8")

            self.assertAlmostEqual(
                locked_threshold(final / "final_model.joblib"), 0.23)

    def test_missing_solution_file_means_no_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "final_model.joblib"
            model.write_bytes(b"stub")
            self.assertIsNone(locked_threshold(model))

    def test_unapplied_rule_means_no_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "final_model.joblib"
            model.write_bytes(b"stub")
            (Path(tmp) / "final_solution.json").write_text(json.dumps({
                "decision_rule": {"applied": False, "threshold": 0.3},
            }), encoding="utf-8")
            self.assertIsNone(locked_threshold(model))

    def test_out_of_range_threshold_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "final_model.joblib"
            model.write_bytes(b"stub")
            (Path(tmp) / "final_solution.json").write_text(json.dumps({
                "decision_rule": {"applied": True, "threshold": 1.5},
            }), encoding="utf-8")
            self.assertIsNone(locked_threshold(model))

    def test_corrupt_json_means_no_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "final_model.joblib"
            model.write_bytes(b"stub")
            (Path(tmp) / "final_solution.json").write_text("{not json",
                                                           encoding="utf-8")
            self.assertIsNone(locked_threshold(model))


class ApplyLockedThresholdTest(unittest.TestCase):
    def test_none_threshold_returns_original_predictions(self):
        labels = np.array([0, 1, 0])
        proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        out = apply_locked_threshold(labels, proba, None)
        np.testing.assert_array_equal(out, labels)

    def test_multiclass_probabilities_are_not_overridden(self):
        labels = np.array([0, 1, 2])
        proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        out = apply_locked_threshold(labels, proba, 0.3)
        np.testing.assert_array_equal(out, labels)

    def test_binary_threshold_override_applies(self):
        labels = np.array([0, 1, 0])
        proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])
        out = apply_locked_threshold(labels, proba, 0.5)
        np.testing.assert_array_equal(out, [0, 1, 0])
        out_low = apply_locked_threshold(labels, proba, 0.2)
        np.testing.assert_array_equal(out_low, [1, 1, 0])

    def test_length_mismatch_falls_back_to_original(self):
        labels = np.array([0, 1, 0])
        proba = np.array([[0.7, 0.3], [0.4, 0.6]])
        out = apply_locked_threshold(labels, proba, 0.2)
        np.testing.assert_array_equal(out, labels)


class NoResearchTest(unittest.TestCase):
    def test_locked_threshold_is_a_pure_lookup(self):
        """Applying a locked threshold must not depend on any labels.

        This is the discipline test: given the same probabilities and the same
        locked scalar, the output is identical no matter what the true labels
        are -- i.e. no implicit re-selection can happen at test time.
        """
        rng = np.random.default_rng(0)
        proba = rng.random((50, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        p1 = proba[:, 1]
        t = 0.33

        first = apply_threshold(p1, t)
        second = apply_threshold(p1, t)
        np.testing.assert_array_equal(first, second)

        # and it matches a hand-computed rule
        np.testing.assert_array_equal(first, (p1 >= t).astype(int))


if __name__ == "__main__":
    unittest.main()
