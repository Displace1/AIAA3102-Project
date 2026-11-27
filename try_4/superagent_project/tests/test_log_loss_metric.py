import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.svm import LinearSVC

from adapters.base import ExperimentResult
from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.image_training import _classification_metrics as image_metrics
from superagent.metrics import (
    METRIC_DIRECTIONS,
    PROBABILITY_METRICS,
    SUPPORTED_METRICS,
    classifier_probabilities,
    classifier_score_bundle,
    classifier_scores,
    is_metric_better,
    log_loss_value,
    metric_direction,
    metric_improvement,
    metric_needs_probabilities,
    result_metric_value,
    test_metric_key,
)
from superagent.model_selector import select_best_model
from superagent.stop_policy import StopPolicy


def binary_data(seed=3, samples=400):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, 5))
    logits = features[:, 0] * 1.3 - features[:, 1] * 0.7
    labels = (logits + rng.normal(0.0, 0.5, samples) > 0).astype(int)
    return features, labels


def multiclass_data(seed=4, samples=600):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, 5))
    scores = np.column_stack(
        [features[:, 0], features[:, 1], -features[:, 0] - features[:, 1]]
    )
    return features, scores.argmax(axis=1)


class DirectionTests(unittest.TestCase):
    def test_log_loss_is_the_only_minimisation_metric(self):
        self.assertEqual(
            {name for name, direction in METRIC_DIRECTIONS.items()
             if direction < 0},
            {"log_loss"},
        )
        self.assertEqual(set(METRIC_DIRECTIONS), SUPPORTED_METRICS)
        self.assertEqual(metric_direction("log_loss"), -1)
        self.assertEqual(metric_direction("accuracy"), 1)

    def test_improvement_is_positive_when_log_loss_drops(self):
        self.assertAlmostEqual(
            metric_improvement(0.42, 0.31, "log_loss"), 0.11
        )
        self.assertAlmostEqual(
            metric_improvement(0.42, 0.31, "accuracy"), -0.11
        )

    def test_is_metric_better_respects_direction(self):
        self.assertTrue(is_metric_better(0.30, 0.40, "log_loss"))
        self.assertFalse(is_metric_better(0.50, 0.40, "log_loss"))
        self.assertTrue(is_metric_better(0.90, 0.80, "accuracy"))
        self.assertTrue(is_metric_better(0.5, None, "log_loss"))
        self.assertFalse(is_metric_better(None, 0.5, "log_loss"))

    def test_only_log_loss_demands_probabilities(self):
        self.assertEqual(PROBABILITY_METRICS, {"log_loss"})
        self.assertTrue(metric_needs_probabilities("log_loss"))
        self.assertFalse(metric_needs_probabilities("roc_auc"))

    def test_result_metric_value_reads_the_log_loss_field(self):
        result = ExperimentResult(
            experiment_id="e",
            model_name="m",
            status="success",
            validation_accuracy=0.8,
            training_time_seconds=1.0,
            artifact_path=None,
            log_loss=0.37,
        )
        self.assertAlmostEqual(result_metric_value(result, "log_loss"), 0.37)
        self.assertEqual(test_metric_key("log_loss"), "test_log_loss")


class LogLossValueTests(unittest.TestCase):
    def test_binary_matches_sklearn(self):
        features, labels = binary_data()
        classifier = LogisticRegression(max_iter=500).fit(features, labels)
        probabilities = classifier_probabilities(classifier, features)
        self.assertEqual(probabilities.shape[1], 2)
        self.assertAlmostEqual(
            log_loss_value(labels, probabilities),
            sklearn_log_loss(labels, probabilities),
            places=12,
        )
        # A single positive-class column must give the same number.
        self.assertAlmostEqual(
            log_loss_value(labels, probabilities[:, 1]),
            sklearn_log_loss(labels, probabilities),
            places=12,
        )

    def test_multiclass_matches_sklearn(self):
        features, labels = multiclass_data()
        classifier = LogisticRegression(max_iter=500).fit(features, labels)
        probabilities = classifier_probabilities(classifier, features)
        self.assertAlmostEqual(
            log_loss_value(labels, probabilities),
            sklearn_log_loss(labels, probabilities),
            places=12,
        )

    def test_margins_are_not_accepted(self):
        features, labels = multiclass_data()
        classifier = LinearSVC(max_iter=5000).fit(features, labels)
        self.assertIsNone(classifier_probabilities(classifier, features))
        # ``classifier_scores`` may still expose margins for rank-based metrics.
        margins = classifier_scores(classifier, features)
        self.assertIsNotNone(margins)
        self.assertIsNone(log_loss_value(labels, margins))
        scores, probabilities = classifier_score_bundle(classifier, features)
        self.assertIsNotNone(scores)
        self.assertIsNone(probabilities)

    def test_single_class_split_has_no_log_loss(self):
        rng = np.random.default_rng(5)
        probabilities = rng.dirichlet([1.0, 1.0], size=40)
        self.assertIsNone(log_loss_value(np.zeros(40, dtype=int), probabilities))

    def test_row_sums_are_renormalised_before_scoring(self):
        labels = np.array([0, 1, 1, 0])
        # Rows sum to 0.9; a naive sklearn call would raise.
        probabilities = np.array(
            [[0.54, 0.36], [0.09, 0.81], [0.36, 0.54], [0.72, 0.18]]
        )
        value = log_loss_value(labels, probabilities)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(
            value, sklearn_log_loss(labels, probabilities / 0.9), places=12
        )

    def test_mismatched_shape_returns_none(self):
        labels = np.array([0, 1, 2, 0])
        probabilities = np.full((4, 2), 0.5)
        self.assertIsNone(log_loss_value(labels, probabilities))
        self.assertIsNone(log_loss_value(labels[:2], np.full((3, 3), 1 / 3)))


class EvaluationIntegrationTests(unittest.TestCase):
    def test_evaluate_classifier_reports_log_loss(self):
        features, labels = binary_data()
        classifier = LogisticRegression(max_iter=500).fit(features, labels)
        metrics = evaluate_classifier(
            classifier, features, labels, features, labels
        )
        self.assertAlmostEqual(
            metrics["log_loss"],
            sklearn_log_loss(labels, classifier.predict_proba(features)),
            places=12,
        )
        self.assertIn("log_loss", result_metric_fields(metrics))

    def test_evaluate_classifier_reports_none_without_probabilities(self):
        features, labels = multiclass_data()
        classifier = LinearSVC(max_iter=5000).fit(features, labels)
        metrics = evaluate_classifier(
            classifier, features, labels, features, labels
        )
        self.assertIsNone(metrics["log_loss"])
        self.assertIsNotNone(metrics["roc_auc"])

    def test_image_metrics_report_log_loss(self):
        labels = [0, 1, 1, 0]
        predictions = [0, 1, 0, 0]
        probabilities = np.array(
            [[0.8, 0.2], [0.1, 0.9], [0.55, 0.45], [0.7, 0.3]]
        )
        metrics = image_metrics(labels, predictions, 2, probabilities)
        self.assertAlmostEqual(
            metrics["log_loss"], sklearn_log_loss(labels, probabilities),
            places=12,
        )


class SelectionAndStopTests(unittest.TestCase):
    def _result(self, name, log_loss, accuracy=0.8, seconds=1.0):
        return ExperimentResult(
            experiment_id=name,
            model_name=name,
            status="success",
            validation_accuracy=accuracy,
            training_time_seconds=seconds,
            artifact_path=None,
            log_loss=log_loss,
        )

    def test_selector_minimises_log_loss(self):
        best = select_best_model(
            [
                self._result("a", 0.40),
                self._result("b", 0.31),
                self._result("c", 0.35),
            ],
            metric="log_loss",
        )
        self.assertEqual(best.experiment_id, "b")

    def test_selector_breaks_ties_by_training_time(self):
        best = select_best_model(
            [self._result("slow", 0.31, seconds=9.0),
             self._result("fast", 0.31, seconds=2.0)],
            metric="log_loss",
        )
        self.assertEqual(best.experiment_id, "fast")

    def test_selector_explains_missing_probabilities(self):
        with self.assertRaises(RuntimeError) as context:
            select_best_model([self._result("a", None)], metric="log_loss")
        message = str(context.exception)
        self.assertIn("log_loss", message)
        self.assertIn("predict_proba", message)

    def _policy(self):
        return StopPolicy(
            max_experiments=10,
            max_runtime_seconds=1000,
            no_improvement_patience=2,
            minimum_improvement=0.001,
            metric="log_loss",
        )

    def test_patience_counts_a_rising_log_loss_as_failure(self):
        results = [
            self._result("a", 0.50),
            self._result("b", 0.52),
            self._result("c", 0.53),
        ]
        decision = self._policy().evaluate(results, elapsed_seconds=1.0)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "no_validation_improvement")

    def test_patience_resets_when_log_loss_drops(self):
        results = [
            self._result("a", 0.50),
            self._result("b", 0.52),
            self._result("c", 0.40),
            self._result("d", 0.41),
        ]
        decision = self._policy().evaluate(results, elapsed_seconds=1.0)
        self.assertFalse(decision.should_stop)


if __name__ == "__main__":
    unittest.main()
