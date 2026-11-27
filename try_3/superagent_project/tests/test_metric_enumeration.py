import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.metrics import (
    SUPPORTED_METRICS,
    classifier_scores,
    result_metric_value,
    roc_auc_value,
    test_metric_key,
    validation_metric_key,
)
from superagent.model_selector import select_best_model
from superagent.task_config import load_task_config
from adapters.base import ExperimentResult


def binary_data(seed=0, samples=240):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, 4))
    logits = features[:, 0] * 1.4 - features[:, 1] * 0.9
    labels = (logits + rng.normal(0.0, 0.6, samples) > 0).astype(int)
    return features, labels


def multiclass_data(seed=1, samples=360):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, 4))
    scores = np.column_stack(
        [features[:, 0], features[:, 1], -features[:, 0] - features[:, 1]]
    )
    labels = scores.argmax(axis=1)
    return features, labels


class MetricRegistryTests(unittest.TestCase):
    def test_supported_metrics_cover_predictions_and_scores(self):
        self.assertEqual(
            SUPPORTED_METRICS,
            {
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "precision",
                "recall",
                "roc_auc",
                "log_loss",
            },
        )

    def test_metric_keys_map_to_result_fields(self):
        self.assertEqual(validation_metric_key("precision"), "precision_macro")
        self.assertEqual(validation_metric_key("recall"), "recall_macro")
        self.assertEqual(validation_metric_key("roc_auc"), "roc_auc")
        self.assertEqual(validation_metric_key("log_loss"), "log_loss")
        self.assertEqual(test_metric_key("accuracy"), "test_accuracy")
        self.assertEqual(test_metric_key("macro_f1"), "f1_macro")
        self.assertEqual(test_metric_key("roc_auc"), "test_roc_auc")
        self.assertEqual(test_metric_key("log_loss"), "test_log_loss")

    def test_unknown_metric_is_rejected_everywhere(self):
        with self.assertRaises(ValueError):
            validation_metric_key("brier_score")
        with self.assertRaises(ValueError):
            test_metric_key("brier_score")


class EvaluationMetricTests(unittest.TestCase):
    def test_binary_logistic_reports_roc_auc(self):
        features, labels = binary_data()
        metrics = evaluate_classifier(
            LogisticRegression(max_iter=500).fit(features, labels),
            features,
            labels,
            features,
            labels,
        )
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertGreater(metrics["roc_auc"], 0.8)
        self.assertIn("roc_auc", result_metric_fields(metrics))

    def test_multiclass_logistic_reports_macro_ovr_roc_auc(self):
        features, labels = multiclass_data()
        metrics = evaluate_classifier(
            LogisticRegression(max_iter=500).fit(features, labels),
            features,
            labels,
            features,
            labels,
        )
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertGreater(metrics["roc_auc"], 0.9)

    def test_margin_model_uses_decision_function(self):
        features, labels = multiclass_data()
        classifier = LinearSVC(max_iter=4000).fit(features, labels)
        scores = classifier_scores(classifier, features)
        self.assertEqual(scores.shape, (len(labels), 3))
        metrics = evaluate_classifier(
            classifier, features, labels, features, labels
        )
        self.assertIsNotNone(metrics["roc_auc"])

    def test_model_without_scores_reports_none(self):
        features, labels = binary_data()

        class LabelsOnly:
            """Mirrors an estimator that exposes only hard predictions."""

            def predict(self, data):
                return np.zeros(len(data), dtype=int)

        classifier = LabelsOnly()
        self.assertIsNone(classifier_scores(classifier, features))
        self.assertIsNone(roc_auc_value(labels, None))
        metrics = evaluate_classifier(
            classifier, features, labels, features, labels
        )
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["validation_accuracy"])

    def test_single_class_target_has_no_roc_auc(self):
        features, labels = binary_data()
        metrics = evaluate_classifier(
            LogisticRegression(max_iter=500).fit(features, labels),
            features,
            np.zeros_like(labels),
            features,
            np.zeros_like(labels),
        )
        self.assertIsNone(metrics["roc_auc"])


def result(name, roc_auc, accuracy=0.8, seconds=1.0):
    return ExperimentResult(
        experiment_id=name,
        model_name=name,
        status="success",
        validation_accuracy=accuracy,
        training_time_seconds=seconds,
        artifact_path=None,
        f1_macro=accuracy,
        balanced_accuracy=accuracy,
        precision_macro=accuracy,
        recall_macro=accuracy,
        roc_auc=roc_auc,
    )


class SelectionMetricTests(unittest.TestCase):
    def test_selects_by_roc_auc_when_available(self):
        best = select_best_model(
            [result("a", 0.91, 0.70), result("b", 0.95, 0.68)],
            metric="roc_auc",
        )
        self.assertEqual(best.experiment_id, "b")

    def test_selector_rejects_metrics_the_adapter_cannot_report(self):
        with self.assertRaises(RuntimeError) as context:
            select_best_model([result("a", None, 0.88)], metric="roc_auc")
        message = str(context.exception)
        self.assertIn("roc_auc", message)
        self.assertIn("predict_proba", message)

    def test_selector_still_reports_real_failures(self):
        failed = ExperimentResult(
            experiment_id="a",
            model_name="a",
            status="failed",
            validation_accuracy=None,
            training_time_seconds=0.0,
            artifact_path=None,
            error_message="boom",
        )
        with self.assertRaises(RuntimeError) as context:
            select_best_model([failed], metric="roc_auc")
        self.assertIn("All experiments failed", str(context.exception))


class TaskConfigMetricTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _config(self, metric):
        paths = {}
        rng = np.random.default_rng(2)
        features = rng.normal(size=(40, 3))
        labels = (features[:, 0] > 0).astype(int)
        for name, values in {
            "train_features": features[:30],
            "train_labels": labels[:30],
            "validation_features": features[30:],
            "validation_labels": labels[30:],
            "test_features": features[30:],
            "test_labels": labels[30:],
        }.items():
            path = self.root / f"{name}.npy"
            np.save(path, values)
            paths[name] = str(path)
        body = {
            "run_type": "development_run",
            "task": {"name": "metric_trial", "metric": metric},
            "data": {"format": "npy_split", **paths},
            "execution": {
                "random_seed": 42,
                "evaluate_test": False,
                "planner_mode": "deterministic",
                "initial_deterministic_experiments": 2,
                "max_experiments": 2,
                "max_runtime_minutes": 5,
                "no_improvement_patience": 2,
                "minimum_improvement": 0.001,
                "maximum_planner_errors": 2,
                "maximum_same_experiment_errors": 2,
                "experiment_timeout_seconds": 120,
                "maximum_memory_mb": 2048,
                "maximum_cpu_threads": 2,
                "resource_poll_interval_seconds": 0.2,
            },
        }
        path = self.root / f"config_{metric}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_accepts_every_supported_metric(self):
        for metric in ("accuracy", "macro_f1", "balanced_accuracy",
                       "precision", "recall", "roc_auc", "log_loss"):
            config = load_task_config(self._config(metric))
            self.assertEqual(config.metric, metric)

    def test_rejects_unsupported_metric(self):
        with self.assertRaises(ValueError) as context:
            load_task_config(self._config("brier_score"))
        self.assertIn("brier_score", str(context.exception))
        self.assertIn("log_loss", str(context.exception))


if __name__ == "__main__":
    unittest.main()
