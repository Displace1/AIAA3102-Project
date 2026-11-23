import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from superagent.metrics import (
    classifier_probabilities,
    classifier_scores,
    log_loss_value,
    roc_auc_value,
    scores_from_probabilities,
)


def evaluate_classifier(classifier, X_train, y_train, X_val, y_val):
    """Return JSON-safe classification metrics for a fitted classifier."""
    train_predictions = classifier.predict(X_train)
    prediction_started = time.perf_counter()
    validation_predictions = classifier.predict(X_val)
    prediction_time = time.perf_counter() - prediction_started

    labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        validation_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_class_recall = precision_recall_fscore_support(
        y_val,
        validation_predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )[1]
    train_accuracy = float(accuracy_score(y_train, train_predictions))
    validation_accuracy = float(
        accuracy_score(y_val, validation_predictions)
    )
    balanced_accuracy = float(
        balanced_accuracy_score(y_val, validation_predictions)
    )
    label_values, label_counts = np.unique(y_val, return_counts=True)
    prediction_values, prediction_counts = np.unique(
        validation_predictions, return_counts=True
    )
    # score-based metrics: None when the estimator exposes no scores.  The
    # probability matrix is extracted once and shared by roc_auc and log_loss;
    # only roc_auc may fall back to decision_function margins.
    probabilities = classifier_probabilities(classifier, X_val)
    scores = (
        scores_from_probabilities(probabilities)
        if probabilities is not None
        else classifier_scores(classifier, X_val)
    )
    roc_auc = roc_auc_value(y_val, scores)
    log_loss_score = log_loss_value(
        y_val,
        probabilities,
        classes=getattr(classifier, "classes_", None),
    )

    return {
        "predictions": validation_predictions,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc": roc_auc,
        "log_loss": log_loss_score,
        "confusion_matrix": confusion_matrix(
            y_val, validation_predictions, labels=labels
        ).astype(int).tolist(),
        "class_recall": {
            str(label): float(value)
            for label, value in zip(labels, per_class_recall)
        },
        "predicted_class_distribution": {
            str(label): int(count)
            for label, count in zip(prediction_values, prediction_counts)
        },
        "majority_class_accuracy": float(label_counts.max() / label_counts.sum()),
        "generalization_gap": train_accuracy - validation_accuracy,
        "prediction_time_seconds": prediction_time,
    }


def result_metric_fields(metrics):
    return {
        key: metrics[key]
        for key in (
            "train_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "roc_auc",
            "log_loss",
            "confusion_matrix",
            "class_recall",
            "predicted_class_distribution",
            "majority_class_accuracy",
            "generalization_gap",
            "prediction_time_seconds",
        )
    }
