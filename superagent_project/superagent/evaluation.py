import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
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

    return {
        "predictions": validation_predictions,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
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
            "confusion_matrix",
            "class_recall",
            "predicted_class_distribution",
            "majority_class_accuracy",
            "generalization_gap",
            "prediction_time_seconds",
        )
    }
