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
from superagent.threshold import (
    binary_positive_probabilities,
    select_threshold,
)


def evaluate_classifier(
    classifier,
    X_train,
    y_train,
    X_val,
    y_val,
    threshold_objective="macro_f1",
):
    """Return JSON-safe classification metrics for a fitted classifier.

    ``threshold_objective`` chooses what the decision-threshold search
    maximises.  It is deliberately independent of the task's selection metric:
    when a task selects on accuracy, tuning for accuracy is a no-op (0.5 is
    already near-optimal), so the interesting use is to tune for macro-F1 or
    balanced accuracy and record the trade-off in ``decision_rule``.
    """
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

    # Decision-threshold tuning.  Chosen on this validation split only; the
    # caller stores the scalar next to the artifact and re-applies it later.
    # ``decision_rule`` records the outcome even when nothing was applied, so an
    # auditor can tell "tuning was skipped" apart from "tuning never ran".
    positive = binary_positive_probabilities(probabilities)
    choice = select_threshold(y_val, positive, objective=threshold_objective)
    tuned_predictions = (
        (positive >= choice.threshold).astype(int)
        if choice.applied and positive is not None
        else None
    )
    decision_rule = {
        "threshold": choice.threshold,
        "applied": choice.applied,
        "objective": choice.objective,
        "reason": choice.reason,
        "baseline_value": choice.baseline_value,
        "tuned_value": choice.tuned_value,
        "validation_samples": int(len(y_val)),
    }
    if tuned_predictions is not None:
        labels_for_tuning = sorted(set(y_val.tolist()) | set(tuned_predictions.tolist()))
        precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
            y_val, tuned_predictions, labels=labels_for_tuning,
            average="macro", zero_division=0,
        )
        decision_rule["tuned_metrics"] = {
            "accuracy": float(accuracy_score(y_val, tuned_predictions)),
            "macro_f1": float(f1_t),
            "macro_precision": float(precision_t),
            "macro_recall": float(recall_t),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_val, tuned_predictions)
            ),
        }

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
        "decision_rule": decision_rule,
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
            "decision_rule",
        )
        if key in metrics
    }
