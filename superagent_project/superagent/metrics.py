import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


# Metrics a task may declare in ``task.metric``.  ``roc_auc`` and ``log_loss``
# need scores rather than hard labels; ``log_loss`` additionally needs genuine
# probabilities, so it is only available from ``predict_proba`` (or the softmax
# of an image model) and never from ``decision_function`` margins.  The other
# five are computed from hard predictions and always available.
SUPPORTED_METRICS = {
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "precision",
    "recall",
    "roc_auc",
    "log_loss",
}

# Metrics that need a score matrix rather than a label vector.
SCORE_METRICS = {"roc_auc", "log_loss"}

# Metrics that need calibrated probabilities.  ROC-AUC is rank based, so raw
# margins are fine there; a log loss computed on margins would be meaningless
# (and can even be negative), so it is deliberately excluded.
PROBABILITY_METRICS = {"log_loss"}

# +1 means "larger is better", -1 means "smaller is better".  Every comparison
# in the system goes through ``metric_improvement`` so a single table keeps
# callers honest; ``max(...)`` on a raw log loss would silently pick the worst
# experiment.
METRIC_DIRECTIONS = {
    "accuracy": 1,
    "macro_f1": 1,
    "balanced_accuracy": 1,
    "precision": 1,
    "recall": 1,
    "roc_auc": 1,
    "log_loss": -1,
}


RESULT_METRIC_ATTRIBUTES = {
    "accuracy": "validation_accuracy",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "roc_auc": "roc_auc",
    "log_loss": "log_loss",
}


def metric_direction(metric):
    """Return +1 for metrics to maximise and -1 for metrics to minimise."""
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported metric: {metric!r}.") from error


def metric_needs_probabilities(metric):
    return metric in PROBABILITY_METRICS


def metric_improvement(previous, current, metric):
    """Signed improvement from ``previous`` to ``current``; positive is better."""
    if previous is None or current is None:
        raise ValueError("metric_improvement requires two numeric values.")
    return metric_direction(metric) * (float(current) - float(previous))


def is_metric_better(candidate, incumbent, metric):
    """True when ``candidate`` is strictly better than ``incumbent``."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return metric_improvement(incumbent, candidate, metric) > 0.0


def result_metric_value(result, metric):
    """Return the configured validation metric from an ExperimentResult."""
    try:
        attribute = RESULT_METRIC_ATTRIBUTES[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported metric: {metric!r}.") from error
    return getattr(result, attribute, None)


def validation_metric_key(metric):
    try:
        return RESULT_METRIC_ATTRIBUTES[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported metric: {metric!r}.") from error


def test_metric_key(metric):
    if metric == "accuracy":
        return "test_accuracy"
    if metric == "roc_auc":
        return "test_roc_auc"
    if metric == "log_loss":
        return "test_log_loss"
    return validation_metric_key(metric)


def classifier_probabilities(classifier, features):
    """Probability matrix of shape ``(n, k)``, or ``None``.

    Only ``predict_proba`` counts.  ``decision_function`` margins are handled
    by ``classifier_scores`` for rank-based metrics, but they are not
    probabilities and must never feed a log loss.
    """
    predict_proba = getattr(classifier, "predict_proba", None)
    if not callable(predict_proba):
        return None
    try:
        probabilities = np.asarray(predict_proba(features), dtype=float)
    except Exception:  # noqa: BLE001 - model-specific failures are expected
        return None
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None
    return probabilities


def scores_from_probabilities(probabilities):
    """Collapse a probability matrix into the shape ``roc_auc_value`` wants."""
    if probabilities is None:
        return None
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities


def _decision_scores(classifier, features):
    decision_function = getattr(classifier, "decision_function", None)
    if not callable(decision_function):
        return None
    try:
        margins = np.asarray(decision_function(features), dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if margins.ndim == 1 or (margins.ndim == 2 and margins.shape[1] >= 2):
        return margins
    return None


def classifier_scores(classifier, features):
    """Probability-like scores: shape ``(n,)`` for binary, ``(n, k)`` otherwise.

    ``predict_proba`` is preferred; ``decision_function`` covers margin-based
    models such as ``LinearSVC``.  Returns ``None`` when neither is usable, which
    makes ``roc_auc`` unavailable instead of inventing a number.
    """
    scores, _ = classifier_score_bundle(classifier, features)
    return scores


def classifier_score_bundle(classifier, features):
    """Return ``(scores, probabilities)`` from a single model call.

    ``probabilities`` is only the strict ``predict_proba`` matrix; it stays
    ``None`` for margin-based models so a log loss is never computed on
    unnormalised margins.
    """
    probabilities = classifier_probabilities(classifier, features)
    if probabilities is not None:
        return scores_from_probabilities(probabilities), probabilities
    return _decision_scores(classifier, features), None


def roc_auc_value(labels, scores):
    """Macro one-vs-rest ROC-AUC, or ``None`` when it cannot be computed."""
    if scores is None:
        return None
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    try:
        if scores.ndim == 1:
            if len(classes) != 2:
                return None
            return float(roc_auc_score(labels, scores))
        if scores.shape[1] == 2 and len(classes) == 2:
            # A two-column probability matrix is a binary problem in disguise.
            return float(roc_auc_score(labels, scores[:, 1]))
        if scores.shape[1] != len(classes):
            return None
        # Macro one-vs-rest, computed per class.  Doing it by hand avoids
        # sklearn's requirement that multiclass scores be probabilities, so raw
        # margins from ``decision_function`` are usable too; per-class AUC is
        # rank based, so a monotone rescaling would not change the result.
        aucs = []
        for index, klass in enumerate(classes):
            positive = (labels == klass).astype(int)
            if positive.min() == positive.max():
                continue
            aucs.append(float(roc_auc_score(positive, scores[:, index])))
        return float(np.mean(aucs)) if aucs else None
    except ValueError:
        return None


def log_loss_value(labels, probabilities, classes=None):
    """Multiclass log loss (cross entropy), or ``None`` when unavailable.

    ``probabilities`` must be a genuine ``(n, k)`` probability matrix, or a
    single column of positive-class probabilities for a binary problem.  Rows
    are clipped and renormalised so a single saturated prediction cannot turn
    the score into infinity.
    """
    if probabilities is None:
        return None
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    observed = np.unique(labels)
    if len(observed) < 2:
        return None

    if probabilities.ndim == 1:
        if len(observed) != 2:
            return None
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            return None
        probabilities = np.column_stack(
            [1.0 - probabilities, probabilities]
        )
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        return None
    if probabilities.shape[1] < 2:
        return None

    # Guard against decision_function margins masquerading as probabilities.
    # Free-form margins are often negative, and a margin row sums to zero, so
    # a value check rejects them before any clipping turns them into a
    # plausible-looking score.  Slightly unnormalised rows are still accepted
    # and fixed up below, because a floating point softmax can drift.
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        return None
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0.0) or np.any(row_sums > 1.0 + 1e-6):
        return None

    if classes is None:
        classes = np.arange(probabilities.shape[1])
    classes = np.asarray(classes)
    if len(classes) != probabilities.shape[1]:
        return None
    if not set(observed.tolist()) <= set(classes.tolist()):
        return None

    matrix = np.clip(probabilities, 1e-15, 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    try:
        # ``eps`` is left at sklearn's default: it is optional on every
        # supported version and was removed from the signature in 1.5, so
        # passing it explicitly is what breaks portability.
        return float(log_loss(labels, matrix, labels=classes))
    except ValueError:
        return None
