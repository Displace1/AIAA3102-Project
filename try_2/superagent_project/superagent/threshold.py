"""Decision-threshold selection and application.

Why this module exists
----------------------
Every try_1 adapter calls ``Pipeline.predict``, which for binary classifiers is
equivalent to "positive if p >= 0.5". On imbalanced data that fixed point can be
far from the operating point the task actually wants: on UCI Bank Marketing the
same fitted logistic model reaches macro-F1 0.688 at t=0.50 and 0.753 at t=0.23,
at the cost of 0.76pp accuracy.

Selection discipline (deliberately narrow)
------------------------------------------
* The threshold is chosen on the VALIDATION split only, never on test.
* It is only attempted for binary problems; a single scalar cannot express a
  K-class decision rule, and pretending otherwise would silently change the
  semantics of ``argmax``.
* It is only attempted when the validation split is large enough to support the
  search. A bootstrap check on this project's own data showed that at n_val=86
  the selected threshold moved between 0.14 and 0.33 across resamples and made
  the test macro-F1 worse in 36% of resamples. At n_val=6782 the same procedure
  never made it worse in 300 resamples. ``MIN_SAMPLES_FOR_THRESHOLD`` encodes
  that finding.

Nothing here mutates an estimator: callers keep the original model and store the
chosen scalar alongside it, so the artifact format is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Empirical floor from the try_2 bootstrap study (see threshold_bootstrap.py).
# Below this many validation rows the argmax of the threshold sweep is noise.
MIN_SAMPLES_FOR_THRESHOLD = 200

DEFAULT_GRID_STEP = 0.01
DEFAULT_GRID_LOW = 0.05
DEFAULT_GRID_HIGH = 0.95

# Objectives a caller may ask for.  Kept explicit so an unknown value fails
# loudly instead of silently falling back to accuracy.
SUPPORTED_OBJECTIVES = ("accuracy", "macro_f1", "balanced_accuracy")


@dataclass(frozen=True)
class ThresholdChoice:
    """Outcome of a threshold search, including why it was (not) applied."""

    threshold: Optional[float]
    objective: str
    applied: bool
    reason: str
    baseline_value: Optional[float] = None
    tuned_value: Optional[float] = None


def binary_positive_probabilities(probabilities):
    """Return the positive-class column of a 2-column probability matrix.

    Returns ``None`` when the input is not a genuine binary probability matrix,
    so callers can skip thresholding instead of inventing a number.
    """
    if probabilities is None:
        return None
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        return None
    if not np.all(np.isfinite(array)):
        return None
    if array.min() < 0.0 or array.max() > 1.0:
        return None
    return array[:, 1]


def _score(labels, predictions, objective):
    if objective == "accuracy":
        return float(np.mean(predictions == labels))
    if objective == "macro_f1":
        # Inline macro F1 so this module has no sklearn dependency in the hot
        # path of the controller, and so the definition is explicit.
        return float(
            np.mean([_f1_for_class(labels, predictions, c)
                     for c in np.unique(labels)])
        )
    if objective == "balanced_accuracy":
        classes = np.unique(labels)
        recalls = [np.mean(predictions[labels == c] == c) for c in classes]
        return float(np.mean(recalls))
    raise ValueError(
        f"Unsupported threshold objective: {objective!r}. "
        f"Expected one of {SUPPORTED_OBJECTIVES}."
    )


def _f1_for_class(labels, predictions, klass):
    tp = float(np.sum((predictions == klass) & (labels == klass)))
    fp = float(np.sum((predictions == klass) & (labels != klass)))
    fn = float(np.sum((predictions != klass) & (labels == klass)))
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def threshold_grid(low=DEFAULT_GRID_LOW, high=DEFAULT_GRID_HIGH,
                   step=DEFAULT_GRID_STEP):
    """Candidate thresholds, inclusive of the endpoints."""
    if not (0.0 < low < high < 1.0):
        raise ValueError("threshold grid must lie strictly inside (0, 1)")
    if step <= 0.0:
        raise ValueError("threshold grid step must be positive")
    count = int(round((high - low) / step)) + 1
    return np.round(np.linspace(low, high, count), 10)


def select_threshold(labels, positive_probabilities, objective="macro_f1",
                     minimum_samples=MIN_SAMPLES_FOR_THRESHOLD,
                     grid=None):
    """Pick the threshold that maximises ``objective`` on validation data.

    Returns a :class:`ThresholdChoice`.  ``applied`` is False whenever the
    search is not safe or not useful, and ``reason`` says which; callers must
    honour ``applied`` rather than reading ``threshold`` blindly.
    """
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported threshold objective: {objective!r}. "
            f"Expected one of {SUPPORTED_OBJECTIVES}."
        )

    labels = np.asarray(labels)
    if positive_probabilities is None:
        return ThresholdChoice(None, objective, False,
                               "estimator exposes no binary probabilities")

    p1 = np.asarray(positive_probabilities, dtype=float)
    if p1.ndim != 1 or p1.shape[0] != labels.shape[0]:
        return ThresholdChoice(None, objective, False,
                               "probability vector length mismatch")

    classes = np.unique(labels)
    if classes.size != 2:
        return ThresholdChoice(
            None, objective, False,
            f"threshold search supports binary tasks only "
            f"(found {classes.size} classes); argmax is kept for multiclass")

    n = int(labels.shape[0])
    if n < minimum_samples:
        return ThresholdChoice(
            None, objective, False,
            f"validation split has {n} rows, below the "
            f"{minimum_samples}-row floor for a reliable threshold search")

    grid = threshold_grid() if grid is None else np.asarray(grid, dtype=float)
    baseline_pred = (p1 >= 0.5).astype(int)
    baseline_value = _score(labels, baseline_pred, objective)

    best_t = 0.5
    best_value = baseline_value
    for t in grid:
        value = _score(labels, (p1 >= t).astype(int), objective)
        if value > best_value:
            best_value, best_t = value, float(t)

    # A threshold only earns its place if it actually improves the objective.
    # Ties keep 0.5 so the behaviour stays identical to the untuned pipeline.
    if not best_value > baseline_value + 1e-12:
        return ThresholdChoice(
            None, objective, False,
            f"no threshold beat the 0.5 baseline on validation "
            f"({objective}={baseline_value:.4f})",
            baseline_value=baseline_value, tuned_value=baseline_value)

    return ThresholdChoice(
        threshold=best_t, objective=objective, applied=True,
        reason=(f"{objective} improved {baseline_value:.4f} -> {best_value:.4f} "
                f"on {n} validation rows"),
        baseline_value=baseline_value, tuned_value=best_value)


def apply_threshold(positive_probabilities, threshold):
    """Hard labels from probabilities at a fixed threshold.

    ``threshold`` None means "no tuning", which reproduces the default
    ``argmax`` decision for a binary classifier.
    """
    p1 = np.asarray(positive_probabilities, dtype=float)
    if threshold is None:
        return (p1 >= 0.5).astype(int)
    return (p1 >= float(threshold)).astype(int)


def positive_class_shares(positive_probabilities, thresholds: Sequence[float]):
    """Share of rows predicted positive at each threshold (monotonicity check)."""
    p1 = np.asarray(positive_probabilities, dtype=float)
    return [float(np.mean(p1 >= t)) for t in thresholds]
