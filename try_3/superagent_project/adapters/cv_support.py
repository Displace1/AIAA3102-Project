"""Attach cross-validation scores to an experiment result.

Adapters call these helpers so the CV wiring stays in one place instead of being
copy-pasted into ten files. Both helpers are no-ops when CV is not requested,
which keeps the default behaviour byte-identical to before.
"""
from __future__ import annotations

from superagent.cross_validation import (
    DEFAULT_FOLDS,
    CrossValidationResult,
    cross_validate_classifier,
)


def cv_requested(hyperparameters):
    """CV is opt-in per experiment so it can be enabled on a subset of runs."""
    return bool((hyperparameters or {}).get("cross_validation", False))


def cv_folds(hyperparameters):
    value = (hyperparameters or {}).get("cv_folds", DEFAULT_FOLDS)
    try:
        folds = int(value)
    except (TypeError, ValueError):
        return DEFAULT_FOLDS
    return max(2, min(folds, 10))


def attach_cross_validation(result, cv: CrossValidationResult):
    """Copy a CrossValidationResult onto an ExperimentResult in place."""
    if cv is None:
        return result
    result.cv_scores = list(cv.scores)
    result.cv_mean = None if cv.mean != cv.mean else float(cv.mean)
    result.cv_std = None if cv.std != cv.std else float(cv.std)
    result.cv_folds = int(cv.folds)
    result.cv_skipped_reason = cv.skipped_reason
    return result


def run_cross_validation(build_estimator, X, y, hyperparameters, seed=42):
    """Convenience wrapper: run CV if requested, else return None."""
    if not cv_requested(hyperparameters):
        return None
    return cross_validate_classifier(
        build_estimator, X, y,
        folds=cv_folds(hyperparameters),
        seed=seed,
    )
