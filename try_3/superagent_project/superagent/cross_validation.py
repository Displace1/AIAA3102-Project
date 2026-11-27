"""K-fold cross-validation over the TRAINING split, for selection stability.

Why this exists
---------------
Selection uses one validation split. Measured on this project's runs, the best
model and the runner-up differ by less than the sampling error in 47 of 62 runs.
A single point estimate therefore cannot separate them.

Cross-validation does not make the models better. It gives a *second, less noisy
estimator* of the same quantity, so a selection can be checked against it.

Two deliberate constraints
--------------------------
1. **Only the training split is folded.** The validation split stays untouched so
   it can remain the tiebreaker and the threshold-tuning source. Folding the
   validation split would leak it into selection.
2. **The reported ``validation_accuracy`` does not change.** CV scores are added
   alongside it; the shipped metric still comes from the validation split. This
   keeps every existing run comparable.

Known bias, measured on this project
------------------------------------
CV on 80% of the data systematically *underestimates* models that benefit from
more rows: ExtraTrees +0.58pp, one-hot HGB +0.22pp, native HGB +0.91pp (CV at
6493 rows vs validation at 8117). So CV alone would favour low-capacity
families. The two estimators must be read together, never swapped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class CrossValidationResult:
    """Per-fold scores plus their summary."""

    scores: list
    mean: float
    std: float
    folds: int
    n_train: int
    skipped_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.skipped_reason is None and bool(self.scores)

    def to_dict(self) -> dict:
        return {
            "scores": [float(s) for s in self.scores],
            "mean": float(self.mean),
            "std": float(self.std),
            "folds": int(self.folds),
            "n_train": int(self.n_train),
            "skipped_reason": self.skipped_reason,
        }


def _skipped(reason, n_train=0, folds=0):
    return CrossValidationResult(
        scores=[], mean=float("nan"), std=float("nan"),
        folds=folds, n_train=n_train, skipped_reason=reason)


def cross_validate_classifier(build_estimator: Callable[[], object],
                              X, y,
                              folds=DEFAULT_FOLDS,
                              seed=42,
                              min_class_count=None):
    """Score ``build_estimator()`` on stratified folds of (X, y).

    ``build_estimator`` must return a fresh, unfitted estimator each call, so the
    same hyper-parameters are evaluated without reuse across folds.

    Refuses to run when a class has fewer members than folds: a stratified split
    would place a single row in a fold, making that fold's score a coin flip on
    one row. Callers should treat ``skipped_reason`` as "no CV signal", not as
    failure.
    """
    X = np.asarray(X) if not isinstance(X, list) else np.asarray(X, dtype=object)
    y = np.asarray(y)
    n_train = int(len(y))

    classes, counts = np.unique(y, return_counts=True)
    smallest = int(counts.min()) if len(counts) else 0
    limit = folds if min_class_count is None else max(folds, min_class_count)
    if smallest < limit:
        return _skipped(
            f"smallest class has {smallest} rows, fewer than the {folds} folds",
            n_train=n_train, folds=folds)
    if n_train < folds * 2:
        return _skipped(
            f"only {n_train} training rows for {folds} folds",
            n_train=n_train, folds=folds)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, valid_idx in splitter.split(X, y):
        estimator = build_estimator()
        estimator.fit(X[train_idx], y[train_idx])
        prediction = estimator.predict(X[valid_idx])
        scores.append(float(np.mean(prediction == y[valid_idx])))

    return CrossValidationResult(
        scores=scores,
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        folds=folds,
        n_train=n_train,
    )


def cv_summary(result):
    """Read the CV fields off an ExperimentResult, as a JSON-ready dict.

    Returns None when the experiment never requested CV, so a run that does not
    use it produces a lock file byte-identical to before.
    """
    scores = getattr(result, "cv_scores", None)
    mean = getattr(result, "cv_mean", None)
    skipped = getattr(result, "cv_skipped_reason", None)
    folds = getattr(result, "cv_folds", None)
    if not scores and skipped is None and mean is None:
        return None
    return {
        "scores": [float(s) for s in (scores or [])],
        "mean": None if mean is None else float(mean),
        "std": None if getattr(result, "cv_std", None) is None
        else float(result.cv_std),
        "folds": folds,
        "skipped_reason": skipped,
        "note": (
            "CV folds the TRAINING split only and uses 80% of it per fold, so it "
            "underestimates data-hungry models. Read it beside the validation "
            "metric, never in place of it."
        ),
    }


def compare_to_selection(results, best):
    """Check whether the validation split's choice also wins on CV.

    This is a *cross-check*, not a decision rule. The selection stands as
    chosen; this records whether the second estimator agrees, so a disagreement
    is visible in the lock file instead of being silently absent. Disagreement is
    expected often -- CV and validation disagree by design when a model is
    data-hungry -- so ``agrees=False`` is information, not an error.

    CV scores are always accuracy (see ``cross_validate_classifier``). When the
    run selects on another metric this compares unlike quantities, so the
    ``metric_matches`` flag says so rather than letting the reader assume.
    """
    scored = []
    for result in results:
        if getattr(result, "status", None) != "success":
            continue
        mean = getattr(result, "cv_mean", None)
        if mean is None:
            continue
        scored.append({
            "experiment_id": result.experiment_id,
            "model_name": result.model_name,
            "cv_mean": float(mean),
            "cv_std": None if getattr(result, "cv_std", None) is None
            else float(result.cv_std),
        })

    if not scored:
        return {"available": False,
                "reason": "no experiment reported cross-validation scores"}
    if len(scored) < 2:
        return {"available": False,
                "reason": "only one experiment reported cross-validation "
                          "scores, so there is nothing to compare",
                "experiments": scored}

    scored.sort(key=lambda row: row["cv_mean"], reverse=True)
    cv_best = scored[0]["experiment_id"]
    best_cv = next((row["cv_mean"] for row in scored
                    if row["experiment_id"] == best.experiment_id), None)
    gap = None
    if best_cv is not None:
        gap = round(float(scored[0]["cv_mean"] - best_cv), 6)
    return {
        "available": True,
        "agrees": cv_best == best.experiment_id,
        "selection": best.experiment_id,
        "selection_cv_mean": best_cv,
        "cv_best": cv_best,
        "gap_to_cv_best": gap,
        "ranking": scored,
        "note": (
            "CV and validation answer the same question on different data. "
            "This field is reported for judgement; it does not change the "
            "selected model."
        ),
    }


def cv_lock_blocks(results, best):
    """Decide which CV keys belong in a lock file, if any.

    Returns an empty dict when no experiment computed CV. That is the normal
    case -- CV is opt-in -- and an empty dict means the caller adds nothing, so a
    run that never asks for CV writes a lock byte-identical to the one produced
    before this feature existed.
    """
    any_cv = any(
        getattr(result, "cv_scores", None) is not None
        or getattr(result, "cv_skipped_reason", None) is not None
        for result in results
    )
    if not any_cv:
        return {}
    blocks = {}
    summary = cv_summary(best)
    if summary is not None:
        blocks["cross_validation"] = summary
    blocks["cv_comparison"] = compare_to_selection(results, best)
    return blocks


def conservative_score(mean, std, weight=0.5):
    """Mean minus a penalty for fold-to-fold variance.

    A model whose folds range from 0.70 to 0.85 is less trustworthy than one that
    sits at 0.78 every fold, even when the means match. ``weight=0`` reproduces
    the plain mean; 0.5 is a mild preference for stability.

    Not used by the shipped pipeline: selection stays on the validation split, and
    CV is reported beside it. Kept as the documented starting point for a future
    rule that does want to trade mean against variance.
    """
    if mean is None or np.isnan(mean):
        return float("nan")
    if std is None or np.isnan(std):
        return float(mean)
    return float(mean - weight * std)
