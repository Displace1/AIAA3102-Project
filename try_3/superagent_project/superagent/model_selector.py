from superagent.metrics import (
    metric_direction,
    metric_needs_probabilities,
    result_metric_value,
)
from superagent.selection_test import (
    DEFAULT_ALPHA,
    evaluate_selection,
)

import numpy as np


def _selection_key(result, metric):
    # Every metric is normalised to "larger is better" before ``max`` runs, so
    # a minimisation metric such as log_loss picks the smallest value instead
    # of the largest.  Ties are still broken by the shorter training time.
    return (
        metric_direction(metric) * result_metric_value(result, metric),
        -result.training_time_seconds,
    )


def _load_predictions(result):
    """Read an experiment's validation predictions, or None when unavailable."""
    path = getattr(result, "val_predictions_path", None)
    if not path:
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def select_best_model(results, metric="accuracy", validation_labels=None,
                      alpha=DEFAULT_ALPHA):
    """Pick the best successful experiment by ``metric``.

    The choice itself is unchanged: ``max`` over the normalised metric with
    training time as the tie-break. When ``validation_labels`` is supplied the
    selection is additionally tested against every rival with an exact McNemar
    test, and the outcome is attached as ``selection_test`` on the returned
    result. That annotation never overrides the choice -- it records whether the
    choice is distinguishable from the alternatives at all.
    """
    successful = [
        result
        for result in results
        if result.status == "success"
        and result_metric_value(result, metric) is not None
    ]

    if not successful:
        finished = [
            result for result in results if result.status == "success"
        ]
        if finished:
            models = ", ".join(
                sorted({result.model_name for result in finished})
            )
            if metric_needs_probabilities(metric):
                raise RuntimeError(
                    f"No successful experiment reported the selection metric "
                    f"{metric!r} ({models}).  {metric} needs a full "
                    "probability matrix, so it is only available for "
                    "estimators that expose predict_proba (or image models "
                    "with a softmax head).  Margin-based estimators such as "
                    "LinearSVC cannot report it."
                )
            raise RuntimeError(
                f"No successful experiment reported the selection metric "
                f"{metric!r} ({models}).  Metrics that need probability "
                "scores, such as roc_auc, are only available for estimators "
                "that expose predict_proba or decision_function."
            )
        failures = "; ".join(
            f"{result.experiment_id}: {result.error_message}"
            for result in results
        )
        raise RuntimeError(f"All experiments failed. {failures}")

    best = max(successful, key=lambda result: _selection_key(result, metric))

    # Annotate the choice with a paired test against every rival. Purely
    # additive: ``best`` is already decided above and is returned unchanged.
    if validation_labels is not None:
        selected_predictions = _load_predictions(best)
        if selected_predictions is not None:
            challengers = []
            for result in successful:
                if result.experiment_id == best.experiment_id:
                    continue
                predictions = _load_predictions(result)
                if predictions is None:
                    continue
                challengers.append(
                    (result.experiment_id, predictions,
                     result_metric_value(result, metric))
                )
            best.selection_test = evaluate_selection(
                best.experiment_id,
                selected_predictions,
                challengers,
                validation_labels,
                result_metric_value(best, metric),
                alpha=alpha,
            ).to_dict()
        else:
            best.selection_test = {
                "verdict": "no_comparison",
                "skipped_reason": "selected experiment recorded no val predictions",
            }

    return best
