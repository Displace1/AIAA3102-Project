from superagent.metrics import (
    metric_direction,
    metric_needs_probabilities,
    result_metric_value,
)



def _selection_key(result, metric):
    # Every metric is normalised to "larger is better" before ``max`` runs, so
    # a minimisation metric such as log_loss picks the smallest value instead
    # of the largest.  Ties are still broken by the shorter training time.
    return (
        metric_direction(metric) * result_metric_value(result, metric),
        -result.training_time_seconds,
    )


def select_best_model(results, metric="accuracy"):
    """Pick the best successful experiment by ``metric``.

    Selection is ``max`` over the normalised metric, with training time as the
    tie-break. This is the try_2 rule: no significance testing, no annotation.
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

    return best
