from superagent.metrics import result_metric_value


def select_best_model(results, metric="accuracy"):
    successful = [
        result
        for result in results
        if result.status == "success"
        and result_metric_value(result, metric) is not None
    ]

    if not successful:
        failures = "; ".join(
            f"{result.experiment_id}: {result.error_message}"
            for result in results
        )
        raise RuntimeError(f"All experiments failed. {failures}")

    return max(
        successful,
        key=lambda result: (
            result_metric_value(result, metric),
            -result.training_time_seconds,
        ),
    )
