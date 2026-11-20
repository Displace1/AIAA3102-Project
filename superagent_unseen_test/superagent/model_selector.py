def select_best_model(results):
    successful = [
        result
        for result in results
        if result.status == "success"
        and result.validation_accuracy is not None
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
            result.validation_accuracy,
            -result.training_time_seconds,
        ),
    )
