SUPPORTED_METRICS = {
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
}


RESULT_METRIC_ATTRIBUTES = {
    "accuracy": "validation_accuracy",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}


def result_metric_value(result, metric):
    """Return the configured validation metric from an ExperimentResult."""
    try:
        attribute = RESULT_METRIC_ATTRIBUTES[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported metric: {metric!r}.") from error
    return getattr(result, attribute, None)


def validation_metric_key(metric):
    if metric == "accuracy":
        return "validation_accuracy"
    if metric == "macro_f1":
        return "f1_macro"
    if metric == "balanced_accuracy":
        return "balanced_accuracy"
    raise ValueError(f"Unsupported metric: {metric!r}.")


def test_metric_key(metric):
    if metric == "accuracy":
        return "test_accuracy"
    if metric == "macro_f1":
        return "f1_macro"
    if metric == "balanced_accuracy":
        return "balanced_accuracy"
    raise ValueError(f"Unsupported metric: {metric!r}.")
