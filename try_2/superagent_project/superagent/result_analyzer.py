class ResultAnalyzer:
    """Produces deterministic, auditable diagnoses from experiment metrics.

    Note on the removed ``possible_overfitting`` flag
    -------------------------------------------------
    Earlier versions emitted ``possible_overfitting`` whenever
    ``generalization_gap >= 0.10``. Measurement across 17 completed runs
    (92 experiments) showed the flag was direction-blind and therefore
    actively harmful:

    * the sign of Spearman(gap, validation accuracy) flipped between runs:
      8 positive, 9 negative, with extremes of +1.000 (task 1) and -0.962
      (20 Newsgroups). A single threshold cannot serve both directions.
    * it fired 9 times, and 4 of those were on the run's BEST model
      (precision as a warning: 0.56). The planners read it as "add
      regularisation", which in task 1 pushed capacity down twice and cost
      2.5pp and 1.2pp.
    * 0.10 sits at the 90th percentile of all observed gaps, but the gap is
      scaled by n_train, model capacity and class count, so the same constant
      can be unreachable for one task and trivially exceeded by another.

    Normalising the gap within a run did not rescue the signal either: across
    17 runs, the high-gap tercile scored HIGHER than the rest (+1.43pp) in 9
    runs and lower in 8.

    The remaining diagnoses below are all directional (they compare a model to
    a concrete reference such as the majority baseline, or to its own other
    classes), so they do not suffer from this ambiguity.
    """

    def analyze(self, result):
        if result.status != "success":
            return [f"experiment_failed:{result.error_type or 'unknown'}"]

        diagnoses = []
        recalls = list((result.class_recall or {}).values())
        if recalls and max(recalls) - min(recalls) >= 0.15:
            diagnoses.append("class_recall_imbalance")
        if (
            recalls
            and min(recalls) < 0.10
            and max(recalls) - min(recalls) > 0.70
        ):
            diagnoses.append("degenerate_model")

        predicted_classes = result.predicted_class_distribution or {}
        if recalls and len(predicted_classes) < len(recalls):
            diagnoses.append("collapsed_predictions")

        if (
            result.validation_accuracy is not None
            and result.majority_class_accuracy is not None
            and result.validation_accuracy - result.majority_class_accuracy
            <= 0.01
        ):
            diagnoses.append("near_majority_baseline")

        if result.feature_count is not None and result.feature_count >= 100000:
            diagnoses.append("high_dimensional_representation")

        if not diagnoses:
            diagnoses.append("no_obvious_metric_warning")
        return diagnoses
