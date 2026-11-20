class ResultAnalyzer:
    """Produces deterministic, auditable diagnoses from experiment metrics."""

    def analyze(self, result):
        if result.status != "success":
            return [f"experiment_failed:{result.error_type or 'unknown'}"]

        diagnoses = []
        if (
            result.generalization_gap is not None
            and result.generalization_gap >= 0.10
        ):
            diagnoses.append("possible_overfitting")

        recalls = list((result.class_recall or {}).values())
        if recalls and max(recalls) - min(recalls) >= 0.15:
            diagnoses.append("class_recall_imbalance")

        if result.feature_count is not None and result.feature_count >= 100000:
            diagnoses.append("high_dimensional_representation")

        if not diagnoses:
            diagnoses.append("no_obvious_metric_warning")
        return diagnoses
