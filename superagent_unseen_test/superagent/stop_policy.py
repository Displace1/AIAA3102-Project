from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str | None


class StopPolicy:
    def __init__(
        self,
        max_experiments,
        max_runtime_seconds,
        no_improvement_patience,
        minimum_improvement,
    ):
        self.max_experiments = max_experiments
        self.max_runtime_seconds = (
            max_runtime_seconds
        )
        self.no_improvement_patience = (
            no_improvement_patience
        )
        self.minimum_improvement = (
            minimum_improvement
        )

    def evaluate(
        self,
        results,
        elapsed_seconds,
    ):
        if len(results) >= self.max_experiments:
            return StopDecision(
                should_stop=True,
                reason="maximum_experiments_reached",
            )

        if elapsed_seconds >= self.max_runtime_seconds:
            return StopDecision(
                should_stop=True,
                reason="maximum_runtime_reached",
            )

        successful = [
            result
            for result in results
            if result.status == "success"
            and result.validation_accuracy is not None
        ]

        if len(successful) <= 1:
            return StopDecision(False, None)

        if self._consecutive_no_improvement(
            successful
        ) >= self.no_improvement_patience:
            return StopDecision(
                should_stop=True,
                reason="no_validation_improvement",
            )

        return StopDecision(False, None)

    def _consecutive_no_improvement(
        self,
        successful,
    ):
        best_before = successful[0].validation_accuracy
        consecutive = 0

        for result in successful[1:]:
            improvement = (
                result.validation_accuracy
                - best_before
            )

            if improvement >= self.minimum_improvement:
                best_before = result.validation_accuracy
                consecutive = 0
            else:
                best_before = max(
                    best_before,
                    result.validation_accuracy,
                )
                consecutive += 1

        return consecutive