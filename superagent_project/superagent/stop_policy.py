from dataclasses import dataclass

from superagent.metrics import (
    is_metric_better,
    metric_improvement,
    result_metric_value,
)


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
        metric="accuracy",
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
        self.metric = metric

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
            and result_metric_value(result, self.metric) is not None
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
        best_before = result_metric_value(successful[0], self.metric)
        consecutive = 0

        for result in successful[1:]:
            value = result_metric_value(result, self.metric)
            # Direction-aware: a drop in log_loss counts as an improvement even
            # though the raw number got smaller.
            improvement = metric_improvement(
                best_before, value, self.metric
            )

            if improvement >= self.minimum_improvement:
                best_before = value
                consecutive = 0
            else:
                if is_metric_better(value, best_before, self.metric):
                    best_before = value
                consecutive += 1

        return consecutive
