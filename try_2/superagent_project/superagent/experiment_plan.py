from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    adapter_name: str
    hyperparameters: Dict[str, Any]
    reason: str

    def to_dict(self):
        return asdict(self)
