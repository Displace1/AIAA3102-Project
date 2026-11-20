from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    status: str
    validation_accuracy: Optional[float]
    training_time_seconds: float
    artifact_path: Optional[str]
    train_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    model_config: Optional[dict] = None
    train_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    confusion_matrix: Optional[list] = None
    class_recall: Optional[dict] = None
    generalization_gap: Optional[float] = None
    epoch_count: Optional[int] = None
    best_epoch: Optional[int] = None
    peak_gpu_memory_mb: Optional[float] = None
    early_stopped: Optional[bool] = None
    prediction_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None
    model_size_bytes: Optional[int] = None
    diagnosis: Optional[list] = None
    error_type: Optional[str] = None
    raw_error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    worker_exit_code: Optional[int] = None

    def to_dict(self):
        return asdict(self)


class ModelAdapter:
    adapter_name = "abstract_adapter"
    model_name = "abstract_model"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        raise NotImplementedError
