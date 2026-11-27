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
    balanced_accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None
    confusion_matrix: Optional[list] = None
    class_recall: Optional[dict] = None
    predicted_class_distribution: Optional[dict] = None
    majority_class_accuracy: Optional[float] = None
    generalization_gap: Optional[float] = None
    epoch_count: Optional[int] = None
    best_epoch: Optional[int] = None
    peak_gpu_memory_mb: Optional[float] = None
    early_stopped: Optional[bool] = None
    stopped_reason: Optional[str] = None
    epoch_seconds: Optional[list] = None
    prediction_time_seconds: Optional[float] = None
    feature_count: Optional[int] = None
    model_size_bytes: Optional[int] = None
    decision_rule: Optional[dict] = None
    # Path to the validation predictions this experiment produced. Selection
    # reads it to run paired significance tests; None means the experiment did
    # not record predictions and cannot be tested.
    val_predictions_path: Optional[str] = None
    # Filled in by select_best_model when validation labels are available:
    # records whether the chosen model is statistically distinguishable from
    # its rivals. None means the test was not run.
    selection_test: Optional[dict] = None
    # K-fold CV over the TRAINING split, as a second estimator of the same
    # quantity. Reported alongside validation_accuracy, never in place of it:
    # CV on 80% of the rows underestimates data-hungry models.
    cv_scores: Optional[list] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_folds: Optional[int] = None
    cv_skipped_reason: Optional[str] = None
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
