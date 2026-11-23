import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from superagent.metrics import (
    log_loss_value,
    metric_direction,
    roc_auc_value,
)


class ImageTrainingError(RuntimeError):
    pass


class GPUOutOfMemoryError(ImageTrainingError):
    pass


# A new epoch is started only if 1.25x the slowest observed epoch still fits in
# the remaining budget.  The margin covers the tail work that happens after the
# last epoch (best-weight reload, validation inference in the adapter, artifact
# save) and the variance caused by a partially warm page cache.
TIME_BUDGET_EPOCH_MARGIN = 1.25


@dataclass(frozen=True)
class ImageTrainingResult:
    best_epoch: int
    epochs_completed: int
    early_stopped: bool
    train_accuracy: float
    validation_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    balanced_accuracy: float
    confusion_matrix: list
    class_recall: dict
    predicted_class_distribution: dict
    majority_class_accuracy: float
    training_time_seconds: float
    peak_gpu_memory_mb: float
    device: str
    best_weights_path: str
    history: list
    # "completed" | "early_stopping" | "time_budget".  Kept separate from
    # ``early_stopped`` because early stopping means "the model converged" while
    # a time-budget stop means "the plan asked for more epochs than the clock
    # allowed"; the planner reacts differently to the two.
    stopped_reason: str = "completed"
    time_budget_seconds: float | None = None
    # ``None`` when the number of classes is below two or the scores were
    # unavailable; a plan selected by ``roc_auc`` then cannot be satisfied.
    roc_auc: float | None = None
    # Cross entropy of the softmax probabilities on the validation split.  A
    # minimisation metric, so it is reported but never compared with ``max``.
    log_loss: float | None = None

    def to_dict(self):
        return asdict(self)


def set_image_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_training_device(requested_device=None):
    if requested_device is None or requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ImageTrainingError("CUDA was requested but is not available.")
    return device


def _classification_metrics(labels, predictions, number_of_classes, scores=None):
    classes = list(range(number_of_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )
    per_class_recall = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        average=None,
        zero_division=0,
    )[1]
    label_values, label_counts = np.unique(labels, return_counts=True)
    prediction_values, prediction_counts = np.unique(
        predictions, return_counts=True
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc": roc_auc_value(labels, scores),
        "log_loss": log_loss_value(
            labels, scores, classes=classes
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classes
        ).astype(int).tolist(),
        "class_recall": {
            str(label): float(value)
            for label, value in zip(classes, per_class_recall)
        },
        "predicted_class_distribution": {
            str(label): int(count)
            for label, count in zip(prediction_values, prediction_counts)
        },
        "majority_class_accuracy": float(label_counts.max() / label_counts.sum()),
    }


def evaluate_image_model(model, data_loader, device, number_of_classes, loss_fn):
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []
    scores = []
    with torch.inference_mode():
        for images, batch_labels in data_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, batch_labels)
            total_loss += float(loss.item()) * len(batch_labels)
            labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            if number_of_classes > 1:
                scores.append(
                    torch.softmax(logits, dim=1).detach().cpu().numpy()
                )
    if not labels:
        raise ImageTrainingError("Evaluation data loader produced no samples.")
    # A single score column is meaningless for ROC-AUC, so binary tasks get the
    # two-column matrix and let ``roc_auc_value`` pick the positive class.
    score_matrix = np.vstack(scores) if scores else None
    return {
        "loss": total_loss / len(labels),
        **_classification_metrics(
            labels, predictions, number_of_classes, score_matrix
        ),
    }


def train_image_classifier(
    model,
    data_loaders,
    output_directory,
    number_of_classes,
    learning_rate,
    epochs,
    weight_decay,
    early_stopping_patience,
    random_seed,
    minimum_improvement=0.0001,
    requested_device="auto",
    use_mixed_precision=True,
    freeze_backbone_epochs=0,
    backbone_parameters=None,
    metric="accuracy",
    time_budget_seconds=None,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least two.")
    if time_budget_seconds is not None and time_budget_seconds < 0:
        raise ValueError("time_budget_seconds cannot be negative.")

    set_image_random_seed(random_seed)
    device = select_training_device(requested_device)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_weights_path = output_directory / "best_state_dict.pt"
    history_path = output_directory / "training_history.json"
    model = model.to(device)

    backbone_parameters = list(backbone_parameters or [])
    if freeze_backbone_epochs > 0:
        for parameter in backbone_parameters:
            parameter.requires_grad = False

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    mixed_precision = bool(use_mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()
    metric_keys = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "roc_auc": "roc_auc",
        "log_loss": "log_loss",
    }
    if metric not in metric_keys:
        raise ValueError(f"Unsupported image selection metric: {metric!r}.")
    selection_key = metric_keys[metric]
    # log_loss is minimised; seeding the incumbent with +inf makes the first
    # epoch a guaranteed improvement without a special case.
    direction = metric_direction(metric)
    best_score = float("-inf") if direction > 0 else float("inf")
    best_epoch = 0
    best_train_accuracy = 0.0
    best_validation_metrics = None
    epochs_without_improvement = 0
    history = []
    early_stopped = False
    stopped_reason = "completed"

    try:
        for epoch in range(1, epochs + 1):
            epoch_started_at = time.perf_counter()
            if epoch == freeze_backbone_epochs + 1:
                for parameter in backbone_parameters:
                    parameter.requires_grad = True

            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for images, labels in data_loaders.train:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=mixed_precision,
                ):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss.item()) * len(labels)
                total_correct += int(
                    (logits.argmax(dim=1) == labels).sum().item()
                )
                total_samples += len(labels)
            if total_samples == 0:
                raise ImageTrainingError(
                    "Training data loader produced no samples."
                )

            train_accuracy = total_correct / total_samples
            validation_metrics = evaluate_image_model(
                model=model,
                data_loader=data_loaders.validation,
                device=device,
                number_of_classes=number_of_classes,
                loss_fn=loss_fn,
            )
            selection_value = validation_metrics.get(selection_key)
            if selection_value is None:
                raise ImageTrainingError(
                    f"Selection metric {metric!r} is not available for this "
                    "task; it needs at least two classes and probability "
                    "scores."
                )
            epoch_record = {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1_macro": validation_metrics["f1_macro"],
                "validation_log_loss": validation_metrics.get("log_loss"),
                "validation_balanced_accuracy": validation_metrics.get(
                    "balanced_accuracy", validation_metrics["recall_macro"]
                ),
                "selection_metric": metric,
                "selection_metric_value": selection_value,
                "backbone_frozen": epoch <= freeze_backbone_epochs,
                # Wall clock for this epoch including its share of the cold
                # cache penalty.  This is the only honest throughput signal on
                # an uncalibrated machine, so it is recorded per epoch.
                "epoch_seconds": time.perf_counter() - epoch_started_at,
            }
            history.append(epoch_record)

            improvement = direction * (selection_value - best_score)
            if improvement >= minimum_improvement:
                best_score = selection_value
                best_epoch = epoch
                best_train_accuracy = train_accuracy
                best_validation_metrics = validation_metrics
                epochs_without_improvement = 0
                _atomic_torch_save(model.state_dict(), best_weights_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    early_stopped = True
                    stopped_reason = "early_stopping"
                    break

            # Time-budget guard.  Stopping here keeps the best weights saved so
            # far and lets the worker send a usable result, instead of being
            # killed by the parent at the timeout with nothing to show.  The
            # slowest epoch so far is used as the next-epoch cost because the
            # first epoch carries the cold page cache penalty and later epochs
            # can be slower again if the dataset does not fit in RAM.
            if time_budget_seconds is not None:
                elapsed = time.perf_counter() - started_at
                remaining = float(time_budget_seconds) - elapsed
                slowest_epoch = max(
                    record["epoch_seconds"] for record in history
                )
                if remaining < slowest_epoch * TIME_BUDGET_EPOCH_MARGIN:
                    stopped_reason = "time_budget"
                    break

        if best_validation_metrics is None or not best_weights_path.is_file():
            raise ImageTrainingError("Training did not produce best weights.")
        state_dict = torch.load(
            best_weights_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        if "out of memory" in str(error).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
            raise GPUOutOfMemoryError(
                "CUDA out of memory during image training."
            ) from error
        raise

    training_time = time.perf_counter() - started_at
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )
    result = ImageTrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        early_stopped=early_stopped,
        train_accuracy=float(best_train_accuracy),
        validation_accuracy=float(best_validation_metrics["accuracy"]),
        precision_macro=float(best_validation_metrics["precision_macro"]),
        recall_macro=float(best_validation_metrics["recall_macro"]),
        f1_macro=float(best_validation_metrics["f1_macro"]),
        balanced_accuracy=float(
            best_validation_metrics.get(
                "balanced_accuracy", best_validation_metrics["recall_macro"]
            )
        ),
        confusion_matrix=best_validation_metrics["confusion_matrix"],
        class_recall=best_validation_metrics["class_recall"],
        predicted_class_distribution=best_validation_metrics.get(
            "predicted_class_distribution", {}
        ),
        majority_class_accuracy=float(
            best_validation_metrics.get("majority_class_accuracy", 0.0)
        ),
        training_time_seconds=training_time,
        peak_gpu_memory_mb=float(peak_gpu_memory_mb),
        device=str(device),
        best_weights_path=str(best_weights_path.resolve()),
        history=history,
        stopped_reason=stopped_reason,
        time_budget_seconds=(
            None if time_budget_seconds is None else float(time_budget_seconds)
        ),
        roc_auc=best_validation_metrics.get("roc_auc"),
        log_loss=best_validation_metrics.get("log_loss"),
    )
    _atomic_write_json(history_path, result.to_dict())
    return result


def _atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
