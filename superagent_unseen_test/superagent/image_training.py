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
    confusion_matrix,
    precision_recall_fscore_support,
)


class ImageTrainingError(RuntimeError):
    pass


class GPUOutOfMemoryError(ImageTrainingError):
    pass


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
    confusion_matrix: list
    class_recall: dict
    training_time_seconds: float
    peak_gpu_memory_mb: float
    device: str
    best_weights_path: str
    history: list

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


def _classification_metrics(labels, predictions, number_of_classes):
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
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classes
        ).astype(int).tolist(),
        "class_recall": {
            str(label): float(value)
            for label, value in zip(classes, per_class_recall)
        },
    }


def evaluate_image_model(model, data_loader, device, number_of_classes, loss_fn):
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []
    with torch.inference_mode():
        for images, batch_labels in data_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, batch_labels)
            total_loss += float(loss.item()) * len(batch_labels)
            labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not labels:
        raise ImageTrainingError("Evaluation data loader produced no samples.")
    return {
        "loss": total_loss / len(labels),
        **_classification_metrics(labels, predictions, number_of_classes),
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
):
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least two.")

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
    best_accuracy = float("-inf")
    best_epoch = 0
    best_train_accuracy = 0.0
    best_validation_metrics = None
    epochs_without_improvement = 0
    history = []
    early_stopped = False

    try:
        for epoch in range(1, epochs + 1):
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
            epoch_record = {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1_macro": validation_metrics["f1_macro"],
                "backbone_frozen": epoch <= freeze_backbone_epochs,
            }
            history.append(epoch_record)

            improvement = validation_metrics["accuracy"] - best_accuracy
            if improvement >= minimum_improvement:
                best_accuracy = validation_metrics["accuracy"]
                best_epoch = epoch
                best_train_accuracy = train_accuracy
                best_validation_metrics = validation_metrics
                epochs_without_improvement = 0
                _atomic_torch_save(model.state_dict(), best_weights_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    early_stopped = True
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
        confusion_matrix=best_validation_metrics["confusion_matrix"],
        class_recall=best_validation_metrics["class_recall"],
        training_time_seconds=training_time,
        peak_gpu_memory_mb=float(peak_gpu_memory_mb),
        device=str(device),
        best_weights_path=str(best_weights_path.resolve()),
        history=history,
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
