import json
import time
from pathlib import Path

import numpy as np
import torch

from adapters.base import ExperimentResult, ModelAdapter
from superagent.image_artifact import (
    load_image_artifact,
    save_image_artifact,
    write_artifact_manifest,
)
from superagent.image_dataset import build_image_dataloaders
from superagent.image_training import (
    select_training_device,
    train_image_classifier,
)


class ImageClassificationAdapter(ModelAdapter):
    """Shared training and artifact logic for image classifiers."""

    architecture_name = "abstract_image_classifier"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def build_model(self, number_of_classes, hyperparameters):
        raise NotImplementedError

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        if data_profile.get("modality") != "image":
            raise ValueError(
                f"{self.adapter_name} requires an image data profile."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        number_of_classes = int(data_profile["number_of_classes"])
        image_size = int(hyperparameters["image_size"])
        loaders = build_image_dataloaders(
            config=config,
            data_profile=data_profile,
            image_size=image_size,
            batch_size=int(hyperparameters["batch_size"]),
            data_augmentation=bool(
                hyperparameters["data_augmentation"]
            ),
        )

        model, backbone_parameters = self.build_model(
            number_of_classes=number_of_classes,
            hyperparameters=hyperparameters,
        )
        training = train_image_classifier(
            model=model,
            data_loaders=loaders,
            output_directory=output_dir,
            number_of_classes=number_of_classes,
            learning_rate=float(hyperparameters["learning_rate"]),
            epochs=int(hyperparameters["epochs"]),
            weight_decay=float(hyperparameters["weight_decay"]),
            early_stopping_patience=int(
                hyperparameters["early_stopping_patience"]
            ),
            random_seed=self.random_seed,
            freeze_backbone_epochs=int(
                hyperparameters.get("freeze_backbone_epochs", 0)
            ),
            backbone_parameters=backbone_parameters,
            metric=config.metric,
        )

        device = select_training_device(training.device)
        predictions, prediction_seconds = self._predict_validation(
            model=model,
            data_loader=loaders.validation,
            device=device,
        )
        np.save(output_dir / "val_predictions.npy", predictions)

        artifact_path = output_dir / "model_package.pt"
        model_config = {
            **hyperparameters,
            "adapter_name": self.adapter_name,
            "architecture": self.architecture_name,
            "number_of_classes": number_of_classes,
            "class_to_index": loaders.class_to_index,
            "index_to_class": loaders.index_to_class,
            "random_seed": self.random_seed,
            "device_used": training.device,
            "best_epoch": training.best_epoch,
            "epochs_completed": training.epochs_completed,
            "early_stopped": training.early_stopped,
            "peak_gpu_memory_mb": training.peak_gpu_memory_mb,
        }
        save_image_artifact(
            model=model,
            path=artifact_path,
            adapter_name=self.adapter_name,
            architecture=self.architecture_name,
            model_config=model_config,
        )
        artifact_package = load_image_artifact(artifact_path)
        write_artifact_manifest(
            artifact_package,
            output_dir / "artifact_manifest.json",
        )
        model_size_bytes = artifact_path.stat().st_size

        metrics = {
            **training.to_dict(),
            "train_samples": int(data_profile["train_samples"]),
            "validation_samples": int(
                data_profile["validation_samples"]
            ),
            "prediction_time_seconds": prediction_seconds,
            "model_size_bytes": model_size_bytes,
            "artifact_path": str(artifact_path.resolve()),
        }
        with (output_dir / "metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=self.model_name,
            status="success",
            validation_accuracy=training.validation_accuracy,
            training_time_seconds=training.training_time_seconds,
            artifact_path=str(artifact_path.resolve()),
            train_samples=int(data_profile["train_samples"]),
            validation_samples=int(data_profile["validation_samples"]),
            model_config=model_config,
            train_accuracy=training.train_accuracy,
            precision_macro=training.precision_macro,
            recall_macro=training.recall_macro,
            f1_macro=training.f1_macro,
            balanced_accuracy=training.balanced_accuracy,
            confusion_matrix=training.confusion_matrix,
            class_recall=training.class_recall,
            predicted_class_distribution=(
                training.predicted_class_distribution
            ),
            majority_class_accuracy=training.majority_class_accuracy,
            generalization_gap=(
                training.train_accuracy - training.validation_accuracy
            ),
            epoch_count=training.epochs_completed,
            best_epoch=training.best_epoch,
            peak_gpu_memory_mb=training.peak_gpu_memory_mb,
            early_stopped=training.early_stopped,
            prediction_time_seconds=prediction_seconds,
            model_size_bytes=model_size_bytes,
        )

    @staticmethod
    def _predict_validation(model, data_loader, device):
        model.eval()
        predictions = []
        started_at = time.perf_counter()
        with torch.inference_mode():
            for images, _ in data_loader:
                logits = model(images.to(device, non_blocking=True))
                predictions.extend(
                    logits.argmax(dim=1).detach().cpu().tolist()
                )
        prediction_seconds = time.perf_counter() - started_at
        return np.asarray(predictions, dtype=np.int64), prediction_seconds
