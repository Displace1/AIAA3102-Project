import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from superagent.image_inspector import SUPPORTED_IMAGE_EXTENSIONS
from superagent.metrics import test_metric_key
from superagent.predictor import ArtifactPredictor, IMAGE_ADAPTERS
from superagent.recovery import configuration_sha256


class FinalEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalEvaluationResult:
    run_directory: str
    selected_experiment: str
    adapter_name: str
    validation_accuracy: float
    test_accuracy: float
    test_samples: int
    final_model_path: str
    predictions_path: str
    metrics_path: str
    selection_metric: str = "accuracy"
    validation_metric_value: float = 0.0
    test_metric_value: float = 0.0

    def to_dict(self):
        return asdict(self)


class FinalEvaluator:
    def __init__(self, predictor=None):
        self.predictor = predictor or ArtifactPredictor()

    def evaluate(self, run_directory, config):
        run_directory = Path(run_directory).expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(
                f"Run directory does not exist: {run_directory}"
            )

        manifest = self._read_json(
            run_directory / "run_manifest.json", "run manifest"
        )
        self._validate_configuration(manifest, config)
        lock_path = run_directory / "solution_lock.json"
        lock = self._read_json(lock_path, "solution lock")
        self._validate_lock(lock, run_directory)

        selected_experiment = lock["selected_experiment"]
        plan_path = (
            run_directory
            / "experiments"
            / selected_experiment
            / "plan.json"
        )
        plan = self._read_json(plan_path, "selected experiment plan")
        if plan.get("experiment_id") != selected_experiment:
            raise FinalEvaluationError(
                "Selected plan ID does not match solution_lock.json."
            )
        adapter_name = plan.get("adapter_name")
        if not adapter_name:
            raise FinalEvaluationError(
                "Selected experiment plan has no adapter_name."
            )

        artifact_path = Path(lock["artifact_path"]).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = run_directory / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Locked model artifact does not exist: {artifact_path}"
            )
        try:
            artifact_path.relative_to(run_directory)
        except ValueError as error:
            raise FinalEvaluationError(
                "Locked model artifact must be inside the run directory."
            ) from error

        artifact_hash = self._sha256(artifact_path)
        state_path = run_directory / "final_evaluation_state.json"
        self._validate_existing_state(
            state_path,
            run_directory / "final_evaluation.json",
            selected_experiment,
            artifact_hash,
        )
        started_at = datetime.now().astimezone().isoformat()
        self._atomic_write_json(
            state_path,
            {
                "status": "started",
                "selected_experiment": selected_experiment,
                "adapter_name": adapter_name,
                "artifact_sha256": artifact_hash,
                "started_at": started_at,
                "test_results_used_for_selection": False,
            },
        )

        prediction_started = time.perf_counter()
        predictions = self.predictor.predict(
            artifact_path=artifact_path,
            adapter_name=adapter_name,
            config=config,
        )
        prediction_time = time.perf_counter() - prediction_started
        labels = self._load_test_labels(config)
        if len(predictions) != len(labels):
            raise FinalEvaluationError(
                "Prediction count does not match test label count: "
                f"{len(predictions)} != {len(labels)}."
            )
        metrics = self._classification_metrics(labels, predictions)
        metrics["prediction_time_seconds"] = prediction_time
        metrics["test_samples"] = int(len(labels))
        selection_metric = lock["selection_metric"]
        selected_test_metric = float(
            metrics[test_metric_key(selection_metric)]
        )
        selected_validation_metric = float(
            lock.get("validation_metric_value", lock["validation_accuracy"])
        )

        final_directory = run_directory / "final"
        final_directory.mkdir(parents=True, exist_ok=True)
        final_model_filename = (
            "final_model.pt"
            if adapter_name in IMAGE_ADAPTERS
            else "final_model.joblib"
        )
        final_model_path = final_directory / final_model_filename
        predictions_npy_path = final_directory / "test_predictions.npy"
        predictions_csv_path = final_directory / "test_predictions.csv"
        metrics_path = final_directory / "test_metrics.json"
        solution_path = final_directory / "final_solution.json"
        hashes_path = final_directory / "artifact_hashes.json"

        self._atomic_copy(artifact_path, final_model_path)
        self._atomic_write_npy(predictions_npy_path, predictions)
        self._atomic_write_predictions_csv(
            predictions_csv_path, predictions
        )
        self._atomic_write_json(metrics_path, metrics)

        final_solution = {
            "run_type": lock.get(
                "run_type",
                manifest.get("run_type", "development_run"),
            ),
            "selected_experiment": selected_experiment,
            "selected_model": lock["selected_model"],
            "adapter_name": adapter_name,
            "selected_by": lock["selection_metric"],
            "validation_accuracy": lock["validation_accuracy"],
            "validation_metric_value": selected_validation_metric,
            "test_accuracy": metrics["test_accuracy"],
            "test_metric_value": selected_test_metric,
            "test_samples": metrics["test_samples"],
            "test_used_for_selection": False,
            "model_path": str(final_model_path),
            "model_filename": final_model_filename,
            "predictions_npy_path": str(predictions_npy_path),
            "predictions_csv_path": str(predictions_csv_path),
            "metrics_path": str(metrics_path),
            "evaluated_at": datetime.now().astimezone().isoformat(),
        }
        self._atomic_write_json(solution_path, final_solution)

        hashes = {
            "source_model_sha256": artifact_hash,
            "final_model_sha256": self._sha256(final_model_path),
            "test_predictions_npy_sha256": self._sha256(
                predictions_npy_path
            ),
            "test_predictions_csv_sha256": self._sha256(
                predictions_csv_path
            ),
            "test_metrics_sha256": self._sha256(metrics_path),
            "final_solution_sha256": self._sha256(solution_path),
        }
        if hashes["source_model_sha256"] != hashes["final_model_sha256"]:
            raise FinalEvaluationError(
                "Copied final model hash does not match the locked artifact."
            )
        self._atomic_write_json(hashes_path, hashes)

        final_record = {
            **final_solution,
            "confusion_matrix": metrics["confusion_matrix"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "artifact_hashes_path": str(hashes_path),
        }
        self._atomic_write_json(
            run_directory / "final_evaluation.json", final_record
        )

        updated_lock = {
            **lock,
            "test_evaluation_completed": True,
            "test_used_for_selection": False,
            "test_accuracy": metrics["test_accuracy"],
            "final_evaluation_path": str(
                run_directory / "final_evaluation.json"
            ),
            "final_model_path": str(final_model_path),
            "test_predictions_path": str(predictions_csv_path),
        }
        self._atomic_write_json(lock_path, updated_lock)
        self._atomic_write_json(
            state_path,
            {
                "status": "completed",
                "selected_experiment": selected_experiment,
                "adapter_name": adapter_name,
                "artifact_sha256": artifact_hash,
                "started_at": started_at,
                "completed_at": datetime.now().astimezone().isoformat(),
                "test_results_used_for_selection": False,
            },
        )

        return FinalEvaluationResult(
            run_directory=str(run_directory),
            selected_experiment=selected_experiment,
            adapter_name=adapter_name,
            validation_accuracy=float(lock["validation_accuracy"]),
            test_accuracy=metrics["test_accuracy"],
            test_samples=metrics["test_samples"],
            final_model_path=str(final_model_path),
            predictions_path=str(predictions_csv_path),
            metrics_path=str(metrics_path),
            selection_metric=selection_metric,
            validation_metric_value=selected_validation_metric,
            test_metric_value=selected_test_metric,
        )

    @staticmethod
    def _validate_lock(lock, run_directory):
        if lock.get("solution_locked") is not True:
            raise FinalEvaluationError("Final solution has not been locked.")
        if lock.get("selected_automatically") is not True:
            raise FinalEvaluationError(
                "Final solution was not selected automatically."
            )
        if lock.get("test_used_for_selection") is not False:
            raise FinalEvaluationError(
                "The lock does not prove that test data was excluded."
            )
        if lock.get("test_evaluation_completed") is not False:
            raise FinalEvaluationError(
                "Test evaluation has already been completed."
            )
        required = {
            "selected_experiment",
            "selected_model",
            "selection_metric",
            "validation_accuracy",
            "artifact_path",
        }
        missing = required - set(lock)
        if missing:
            raise FinalEvaluationError(
                f"solution_lock.json is missing fields: {sorted(missing)}"
            )

    @staticmethod
    def _validate_configuration(manifest, config):
        if manifest.get("task_name") != config.name:
            raise FinalEvaluationError(
                "Run task name does not match the selected configuration."
            )
        expected_hash = manifest.get("config_sha256")
        if expected_hash is not None:
            if configuration_sha256(config.source_path) != expected_hash:
                raise FinalEvaluationError(
                    "Task configuration changed after the run was created."
                )
        else:
            original = Path(manifest["config_path"]).expanduser().resolve()
            if original != Path(config.source_path).expanduser().resolve():
                raise FinalEvaluationError(
                    "Task configuration path does not match the run."
                )

    @staticmethod
    def _validate_existing_state(
        path, final_evaluation_path, experiment_id, artifact_hash
    ):
        if not path.exists():
            if final_evaluation_path.exists():
                raise FinalEvaluationError(
                    "final_evaluation.json exists without an evaluation state."
                )
            return
        state = FinalEvaluator._read_json(path, "final evaluation state")
        if state.get("status") == "completed":
            raise FinalEvaluationError(
                "Final evaluation state is already completed."
            )
        if (
            state.get("selected_experiment") != experiment_id
            or state.get("artifact_sha256") != artifact_hash
        ):
            raise FinalEvaluationError(
                "Interrupted final evaluation does not match the locked model."
            )

    @staticmethod
    def _load_test_labels(config):
        if config.data["format"] == "npy_split":
            return np.asarray(
                np.load(config.data["test_labels"], allow_pickle=True),
                dtype=int,
            )
        if config.data["format"] == "csv_split":
            labels = []
            target = config.data["target_column"]
            with open(
                config.data["test"],
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None or target not in reader.fieldnames:
                    raise FinalEvaluationError(
                        f"Test CSV is missing target column: {target}"
                    )
                for row in reader:
                    labels.append(int(row[target]))
            return np.asarray(labels, dtype=int)
        if config.data["format"] == "image_directory_split":
            train_directory = Path(config.data["train_directory"])
            test_directory = Path(config.data["test_directory"])
            class_names = sorted(
                path.name
                for path in train_directory.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
            class_to_index = {
                name: index for index, name in enumerate(class_names)
            }
            image_paths = sorted(
                path
                for path in test_directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            labels = []
            for path in image_paths:
                relative_parts = path.relative_to(test_directory).parts
                class_name = relative_parts[0] if len(relative_parts) > 1 else None
                if class_name not in class_to_index:
                    raise FinalEvaluationError(
                        "Final image evaluation requires test images under "
                        "class-named subdirectories matching the training set. "
                        f"Invalid test image: {path}"
                    )
                labels.append(class_to_index[class_name])
            if not labels:
                raise FinalEvaluationError(
                    "Final image evaluation found no labeled test images."
                )
            return np.asarray(labels, dtype=int)
        raise FinalEvaluationError(
            f"Unsupported data format: {config.data['format']}"
        )

    @staticmethod
    def _classification_metrics(labels, predictions):
        classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
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
            "test_accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
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

    @staticmethod
    def _read_json(path, description):
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                value = json.load(file)
        except FileNotFoundError as error:
            raise FinalEvaluationError(
                f"Missing {description}: {path}"
            ) from error
        except json.JSONDecodeError as error:
            raise FinalEvaluationError(
                f"Invalid JSON in {description}: {path}"
            ) from error
        if not isinstance(value, dict):
            raise FinalEvaluationError(
                f"Expected a JSON object in {description}: {path}"
            )
        return value

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _atomic_copy(source, destination):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with Path(source).open("rb") as source_file, temporary.open(
                "wb"
            ) as destination_file:
                shutil.copyfileobj(source_file, destination_file)
                destination_file.flush()
                os.fsync(destination_file.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _atomic_write_npy(path, values):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as file:
                np.save(file, values)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _atomic_write_predictions_csv(path, predictions):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["row_id", "prediction"])
                for row_id, prediction in enumerate(predictions):
                    writer.writerow([row_id, int(prediction)])
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
