import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from superagent.predictor import ArtifactPredictor, IMAGE_ADAPTERS


class FinalModelInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalPredictionResult:
    run_directory: str
    selected_experiment: str
    adapter_name: str
    input_path: str
    prediction_count: int
    predictions_csv_path: str
    predictions_npy_path: str
    manifest_path: str

    def to_dict(self):
        return asdict(self)


class FinalModelRunner:
    """Run a verified exported final model on unlabeled input data."""

    def __init__(self, predictor=None):
        self.predictor = predictor or ArtifactPredictor()

    def predict(self, run_directory, input_path, output_directory):
        run_directory = Path(run_directory).expanduser().resolve()
        input_path = Path(input_path).expanduser().resolve()
        output_directory = Path(output_directory).expanduser().resolve()

        if not run_directory.is_dir():
            raise FileNotFoundError(
                f"Final run directory does not exist: {run_directory}"
            )
        if not input_path.exists():
            raise FileNotFoundError(
                f"Prediction input does not exist: {input_path}"
            )
        if output_directory.is_file():
            raise FinalModelInferenceError(
                f"Prediction output path is a file: {output_directory}"
            )
        if output_directory.exists() and any(output_directory.iterdir()):
            raise FinalModelInferenceError(
                f"Prediction output directory is not empty: {output_directory}"
            )

        final_directory = run_directory / "final"
        solution = self._read_json(
            final_directory / "final_solution.json", "final solution"
        )
        hashes = self._read_json(
            final_directory / "artifact_hashes.json", "artifact hashes"
        )
        lock = self._read_json(
            run_directory / "solution_lock.json", "solution lock"
        )
        self._validate_package(lock, solution)
        adapter_name = solution["adapter_name"]
        default_model_filename = (
            "final_model.pt"
            if adapter_name in IMAGE_ADAPTERS
            else "final_model.joblib"
        )
        model_filename = solution.get(
            "model_filename", default_model_filename
        )
        if (
            not isinstance(model_filename, str)
            or Path(model_filename).name != model_filename
        ):
            raise FinalModelInferenceError(
                "final_solution.json contains an unsafe model filename."
            )
        model_path = final_directory / model_filename

        expected_hash = hashes.get("final_model_sha256")
        if not expected_hash:
            raise FinalModelInferenceError(
                "artifact_hashes.json has no final_model_sha256."
            )
        actual_hash = self._sha256(model_path)
        if actual_hash != expected_hash:
            raise FinalModelInferenceError(
                "Final model SHA-256 does not match artifact_hashes.json."
            )

        prediction_batch = self.predictor.predict_input_with_metadata(
            artifact_path=model_path,
            adapter_name=adapter_name,
            input_path=input_path,
        )
        predictions = np.asarray(
            prediction_batch.predictions, dtype=int
        ).reshape(-1)

        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "predictions.csv"
        npy_path = output_directory / "predictions.npy"
        manifest_path = output_directory / "prediction_manifest.json"
        self._atomic_write_npy(npy_path, predictions)
        self._atomic_write_csv(
            csv_path,
            predictions,
            input_paths=prediction_batch.input_paths,
            class_names=prediction_batch.class_names,
        )

        manifest = {
            "selected_experiment": solution["selected_experiment"],
            "adapter_name": adapter_name,
            "model_path": str(model_path),
            "model_sha256": actual_hash,
            "input_path": str(input_path),
            "input_kind": "directory" if input_path.is_dir() else "file",
            "input_sha256": self._sha256_input(input_path),
            "prediction_count": int(len(predictions)),
            "input_paths_included": prediction_batch.input_paths is not None,
            "class_names_included": prediction_batch.class_names is not None,
            "predictions_csv_path": str(csv_path),
            "predictions_csv_sha256": self._sha256(csv_path),
            "predictions_npy_path": str(npy_path),
            "predictions_npy_sha256": self._sha256(npy_path),
            "created_at": datetime.now().astimezone().isoformat(),
            "labels_read": False,
            "llm_called": False,
        }
        self._atomic_write_json(manifest_path, manifest)

        return FinalPredictionResult(
            run_directory=str(run_directory),
            selected_experiment=solution["selected_experiment"],
            adapter_name=adapter_name,
            input_path=str(input_path),
            prediction_count=int(len(predictions)),
            predictions_csv_path=str(csv_path),
            predictions_npy_path=str(npy_path),
            manifest_path=str(manifest_path),
        )

    @staticmethod
    def _validate_package(lock, solution):
        if lock.get("solution_locked") is not True:
            raise FinalModelInferenceError("Final solution is not locked.")
        if lock.get("test_evaluation_completed") is not True:
            raise FinalModelInferenceError(
                "Final test evaluation has not been completed."
            )
        if lock.get("test_used_for_selection") is not False:
            raise FinalModelInferenceError(
                "Final package violates the no-test-selection rule."
            )
        required = {"selected_experiment", "adapter_name"}
        missing = required - set(solution)
        if missing:
            raise FinalModelInferenceError(
                f"final_solution.json is missing fields: {sorted(missing)}"
            )
        if solution["selected_experiment"] != lock.get("selected_experiment"):
            raise FinalModelInferenceError(
                "Final solution and solution lock select different experiments."
            )

    @staticmethod
    def _read_json(path, description):
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise FinalModelInferenceError(
                f"Cannot read {description} {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise FinalModelInferenceError(
                f"Expected a JSON object in {description}: {path}"
            )
        return value

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        try:
            with Path(path).open("rb") as file:
                for chunk in iter(lambda: file.read(65536), b""):
                    digest.update(chunk)
        except FileNotFoundError as error:
            raise FinalModelInferenceError(
                f"Required file does not exist: {path}"
            ) from error
        return digest.hexdigest()

    @staticmethod
    def _sha256_input(path):
        path = Path(path)
        if path.is_file():
            return FinalModelRunner._sha256(path)
        if not path.is_dir():
            raise FinalModelInferenceError(
                f"Prediction input does not exist: {path}"
            )

        digest = hashlib.sha256()
        files = sorted(value for value in path.rglob("*") if value.is_file())
        if not files:
            raise FinalModelInferenceError(
                f"Prediction input directory is empty: {path}"
            )
        for file_path in files:
            relative_path = file_path.relative_to(path).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as file:
                for chunk in iter(lambda: file.read(65536), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
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
    def _atomic_write_csv(
        path,
        predictions,
        input_paths=None,
        class_names=None,
    ):
        has_image_metadata = input_paths is not None or class_names is not None
        if has_image_metadata:
            if input_paths is None or class_names is None:
                raise FinalModelInferenceError(
                    "Image prediction paths and class names must be provided together."
                )
            if len(input_paths) != len(predictions) or len(class_names) != len(
                predictions
            ):
                raise FinalModelInferenceError(
                    "Image prediction metadata count does not match predictions."
                )
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                if has_image_metadata:
                    writer.writerow(
                        ["row_id", "input_path", "prediction", "class_name"]
                    )
                    for row_id, (input_path, prediction, class_name) in enumerate(
                        zip(input_paths, predictions, class_names)
                    ):
                        writer.writerow(
                            [row_id, input_path, int(prediction), class_name]
                        )
                else:
                    writer.writerow(["row_id", "prediction"])
                    for row_id, prediction in enumerate(predictions):
                        writer.writerow([row_id, int(prediction)])
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
