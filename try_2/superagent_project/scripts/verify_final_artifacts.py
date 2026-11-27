import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from superagent.image_artifact import (
    build_model_from_artifact,
    load_image_artifact,
)
from superagent.predictor import IMAGE_ADAPTERS


class FinalArtifactVerificationError(RuntimeError):
    pass


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise FinalArtifactVerificationError(
            f"Cannot read required JSON file {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise FinalArtifactVerificationError(
            f"Expected JSON object: {path}"
        )
    return value


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(65536), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise FinalArtifactVerificationError(
            f"Required artifact is missing: {path}"
        ) from error
    return digest.hexdigest()


def verify_final_artifacts(run_directory):
    run_directory = Path(run_directory).expanduser().resolve()
    final_directory = run_directory / "final"
    lock = _read_json(run_directory / "solution_lock.json")
    metrics = _read_json(final_directory / "test_metrics.json")
    solution = _read_json(final_directory / "final_solution.json")
    hashes = _read_json(final_directory / "artifact_hashes.json")

    if lock.get("test_evaluation_completed") is not True:
        raise FinalArtifactVerificationError(
            "solution_lock.json does not mark test evaluation complete."
        )
    if lock.get("test_used_for_selection") is not False:
        raise FinalArtifactVerificationError(
            "solution_lock.json does not preserve the no-test-selection rule."
        )

    source_model_path = Path(lock["artifact_path"]).expanduser().resolve()
    adapter_name = solution.get("adapter_name")
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
        raise FinalArtifactVerificationError(
            "final_solution.json contains an unsafe model filename."
        )
    final_model_path = final_directory / model_filename
    predictions_npy_path = final_directory / "test_predictions.npy"
    predictions_csv_path = final_directory / "test_predictions.csv"
    metrics_path = final_directory / "test_metrics.json"
    solution_path = final_directory / "final_solution.json"
    expected_hashes = {
        "source_model_sha256": _sha256(source_model_path),
        "final_model_sha256": _sha256(final_model_path),
        "test_predictions_npy_sha256": _sha256(predictions_npy_path),
        "test_predictions_csv_sha256": _sha256(predictions_csv_path),
        "test_metrics_sha256": _sha256(metrics_path),
        "final_solution_sha256": _sha256(solution_path),
    }
    for name, actual in expected_hashes.items():
        if hashes.get(name) != actual:
            raise FinalArtifactVerificationError(
                f"SHA-256 mismatch for {name}."
            )
    if (
        expected_hashes["source_model_sha256"]
        != expected_hashes["final_model_sha256"]
    ):
        raise FinalArtifactVerificationError(
            "Exported model differs from the locked source model."
        )

    try:
        if adapter_name in IMAGE_ADAPTERS:
            build_model_from_artifact(load_image_artifact(final_model_path))
        else:
            joblib.load(final_model_path)
    except Exception as error:
        raise FinalArtifactVerificationError(
            f"Final model cannot be reloaded: {error}"
        ) from error

    predictions_npy = np.asarray(np.load(predictions_npy_path), dtype=int)
    predictions_csv = []
    with predictions_csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["row_id", "prediction"]:
            raise FinalArtifactVerificationError(
                "Prediction CSV header is invalid."
            )
        for expected_row_id, row in enumerate(reader):
            if int(row["row_id"]) != expected_row_id:
                raise FinalArtifactVerificationError(
                    "Prediction CSV row IDs are not sequential."
                )
            predictions_csv.append(int(row["prediction"]))
    predictions_csv = np.asarray(predictions_csv, dtype=int)
    if not np.array_equal(predictions_npy, predictions_csv):
        raise FinalArtifactVerificationError(
            "NPY and CSV predictions are different."
        )
    if len(predictions_npy) != int(metrics["test_samples"]):
        raise FinalArtifactVerificationError(
            "Prediction count does not match test_metrics.json."
        )
    if solution.get("test_accuracy") != metrics.get("test_accuracy"):
        raise FinalArtifactVerificationError(
            "Final solution and metrics contain different test accuracy."
        )

    return {
        "verified": True,
        "run_directory": str(run_directory),
        "selected_experiment": solution["selected_experiment"],
        "test_samples": int(metrics["test_samples"]),
        "test_accuracy": float(metrics["test_accuracy"]),
        "final_model_sha256": expected_hashes["final_model_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify exported final model and prediction artifacts."
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    result = verify_final_artifacts(args.run_directory)
    print("Final artifacts verified successfully.")
    print(f"Selected experiment: {result['selected_experiment']}")
    print(f"Test samples: {result['test_samples']}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    print(f"Model SHA-256: {result['final_model_sha256']}")


if __name__ == "__main__":
    main()
