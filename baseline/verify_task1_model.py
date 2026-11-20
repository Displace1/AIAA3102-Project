import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def normalize_categorical_array(array):
    """Use exactly the same normalization as the training script."""
    result = np.empty(array.shape, dtype=object)

    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            value = array[row_index, column_index]

            is_missing = (
                value is None
                or (
                    isinstance(value, (float, np.floating))
                    and np.isnan(value)
                )
                or str(value).strip().lower() in {
                    "",
                    "nan",
                    "none",
                    "?",
                }
            )

            result[row_index, column_index] = (
                "__MISSING__" if is_missing else str(value)
            )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing C_val.npy and y_val.npy",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/task1"),
        help="Directory containing pipeline.joblib",
    )

    args = parser.parse_args()

    model_path = args.model_dir / "pipeline.joblib"
    saved_predictions_path = args.model_dir / "val_predictions.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not saved_predictions_path.exists():
        raise FileNotFoundError(
            f"Saved predictions not found: {saved_predictions_path}"
        )

    # Load the saved pipeline.
    pipeline = joblib.load(model_path)

    # Load validation data only.
    X_val = np.load(
        args.data_dir / "C_val.npy",
        allow_pickle=True,
    )
    y_val = np.load(
        args.data_dir / "y_val.npy",
        allow_pickle=True,
    ).astype(int)

    X_val = normalize_categorical_array(X_val)

    # Predict using the reloaded model.
    reloaded_predictions = pipeline.predict(X_val)
    reloaded_predictions = np.asarray(
        reloaded_predictions,
        dtype=int,
    )

    # Load predictions produced during training.
    saved_predictions = np.load(
        saved_predictions_path,
        allow_pickle=False,
    ).astype(int)

    if len(reloaded_predictions) != len(y_val):
        raise AssertionError(
            "Prediction count does not match validation sample count."
        )

    predictions_identical = np.array_equal(
        reloaded_predictions,
        saved_predictions,
    )

    accuracy = accuracy_score(
        y_val,
        reloaded_predictions,
    )

    print(f"Model path: {model_path.resolve()}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Reloaded validation accuracy: {accuracy:.6f}")
    print(
        "Reloaded predictions equal saved predictions: "
        f"{predictions_identical}"
    )
    print(
        "Prediction values:",
        np.unique(reloaded_predictions).tolist(),
    )

    if not predictions_identical:
        raise AssertionError(
            "Reloaded predictions differ from saved predictions."
        )

    print("Task 1 model reload verification passed.")


if __name__ == "__main__":
    main()