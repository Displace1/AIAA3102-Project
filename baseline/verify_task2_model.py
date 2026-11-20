import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def load_csv(path):
    headlines_a = []
    headlines_b = []
    labels = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "headline_A",
            "headline_B",
            "label_pairwise",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} is missing required columns. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            headlines_a.append(row["headline_A"].strip())
            headlines_b.append(row["headline_B"].strip())
            labels.append(int(row["label_pairwise"]))

    return (
        headlines_a,
        headlines_b,
        np.asarray(labels, dtype=int),
    )


def build_pair_features(
    vectorizer,
    headlines_a,
    headlines_b,
):
    features_a = vectorizer.transform(headlines_a)
    features_b = vectorizer.transform(headlines_b)

    return features_a - features_b


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing val.csv",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/task2"),
        help="Directory containing model.joblib",
    )

    args = parser.parse_args()

    model_path = args.model_dir / "model.joblib"
    saved_predictions_path = args.model_dir / "val_predictions.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not saved_predictions_path.exists():
        raise FileNotFoundError(
            f"Saved predictions not found: {saved_predictions_path}"
        )

    # Load the model artifact.
    artifact = joblib.load(model_path)

    required_keys = {
        "vectorizer",
        "classifier",
        "feature_method",
    }

    missing_keys = required_keys - set(artifact.keys())

    if missing_keys:
        raise KeyError(
            f"Model artifact is missing keys: {sorted(missing_keys)}"
        )

    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]

    # Load validation data.
    val_a, val_b, y_val = load_csv(
        args.data_dir / "val.csv"
    )

    # Build features using the loaded vectorizer.
    X_val = build_pair_features(
        vectorizer,
        val_a,
        val_b,
    )

    # Predict with the loaded classifier.
    reloaded_predictions = classifier.predict(X_val)
    reloaded_predictions = np.asarray(
        reloaded_predictions,
        dtype=int,
    )

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
    print(f"Feature method: {artifact['feature_method']}")
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

    print("Task 2 model reload verification passed.")


if __name__ == "__main__":
    main()