import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def normalize_categorical_array(array):
    """Convert every value to a categorical string and normalize missing values."""
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
                or str(value).strip().lower() in {"", "nan", "none", "?"}
            )

            result[row_index, column_index] = (
                "__MISSING__" if is_missing else str(value)
            )

    return result


def load_split(data_dir, split):
    features = np.load(
        data_dir / f"C_{split}.npy",
        allow_pickle=True,
    )
    labels = np.load(
        data_dir / f"y_{split}.npy",
        allow_pickle=True,
    )

    features = normalize_categorical_array(features)
    labels = labels.astype(int)

    return features, labels


def build_pipeline():
    return Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task1"),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Only use this after the model configuration is locked.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(args.data_dir, "train")
    X_val, y_val = load_split(args.data_dir, "val")

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation accuracy: {val_accuracy:.6f}")
    print(classification_report(y_val, val_predictions, digits=4))

    joblib.dump(
        pipeline,
        args.output_dir / "pipeline.joblib",
    )

    np.save(
        args.output_dir / "val_predictions.npy",
        val_predictions,
    )

    metrics = {
        "model": "one_hot_logistic_regression",
        "random_seed": 42,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "validation_accuracy": float(val_accuracy),
    }

    if args.evaluate_test:
        X_test, y_test = load_split(args.data_dir, "test")
        test_predictions = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"Test accuracy: {test_accuracy:.6f}")
        print(classification_report(y_test, test_predictions, digits=4))

        metrics["test_samples"] = int(len(y_test))
        metrics["test_accuracy"] = float(test_accuracy)

        np.save(
            args.output_dir / "test_predictions.npy",
            test_predictions,
        )

    with open(
        args.output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()