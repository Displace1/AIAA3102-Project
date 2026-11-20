import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_csv(path):
    headlines_a = []
    headlines_b = []
    labels = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
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

    return headlines_a, headlines_b, np.asarray(labels, dtype=int)


def build_pair_features(vectorizer, headlines_a, headlines_b):
    features_a = vectorizer.transform(headlines_a)
    features_b = vectorizer.transform(headlines_b)

    return features_a - features_b


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
        default=Path("outputs/task2"),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Only use this after the model configuration is locked.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_a, train_b, y_train = load_csv(
        args.data_dir / "train.csv"
    )
    val_a, val_b, y_val = load_csv(
        args.data_dir / "val.csv"
    )

    # Fit the vectorizer only on training text.
    training_corpus = train_a + train_b

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=50000,
        sublinear_tf=True,
    )

    vectorizer.fit(training_corpus)

    X_train = build_pair_features(
        vectorizer,
        train_a,
        train_b,
    )
    X_val = build_pair_features(
        vectorizer,
        val_a,
        val_b,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation accuracy: {val_accuracy:.6f}")
    print(classification_report(y_val, val_predictions, digits=4))

    artifact = {
        "vectorizer": vectorizer,
        "classifier": model,
        "feature_method": "tfidf_A_minus_tfidf_B",
    }

    joblib.dump(
        artifact,
        args.output_dir / "model.joblib",
    )

    np.save(
        args.output_dir / "val_predictions.npy",
        val_predictions,
    )

    metrics = {
        "model": "pairwise_tfidf_logistic_regression",
        "feature_method": "tfidf_A_minus_tfidf_B",
        "random_seed": 42,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "validation_accuracy": float(val_accuracy),
    }

    if args.evaluate_test:
        test_a, test_b, y_test = load_csv(
            args.data_dir / "test.csv"
        )

        X_test = build_pair_features(
            vectorizer,
            test_a,
            test_b,
        )
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"Test accuracy: {test_accuracy:.6f}")
        print(classification_report(y_test, test_predictions, digits=4))

        metrics["test_samples"] = int(len(y_test))
        metrics["test_accuracy"] = float(test_accuracy)

        with open(
            args.output_dir / "test_predictions.csv",
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["label_pairwise"])

            for prediction in test_predictions:
                writer.writerow([int(prediction)])

    with open(
        args.output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()