import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from adapters.base import ExperimentResult, ModelAdapter
from superagent.evaluation import evaluate_classifier, result_metric_fields


def load_text_pair_csv(path, feature_columns, target_column):
    columns, labels = load_text_csv(path, feature_columns, target_column)
    if len(columns) != 2:
        raise ValueError("Text-pair loading requires exactly two columns.")
    return columns[0], columns[1], labels


def load_text_csv(path, feature_columns, target_column):
    text_columns = [[] for _ in feature_columns]
    labels = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for index, column in enumerate(feature_columns):
                text_columns[index].append(row[column].strip())
            labels.append(int(row[target_column]))

    return text_columns, np.asarray(labels, dtype=int)


def build_pair_features(vectorizer, first_texts, second_texts):
    return vectorizer.transform(first_texts) - vectorizer.transform(
        second_texts
    )


def build_text_features(vectorizer, text_columns):
    if len(text_columns) == 1:
        return vectorizer.transform(text_columns[0])
    if len(text_columns) == 2:
        return build_pair_features(
            vectorizer, text_columns[0], text_columns[1]
        )
    raise ValueError("Text tasks require one or two feature columns.")


class TextPairBaselineAdapter(ModelAdapter):
    adapter_name = "text_pair_word_tfidf_logistic"
    model_name = "pairwise_tfidf_logistic_regression"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        output_dir = Path(output_dir)
        feature_columns = data_profile["feature_columns"]
        target_column = data_profile["target_column"]

        train_columns, y_train = load_text_csv(
            config.data["train"], feature_columns, target_column
        )
        val_columns, y_val = load_text_csv(
            config.data["validation"], feature_columns, target_column
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(
                hyperparameters["ngram_min"],
                hyperparameters["ngram_max"],
            ),
            min_df=hyperparameters["min_df"],
            max_df=hyperparameters["max_df"],
            max_features=hyperparameters["max_features"],
            sublinear_tf=hyperparameters["sublinear_tf"],
        )
        training_started = time.perf_counter()
        vectorizer.fit(
            [text for column in train_columns for text in column]
        )
        X_train = build_text_features(vectorizer, train_columns)
        X_val = build_text_features(vectorizer, val_columns)

        classifier = LogisticRegression(
            C=hyperparameters["C"],
            max_iter=hyperparameters["max_iter"],
            random_state=self.random_seed,
        )
        classifier.fit(X_train, y_train)
        training_time_seconds = time.perf_counter() - training_started
        metrics = evaluate_classifier(classifier, X_train, y_train, X_val, y_val)
        predictions = metrics["predictions"].astype(int)

        model_path = output_dir / "model.joblib"
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "feature_method": (
                    "tfidf_single"
                    if len(feature_columns) == 1
                    else "tfidf_A_minus_tfidf_B"
                ),
                "feature_columns": feature_columns,
                "target_column": target_column,
            },
            model_path,
        )
        model_size_bytes = model_path.stat().st_size
        feature_count = len(vectorizer.vocabulary_)
        np.save(output_dir / "val_predictions.npy", predictions)

        model_config = {
            **hyperparameters,
            "feature_method": (
                "tfidf_single"
                if len(feature_columns) == 1
                else "tfidf_A_minus_tfidf_B"
            ),
            "classifier": "LogisticRegression",
            "random_seed": self.random_seed,
        }
        with (output_dir / "metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    **{key: value for key, value in metrics.items() if key != "predictions"},
                    "train_samples": len(y_train),
                    "validation_samples": len(y_val),
                    "vocabulary_size": feature_count,
                    "model_size_bytes": model_size_bytes,
                },
                file,
                indent=2,
            )

        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=self.model_name,
            status="success",
            validation_accuracy=metrics["validation_accuracy"],
            training_time_seconds=training_time_seconds,
            artifact_path=str(model_path.resolve()),
            train_samples=int(len(y_train)),
            validation_samples=int(len(y_val)),
            model_config=model_config,
            feature_count=feature_count,
            model_size_bytes=model_size_bytes,
            **result_metric_fields(metrics),
        )


class TextWordTfidfAdapter(TextPairBaselineAdapter):
    adapter_name = "text_word_tfidf_logistic"
    model_name = "text_tfidf_logistic_regression"
