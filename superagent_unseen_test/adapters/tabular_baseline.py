import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from adapters.base import ExperimentResult, ModelAdapter
from superagent.evaluation import evaluate_classifier, result_metric_fields


def normalize_categorical_array(array):
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
                or str(value).strip().lower()
                in {"", "nan", "none", "?"}
            )
            result[row_index, column_index] = (
                "__MISSING__" if is_missing else str(value)
            )
    return result


class TabularBaselineAdapter(ModelAdapter):
    adapter_name = "tabular_one_hot_logistic"
    model_name = "one_hot_logistic_regression"

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

        X_train = normalize_categorical_array(
            np.load(config.data["train_features"], allow_pickle=True)
        )
        y_train = np.load(
            config.data["train_labels"], allow_pickle=True
        ).astype(int)
        X_val = normalize_categorical_array(
            np.load(
                config.data["validation_features"], allow_pickle=True
            )
        )
        y_val = np.load(
            config.data["validation_labels"], allow_pickle=True
        ).astype(int)

        pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=hyperparameters["C"],
                        max_iter=hyperparameters["max_iter"],
                        random_state=self.random_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_train, y_train, X_val, y_val)
        predictions = metrics["predictions"].astype(int)

        model_path = output_dir / "pipeline.joblib"
        joblib.dump(pipeline, model_path)
        model_size_bytes = model_path.stat().st_size
        feature_count = len(pipeline.named_steps["encoder"].get_feature_names_out())
        np.save(output_dir / "val_predictions.npy", predictions)

        model_config = {
            **hyperparameters,
            "preprocessing": "categorical_string_normalization",
            "encoder": "OneHotEncoder(handle_unknown='ignore')",
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
                    "feature_count": feature_count,
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
            training_time_seconds=0.0,
            artifact_path=str(model_path.resolve()),
            train_samples=int(len(y_train)),
            validation_samples=int(len(y_val)),
            model_config=model_config,
            feature_count=feature_count,
            model_size_bytes=model_size_bytes,
            **result_metric_fields(metrics),
        )
