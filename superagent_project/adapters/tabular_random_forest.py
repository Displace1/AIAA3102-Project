import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from adapters.base import ExperimentResult, ModelAdapter
from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.tabular_data import (
    build_tabular_preprocessor,
    load_tabular_split,
    package_tabular_model,
    transformed_feature_count,
)


class TabularRandomForestAdapter(ModelAdapter):
    adapter_name = "tabular_random_forest"
    model_name = "one_hot_random_forest"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def run(self, experiment_id, config, data_profile, output_dir, hyperparameters):
        output_dir = Path(output_dir)
        X_train, y_train = load_tabular_split(config, data_profile, "train")
        X_val, y_val = load_tabular_split(config, data_profile, "validation")
        numeric_preprocessing = hyperparameters.get(
            "numeric_preprocessing", "none"
        )
        categorical_preprocessing = hyperparameters.get(
            "categorical_preprocessing", "onehot"
        )
        preprocessor = build_tabular_preprocessor(
            data_profile,
            numeric_preprocessing=numeric_preprocessing,
            categorical_preprocessing=categorical_preprocessing,
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=hyperparameters["n_estimators"],
                        max_depth=hyperparameters["max_depth"],
                        max_features=hyperparameters["max_features"],
                        min_samples_leaf=hyperparameters["min_samples_leaf"],
                        class_weight=hyperparameters["class_weight"],
                        random_state=self.random_seed,
                        n_jobs=config.maximum_cpu_threads,
                    ),
                ),
            ]
        )
        training_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time_seconds = time.perf_counter() - training_started
        metrics = evaluate_classifier(pipeline, X_train, y_train, X_val, y_val)
        predictions = metrics.pop("predictions").astype(int)

        model_path = output_dir / "pipeline.joblib"
        joblib.dump(
            package_tabular_model(
                pipeline, data_profile, config.data["format"]
            ),
            model_path,
        )
        np.save(output_dir / "val_predictions.npy", predictions)
        feature_count = transformed_feature_count(
            pipeline.named_steps["preprocessor"], X_train.shape[1]
        )
        model_size_bytes = model_path.stat().st_size
        model_config = {
            **hyperparameters,
            "numeric_preprocessing": numeric_preprocessing,
            "categorical_preprocessing": categorical_preprocessing,
            "preprocessing": "type_aware_column_transformer_sparse",
            "classifier": "RandomForestClassifier",
            "random_seed": self.random_seed,
            "n_jobs": config.maximum_cpu_threads,
        }
        payload = {
            **metrics,
            "train_samples": len(y_train),
            "validation_samples": len(y_val),
            "feature_count": feature_count,
            "model_size_bytes": model_size_bytes,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

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
