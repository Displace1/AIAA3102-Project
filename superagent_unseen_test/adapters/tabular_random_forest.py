import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from adapters.base import ExperimentResult, ModelAdapter
from adapters.tabular_baseline import normalize_categorical_array
from superagent.evaluation import evaluate_classifier, result_metric_fields


class TabularRandomForestAdapter(ModelAdapter):
    adapter_name = "tabular_random_forest"
    model_name = "one_hot_random_forest"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def run(self, experiment_id, config, data_profile, output_dir, hyperparameters):
        output_dir = Path(output_dir)
        X_train = normalize_categorical_array(
            np.load(config.data["train_features"], allow_pickle=True)
        )
        y_train = np.load(config.data["train_labels"], allow_pickle=True).astype(int)
        X_val = normalize_categorical_array(
            np.load(config.data["validation_features"], allow_pickle=True)
        )
        y_val = np.load(config.data["validation_labels"], allow_pickle=True).astype(int)

        pipeline = Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_train, y_train, X_val, y_val)
        predictions = metrics.pop("predictions").astype(int)

        model_path = output_dir / "pipeline.joblib"
        joblib.dump(pipeline, model_path)
        np.save(output_dir / "val_predictions.npy", predictions)
        feature_count = len(pipeline.named_steps["encoder"].get_feature_names_out())
        model_size_bytes = model_path.stat().st_size
        model_config = {
            **hyperparameters,
            "encoder": "OneHotEncoder",
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
            training_time_seconds=0.0,
            artifact_path=str(model_path.resolve()),
            train_samples=int(len(y_train)),
            validation_samples=int(len(y_val)),
            model_config=model_config,
            feature_count=feature_count,
            model_size_bytes=model_size_bytes,
            **result_metric_fields(metrics),
        )
