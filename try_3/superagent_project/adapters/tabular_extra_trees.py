import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from adapters.base import ExperimentResult, ModelAdapter
from adapters.cv_support import (
    attach_cross_validation,
    run_cross_validation,
)
from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.tabular_data import (
    build_tabular_preprocessor,
    load_tabular_split,
    package_tabular_model,
    transformed_feature_count,
)


class TabularExtraTreesAdapter(ModelAdapter):
    adapter_name = "tabular_extra_trees"
    model_name = "one_hot_extra_trees"

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
        X_train, y_train = load_tabular_split(config, data_profile, "train")
        X_val, y_val = load_tabular_split(config, data_profile, "validation")
        numeric_preprocessing = hyperparameters.get(
            "numeric_preprocessing", "none"
        )
        categorical_preprocessing = hyperparameters.get(
            "categorical_preprocessing", "onehot"
        )
        ordinal_preprocessing = hyperparameters.get(
            "ordinal_preprocessing", "ordinal"
        )

        classifier_settings = dict(
            n_estimators=hyperparameters["n_estimators"],
            max_features=hyperparameters["max_features"],
            min_samples_leaf=hyperparameters["min_samples_leaf"],
            class_weight=hyperparameters["class_weight"],
            random_state=self.random_seed,
            n_jobs=config.maximum_cpu_threads,
        )

        def build_fresh_pipeline():
            # A fresh preprocessor per fold: fitting it once and reusing it would
            # let fold-specific category vocabularies leak between folds.
            return Pipeline(
                steps=[
                    (
                        "preprocessor",
                        build_tabular_preprocessor(
                            data_profile,
                            numeric_preprocessing=numeric_preprocessing,
                            categorical_preprocessing=categorical_preprocessing,
                            ordinal_preprocessing=ordinal_preprocessing,
                        ),
                    ),
                    ("classifier", ExtraTreesClassifier(**classifier_settings)),
                ]
            )

        # CV runs on the TRAINING split only, before the final fit, so the
        # validation split stays untouched for selection and threshold tuning.
        cv = run_cross_validation(
            build_fresh_pipeline, X_train, y_train, hyperparameters,
            seed=self.random_seed,
        )

        pipeline = build_fresh_pipeline()
        training_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time_seconds = time.perf_counter() - training_started
        metrics = evaluate_classifier(pipeline, X_train, y_train, X_val, y_val)
        predictions = metrics["predictions"].astype(int)

        model_path = output_dir / "pipeline.joblib"
        joblib.dump(
            package_tabular_model(
                pipeline, data_profile, config.data["format"]
            ),
            model_path,
        )
        model_size_bytes = model_path.stat().st_size
        feature_count = transformed_feature_count(
            pipeline.named_steps["preprocessor"], X_train.shape[1]
        )
        np.save(output_dir / "val_predictions.npy", predictions)

        model_config = {
            **hyperparameters,
            "numeric_preprocessing": numeric_preprocessing,
            "categorical_preprocessing": categorical_preprocessing,
            "ordinal_preprocessing": ordinal_preprocessing,
            "preprocessing": "type_aware_column_transformer_sparse",
            "classifier": "ExtraTreesClassifier",
            "random_seed": self.random_seed,
            "n_jobs": config.maximum_cpu_threads,
        }
        metrics_document = {
            **{key: value for key, value in metrics.items() if key != "predictions"},
            "train_samples": len(y_train),
            "validation_samples": len(y_val),
            "feature_count": feature_count,
            "model_size_bytes": model_size_bytes,
        }
        if cv is not None:
            metrics_document["cross_validation"] = cv.to_dict()
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics_document, file, indent=2)

        result = ExperimentResult(
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
        return attach_cross_validation(result, cv)
