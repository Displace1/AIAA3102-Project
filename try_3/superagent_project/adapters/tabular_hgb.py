"""Histogram gradient boosting adapter for tabular classification.

Why this adapter exists
-----------------------
The shipped tabular families are LogisticRegression / ExtraTrees / RandomForest.
On Covertype (100k subsample) they reach val accuracy 0.8501 / 0.8957, while a
histogram gradient booster reaches 0.9101 -- see analysis/hgb_prechecks.md.

The bigger motivation is memory. ``ExtraTrees(400, leaf=2)`` on the full 407k-row
Covertype training split extrapolates to ~5.6 GB, which exceeds the worker's
4 GB limit, so the task cannot be run at full size with the existing zoo at all.
HGB bins each feature into at most 255 quantiles once and grows shallow trees
over those bins, so model size scales with bin count x tree count rather than
with row count. On the same full split it fits in a few minutes and a few MB.

Two deliberate departures from the other tabular adapters
---------------------------------------------------------
* Its own preprocessor: raw numeric values, dense one-hot categories. Reusing
  ``build_tabular_preprocessor`` with ``numeric_preprocessing="standard"`` costs
  ~10pp of accuracy, because HGB is invariant to monotone transforms and
  standardising binary columns can merge their two bin edges.
* Early stopping splits a fraction off the TRAINING data. The validation split
  given by the task is never used for stopping, so no information leaks into
  selection.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from adapters.base import ExperimentResult, ModelAdapter
from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.tabular_data import (
    build_hgb_preprocessor,
    load_tabular_split,
    package_tabular_model,
    to_dense_float,
    transformed_feature_count,
)


class DenseTransform:
    """Pipeline step that densifies the sparse output of the preprocessor.

    Placed between preprocessing and the classifier so the fitted artifact
    remains a plain sklearn Pipeline, unpicklable with only sklearn imported.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return to_dense_float(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


class TabularHGBAdapter(ModelAdapter):
    adapter_name = "tabular_hist_gradient_boosting"
    model_name = "one_hot_hist_gradient_boosting"

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

        learning_rate = float(hyperparameters.get("learning_rate", 0.05))
        max_leaf_nodes = int(hyperparameters.get("max_leaf_nodes", 63))
        min_samples_leaf = int(hyperparameters.get("min_samples_leaf", 10))
        l2_regularization = float(
            hyperparameters.get("l2_regularization", 1.0)
        )
        max_iter = int(hyperparameters.get("max_iter", 600))
        early_stopping = bool(hyperparameters.get("early_stopping", True))
        class_weight = hyperparameters.get("class_weight")

        classifier_kwargs = dict(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=self.random_seed,
        )
        if early_stopping:
            # Carved from the TRAINING split, never from the task's validation
            # split, so early-stop decisions stay inside training.
            classifier_kwargs["validation_fraction"] = float(
                hyperparameters.get("validation_fraction", 0.1)
            )
            classifier_kwargs["n_iter_no_change"] = int(
                hyperparameters.get("n_iter_no_change", 30)
            )
        if class_weight is not None:
            # Older scikit-learn releases reject this argument outright; fail
            # loudly instead of silently training an unweighted model.
            try:
                HistGradientBoostingClassifier(class_weight=class_weight)
            except TypeError as error:
                raise ValueError(
                    "This scikit-learn version does not support class_weight "
                    "on HistGradientBoostingClassifier; omit the parameter."
                ) from error
            classifier_kwargs["class_weight"] = class_weight

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_hgb_preprocessor(data_profile)),
                ("densify", DenseTransform()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(**classifier_kwargs),
                ),
            ]
        )

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

        fitted_classifier = pipeline.named_steps["classifier"]
        n_iter = int(getattr(fitted_classifier, "n_iter_", -1))
        model_config = {
            **hyperparameters,
            "preprocessing": "hgb_dense_raw_numeric_onehot_categorical",
            "classifier": "HistGradientBoostingClassifier",
            "random_seed": self.random_seed,
            "n_iter_": n_iter,
            "early_stopping_source": (
                "training_split" if early_stopping else "disabled"
            ),
        }
        with (output_dir / "metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "predictions"
                    },
                    "train_samples": len(y_train),
                    "validation_samples": len(y_val),
                    "feature_count": feature_count,
                    "model_size_bytes": model_size_bytes,
                    "n_iter_": n_iter,
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
