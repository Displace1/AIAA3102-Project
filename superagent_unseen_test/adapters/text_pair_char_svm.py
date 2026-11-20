import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from adapters.base import ExperimentResult, ModelAdapter
from adapters.text_pair_baseline import build_pair_features, load_text_pair_csv
from superagent.evaluation import evaluate_classifier, result_metric_fields


class TextPairCharSvmAdapter(ModelAdapter):
    adapter_name = "text_pair_char_tfidf_svm"
    model_name = "pairwise_char_tfidf_linear_svm"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def run(self, experiment_id, config, data_profile, output_dir, hyperparameters):
        output_dir = Path(output_dir)
        columns = data_profile["feature_columns"]
        target = data_profile["target_column"]
        train_a, train_b, y_train = load_text_pair_csv(
            config.data["train"], columns, target
        )
        val_a, val_b, y_val = load_text_pair_csv(
            config.data["validation"], columns, target
        )
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(hyperparameters["ngram_min"], hyperparameters["ngram_max"]),
            min_df=hyperparameters["min_df"],
            max_features=hyperparameters["max_features"],
            sublinear_tf=hyperparameters["sublinear_tf"],
        )
        vectorizer.fit(train_a + train_b)
        X_train = build_pair_features(vectorizer, train_a, train_b)
        X_val = build_pair_features(vectorizer, val_a, val_b)
        classifier = LinearSVC(
            C=hyperparameters["C"],
            max_iter=hyperparameters["max_iter"],
            random_state=self.random_seed,
        )
        classifier.fit(X_train, y_train)
        metrics = evaluate_classifier(classifier, X_train, y_train, X_val, y_val)
        predictions = metrics.pop("predictions").astype(int)

        model_path = output_dir / "model.joblib"
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "feature_method": "char_tfidf_A_minus_char_tfidf_B",
                "feature_columns": columns,
                "target_column": target,
            },
            model_path,
        )
        np.save(output_dir / "val_predictions.npy", predictions)
        feature_count = len(vectorizer.vocabulary_)
        model_size_bytes = model_path.stat().st_size
        model_config = {
            **hyperparameters,
            "analyzer": "char_wb",
            "classifier": "LinearSVC",
            "random_seed": self.random_seed,
        }
        payload = {
            **metrics,
            "train_samples": len(y_train),
            "validation_samples": len(y_val),
            "vocabulary_size": feature_count,
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
