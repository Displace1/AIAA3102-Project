import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from adapters.tabular_baseline import normalize_categorical_array
from adapters.text_pair_baseline import build_text_features
from superagent.image_artifact import (
    build_model_from_artifact,
    load_image_artifact,
)
from superagent.image_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_image_inference_loader,
)
from superagent.image_training import select_training_device
from superagent.metrics import classifier_score_bundle
from superagent.tabular_data import load_tabular_csv
from superagent.threshold import apply_threshold, binary_positive_probabilities


TABULAR_ADAPTERS = {
    "tabular_one_hot_logistic",
    "tabular_extra_trees",
    "tabular_random_forest",
    "tabular_hist_gradient_boosting",
}

TEXT_PAIR_ADAPTERS = {
    "text_pair_word_tfidf_logistic",
    "text_pair_char_tfidf_logistic",
    "text_pair_char_tfidf_svm",
    "text_word_tfidf_logistic",
    "text_char_tfidf_logistic",
    "text_char_tfidf_svm",
}

IMAGE_ADAPTERS = {
    "image_small_cnn",
    "image_resnet18",
    "image_mobilenet_v3_small",
}


class PredictionError(RuntimeError):
    pass


def locked_threshold(artifact_path):
    """Return the decision threshold stored beside an exported artifact.

    The scalar lives in ``final_solution.json`` in the same directory as the
    model, so the artifact file format itself is unchanged.  A missing file or a
    run that never tuned a threshold both yield ``None``, meaning "use the
    model's own argmax decision", which is exactly the old behaviour.
    """
    artifact_path = Path(artifact_path)
    solution_path = artifact_path.parent / "final_solution.json"
    if not solution_path.is_file():
        return None
    try:
        with solution_path.open(encoding="utf-8") as handle:
            solution = json.load(handle)
    except (OSError, ValueError):
        return None
    rule = solution.get("decision_rule")
    if not isinstance(rule, dict) or not rule.get("applied"):
        return None
    threshold = rule.get("threshold")
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return None
    if not 0.0 < value < 1.0:
        return None
    return value


def apply_locked_threshold(predictions, probabilities, threshold):
    """Override argmax labels with the locked threshold, when usable."""
    if threshold is None:
        return predictions
    positive = binary_positive_probabilities(probabilities)
    if positive is None or len(positive) != len(predictions):
        return predictions
    return apply_threshold(positive, threshold)


@dataclass(frozen=True)
class PredictionBatch:
    predictions: np.ndarray
    input_paths: Optional[list] = None
    class_names: Optional[list] = None
    # Probability-like scores so the final evaluation can report roc_auc;
    # ``None`` when the locked model exposes no probabilities.
    scores: Optional[np.ndarray] = None
    # Strict probabilities (``predict_proba`` / softmax) so the final
    # evaluation can report log_loss.  Kept apart from ``scores`` because a
    # binary ``scores`` column may be a decision_function margin.
    probabilities: Optional[np.ndarray] = None


class ArtifactPredictor:
    """Load one locked artifact and produce test predictions."""

    def predict(self, artifact_path, adapter_name, config):
        return self.predict_batch(artifact_path, adapter_name, config).predictions

    def predict_batch(self, artifact_path, adapter_name, config):
        """Same as :meth:`predict` but keeps the scores for score-based metrics."""
        return self.predict_input_with_metadata(
            artifact_path=artifact_path,
            adapter_name=adapter_name,
            input_path=self._input_path(adapter_name, config),
        )

    @staticmethod
    def _input_path(adapter_name, config):
        if adapter_name in TABULAR_ADAPTERS:
            return (
                config.data["test_features"]
                if config.data["format"] == "npy_split"
                else config.data["test"]
            )
        if adapter_name in TEXT_PAIR_ADAPTERS:
            return config.data["test"]
        if adapter_name in IMAGE_ADAPTERS:
            return config.data["test_directory"]
        raise PredictionError(
            f"No final prediction implementation for adapter: {adapter_name}"
        )

    def predict_input(self, artifact_path, adapter_name, input_path):
        return self.predict_input_with_metadata(
            artifact_path=artifact_path,
            adapter_name=adapter_name,
            input_path=input_path,
        ).predictions

    def predict_input_with_metadata(
        self, artifact_path, adapter_name, input_path
    ):
        artifact_path = Path(artifact_path).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Selected model artifact does not exist: {artifact_path}"
            )
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Prediction input does not exist: {input_path}")

        if adapter_name in TABULAR_ADAPTERS:
            self._require_file(input_path, "Tabular")
            return self._predict_tabular(artifact_path, input_path)
        if adapter_name in TEXT_PAIR_ADAPTERS:
            self._require_file(input_path, "Text-pair")
            return self._predict_text_pair(artifact_path, input_path)
        if adapter_name in IMAGE_ADAPTERS:
            return self._predict_image(artifact_path, input_path)
        raise PredictionError(
            f"No final prediction implementation for adapter: {adapter_name}"
        )

    @staticmethod
    def _predict_tabular(artifact_path, input_path):
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict) and artifact.get(
            "artifact_type"
        ) == "tabular_classifier":
            model = artifact.get("model")
            feature_columns = artifact.get("feature_columns")
            if not hasattr(model, "predict") or not isinstance(
                feature_columns, list
            ):
                raise PredictionError("Invalid tabular model package.")
            if input_path.suffix.lower() != ".csv":
                raise PredictionError(
                    "This tabular model expects an unlabeled CSV input."
                )
            features = load_tabular_csv(
                input_path,
                feature_columns=feature_columns,
                require_target=False,
            )
            scores, probabilities = classifier_score_bundle(model, features)
            threshold = locked_threshold(artifact_path)
            labels = np.asarray(model.predict(features), dtype=int)
            return PredictionBatch(
                predictions=np.asarray(
                    apply_locked_threshold(labels, probabilities, threshold),
                    dtype=int,
                ),
                scores=scores,
                probabilities=probabilities,
            )

        model = artifact
        if not hasattr(model, "predict"):
            raise PredictionError(
                "Tabular artifact does not expose predict()."
            )
        features = np.load(
            input_path, allow_pickle=True
        )
        if not getattr(model, "superagent_type_aware", False):
            features = normalize_categorical_array(features)
        scores, probabilities = classifier_score_bundle(model, features)
        threshold = locked_threshold(artifact_path)
        labels = np.asarray(model.predict(features), dtype=int)
        return PredictionBatch(
            predictions=np.asarray(
                apply_locked_threshold(labels, probabilities, threshold),
                dtype=int,
            ),
            scores=scores,
            probabilities=probabilities,
        )

    @staticmethod
    def _predict_text_pair(artifact_path, input_path):
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict):
            raise PredictionError("Text artifact must be a dictionary.")
        required = {"vectorizer", "classifier", "feature_columns"}
        missing = required - set(artifact)
        if missing:
            raise PredictionError(
                f"Text artifact is missing fields: {sorted(missing)}"
            )

        feature_columns = artifact["feature_columns"]
        if not isinstance(feature_columns, list) or len(feature_columns) not in {
            1,
            2,
        }:
            raise PredictionError(
                "Text artifact requires one or two feature columns."
            )
        text_columns = [[] for _ in feature_columns]
        with open(
            input_path, "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise PredictionError("Test CSV has no header.")
            for column in feature_columns:
                if column not in reader.fieldnames:
                    raise PredictionError(
                        f"Test CSV is missing feature column: {column}"
                    )
            for row in reader:
                for index, column in enumerate(feature_columns):
                    text_columns[index].append(row[column].strip())

        features = build_text_features(artifact["vectorizer"], text_columns)
        classifier = artifact["classifier"]
        scores, probabilities = classifier_score_bundle(classifier, features)
        threshold = locked_threshold(artifact_path)
        labels = np.asarray(classifier.predict(features), dtype=int)
        return PredictionBatch(
            predictions=np.asarray(
                apply_locked_threshold(labels, probabilities, threshold),
                dtype=int,
            ),
            scores=scores,
            probabilities=probabilities,
        )

    @staticmethod
    def _predict_image(artifact_path, input_path):
        package = load_image_artifact(artifact_path)
        preprocessing = package["preprocessing"]
        expected_mean = list(IMAGENET_MEAN)
        expected_std = list(IMAGENET_STD)
        if preprocessing.get("color_mode") != "RGB":
            raise PredictionError("Only RGB image artifacts are supported.")
        if preprocessing.get("normalization_mean") != expected_mean or (
            preprocessing.get("normalization_std") != expected_std
        ):
            raise PredictionError(
                "Image artifact uses unsupported normalization settings."
            )

        image_size = int(preprocessing["image_size"])
        configured_batch_size = int(
            package["model_config"].get("batch_size", 32)
        )
        loader = build_image_inference_loader(
            input_path=input_path,
            image_size=image_size,
            batch_size=max(1, min(configured_batch_size, 64)),
            num_workers=0,
        )
        device = select_training_device("auto")
        model = build_model_from_artifact(package).to(device)
        predictions = []
        input_paths = []
        scores = []
        with torch.inference_mode():
            for images, batch_paths in loader:
                logits = model(images.to(device, non_blocking=True))
                predictions.extend(
                    logits.argmax(dim=1).detach().cpu().tolist()
                )
                scores.append(
                    torch.softmax(logits, dim=1).detach().cpu().numpy()
                )
                input_paths.extend(str(path) for path in batch_paths)

        index_to_class = package["model_config"].get("index_to_class")
        if not isinstance(index_to_class, dict):
            raise PredictionError(
                "Image artifact is missing index_to_class mapping."
            )
        class_names = []
        for prediction in predictions:
            class_name = index_to_class.get(prediction)
            if class_name is None:
                class_name = index_to_class.get(str(prediction))
            if class_name is None:
                raise PredictionError(
                    "Image artifact has no class name for prediction index "
                    f"{prediction}."
                )
            class_names.append(str(class_name))
        score_matrix = np.vstack(scores) if scores else None
        return PredictionBatch(
            predictions=np.asarray(predictions, dtype=int),
            input_paths=input_paths,
            class_names=class_names,
            scores=score_matrix,
            # The image head always ends in softmax, so the same matrix is a
            # genuine probability distribution.
            probabilities=score_matrix,
        )

    @staticmethod
    def _require_file(path, modality_name):
        if not path.is_file():
            raise PredictionError(
                f"{modality_name} prediction input must be a file: {path}"
            )
