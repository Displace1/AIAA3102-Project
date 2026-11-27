"""Text classification on pretrained sentence embeddings.

Why this adapter exists
-----------------------
The three shipped text adapters all use TF-IDF, which is pure surface statistics:
"car" and "automobile" are orthogonal to it. Measured on this project's own data
with an identical classifier, a frozen sentence encoder beat TF-IDF on both text
tasks and never once lost across 30 fold-runs (see try_4 results):

    20 Newsgroups   +3.72pp (sd 1.63, 15/15 folds better)
    task2 Headlines +1.77pp (sd 0.71, 14/15 better, 1 tie, 0 losses)

Design constraints
------------------
1. **One frozen encoder, no fine-tuning.** Fine-tuning would need a GPU this
   machine does not have, and would make the comparison against TF-IDF a
   comparison of training budgets rather than of representations.
2. **The encoder is loaded once and cached on the adapter instance.** Loading
   costs ~7s; re-loading per experiment would dominate small tasks.
3. **Offline by default.** ``local_files_only=True`` with a model *id* does not
   actually stop huggingface_hub from probing the network -- measured at 150s of
   retries against a model already on disk. Pinning the local snapshot path, or
   setting HF_HUB_OFFLINE, brings that to 7s. Both are handled here.
4. **Graceful degradation.** torch / sentence-transformers are optional
   dependencies. When they are missing the experiment fails with a clear reason
   and the rest of the run continues; it must not crash the agent.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from adapters.base import ExperimentResult, ModelAdapter
from adapters import embedding_cache
from superagent.evaluation import evaluate_classifier, result_metric_fields
from superagent.label_encoding import encode_labels

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CACHE = (Path(os.path.expanduser("~")) / ".cache" / "huggingface"
                 / "hub" / "models--sentence-transformers--all-mpnet-base-v2"
                 / "snapshots")


class EmbeddingUnavailable(RuntimeError):
    """Raised when the encoder cannot be loaded. Always a clean, reported failure."""


def _snapshot_paths(model_name=DEFAULT_MODEL):
    """Candidate local directories for ``model_name``.

    A model id is tried last: it works, but only after a 150s network probe
    unless HF_HUB_OFFLINE is set. Preferring a directory keeps runs fast and
    keeps an offline machine from burning retries on every experiment.
    """
    slug = "models--" + model_name.replace("/", "--")
    root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    found = []
    if root.is_dir():
        for candidate in sorted(root.iterdir()):
            if (candidate / "model.safetensors").is_file() or (
                    candidate / "pytorch_model.bin").is_file():
                found.append(str(candidate))
    return found


def load_encoder(model_name=DEFAULT_MODEL, device="cpu"):
    """Load a frozen sentence encoder, offline when possible.

    Import errors are converted to ``EmbeddingUnavailable`` so the caller can
    report "this adapter is unusable here" instead of a traceback.
    """
    # These stop huggingface_hub from probing the network for optional files.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as error:                          # noqa: BLE001
        raise EmbeddingUnavailable(
            "sentence-transformers is not installed "
            f"({type(error).__name__}: {error}). "
            "Install it, or use a TF-IDF text adapter instead."
        ) from error

    candidates = _snapshot_paths(model_name) + [model_name]
    last_error = None
    for candidate in candidates:
        try:
            return SentenceTransformer(candidate, device=device)
        except Exception as error:                      # noqa: BLE001
            last_error = error
            continue
    raise EmbeddingUnavailable(
        f"Could not load {model_name} from any local snapshot or the hub: "
        f"{type(last_error).__name__}: {last_error}"
    )


def encode_texts(encoder, texts, batch_size=32):
    """L2-normalised embeddings, so cosine similarity is a dot product."""
    return encoder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def pair_features(first, second):
    """Features for a two-text task: [A, A-B, A*B].

    A-B encodes the difference (useful for "which headline is better"), A*B the
    interaction, and A the absolute position. Measured best of the options tried
    in try_4/analysis/checks/embedding_vs_tfidf.py.
    """
    return np.hstack([first, first - second, first * second])


def load_text_columns(path, feature_columns, target_column, label_mapping=None):
    columns = [[] for _ in feature_columns]
    labels = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for column in feature_columns:
            if column not in reader.fieldnames:
                raise ValueError(f"Missing feature column {column!r} in {path}")
        for row in reader:
            for index, column in enumerate(feature_columns):
                columns[index].append((row.get(column) or "").strip())
            if target_column:
                labels.append(row[target_column])
    y = encode_labels(labels, label_mapping) if target_column else None
    return columns, y


class TextEmbeddingLogisticAdapter(ModelAdapter):
    adapter_name = "text_embedding_logistic"
    model_name = "text_pretrained_embedding_logistic"

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self._encoder = None

    def encoder(self, model_name):
        # Cached across experiments: loading is ~7s and the encoder never
        # changes, so a per-experiment reload is pure waste.
        if self._encoder is None:
            self._encoder = load_encoder(model_name)
        return self._encoder

    def embedded_columns(self, encoder, columns, paths, model_name, cache_dir):
        """Embed each text column, reusing the on-disk cache when possible.

        Returns ``(embeddings, cache_info)``. The cache is best-effort by
        design: any failure (unwritable directory, corrupt file, changed data)
        falls through to encoding, because an optimisation must never be able to
        break an experiment.
        """
        info = {"enabled": bool(cache_dir), "columns": []}
        if not cache_dir:
            return [encode_texts(encoder, column) for column in columns], info

        # The dimension belongs in the cache key, but a wrapper or test double
        # may not advertise it. Encode one row to learn the width, then look up
        # the cache with that width. Probing one row costs nothing; encoding the
        # whole column first would defeat the cache entirely.
        dimension = None
        if hasattr(encoder, "get_sentence_embedding_dimension"):
            dimension = int(encoder.get_sentence_embedding_dimension())
        if dimension is None:
            dimension = int(encode_texts(encoder, columns[0][:1]).shape[1])
        info["dimension"] = dimension

        embeddings = []
        for index, column in enumerate(columns):
            key = embedding_cache.cache_key(
                paths, [f"col{index}"], model_name, dimension)
            with embedding_cache.TimedLookup() as timer:
                cached = embedding_cache.load(
                    cache_dir, key, f"col{index}", expected_rows=len(column))
            if cached is not None:
                timer.hit = True
                timer.rows = int(cached.shape[0])
                embeddings.append(cached)
                info["columns"].append({"index": index, **timer.as_dict(),
                                        "key": key})
                continue
            column_embeddings = encode_texts(encoder, column)
            try:
                embedding_cache.store(cache_dir, key, f"col{index}",
                                      column_embeddings)
            except OSError:
                pass          # read-only run directory: encode, do not fail
            timer.rows = int(column_embeddings.shape[0])
            embeddings.append(column_embeddings)
            info["columns"].append({"index": index, **timer.as_dict(),
                                    "key": key})
        return embeddings, info

    def run(self, experiment_id, config, data_profile, output_dir,
            hyperparameters):
        output_dir = Path(output_dir)
        feature_columns = data_profile["feature_columns"]
        target_column = data_profile["target_column"]
        label_mapping = data_profile.get("label_mapping")

        train_columns, y_train = load_text_columns(
            config.data["train"], feature_columns, target_column, label_mapping)
        val_columns, y_val = load_text_columns(
            config.data["validation"], feature_columns, target_column,
            label_mapping)

        model_name = hyperparameters.get("embedding_model", DEFAULT_MODEL)
        training_started = time.perf_counter()
        encoder = self.encoder(model_name)

        cache_dir = hyperparameters.get("embedding_cache_dir")
        train_emb, train_cache = self.embedded_columns(
            encoder, train_columns, [config.data["train"]], model_name,
            cache_dir)
        val_emb, val_cache = self.embedded_columns(
            encoder, val_columns, [config.data["validation"]], model_name,
            cache_dir)
        if len(train_columns) == 2:
            X_train = pair_features(*train_emb)
            X_val = pair_features(*val_emb)
        else:
            X_train, X_val = train_emb[0], val_emb[0]

        vectorizer = None
        if hyperparameters.get("concat_tfidf", False):
            # TF-IDF on top of the embeddings. Measured to add ~0.7-1.0pp.
            vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
                max_features=hyperparameters.get("tfidf_max_features", 50000),
                sublinear_tf=True)
            vectorizer.fit([t for column in train_columns for t in column])
            tf_train = [vectorizer.transform(c) for c in train_columns]
            tf_val = [vectorizer.transform(c) for c in val_columns]
            if len(train_columns) == 2:
                tf_train = [tf_train[0] - tf_train[1]]
                tf_val = [tf_val[0] - tf_val[1]]
            X_train = np.hstack([X_train, tf_train[0].toarray()])
            X_val = np.hstack([X_val, tf_val[0].toarray()])

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
                "classifier": classifier,
                "vectorizer": vectorizer,
                "embedding_model": model_name,
                "feature_method": (
                    "embedding_single" if len(feature_columns) == 1
                    else "embedding_A_AminusB_AxB"),
                "concat_tfidf": bool(hyperparameters.get("concat_tfidf", False)),
                "feature_columns": feature_columns,
                "target_column": target_column,
            },
            model_path,
        )
        model_size_bytes = model_path.stat().st_size
        np.save(output_dir / "val_predictions.npy", predictions)

        feature_count = int(X_train.shape[1])
        model_config = {
            "C": hyperparameters["C"],
            "max_iter": hyperparameters["max_iter"],
            "embedding_model": model_name,
            "concat_tfidf": bool(hyperparameters.get("concat_tfidf", False)),
            "feature_method": (
                "embedding_single" if len(feature_columns) == 1
                else "embedding_A_AminusB_AxB"),
            "classifier": "LogisticRegression",
            "random_seed": self.random_seed,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    **{k: v for k, v in metrics.items() if k != "predictions"},
                    "train_samples": len(y_train),
                    "validation_samples": len(y_val),
                    "feature_count": feature_count,
                    "embedding_dim": int(train_emb[0].shape[1]),
                    "model_size_bytes": model_size_bytes,
                    # Recorded so a reviewer can tell whether a fast experiment
                    # was fast because of the cache or because it did less work.
                    "embedding_cache": {"train": train_cache, "validation": val_cache},
                },
                file, indent=2)

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


def predict_with_artifact(artifact, text_columns, encoder):
    """Score already-loaded text columns with a saved artifact.

    Kept here (not in predictor.py) so the encoder knowledge lives in one place;
    predictor.py calls this for the text_embedding_* adapter names.
    """
    emb = [encode_texts(encoder, c) for c in text_columns]
    X = pair_features(*emb) if len(text_columns) == 2 else emb[0]
    if artifact.get("concat_tfidf") and artifact.get("vectorizer") is not None:
        vectorizer = artifact["vectorizer"]
        tf = [vectorizer.transform(c) for c in text_columns]
        if len(text_columns) == 2:
            tf = [tf[0] - tf[1]]
        X = np.hstack([X, tf[0].toarray()])
    return artifact["classifier"], X
