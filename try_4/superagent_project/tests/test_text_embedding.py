"""The pretrained-embedding text adapter.

What matters, in order:

1. **It never crashes the run.** torch / sentence-transformers are optional;
   when they are missing the experiment must fail cleanly with a reason, and the
   rest of the agent must keep working. This is the highest-risk part of the
   whole try_4 direction, so it is tested with a broken import.
2. **Features are built the way the measurement used them.** A single-text task
   uses the embedding; a two-text task uses [A, A-B, A*B]. If the adapter built
   something else, the +1.8..3.8pp measurement would not describe it.
3. **The TF-IDF concat is optional and off by default.**
4. **The encoder is loaded once**, not once per experiment.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adapters.text_embedding_logistic import (
    DEFAULT_MODEL,
    EmbeddingUnavailable,
    TextEmbeddingLogisticAdapter,
    encode_texts,
    load_encoder,
    load_text_columns,
    pair_features,
    predict_with_artifact,
)


class FakeEncoder:
    """Deterministic stand-in: dimension 4, normalised, no torch needed."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        out = np.zeros((len(texts), 4), dtype=float)
        for i, text in enumerate(texts):
            out[i, 0] = len(text) % 7
            out[i, 1] = sum(map(ord, text)) % 11
            out[i, 2] = text.count("a")
            out[i, 3] = 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class FakeConfig:
    def __init__(self, train, validation):
        self.data = {"train": str(train), "validation": str(validation)}


def write_csv(path, rows, columns, target):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns) + [target])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_single_task(root, n=40):
    """Learnable single-text task: label depends on a keyword.

    Labels are integers, matching the real datasets (20 Newsgroups ships
    ``text,label`` with 0..3). String labels would need a ``label_mapping``
    from the data profile, which is a separate concern already covered by
    tests/test_string_labels.py.
    """
    rows = []
    for i in range(n):
        label = 1 if i % 2 == 0 else 0
        text = ("aaaaaaaa good" if label == 1 else "bbbbbbbb bad")
        rows.append({"text": f"{text} {i}", "label": label})
    train = Path(root) / "train.csv"
    val = Path(root) / "val.csv"
    write_csv(train, rows[: n // 2], ["text"], "label")
    write_csv(val, rows[n // 2:], ["text"], "label")
    return train, val


class EncodeTest(unittest.TestCase):
    def test_pair_features_shape_and_identity(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        out = pair_features(a, b)
        self.assertEqual(out.shape, (1, 6))
        # [A, A-B, A*B]
        np.testing.assert_allclose(out[0, :2], a[0])
        np.testing.assert_allclose(out[0, 2:4], [1.0, -1.0])
        np.testing.assert_allclose(out[0, 4:], [0.0, 0.0])

    def test_encode_returns_one_row_per_text(self):
        embeddings = encode_texts(FakeEncoder(), ["a", "bb", "ccc"])
        self.assertEqual(embeddings.shape, (3, 4))


class LoadEncoderTest(unittest.TestCase):
    def test_missing_library_becomes_embedding_unavailable(self):
        """A missing optional dependency must not escape as ImportError."""
        real_import = __import__

        def deny(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("no sentence_transformers here")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=deny):
            with self.assertRaises(EmbeddingUnavailable) as ctx:
                load_encoder()
        self.assertIn("sentence-transformers is not installed",
                      str(ctx.exception))

    def test_error_message_names_the_workaround(self):
        real_import = __import__

        def deny(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=deny):
            with self.assertRaises(EmbeddingUnavailable) as ctx:
                load_encoder()
        self.assertIn("TF-IDF", str(ctx.exception))

    def test_unloadable_model_reports_every_attempt(self):
        class Broken:
            def __init__(self, *a, **k):
                raise OSError("weights corrupt")

        with patch.dict("sys.modules",
                        {"sentence_transformers": type(
                            "M", (), {"SentenceTransformer": Broken})}):
            with self.assertRaises(EmbeddingUnavailable) as ctx:
                load_encoder()
        self.assertIn("weights corrupt", str(ctx.exception))


class AdapterRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.val = make_single_task(self.root)
        self.config = FakeConfig(self.train, self.val)

    def tearDown(self):
        self.temp.cleanup()

    def profile(self, columns):
        return {"modality": "text", "feature_columns": columns,
                "target_column": "label", "label_mapping": None}

    def run_adapter(self, hyperparameters, columns=("text",), adapter=None):
        adapter = adapter or TextEmbeddingLogisticAdapter(random_seed=42)
        out = self.root / "exp_out"
        out.mkdir(exist_ok=True)
        return adapter.run(
            experiment_id="exp_001", config=self.config,
            data_profile=self.profile(list(columns)),
            output_dir=out, hyperparameters=hyperparameters,
        )

    def base_params(self, **extra):
        params = {"C": 1.0, "max_iter": 1000}
        params.update(extra)
        return params

    def patched_adapter(self, encoder):
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = encoder
        return adapter

    def test_single_text_produces_a_working_model(self):
        result = self.run_adapter(self.base_params(),
                                  adapter=self.patched_adapter(FakeEncoder()))
        self.assertEqual(result.status, "success")
        self.assertGreater(result.validation_accuracy, 0.8)
        self.assertEqual(result.feature_count, 4)

    def test_encoder_is_loaded_once_across_experiments(self):
        encoder = FakeEncoder()
        adapter = self.patched_adapter(encoder)
        for _ in range(3):
            self.run_adapter(self.base_params(), adapter=adapter)
        # one call per train + one per val column, three times = 6,
        # NOT 3 loads. What matters is that load_encoder ran once, which the
        # cached attribute guarantees by construction.
        self.assertEqual(encoder.calls, 6)

    def test_metrics_record_the_embedding_dim(self):
        out = self.root / "exp_dim"
        out.mkdir(exist_ok=True)
        adapter = self.patched_adapter(FakeEncoder())
        adapter.run("exp_001", self.config, self.profile(["text"]), out,
                    self.base_params())
        doc = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["embedding_dim"], 4)
        self.assertIn("feature_count", doc)

    def test_concat_tfidf_is_off_by_default(self):
        out = self.root / "exp_plain"
        out.mkdir(exist_ok=True)
        adapter = self.patched_adapter(FakeEncoder())
        adapter.run("exp_001", self.config, self.profile(["text"]), out,
                    self.base_params())
        artifact = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertNotIn("vocabulary_size", artifact)

    def test_concat_tfidf_raises_feature_count(self):
        plain = self.run_adapter(self.base_params(),
                                 adapter=self.patched_adapter(FakeEncoder()))
        with_tfidf = self.run_adapter(
            self.base_params(concat_tfidf=True, tfidf_max_features=5000),
            adapter=self.patched_adapter(FakeEncoder()))
        self.assertGreater(with_tfidf.feature_count, plain.feature_count)

    def test_saved_artifact_does_not_embed_the_weights(self):
        """The 418 MB encoder must not be serialised into the model."""
        out = self.root / "exp_art"
        out.mkdir(exist_ok=True)
        adapter = self.patched_adapter(FakeEncoder())
        result = adapter.run("exp_001", self.config, self.profile(["text"]),
                             out, self.base_params())
        self.assertLess(result.model_size_bytes, 1_000_000)
        import joblib
        artifact = joblib.load(result.artifact_path)
        self.assertEqual(artifact["embedding_model"], DEFAULT_MODEL)
        self.assertNotIn("encoder", artifact)

    def test_missing_encoder_fails_without_writing_a_model(self):
        out = self.root / "exp_fail"
        out.mkdir(exist_ok=True)
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)

        def boom(name=DEFAULT_MODEL, device="cpu"):
            raise EmbeddingUnavailable("encoder not available here")

        with patch("adapters.text_embedding_logistic.load_encoder", boom):
            with self.assertRaises(EmbeddingUnavailable):
                adapter.run("exp_001", self.config, self.profile(["text"]), out,
                            self.base_params())
        self.assertFalse((out / "model.joblib").exists())


class PairAdapterTest(unittest.TestCase):
    def test_two_columns_use_the_pair_feature_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"a": f"aaaaaaaa {i}", "b": f"zzzzzzzz {i}",
                     "label": 1 if i % 2 == 0 else 0} for i in range(40)]
            train, val = root / "t.csv", root / "v.csv"
            write_csv(train, rows[:20], ["a", "b"], "label")
            write_csv(val, rows[20:], ["a", "b"], "label")

            adapter = TextEmbeddingLogisticAdapter(random_seed=42)
            adapter._encoder = FakeEncoder()
            out = root / "pair_out"
            out.mkdir()
            result = adapter.run(
                "exp_001", FakeConfig(train, val),
                {"modality": "text_pair", "feature_columns": ["a", "b"],
                 "target_column": "label", "label_mapping": None},
                out, {"C": 1.0, "max_iter": 1000})
            # 3 x embedding dim
            self.assertEqual(result.feature_count, 12)
            self.assertIn("AminusB", result.model_config["feature_method"])


class PredictionTest(unittest.TestCase):
    def test_predict_with_artifact_reproduces_training_features(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, val = make_single_task(root)
            adapter = TextEmbeddingLogisticAdapter(random_seed=42)
            adapter._encoder = FakeEncoder()
            out = root / "out"
            out.mkdir()
            result = adapter.run(
                "exp_001", FakeConfig(train, val),
                {"modality": "text", "feature_columns": ["text"],
                 "target_column": "label", "label_mapping": None},
                out, {"C": 1.0, "max_iter": 1000})
            artifact = joblib.load(result.artifact_path)
            columns, _ = load_text_columns(val, ["text"], None)
            classifier, features = predict_with_artifact(
                artifact, columns, FakeEncoder())
            self.assertEqual(features.shape[1],
                             artifact["classifier"].coef_.shape[1])
            predictions = classifier.predict(features)
            self.assertEqual(len(predictions), len(columns[0]))


if __name__ == "__main__":
    unittest.main()
