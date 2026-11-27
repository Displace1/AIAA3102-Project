"""Can an embedding model actually be used to predict?

This exists because it could NOT, and no test caught it.

The failure: `ArtifactPredictor._input_path` matched tabular, text-pair and image
adapters, but not the new embedding adapters. Training worked, evaluation during
the run worked, the model locked -- and then final evaluation died with
"No final prediction implementation for adapter: text_pair_embedding_logistic".

That is the worst shape of bug in this project: everything looks green until the
one step that produces the deliverable. A model that cannot predict is not a
model.

So these tests check the whole route from a trained artifact to a prediction
batch, for both text adapters, including the pair layout.
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters.text_embedding_logistic import TextEmbeddingLogisticAdapter
from superagent.predictor import (
    TEXT_EMBEDDING_ADAPTERS,
    ArtifactPredictor,
    PredictionError,
)
from tests.test_text_embedding import (
    FakeConfig,
    FakeEncoder,
    make_single_task,
    write_csv,
)


class FakeFinalConfig:
    """Mimics the task config the final evaluator passes down."""

    def __init__(self, test_path):
        self.data = {"format": "csv_split", "test": str(test_path)}


class InputPathTest(unittest.TestCase):
    def test_embedding_adapters_are_routed_to_the_test_csv(self):
        for name in ("text_embedding_logistic",
                     "text_pair_embedding_logistic"):
            path = ArtifactPredictor._input_path(
                name, FakeFinalConfig("/tmp/some_test.csv"))
            self.assertEqual(path, "/tmp/some_test.csv")

    def test_unknown_adapter_still_raises(self):
        with self.assertRaises(PredictionError):
            ArtifactPredictor._input_path("nope_not_real",
                                  FakeFinalConfig("/tmp/x.csv"))

    def test_embedding_set_is_disjoint_from_the_tfidf_set(self):
        from superagent.predictor import TEXT_PAIR_ADAPTERS
        self.assertFalse(TEXT_EMBEDDING_ADAPTERS & TEXT_PAIR_ADAPTERS)


class EndToEndPredictionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def train_and_predict(self, columns, label_col, profile, rows):
        train, val = self.root / "tr.csv", self.root / "va.csv"
        write_csv(train, rows[: len(rows) // 2], columns, label_col)
        write_csv(val, rows[len(rows) // 2:], columns, label_col)

        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = FakeEncoder()
        out = self.root / "run"
        out.mkdir()
        result = adapter.run(
            "exp_001", FakeConfig(train, val), profile, out,
            {"C": 1.0, "max_iter": 1000})

        # Predict on the same CSV shape a real test split would have (no labels).
        test_csv = self.root / "test.csv"
        with open(test_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows[:5]:
                writer.writerow({c: row[c] for c in columns})

        # The predictor reloads the encoder itself; point it at the double.
        import adapters.text_embedding_logistic as module
        original = module.load_encoder
        module.load_encoder = lambda *a, **k: FakeEncoder()
        try:
            batch = ArtifactPredictor().predict_batch(
                adapter_name="text_embedding_logistic",
                artifact_path=Path(result.artifact_path),
                config=FakeFinalConfig(test_csv),
            )
        finally:
            module.load_encoder = original
        return batch, result

    def test_single_text_model_predicts(self):
        rows = [{"text": f"aaaaaaaa {i}", "label": i % 2} for i in range(40)]
        batch, _ = self.train_and_predict(
            ["text"], "label",
            {"modality": "text", "feature_columns": ["text"],
             "target_column": "label", "label_mapping": None},
            rows)
        self.assertEqual(len(batch.predictions), 5)

    def test_pair_model_predicts(self):
        rows = [{"a": f"aaaaaaaa {i}", "b": f"zzzzzzzz {i}",
                 "label": i % 2} for i in range(40)]
        batch, result = self.train_and_predict(
            ["a", "b"], "label",
            {"modality": "text_pair", "feature_columns": ["a", "b"],
             "target_column": "label", "label_mapping": None},
            rows)
        self.assertEqual(len(batch.predictions), 5)
        # The pair layout must survive the round trip: 3 x embedding dim.
        self.assertEqual(result.feature_count, 12)

    def test_missing_feature_column_is_a_clear_error(self):
        rows = [{"text": f"aaaaaaaa {i}", "label": i % 2} for i in range(40)]
        train, val = self.root / "t2.csv", self.root / "v2.csv"
        write_csv(train, rows[:20], ["text"], "label")
        write_csv(val, rows[20:], ["text"], "label")
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = FakeEncoder()
        out = self.root / "run2"
        out.mkdir()
        result = adapter.run(
            "exp_001", FakeConfig(train, val),
            {"modality": "text", "feature_columns": ["text"],
             "target_column": "label", "label_mapping": None},
            out, {"C": 1.0, "max_iter": 1000})

        wrong = self.root / "wrong.csv"
        wrong.write_text("other,column\nfoo,bar\n", encoding="utf-8")

        import adapters.text_embedding_logistic as module
        original = module.load_encoder
        module.load_encoder = lambda *a, **k: FakeEncoder()
        try:
            with self.assertRaises(PredictionError) as ctx:
                ArtifactPredictor().predict_batch(
                    adapter_name="text_embedding_logistic",
                    artifact_path=Path(result.artifact_path),
                    config=FakeFinalConfig(wrong),
                )
        finally:
            module.load_encoder = original
        self.assertIn("missing feature column", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
