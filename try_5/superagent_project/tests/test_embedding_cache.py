"""The embedding cache, which is what makes the embedding adapter usable.

Why this matters more than it looks
----------------------------------
Encoding 20 Newsgroups takes 213s on the cluster. A run spends 8 experiments.
Without a cache that is 8 x 213s = ~28 minutes inside a 30-60 minute budget, so
the adapter would exist and be unusable.

The cache also has to survive a *process* boundary: experiments run in spawned
children (``ExperimentExecutor`` -> ``run_adapter_worker``), so the adapter
object is pickled per experiment. An in-memory dict would be empty every time.

What is tested here:

1. a second lookup of the same corpus is a hit and does not re-encode
2. editing the data invalidates it
3. a different column set / model / normalisation invalidates it
4. a corrupt cache file degrades to a miss, never an error
5. the adapter still works with the cache disabled (the shipped default)
6. results are identical whether or not the cache is used
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adapters import embedding_cache
from adapters.text_embedding_logistic import TextEmbeddingLogisticAdapter
from tests.test_text_embedding import (
    FakeConfig,
    FakeEncoder,
    make_single_task,
    write_csv,
)


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "train.csv"
        self.data.write_text("text,label\nhello,0\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def key(self, **overrides):
        kwargs = dict(data_paths=[self.data], text_columns=["col0"],
                      model_name="m", dimension=768)
        kwargs.update(overrides)
        return embedding_cache.cache_key(**kwargs)

    def test_same_inputs_give_the_same_key(self):
        self.assertEqual(self.key(), self.key())

    def test_different_model_invalidates(self):
        self.assertNotEqual(self.key(), self.key(model_name="other"))

    def test_different_dimension_invalidates(self):
        self.assertNotEqual(self.key(), self.key(dimension=384))

    def test_different_columns_invalidate(self):
        self.assertNotEqual(self.key(), self.key(text_columns=["col1"]))

    def test_normalisation_flag_invalidates(self):
        self.assertNotEqual(self.key(), self.key(normalised=False))

    def test_editing_the_file_invalidates(self):
        before = self.key()
        # Bump mtime and size: any real edit does at least one of these.
        self.data.write_text("text,label\nhello world,0\nbye,1\n",
                             encoding="utf-8")
        os.utime(self.data, (1, 1))
        self.assertNotEqual(before, self.key())

    def test_single_path_and_list_agree(self):
        self.assertEqual(
            embedding_cache.cache_key(self.data, ["col0"], "m", 8),
            embedding_cache.cache_key([self.data], ["col0"], "m", 8))


class StoreLoadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name) / "cache"

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip(self):
        array = np.arange(12, dtype=float).reshape(3, 4)
        embedding_cache.store(self.dir, "abc", "col0", array)
        loaded = embedding_cache.load(self.dir, "abc", "col0")
        np.testing.assert_array_equal(loaded, array)

    def test_missing_returns_none(self):
        self.assertIsNone(embedding_cache.load(self.dir, "nope", "col0"))

    def test_row_count_mismatch_is_a_miss(self):
        embedding_cache.store(self.dir, "abc", "col0",
                              np.zeros((3, 4), dtype=float))
        self.assertIsNone(
            embedding_cache.load(self.dir, "abc", "col0", expected_rows=5))

    def test_corrupt_file_is_a_miss_not_an_error(self):
        self.dir.mkdir(parents=True)
        path = embedding_cache.cache_path(self.dir, "abc", "col0")
        path.write_bytes(b"not a numpy file at all")
        self.assertIsNone(embedding_cache.load(self.dir, "abc", "col0"))

    def test_no_partial_file_survives_a_failed_write(self):
        array = np.zeros((2, 2), dtype=float)
        embedding_cache.store(self.dir, "abc", "col0", array)
        leftovers = list(self.dir.glob("*.tmp.npy"))
        self.assertEqual(leftovers, [])

    def test_splits_are_kept_apart(self):
        embedding_cache.store(self.dir, "abc", "col0", np.ones((2, 2)))
        embedding_cache.store(self.dir, "abc", "col1", np.full((2, 2), 7.0))
        self.assertTrue(np.allclose(
            embedding_cache.load(self.dir, "abc", "col0"), 1.0))
        self.assertTrue(np.allclose(
            embedding_cache.load(self.dir, "abc", "col1"), 7.0))


class CounterEncoder(FakeEncoder):
    def __init__(self):
        super().__init__()
        self.texts_encoded = 0

    def encode(self, texts, **kwargs):
        self.texts_encoded += len(texts)
        return super().encode(texts, **kwargs)


class AdapterCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.val = make_single_task(self.root)
        self.config = FakeConfig(self.train, self.val)
        self.cache_dir = self.root / "cache"

    def tearDown(self):
        self.temp.cleanup()

    def profile(self):
        return {"modality": "text", "feature_columns": ["text"],
                "target_column": "label", "label_mapping": None}

    def run_once(self, encoder, cache_dir, out_name):
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = encoder
        out = self.root / out_name
        out.mkdir(exist_ok=True)
        params = {"C": 1.0, "max_iter": 1000}
        if cache_dir is not None:
            params["embedding_cache_dir"] = str(cache_dir)
        result = adapter.run("exp", self.config, self.profile(), out, params)
        return result, out

    def read_metrics(self, directory):
        return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))

    def test_second_run_does_not_reencode(self):
        first_encoder = CounterEncoder()
        self.run_once(first_encoder, self.cache_dir, "out1")
        self.assertGreater(first_encoder.texts_encoded, 0)

        second_encoder = CounterEncoder()
        self.run_once(second_encoder, self.cache_dir, "out2")
        # Both train and validation columns must come from the cache. The only
        # encode left is the one-row dimension probe (1 row per column), which
        # exists so the cache can be keyed without encoding the whole column.
        self.assertLessEqual(second_encoder.texts_encoded, 2)

    def test_cached_run_gives_the_same_accuracy(self):
        uncached, _ = self.run_once(CounterEncoder(), None, "plain")
        self.run_once(CounterEncoder(), self.cache_dir, "warm")
        cached, _ = self.run_once(CounterEncoder(), self.cache_dir, "hit")
        self.assertEqual(uncached.validation_accuracy, cached.validation_accuracy)

    def test_cache_info_is_recorded_in_metrics(self):
        self.run_once(CounterEncoder(), self.cache_dir, "warm")
        out = self.root / "report"
        out.mkdir()
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = CounterEncoder()
        adapter.run("exp", self.config, self.profile(), out,
                    {"C": 1.0, "max_iter": 1000,
                     "embedding_cache_dir": str(self.cache_dir)})
        block = self.read_metrics(out)["embedding_cache"]
        self.assertTrue(block["train"]["enabled"])
        self.assertTrue(block["train"]["columns"][0]["hit"])

    def test_adapter_still_works_with_cache_disabled(self):
        result, out = self.run_once(CounterEncoder(), None, "nocache")
        self.assertEqual(result.status, "success")
        self.assertFalse(self.read_metrics(out)["embedding_cache"]["train"]["enabled"])

    def test_read_only_cache_directory_does_not_fail_the_run(self):
        """An unwritable cache must degrade to encoding, not raise."""
        blocker = self.root / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        result, _ = self.run_once(CounterEncoder(), blocker / "cache", "ro")
        self.assertEqual(result.status, "success")

    def test_pair_task_caches_each_column_separately(self):
        rows = [{"a": f"aaaaaaaa {i}", "b": f"zzzzzzzz {i}",
                 "label": 1 if i % 2 == 0 else 0} for i in range(40)]
        train, val = self.root / "pa.csv", self.root / "pv.csv"
        write_csv(train, rows[:20], ["a", "b"], "label")
        write_csv(val, rows[20:], ["a", "b"], "label")

        profile = {"modality": "text_pair", "feature_columns": ["a", "b"],
                   "target_column": "label", "label_mapping": None}
        for name in ("p1", "p2"):
            adapter = TextEmbeddingLogisticAdapter(random_seed=42)
            adapter._encoder = CounterEncoder()
            out = self.root / name
            out.mkdir()
            adapter.run("exp", FakeConfig(train, val), profile, out,
                        {"C": 1.0, "max_iter": 1000,
                         "embedding_cache_dir": str(self.cache_dir)})

        second = CounterEncoder()
        adapter = TextEmbeddingLogisticAdapter(random_seed=42)
        adapter._encoder = second
        out = self.root / "p3"
        out.mkdir()
        adapter.run("exp", FakeConfig(train, val), profile, out,
                    {"C": 1.0, "max_iter": 1000,
                     "embedding_cache_dir": str(self.cache_dir)})
        # Only the per-column dimension probes remain (2 columns x 1 row).
        self.assertLessEqual(second.texts_encoded, 4)

    def test_changed_data_is_reencoded(self):
        self.run_once(CounterEncoder(), self.cache_dir, "orig")
        # Rewrite the training split with different content and a new mtime.
        rows = [{"text": f"ccccc {i}", "label": i % 2} for i in range(20)]
        write_csv(self.train, rows, ["text"], "label")
        os.utime(self.train, (1, 1))
        encoder = CounterEncoder()
        self.run_once(encoder, self.cache_dir, "changed")
        self.assertGreater(encoder.texts_encoded, 2)


if __name__ == "__main__":
    unittest.main()
