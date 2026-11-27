"""Tests for the HistGradientBoosting tabular adapter."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from superagent.experiment_registry import ExperimentRegistry
from superagent.plan_validator import PlanValidator
from superagent.tabular_data import (
    build_hgb_preprocessor,
    estimated_dense_bytes,
    to_dense_float,
)


def synthetic_profile(n_numeric=3, n_categorical=2, n_levels=4, seed=0):
    rng = np.random.default_rng(seed)
    n = 120
    numeric = rng.normal(size=(n, n_numeric))
    categorical = np.array(
        [[f"L{rng.integers(0, n_levels)}" for _ in range(n_categorical)]
         for _ in range(n)],
        dtype=object,
    )
    X = np.hstack([numeric.astype(object), categorical])
    y = (numeric[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    profile = {
        "modality": "tabular",
        "numeric_column_indices": list(range(n_numeric)),
        "ordinal_column_indices": [],
        "categorical_column_indices": list(
            range(n_numeric, n_numeric + n_categorical)
        ),
        "class_labels": None,
    }
    return X, y, profile


class PreprocessorTest(unittest.TestCase):
    def test_output_is_dense_and_unscaled(self):
        X, _, profile = synthetic_profile()
        pre = build_hgb_preprocessor(profile)
        out = pre.fit_transform(X)
        # dense: has .shape and no .toarray
        self.assertFalse(hasattr(out, "toarray"))
        self.assertEqual(out.shape[0], X.shape[0])
        # unscaled: the first numeric column must keep its original values
        np.testing.assert_allclose(out[:, 0], X[:, 0].astype(float), rtol=0)

    def test_missing_numeric_values_stay_nan(self):
        X, _, profile = synthetic_profile()
        X[0, 0] = np.nan
        pre = build_hgb_preprocessor(profile)
        out = pre.fit_transform(X)
        self.assertTrue(np.isnan(out[0, 0]),
                        "HGB learns from missing values; do not impute here")

    def test_rejects_overlapping_column_indices(self):
        _, _, profile = synthetic_profile()
        profile["categorical_column_indices"] = [0, 1]
        with self.assertRaises(ValueError):
            build_hgb_preprocessor(profile)

    def test_rejects_empty_column_assignment(self):
        with self.assertRaises(ValueError):
            build_hgb_preprocessor({
                "numeric_column_indices": [],
                "ordinal_column_indices": [],
                "categorical_column_indices": [],
            })

    def test_onehot_expands_categories(self):
        X, _, profile = synthetic_profile(n_categorical=1, n_levels=4)
        pre = build_hgb_preprocessor(profile)
        out = pre.fit_transform(X)
        # 3 numeric + 4 one-hot levels
        self.assertEqual(out.shape[1], 7)


class DenseGuardTest(unittest.TestCase):
    def test_estimates_bytes(self):
        self.assertEqual(estimated_dense_bytes(1000, 50), 1000 * 50 * 8)

    def test_rejects_matrix_over_budget(self):
        import scipy.sparse as sp
        big = sp.csr_matrix((1000, 1000))
        with self.assertRaises(ValueError) as ctx:
            to_dense_float(big, row_budget_bytes=1024)
        self.assertIn("dense feature matrix", str(ctx.exception))

    def test_converts_sparse_within_budget(self):
        import scipy.sparse as sp
        small = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        out = to_dense_float(small)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_allclose(out, [[1, 0], [0, 2]])


class RegistryAndValidatorTest(unittest.TestCase):
    def test_registry_builds_the_adapter(self):
        adapter = ExperimentRegistry(42).adapter_for(
            "tabular_hist_gradient_boosting")
        self.assertEqual(adapter.adapter_name,
                         "tabular_hist_gradient_boosting")

    def test_validator_exposes_parameters(self):
        specs = PlanValidator.allowed_adapter_specs("tabular")
        self.assertIn("tabular_hist_gradient_boosting", specs)
        params = specs["tabular_hist_gradient_boosting"]["parameters"]
        for expected in ("learning_rate", "max_leaf_nodes", "max_iter",
                         "early_stopping"):
            self.assertIn(expected, params)


class EndToEndTest(unittest.TestCase):
    def test_adapter_runs_and_returns_result(self):
        X, y, profile = synthetic_profile()
        n = len(y)
        cuts = [int(n * 0.6), int(n * 0.8)]
        splits = {
            "train": (X[:cuts[0]], y[:cuts[0]]),
            "validation": (X[cuts[0]:cuts[1]], y[cuts[0]:cuts[1]]),
        }

        class _Config:
            random_seed = 42
            maximum_cpu_threads = 2
            data = {"format": "npy_split"}
            name = "synthetic"
            metric = "accuracy"
            source_path = "synthetic.json"

        with tempfile.TemporaryDirectory() as tmp:
            import adapters.tabular_hgb as mod
            original = mod.load_tabular_split
            mod.load_tabular_split = (
                lambda config, p, split: splits[split])
            try:
                adapter = mod.TabularHGBAdapter(42)
                result = adapter.run(
                    "exp_001", _Config(), profile, Path(tmp),
                    {"learning_rate": 0.1, "max_leaf_nodes": 15,
                     "min_samples_leaf": 5, "l2_regularization": 0.0,
                     "max_iter": 60, "early_stopping": False},
                )
            finally:
                mod.load_tabular_split = original

            self.assertEqual(result.status, "success")
            self.assertIsNotNone(result.validation_accuracy)
            self.assertGreaterEqual(result.validation_accuracy, 0.0)
            self.assertLessEqual(result.validation_accuracy, 1.0)
            self.assertTrue((Path(tmp) / "pipeline.joblib").is_file())
            self.assertTrue((Path(tmp) / "val_predictions.npy").is_file())

            metrics = json.loads(
                (Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("decision_rule", metrics)
            self.assertEqual(metrics["early_stopping_source"]
                             if "early_stopping_source" in metrics
                             else result.model_config["early_stopping_source"],
                             "disabled")
            self.assertEqual(result.model_config["early_stopping_source"],
                             "disabled")

    def test_early_stopping_records_training_source(self):
        """Early stopping must draw its holdout from TRAIN, not validation."""
        X, y, profile = synthetic_profile()
        cuts = [int(len(y) * 0.6), int(len(y) * 0.8)]
        splits = {"train": (X[:cuts[0]], y[:cuts[0]]),
                  "validation": (X[cuts[0]:cuts[1]], y[cuts[0]:cuts[1]])}

        class _Config:
            random_seed = 42
            maximum_cpu_threads = 2
            data = {"format": "npy_split"}
            name = "synthetic"
            metric = "accuracy"
            source_path = "synthetic.json"

        with tempfile.TemporaryDirectory() as tmp:
            import adapters.tabular_hgb as mod
            original = mod.load_tabular_split
            mod.load_tabular_split = lambda config, p, split: splits[split]
            try:
                result = mod.TabularHGBAdapter(42).run(
                    "exp_002", _Config(), profile, Path(tmp),
                    {"learning_rate": 0.1, "max_leaf_nodes": 15,
                     "max_iter": 80, "early_stopping": True,
                     "validation_fraction": 0.1, "n_iter_no_change": 5},
                )
            finally:
                mod.load_tabular_split = original

            self.assertEqual(result.model_config["early_stopping_source"],
                             "training_split")


if __name__ == "__main__":
    unittest.main()
