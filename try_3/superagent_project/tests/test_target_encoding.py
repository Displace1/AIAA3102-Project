"""Target encoding must not leak labels, and must be reachable by the planner.

The leak test is the important one. sklearn's TargetEncoder only cross-fits
inside ``fit_transform(X, y)``; a Pipeline calls ``fit_transform(X)`` without
labels, so using the raw sklearn class inside a pipeline silently encodes each
row with its own target. That is measured here with pure-noise labels: if
encoding leaks, the training AUC rises above chance while validation stays at
chance.
"""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from superagent.plan_validator import PARAMETER_RULES, PlanValidator
from superagent.tabular_data import build_tabular_preprocessor
from superagent.target_encoding import CrossFittedTargetEncoder


def noise_case(n=3000, n_levels=50, seed=0):
    """Labels independent of the feature: any train signal is leakage."""
    rng = np.random.default_rng(seed)
    X = np.array([[f"L{rng.integers(0, n_levels)}"] for _ in range(n)],
                 dtype=object)
    y = rng.integers(0, 2, n)
    profile = {
        "numeric_column_indices": [],
        "ordinal_column_indices": [],
        "categorical_column_indices": [0],
        "number_of_classes": 2,
    }
    return X, y, profile


class NoLeakageTest(unittest.TestCase):
    def test_target_encoding_matches_onehot_on_noise(self):
        """Both encodings must stay near chance on pure-noise labels.

        A comparison against one-hot rather than an absolute threshold: with
        few rows per level an unregularised linear model overfits the one-hot
        matrix too, so the fair test is whether target encoding does any WORSE.
        """
        X, y, profile = noise_case()
        tr, va = slice(0, 2000), slice(2000, None)
        scores = {}
        for mode in ("onehot", "target"):
            pre = build_tabular_preprocessor(
                profile, categorical_preprocessing=mode)
            pipe = Pipeline([
                ("prep", pre),
                ("clf", LogisticRegression(max_iter=2000, C=0.01)),
            ])
            pipe.fit(X[tr], y[tr])
            scores[mode] = (
                roc_auc_score(y[tr], pipe.predict_proba(X[tr])[:, 1]),
                roc_auc_score(y[va], pipe.predict_proba(X[va])[:, 1]),
            )

        onehot_train, _ = scores["onehot"]
        target_train, target_val = scores["target"]
        self.assertLess(
            abs(target_train - onehot_train), 0.10,
            "target encoding's training AUC must track one-hot's; a large "
            "excess means the fold-wise split is not happening "
            f"(onehot={onehot_train:.3f}, target={target_train:.3f})")
        self.assertLess(
            target_val, 0.60,
            "validation AUC on pure noise must stay near chance, "
            f"got {target_val:.3f}")

    def test_manual_fit_transform_is_cross_fitted(self):
        """fit_transform must differ from fit().transform() on the same rows."""
        rng = np.random.default_rng(0)
        X = np.array([[f"L{rng.integers(0, 30)}"] for _ in range(300)],
                     dtype=object)
        y = rng.integers(0, 2, 300)
        encoder = CrossFittedTargetEncoder(random_state=42)
        out_of_fold = encoder.fit_transform(X, y)
        in_sample = encoder.transform(X)
        self.assertFalse(
            np.allclose(out_of_fold, in_sample),
            "out-of-fold encodings must differ from the full-fit encodings; "
            "if they are identical the cross-fitting never ran")

    def test_requires_labels(self):
        encoder = CrossFittedTargetEncoder()
        X = np.array([["a"], ["b"]], dtype=object)
        with self.assertRaises(ValueError):
            encoder.fit(X, None)
        with self.assertRaises(ValueError):
            encoder.fit_transform(X, None)


class ReachabilityTest(unittest.TestCase):
    def test_planner_can_choose_target_encoding(self):
        specs = PlanValidator.allowed_adapter_specs("tabular", metric="accuracy")
        for adapter in ("tabular_extra_trees", "tabular_random_forest"):
            choices = specs[adapter]["parameters"]["categorical_preprocessing"]["choices"]
            self.assertIn("target", choices, f"{adapter} should offer target")

    def test_rule_table_lists_target(self):
        # HGB has its own dense preprocessor and does not expose this knob.
        for adapter in ("tabular_extra_trees", "tabular_random_forest"):
            choices = PARAMETER_RULES[adapter]["parameters"][
                "categorical_preprocessing"]["choices"]
            self.assertIn("target", choices)

    def test_hgb_does_not_advertise_categorical_preprocessing(self):
        """HGB owns its dense preprocessing, so the knob must not appear."""
        specs = PlanValidator.allowed_adapter_specs("tabular", metric="accuracy")
        params = specs["tabular_hist_gradient_boosting"]["parameters"]
        self.assertNotIn("categorical_preprocessing", params)


class BuildTest(unittest.TestCase):
    def test_binary_and_multiclass_pick_correct_target_type(self):
        X, y, profile = noise_case(n=400, n_levels=10)
        pre_bin = build_tabular_preprocessor(
            profile, categorical_preprocessing="target")
        pre_bin.fit(X, y)
        enc = pre_bin.named_transformers_["categorical"].named_steps["encode"]
        self.assertEqual(enc.target_type, "binary")

        y3 = np.arange(len(y)) % 3
        profile3 = dict(profile, number_of_classes=3)
        pre_multi = build_tabular_preprocessor(
            profile3, categorical_preprocessing="target")
        pre_multi.fit(X, y3)
        enc3 = pre_multi.named_transformers_["categorical"].named_steps["encode"]
        self.assertEqual(enc3.target_type, "multiclass")

    def test_unknown_mode_still_rejected(self):
        _, _, profile = noise_case(n=100, n_levels=5)
        with self.assertRaises(ValueError):
            build_tabular_preprocessor(
                profile, categorical_preprocessing="hashing")


if __name__ == "__main__":
    unittest.main()
