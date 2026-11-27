import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from superagent.data_inspector import DataInspector
from superagent.tabular_data import (
    ORDINAL_MIN_SHAPE_SCORE,
    ORDINAL_SIGNIFICANCE_LEVEL,
    build_tabular_preprocessor,
    classify_tabular_columns,
    infer_tabular_column_types,
    ordinal_monotone_p_value,
    ordinal_monotone_shape,
    ordinal_monotonicity,
    transformed_feature_count,
)


def ordinal_column(levels=16, repeats=250):
    return np.repeat(np.arange(1, levels + 1), repeats)


def monotone_labels(column, cut=None):
    cut = cut if cut is not None else int(column.max() * 0.75)
    return (column >= cut).astype(int)


def ushape_labels(column):
    low = column <= 3
    high = column >= column.max() - 2
    return (low | high).astype(int)


class InferenceTests(unittest.TestCase):
    def test_binary_integers_become_ordinal(self):
        values = np.asarray([[0], [1], [1], [0]] * 10, dtype=object)
        numeric, ordinal, categorical = infer_tabular_column_types(values)
        self.assertEqual(ordinal, [0])
        self.assertEqual(numeric, [])
        self.assertEqual(categorical, [])

    def test_binary_floats_and_strings_become_ordinal(self):
        floats = np.asarray([[0.0], [1.0], [1.0], [0.0]] * 5, dtype=object)
        strings = np.asarray([["0"], ["1"], ["1"], ["0"]] * 5, dtype=object)
        self.assertEqual(infer_tabular_column_types(floats)[1], [0])
        self.assertEqual(infer_tabular_column_types(strings)[1], [0])

    def test_monotone_integer_levels_become_ordinal(self):
        column = ordinal_column()
        labels = monotone_labels(column)
        matrix = column.reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(matrix, target=labels)
        self.assertEqual(decisions[0].kind, "ordinal")
        self.assertEqual(decisions[0].reason, "monotone_levels")
        self.assertLess(
            decisions[0].monotone_p_value, ORDINAL_SIGNIFICANCE_LEVEL
        )
        self.assertGreaterEqual(
            decisions[0].shape_score, ORDINAL_MIN_SHAPE_SCORE
        )

    def test_ushaped_integer_levels_stay_categorical(self):
        column = ordinal_column()
        labels = ushape_labels(column)
        matrix = column.reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(matrix, target=labels)
        self.assertEqual(decisions[0].kind, "categorical")
        self.assertLess(decisions[0].shape_score, ORDINAL_MIN_SHAPE_SCORE)

    def test_alternating_integer_effect_stays_categorical(self):
        # [high, 0, low, 0, high, 0, low, 0]: the two peaks sit in the lower
        # half of the rank space, so |Spearman| looks positive even though the
        # curve reverses violently.  Only the shape check rejects it.
        column = np.tile(np.arange(1, 9), 500)
        effect = np.asarray([1.8, 0.0, -1.6, 0.0, 1.8, 0.0, -1.6, 0.0])
        rng = np.random.default_rng(11)
        logits = effect[column - 1] + rng.normal(0.0, 1.2, column.shape[0])
        labels = (logits > 0).astype(int)
        matrix = column.reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(matrix, target=labels)
        self.assertEqual(decisions[0].kind, "categorical")
        self.assertEqual(decisions[0].reason, "non_monotone_shape")
        self.assertLess(decisions[0].shape_score, ORDINAL_MIN_SHAPE_SCORE)

    def test_rare_positive_class_still_ordinal(self):
        # A 3% positive rate deflates |Spearman| far below any usable
        # threshold, while the monotone trend is genuinely there.
        column = np.tile(np.arange(1, 11), 600)
        rng = np.random.default_rng(5)
        logit = 0.13 * (column - 5.5) - 3.5 + rng.normal(0.0, 0.5, column.shape[0])
        labels = (rng.random(column.shape[0]) < 1 / (1 + np.exp(-logit))).astype(int)
        matrix = column.reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(matrix, target=labels)
        self.assertEqual(decisions[0].kind, "ordinal")
        self.assertLess(decisions[0].monotonicity, 0.10)

    def test_integer_levels_without_target_stay_categorical(self):
        column = ordinal_column(levels=8, repeats=10)
        matrix = column.reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(matrix)
        self.assertEqual(decisions[0].kind, "categorical")

    def test_sparse_integer_codes_stay_categorical(self):
        codes = np.asarray([[0], [1], [2], [7], [9]] * 20, dtype=object)
        decisions = classify_tabular_columns(
            codes, target=np.asarray([0, 1, 0, 1, 0] * 20)
        )
        self.assertEqual(decisions[0].kind, "categorical")
        self.assertEqual(decisions[0].reason, "sparse_integer_levels")

    def test_high_cardinality_integers_stay_numeric(self):
        identifiers = np.arange(1000, 5000).reshape(-1, 1).astype(object)
        decisions = classify_tabular_columns(identifiers)
        self.assertEqual(decisions[0].kind, "numeric")
        self.assertEqual(decisions[0].reason, "high_cardinality_integer")

    def test_continuous_and_text_columns_are_unchanged(self):
        rng = np.random.default_rng(7)
        continuous = rng.normal(size=400).reshape(-1, 1).astype(object)
        text = np.asarray([["red"], ["blue"], ["green"], ["red"]] * 100, dtype=object)
        self.assertEqual(
            classify_tabular_columns(continuous)[0].kind, "numeric"
        )
        self.assertEqual(classify_tabular_columns(text)[0].kind, "categorical")

    def test_explicit_declaration_wins_over_heuristics(self):
        column = ordinal_column(levels=8, repeats=10)
        matrix = column.reshape(-1, 1).astype(object)
        numeric, ordinal, categorical = infer_tabular_column_types(
            matrix, explicit_categorical=[0]
        )
        self.assertEqual(categorical, [0])
        self.assertEqual(ordinal, [])
        numeric, ordinal, categorical = infer_tabular_column_types(
            column.astype(object).reshape(-1, 1), explicit_ordinal=[0]
        )
        self.assertEqual(ordinal, [0])
        self.assertEqual(classify_tabular_columns(
            matrix, explicit_ordinal=[0]
        )[0].reason, "declared")

    def test_explicit_declaration_must_be_disjoint_and_complete(self):
        matrix = np.zeros((4, 2), dtype=object)
        with self.assertRaises(ValueError):
            infer_tabular_column_types(
                matrix, explicit_numeric=[0], explicit_ordinal=[0, 1]
            )
        with self.assertRaises(ValueError):
            infer_tabular_column_types(matrix, explicit_numeric=[0])

    def test_partition_covers_every_column_exactly_once(self):
        rng = np.random.default_rng(3)
        matrix = np.column_stack(
            [
                rng.normal(size=300),
                rng.integers(0, 2, size=300),
                rng.integers(1, 7, size=300),
                [f"c{index % 5}" for index in range(300)],
            ]
        ).astype(object)
        numeric, ordinal, categorical = infer_tabular_column_types(
            matrix, target=rng.integers(0, 2, size=300)
        )
        combined = numeric + ordinal + categorical
        self.assertEqual(sorted(combined), [0, 1, 2, 3])
        self.assertEqual(len(combined), len(set(combined)))

    def test_monotonicity_helper_extremes(self):
        values = list(range(1, 21))
        self.assertGreater(ordinal_monotonicity(values, [0] * 10 + [1] * 10), 0.85)
        alternating = [index % 2 for index in range(20)]
        self.assertLess(ordinal_monotonicity(values, alternating), 0.2)
        self.assertEqual(ordinal_monotonicity(values, None), 0.0)

    def test_p_value_helper_helpers_are_uninformative_without_target(self):
        values = list(range(1, 21))
        self.assertEqual(ordinal_monotone_p_value(values, None), 1.0)
        self.assertEqual(ordinal_monotone_shape(values, None), 0.0)

    def test_p_value_helper_separates_monotone_from_alternating(self):
        column = np.tile(np.arange(1, 17), 200)
        monotone = monotone_labels(column)
        alternating = np.tile([0, 1, 0, 0, 1, 1, 0, 1], 400)
        self.assertLess(ordinal_monotone_p_value(column, monotone), 1e-6)
        self.assertGreaterEqual(
            ordinal_monotone_shape(column, monotone), ORDINAL_MIN_SHAPE_SCORE
        )
        self.assertLess(
            ordinal_monotone_shape(column, alternating), ORDINAL_MIN_SHAPE_SCORE
        )


class PreprocessorTests(unittest.TestCase):
    def _matrix(self, rows=64):
        column = np.tile(np.arange(1, 17), rows // 16)
        return np.column_stack(
            [
                column,
                np.linspace(0.0, 1.0, len(column)),
                ["a" if index % 2 else "b" for index in range(len(column))],
            ]
        ).astype(object)

    def _profile(self):
        return {
            "numeric_column_indices": [1],
            "ordinal_column_indices": [0],
            "categorical_column_indices": [2],
        }

    def test_ordinal_column_expands_to_one_feature(self):
        data = self._matrix()
        preprocessor = build_tabular_preprocessor(
            self._profile(), numeric_preprocessing="none"
        )
        preprocessor.fit(data)
        self.assertEqual(transformed_feature_count(preprocessor, -1), 4)

    def test_onehot_switch_expands_ordinal_column(self):
        data = self._matrix()
        preprocessor = build_tabular_preprocessor(
            self._profile(),
            numeric_preprocessing="none",
            ordinal_preprocessing="onehot",
        )
        preprocessor.fit(data)
        # 16 one-hot columns + 1 numeric + 2 categorical levels
        self.assertEqual(transformed_feature_count(preprocessor, -1), 19)

    def test_ordinal_order_follows_numeric_value_not_string_order(self):
        data = np.asarray([["2"], ["10"], ["9"]], dtype=object)
        profile = {"ordinal_column_indices": [0]}
        preprocessor = build_tabular_preprocessor(
            profile, numeric_preprocessing="none"
        )
        transformed = np.asarray(preprocessor.fit_transform(data), dtype=float)
        self.assertEqual(transformed[:, 0].tolist(), [2.0, 10.0, 9.0])

    def test_missing_ordinal_level_is_imputed_with_median(self):
        data = np.asarray([["1"], ["2"], [""], ["4"]], dtype=object)
        profile = {"ordinal_column_indices": [0]}
        preprocessor = build_tabular_preprocessor(
            profile, numeric_preprocessing="none"
        )
        transformed = np.asarray(preprocessor.fit_transform(data), dtype=float)
        self.assertEqual(transformed[2, 0], 2.0)

    def test_unknown_ordinal_preprocessing_is_rejected(self):
        with self.assertRaises(ValueError):
            build_tabular_preprocessor(
                self._profile(), ordinal_preprocessing="target"
            )

    def test_overlapping_indices_are_rejected(self):
        profile = {"numeric_column_indices": [0], "ordinal_column_indices": [0]}
        with self.assertRaises(ValueError):
            build_tabular_preprocessor(profile)

    def test_profile_without_ordinal_key_still_works(self):
        data = self._matrix()
        profile = {
            "numeric_column_indices": [0, 1],
            "categorical_column_indices": [2],
        }
        preprocessor = build_tabular_preprocessor(
            profile, numeric_preprocessing="none"
        )
        preprocessor.fit(data)
        self.assertGreater(transformed_feature_count(preprocessor, -1), 2)


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _npy_config(self, features, labels, stem="task"):
        paths = {}
        half = len(features) // 2
        for name, values in {
            "train_features": features[:half],
            "train_labels": labels[:half],
            "validation_features": features[half:],
            "validation_labels": labels[half:],
        }.items():
            path = self.root / f"{stem}_{name}.npy"
            np.save(path, values)
            paths[name] = str(path)
        return SimpleNamespace(
            data={"format": "npy_split", **paths},
            metric="accuracy",
            maximum_cpu_threads=1,
        )

    def test_npy_profile_exposes_ordinal_channel(self):
        # The ordinal band is ``k < max(10, min(50, 0.005 * n_train))``, so a
        # 16-level column needs at least 4000 training rows to be in it.
        levels = np.tile(np.arange(1, 17), 500)
        binary = np.tile([0, 1], 4000)
        features = np.column_stack(
            [levels, binary, np.linspace(0, 1, 8000)]
        ).astype(object)
        labels = monotone_labels(levels)
        profile = DataInspector().inspect(self._npy_config(features, labels))

        self.assertEqual(profile["ordinal_column_indices"], [0, 1])
        self.assertEqual(profile["ordinal_feature_count"], 2)
        self.assertEqual(profile["numeric_column_indices"], [2])
        self.assertEqual(
            profile["column_type_counts"],
            {"numeric": 1, "ordinal": 2, "categorical": 0},
        )
        self.assertLessEqual(len(profile["ordinal_evidence"]), 8)
        self.assertEqual(profile["ordinal_evidence"][1]["reason"], "binary")

    def test_csv_profile_honours_declared_ordinal_columns(self):
        rows = []
        for index in range(240):
            rows.append(
                {
                    "grade": str(index % 6),
                    "score": f"{index / 7:.3f}",
                    "city": ["a", "b", "c"][index % 3],
                    "label": str(index % 2),
                }
            )
        train = self.root / "train.csv"
        validation = self.root / "validation.csv"
        for path, chunk in ((train, rows[:160]), (validation, rows[160:])):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["grade", "score", "city", "label"]
                )
                writer.writeheader()
                writer.writerows(chunk)

        config = SimpleNamespace(
            data={
                "format": "csv_split",
                "train": str(train),
                "validation": str(validation),
                "target_column": "label",
                "columns": {
                    "numeric": ["score"],
                    "ordinal": ["grade"],
                    "categorical": ["city"],
                },
            },
            metric="accuracy",
            maximum_cpu_threads=1,
        )
        profile = DataInspector().inspect(config)
        self.assertEqual(profile["ordinal_columns"], ["grade"])
        self.assertEqual(profile["categorical_columns"], ["city"])
        self.assertEqual(profile["numeric_columns"], ["score"])
        self.assertEqual(profile["ordinal_evidence"][0]["column"], "grade")
        self.assertEqual(profile["ordinal_evidence"][0]["reason"], "declared")


if __name__ == "__main__":
    unittest.main()
