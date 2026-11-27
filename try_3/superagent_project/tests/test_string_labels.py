import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.tabular_baseline import TabularBaselineAdapter
from superagent.data_inspector import DataInspector
from superagent.final_evaluator import FinalEvaluator
from superagent.label_encoding import decode_labels, describe_labels, encode_labels
from superagent.tabular_data import load_tabular_split


class DescribeLabelsTests(unittest.TestCase):
    def test_integral_numbers_keep_their_values(self):
        fields = describe_labels([0, 1, 1, 0])
        self.assertEqual(fields["label_encoding"], "identity")
        self.assertIsNone(fields["label_mapping"])
        self.assertEqual(encode_labels([1, 0, 1], None).tolist(), [1, 0, 1])

    def test_numeric_strings_are_treated_as_numbers(self):
        fields = describe_labels(["0", "1", "1"])
        self.assertEqual(fields["label_encoding"], "identity")
        self.assertEqual(encode_labels(["1", "0"], None).tolist(), [1, 0])

    def test_non_zero_based_integers_are_not_shifted(self):
        fields = describe_labels([1, 2, 3, 2])
        self.assertEqual(fields["label_encoding"], "identity")
        self.assertEqual(encode_labels([3, 1], None).tolist(), [3, 1])

    def test_string_labels_get_a_sorted_deterministic_mapping(self):
        fields = describe_labels(["yes", "no", "yes", "maybe"])
        self.assertEqual(fields["label_encoding"], "string_mapping")
        self.assertEqual(fields["label_mapping"], {"maybe": 0, "no": 1, "yes": 2})
        self.assertEqual(fields["class_labels"], ["maybe", "no", "yes"])
        # Row order must not influence the codes.
        shuffled = describe_labels(["maybe", "yes", "no", "yes"])
        self.assertEqual(shuffled["label_mapping"], fields["label_mapping"])

    def test_labels_that_only_look_numeric_fall_back_to_mapping(self):
        # "1" and "01" are the same number but different labels, so an
        # identity encoding would silently merge two classes.
        fields = describe_labels(["1", "01", "2"])
        self.assertEqual(fields["label_encoding"], "string_mapping")
        self.assertEqual(len(fields["class_labels"]), 3)

    def test_unknown_label_is_reported(self):
        mapping = describe_labels(["yes", "no"])["label_mapping"]
        with self.assertRaises(ValueError) as context:
            encode_labels(["yes", "unknown"], mapping)
        self.assertIn("unknown", str(context.exception))

    def test_missing_mapping_for_string_labels_is_explicit(self):
        with self.assertRaises(ValueError) as context:
            encode_labels(["yes", "no"], None)
        self.assertIn("mapping", str(context.exception))

    def test_decoding_restores_original_labels(self):
        self.assertEqual(
            decode_labels([0, 1], ["no", "yes"]), ["no", "yes"]
        )
        self.assertEqual(decode_labels([0, 1], None), [0, 1])


class StringLabelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _npy_config(self, labels, informative=None):
        rng = np.random.default_rng(3)
        signal = (
            np.asarray(informative, dtype=float)
            if informative is not None
            else rng.normal(size=len(labels))
        )
        features = np.column_stack(
            [signal + rng.normal(scale=0.3, size=len(labels)), rng.normal(size=len(labels))]
        ).astype(object)
        half = len(labels) // 2
        paths = {}
        for name, values in {
            "train_features": features[:half],
            "train_labels": labels[:half],
            "validation_features": features[half:],
            "validation_labels": labels[half:],
            "test_features": features[half:],
            "test_labels": labels[half:],
        }.items():
            path = self.root / f"{name}.npy"
            np.save(path, values)
            paths[name] = str(path)
        return SimpleNamespace(
            data={"format": "npy_split", **paths},
            metric="accuracy",
            maximum_cpu_threads=1,
        )

    def test_profile_records_the_mapping_and_keeps_readable_classes(self):
        labels = np.asarray(["no", "yes"] * 40, dtype=object)
        config = self._npy_config(labels)
        profile = DataInspector().inspect(config)

        self.assertEqual(profile["label_encoding"], "string_mapping")
        self.assertEqual(profile["label_mapping"], {"no": 0, "yes": 1})
        self.assertEqual(profile["class_labels"], ["no", "yes"])
        # The profile describes the training split, which is the first half.
        self.assertEqual(
            profile["class_distribution"], {"no": 20, "yes": 20}
        )

    def test_split_loader_encodes_every_split_with_one_mapping(self):
        labels = np.asarray(["no", "yes"] * 40, dtype=object)
        config = self._npy_config(labels)
        profile = DataInspector().inspect(config)

        _, y_train = load_tabular_split(config, profile, "train")
        _, y_val = load_tabular_split(config, profile, "validation")
        self.assertEqual(y_train.dtype.kind, "i")
        self.assertEqual(y_val.dtype.kind, "i")
        self.assertEqual(sorted(set(y_train.tolist())), [0, 1])
        self.assertEqual(sorted(set(y_val.tolist())), [0, 1])

    def test_compatible_label_keeps_the_identity_encoding(self):
        labels = np.arange(120) % 2
        config = self._npy_config(labels)
        profile = DataInspector().inspect(config)
        self.assertEqual(profile["label_encoding"], "identity")
        self.assertIsNone(profile["label_mapping"])

    def test_adapter_trains_end_to_end_on_string_labels(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(size=240)
        labels = np.where(signal > 0, "yes", "no").astype(object)
        config = self._npy_config(labels, informative=signal)
        profile = DataInspector().inspect(config)
        output = self.root / "experiment"
        output.mkdir()

        result = TabularBaselineAdapter(42).run(
            "exp_001",
            config,
            profile,
            output,
            {"max_iter": 500, "C": 1.0},
        )

        self.assertEqual(result.status, "success")
        self.assertGreater(result.validation_accuracy, 0.7)
        self.assertIsNotNone(result.roc_auc)
        # Predictions stay integer codes; decoding is a reporting concern.
        stored = np.load(output / "val_predictions.npy", allow_pickle=True)
        self.assertEqual(np.asarray(stored).dtype.kind, "i")


class FinalEvaluationLabelTests(unittest.TestCase):
    def test_class_recall_uses_original_label_names(self):
        metrics = FinalEvaluator._classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 1, 1, 1]),
            class_labels=["no", "yes"],
        )
        self.assertEqual(set(metrics["class_recall"]), {"no", "yes"})
        self.assertIsNone(metrics["test_roc_auc"])

    def test_predictions_csv_decodes_string_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            FinalEvaluator._atomic_write_predictions_csv(
                path, np.asarray([0, 1, 1]), ["no", "yes"]
            )
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["prediction"] for row in rows], ["no", "yes", "yes"])

    def test_predictions_csv_keeps_codes_without_a_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            FinalEvaluator._atomic_write_predictions_csv(
                path, np.asarray([0, 1, 1])
            )
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["prediction"] for row in rows], ["0", "1", "1"])


if __name__ == "__main__":
    unittest.main()
