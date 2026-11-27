"""Tests for the measured-throughput cache (L2).

The point of the cache is that a machine the analytic model knows nothing about
can still gate plans, because it measured its own per-epoch cost.  These tests
pin: (a) the recorded number is the slowest epoch, not the mean, (b) the cache
survives a round trip through JSON, (c) a hit wins over the analytic model and
works even when the fingerprint is foreign, (d) a different dataset shape falls
back instead of reusing a number that does not apply.
"""

import json
import tempfile
import unittest
from pathlib import Path

from adapters.base import ExperimentResult
from superagent.image_runtime import (
    FIXED_SETUP_SECONDS,
    MEASURED_THROUGHPUT_UPLIFT,
    estimate_image_training_seconds,
    estimate_image_training_seconds_with_source,
)
from superagent.plan_validator import PlanValidationError, PlanValidator
from superagent.experiment_plan import ExperimentPlan
from superagent.throughput_cache import (
    ThroughputCache,
    attach_measured_throughput,
    measurement_from_result,
    measured_throughput_key,
)


def profile(train=5000, validation=1000, timeout=1800, os_family="windows",
            device="cpu"):
    return {
        "modality": "image",
        "train_samples": train,
        "validation_samples": validation,
        "image_summary": {
            "train": {"median_width": 32, "median_height": 32},
            "validation": {"median_width": 32, "median_height": 32},
        },
        "execution_context": {
            "experiment_timeout_seconds": timeout,
            "maximum_cpu_threads": 2,
            "os_family": os_family,
            "training_device": device,
            "cuda_available": device == "cuda",
            "torch_threads": 12,
        },
    }


def hyperparameters(size=32, epochs=6, batch=32):
    return {
        "image_size": size,
        "batch_size": batch,
        "learning_rate": 0.001,
        "epochs": epochs,
        "data_augmentation": True,
        "early_stopping_patience": 4,
        "weight_decay": 0.0001,
    }


def plan(size=32, epochs=6, batch=32):
    return ExperimentPlan(
        experiment_id="exp_002",
        adapter_name="image_small_cnn",
        hyperparameters=hyperparameters(size, epochs, batch),
        reason="Throughput cache test plan.",
    )


def result_with_epochs(epoch_seconds, epochs=None):
    return ExperimentResult(
        experiment_id="exp_002",
        model_name="small_cnn",
        status="success",
        validation_accuracy=0.8,
        training_time_seconds=sum(epoch_seconds),
        artifact_path=None,
        epoch_count=len(epoch_seconds) if epochs is None else epochs,
        model_config={"history": []},
        epoch_seconds=list(epoch_seconds),
        stopped_reason="early_stopping",
    )


class MeasurementRecordingTests(unittest.TestCase):
    def test_uses_slowest_epoch_not_the_mean(self):
        entry = measurement_from_result(
            plan(),
            profile(),
            result_with_epochs([3.0, 9.3, 7.8]),
        )

        self.assertIsNotNone(entry)
        self.assertAlmostEqual(entry["mean_epoch_seconds"], 6.7, places=3)
        self.assertAlmostEqual(entry["max_epoch_seconds"], 9.3, places=3)
        self.assertEqual(entry["epochs_completed"], 3)

    def test_falls_back_to_training_time_without_epoch_timings(self):
        result = result_with_epochs([])
        result.epoch_count = 4
        result.training_time_seconds = 40.0

        entry = measurement_from_result(plan(), profile(), result)

        self.assertIsNotNone(entry)
        self.assertAlmostEqual(entry["mean_epoch_seconds"], 10.0, places=3)
        self.assertAlmostEqual(entry["max_epoch_seconds"], 10.0, places=3)

    def test_failed_experiment_is_not_recorded(self):
        result = result_with_epochs([1.0])
        result.status = "failed"

        self.assertIsNone(measurement_from_result(plan(), profile(), result))

    def test_zero_epoch_result_is_not_recorded(self):
        result = result_with_epochs([], epochs=0)

        self.assertIsNone(measurement_from_result(plan(), profile(), result))


class CachePersistenceTests(unittest.TestCase):
    def test_round_trip_through_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "throughput_cache.json"
            entry = measurement_from_result(
                plan(), profile(), result_with_epochs([3.0, 9.3])
            )
            self.assertTrue(ThroughputCache(path).record(entry))

            reloaded = ThroughputCache(path)
            self.assertEqual(reloaded.get(entry["key"]), entry)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)

    def test_corrupt_cache_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "throughput_cache.json"
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(ThroughputCache(path).entries(), {})


class MeasuredEstimateTests(unittest.TestCase):
    def _profile_with_measurement(self, **overrides):
        base = profile(**overrides)
        entry = measurement_from_result(
            plan(),
            base,
            result_with_epochs([3.0, 9.3, 7.8]),
        )
        return attach_measured_throughput(base, {entry["key"]: entry})

    def test_hit_replaces_the_analytic_model(self):
        measured_profile = self._profile_with_measurement()

        seconds, source = estimate_image_training_seconds_with_source(
            "image_small_cnn", hyperparameters(epochs=6), measured_profile
        )

        self.assertEqual(source, "measured")
        expected = FIXED_SETUP_SECONDS + 6 * 9.3 * MEASURED_THROUGHPUT_UPLIFT
        self.assertAlmostEqual(seconds, expected, places=3)
        self.assertEqual(
            estimate_image_training_seconds(
                "image_small_cnn",
                hyperparameters(epochs=6),
                measured_profile,
            ),
            seconds,
        )

    def test_hit_works_on_an_uncalibrated_machine(self):
        measured_profile = self._profile_with_measurement(os_family="linux")

        seconds, source = estimate_image_training_seconds_with_source(
            "image_small_cnn", hyperparameters(epochs=6), measured_profile
        )

        self.assertEqual(source, "measured")
        self.assertGreater(seconds, 0.0)

    def test_same_machine_without_measurement_uses_the_model(self):
        seconds, source = estimate_image_training_seconds_with_source(
            "image_small_cnn", hyperparameters(epochs=6), profile()
        )

        self.assertEqual(source, "model")
        self.assertIsNotNone(seconds)

    def test_different_dataset_shape_falls_back(self):
        measured_profile = self._profile_with_measurement()
        measured_profile["train_samples"] = 50000

        seconds, source = estimate_image_training_seconds_with_source(
            "image_small_cnn", hyperparameters(epochs=6), measured_profile
        )

        self.assertEqual(source, "model")
        self.assertIsNotNone(seconds)

    def test_different_image_size_falls_back_to_none_on_foreign_os(self):
        measured_profile = self._profile_with_measurement(os_family="linux")

        seconds, source = estimate_image_training_seconds_with_source(
            "image_small_cnn", hyperparameters(size=128, epochs=6),
            measured_profile,
        )

        self.assertIsNone(seconds)
        self.assertIsNone(source)

    def test_validator_reports_the_measurement_as_the_source(self):
        # 9.3s/epoch measured, 20 epochs -> 216s > 0.8 * 200s.
        measured_profile = self._profile_with_measurement(timeout=200)

        with self.assertRaises(PlanValidationError) as captured:
            PlanValidator().validate(
                plan=plan(epochs=20),
                data_profile=measured_profile,
            )

        message = str(captured.exception)
        self.assertIn("measured run on this data and machine", message)
        self.assertIn("image_small_cnn@32px", message)


class KeyTests(unittest.TestCase):
    def test_key_covers_dataset_and_machine(self):
        base = profile()
        first = measured_throughput_key("image_small_cnn", hyperparameters(), base)
        second = measured_throughput_key(
            "image_small_cnn", hyperparameters(), profile(train=1000)
        )
        third = measured_throughput_key(
            "image_small_cnn", hyperparameters(), profile(os_family="linux")
        )

        self.assertIsNotNone(first)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_key_rejects_incomplete_hyperparameters(self):
        self.assertIsNone(
            measured_throughput_key("image_small_cnn", {}, profile())
        )
        self.assertIsNone(
            measured_throughput_key(
                "image_small_cnn", {"image_size": "medium"}, profile()
            )
        )


if __name__ == "__main__":
    unittest.main()
