"""Tests for the time-budget guard and the runtime fingerprint.

The budget guard exists because the runtime estimator can only ever be an
estimate: on an uncalibrated machine it returns ``None`` and the only
protection left is the executor's hard kill.  These tests pin the behaviour
that makes that acceptable — a worker stops at an epoch boundary and still
returns a usable model instead of being killed with nothing to show.
"""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from adapters.base import ExperimentResult, ModelAdapter
from superagent.executor import ExperimentExecutor
from superagent.image_dataset import build_image_dataloaders
from superagent.image_inspector import ImageInspector
from superagent.image_runtime import estimate_image_training_seconds
from superagent.image_training import train_image_classifier
from superagent.runtime_environment import (
    EXECUTOR_DEADLINE_RESERVE_SECONDS,
    TRAINING_TAIL_RESERVE_SECONDS,
    attach_experiment_deadline,
    describe_runtime,
    os_family,
    remaining_training_budget,
)

TRAIN_IMAGES_PER_CLASS = 4
VALIDATION_IMAGES_PER_CLASS = 2


class TinyImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, inputs):
        return self.network(inputs)


class DeadlineEchoAdapter(ModelAdapter):
    """Reports the deadline the executor published, to prove the plumbing."""

    model_name = "deadline_echo"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        context = data_profile.get("execution_context") or {}
        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=self.model_name,
            status="success",
            validation_accuracy=1.0,
            training_time_seconds=0.0,
            artifact_path=None,
            model_config={"execution_context": context},
        )


def image_profile(train=5000, validation=1000, timeout=1800, **context):
    execution_context = {
        "experiment_timeout_seconds": timeout,
        "maximum_cpu_threads": 2,
        "os_family": os_family(),
        "training_device": "cpu",
        "cuda_available": False,
    }
    execution_context.update(context)
    return {
        "modality": "image",
        "train_samples": train,
        "validation_samples": validation,
        "image_summary": {
            "train": {"median_width": 32, "median_height": 32},
            "validation": {"median_width": 32, "median_height": 32},
        },
        "execution_context": execution_context,
    }


def hyperparameters(size=32, epochs=4, batch=32):
    return {
        "image_size": size,
        "batch_size": batch,
        "learning_rate": 0.001,
        "epochs": epochs,
        "data_augmentation": True,
        "early_stopping_patience": 4,
        "weight_decay": 0.0001,
    }


class TimeBudgetTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train = self.root / "train"
        self.validation = self.root / "validation"
        for split in (self.train, self.validation):
            for class_name in ("dark", "light"):
                (split / class_name).mkdir(parents=True)
        for index in range(TRAIN_IMAGES_PER_CLASS):
            self._image(self.train / "dark" / f"dark_{index}.png", index)
            self._image(
                self.train / "light" / f"light_{index}.png", 255 - index
            )
        for index in range(VALIDATION_IMAGES_PER_CLASS):
            self._image(self.validation / "dark" / f"dark_{index}.png", 0)
            self._image(
                self.validation / "light" / f"light_{index}.png", 255
            )
        self.config = SimpleNamespace(
            random_seed=42,
            maximum_cpu_threads=1,
            metric="accuracy",
            data={
                "train_directory": str(self.train),
                "validation_directory": str(self.validation),
                "test_directory": str(self.validation),
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _image(path, value):
        Image.new("RGB", (16, 16), color=(value, value, value)).save(path)

    def _loaders(self):
        profile = ImageInspector().inspect(self.config)
        return build_image_dataloaders(
            config=self.config,
            data_profile=profile,
            image_size=16,
            batch_size=4,
            data_augmentation=False,
            num_workers=0,
        )

    def _train(self, epochs=4, time_budget_seconds=None, name="output",
               early_stopping_patience=None):
        return train_image_classifier(
            model=TinyImageClassifier(),
            data_loaders=self._loaders(),
            output_directory=self.root / name,
            number_of_classes=2,
            learning_rate=0.01,
            epochs=epochs,
            weight_decay=0.0,
            early_stopping_patience=(
                epochs
                if early_stopping_patience is None
                else early_stopping_patience
            ),
            random_seed=42,
            requested_device="cpu",
            use_mixed_precision=False,
            time_budget_seconds=time_budget_seconds,
        )

    def test_exhausted_budget_stops_after_one_epoch_with_weights(self):
        result = self._train(
            epochs=10, time_budget_seconds=0.0, name="budget_stop"
        )

        self.assertEqual(result.epochs_completed, 1)
        self.assertEqual(result.stopped_reason, "time_budget")
        self.assertFalse(result.early_stopped)
        self.assertEqual(result.time_budget_seconds, 0.0)
        self.assertTrue(Path(result.best_weights_path).is_file())
        self.assertGreater(result.validation_accuracy, 0.0)

    def test_generous_budget_completes_every_epoch(self):
        result = self._train(
            epochs=3, time_budget_seconds=600.0, name="budget_generous"
        )

        self.assertEqual(result.epochs_completed, 3)
        self.assertEqual(result.stopped_reason, "completed")

    def test_unbudgeted_run_keeps_historical_behaviour(self):
        result = self._train(epochs=2, name="budget_absent")

        self.assertEqual(result.epochs_completed, 2)
        self.assertEqual(result.stopped_reason, "completed")
        self.assertIsNone(result.time_budget_seconds)

    def test_early_stopping_reason_wins_over_budget(self):
        fixed_metrics = {
            "loss": 0.5,
            "accuracy": 0.75,
            "precision_macro": 0.75,
            "recall_macro": 0.75,
            "f1_macro": 0.75,
            "confusion_matrix": [[1, 1], [0, 2]],
            "class_recall": {"0": 0.5, "1": 1.0},
        }
        with patch(
            "superagent.image_training.evaluate_image_model",
            return_value=fixed_metrics,
        ):
            result = self._train(
                epochs=5,
                time_budget_seconds=600.0,
                name="early_stopping_reason",
                early_stopping_patience=1,
            )

        self.assertTrue(result.early_stopped)
        self.assertEqual(result.stopped_reason, "early_stopping")
        self.assertEqual(result.epochs_completed, 2)

    def test_every_epoch_records_its_wall_clock(self):
        result = self._train(epochs=2, name="epoch_seconds")

        self.assertEqual(len(result.history), 2)
        for record in result.history:
            self.assertGreaterEqual(record["epoch_seconds"], 0.0)


class BudgetPlumbingTests(unittest.TestCase):
    def test_remaining_budget_subtracts_tail_reserve(self):
        now = 1000.0
        deadline = now + 100.0

        self.assertEqual(
            remaining_training_budget(deadline, now=now),
            100.0 - TRAINING_TAIL_RESERVE_SECONDS,
        )
        self.assertEqual(
            remaining_training_budget(deadline, now=now, tail_reserve=0.0),
            100.0,
        )
        self.assertIsNone(remaining_training_budget(None, now=now))

    def test_past_deadline_yields_zero_not_negative(self):
        self.assertEqual(
            remaining_training_budget(1000.0, now=2000.0),
            0.0,
        )

    def test_attach_deadline_copies_the_profile(self):
        original = {"execution_context": {"os_family": "windows"}}
        before = time.time()

        updated = attach_experiment_deadline(original, 600)

        deadline = updated["execution_context"]["wall_clock_deadline"]
        self.assertAlmostEqual(
            deadline,
            before + 600 - EXECUTOR_DEADLINE_RESERVE_SECONDS,
            delta=2.0,
        )
        self.assertEqual(
            updated["execution_context"]["experiment_timeout_seconds"], 600
        )
        self.assertNotIn("wall_clock_deadline", original["execution_context"])
        self.assertEqual(updated["execution_context"]["os_family"], "windows")

    def test_executor_publishes_deadline_to_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                experiment_timeout_seconds=30,
                maximum_memory_mb=1024,
                maximum_cpu_threads=1,
                resource_poll_interval_seconds=0.02,
            )
            result = ExperimentExecutor().execute(
                experiment_id="deadline_test",
                adapter=DeadlineEchoAdapter(),
                plan=SimpleNamespace(hyperparameters={}),
                config=config,
                data_profile={"execution_context": {"os_family": "windows"}},
                experiment_dir=Path(directory),
            )

        self.assertEqual(result.status, "success")
        context = result.model_config["execution_context"]
        self.assertIn("wall_clock_deadline", context)
        # 30s timeout minus the executor reserve, minus whatever the worker
        # spent starting up; assert the window rather than an exact value.
        remaining = remaining_training_budget(
            context["wall_clock_deadline"], tail_reserve=0.0
        )
        self.assertGreater(remaining, 0.0)
        self.assertLessEqual(remaining, 30.0)


class RuntimeFingerprintTests(unittest.TestCase):
    def test_describe_runtime_exposes_the_gate_keys(self):
        runtime = describe_runtime()

        for key in (
            "os_family",
            "training_device",
            "os_family",
            "cuda_available",
            "npu_available",
            "torch_threads",
            "cpu_count",
        ):
            self.assertIn(key, runtime)
        self.assertEqual(runtime["os_family"], os_family())
        self.assertIn(runtime["training_device"], ("cpu", "cuda"))
        self.assertGreaterEqual(runtime["torch_threads"], 1)

    def test_estimator_serves_the_calibrated_machine(self):
        estimate = estimate_image_training_seconds(
            "image_small_cnn",
            hyperparameters(size=32, epochs=3),
            image_profile(os_family="windows", training_device="cpu"),
        )

        self.assertIsNotNone(estimate)
        self.assertGreater(estimate, 0.0)

    def test_estimator_refuses_a_foreign_os(self):
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_small_cnn",
                hyperparameters(),
                image_profile(os_family="linux"),
            )
        )

    def test_estimator_refuses_a_foreign_device(self):
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_small_cnn",
                hyperparameters(),
                image_profile(
                    os_family="windows",
                    training_device="cuda",
                    cuda_available=True,
                ),
            )
        )

    def test_inspector_records_the_fingerprint_in_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation"):
                for class_name in ("dark", "light"):
                    (root / split / class_name).mkdir(parents=True)
                    Image.new(
                        "RGB", (8, 8), color=(1, 1, 1)
                    ).save(root / split / class_name / "a.png")
            config = SimpleNamespace(
                random_seed=42,
                maximum_cpu_threads=2,
                metric="accuracy",
                data={
                    "train_directory": str(root / "train"),
                    "validation_directory": str(root / "validation"),
                    "test_directory": str(root / "validation"),
                },
            )
            profile = ImageInspector().inspect(config)

        context = profile["execution_context"]
        self.assertEqual(context["os_family"], os_family())
        self.assertIn(context["training_device"], ("cpu", "cuda"))
        self.assertIn("cuda_available", context)


if __name__ == "__main__":
    unittest.main()
