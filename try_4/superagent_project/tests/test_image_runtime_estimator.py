"""Tests for the calibrated image runtime estimator and the gate built on it.

The numeric expectations come from real runs recorded under
``test_tasks/results/`` (P1-C, Fashion-MNIST, 5000 train / 1000 validation):
training_time_seconds of 262.8s (small_cnn, 15 epochs), 215.4s (mobilenet,
8 epochs) and 199.1s (small_cnn, 11 epochs after early stopping).
"""

import json
import unittest

from superagent.experiment_plan import ExperimentPlan
from superagent.image_runtime import (
    estimate_image_training_seconds,
    recommended_image_size,
)
from superagent.llm.experiment_planner import LLMExperimentPlanner
from superagent.plan_validator import PlanValidationError, PlanValidator


def profile(train=5000, validation=1000, timeout=1800, cuda=False):
    return {
        "modality": "image",
        "task_type": "multiclass_classification",
        "train_samples": train,
        "validation_samples": validation,
        "class_count": 10,
        "metric": "accuracy",
        "image_summary": {
            "train": {"median_width": 28, "median_height": 28},
            "validation": {"median_width": 28, "median_height": 28},
        },
        "execution_context": {
            "experiment_timeout_seconds": timeout,
            "maximum_cpu_threads": 2,
            "cuda_available": cuda,
        },
    }


def hyperparameters(size, epochs, batch=32):
    values = {
        "image_size": size,
        "batch_size": batch,
        "learning_rate": 0.001,
        "epochs": epochs,
        "data_augmentation": True,
        "early_stopping_patience": 4,
        "weight_decay": 0.0001,
    }
    return values


def pretrained_hyperparameters(size, epochs, batch=32):
    values = hyperparameters(size, epochs, batch)
    values["pretrained"] = True
    values["freeze_backbone_epochs"] = 0
    return values


def plan(adapter, size, epochs, batch=32):
    if adapter == "image_small_cnn":
        values = hyperparameters(size, epochs, batch)
    else:
        values = pretrained_hyperparameters(size, epochs, batch)
    return ExperimentPlan(
        experiment_id="exp_009",
        adapter_name=adapter,
        hyperparameters=values,
        reason="Estimator test plan.",
    )


class ImageRuntimeEstimatorTests(unittest.TestCase):
    def test_matches_measured_small_cnn_runs(self):
        # Predicted/measured must sit just above 1.0: optimistic estimates let
        # doomed plans through, pessimistic ones only cost a retry.
        for epochs, measured in ((15, 262.8), (11, 199.1)):
            estimated = estimate_image_training_seconds(
                "image_small_cnn",
                {"image_size": 32, "epochs": epochs, "batch_size": 32},
                profile(),
            )
            ratio = estimated / measured
            self.assertGreaterEqual(ratio, 1.0, (epochs, estimated))
            self.assertLessEqual(ratio, 1.15, (epochs, estimated))

    def test_matches_measured_mobilenet_run(self):
        estimated = estimate_image_training_seconds(
            "image_mobilenet_v3_small",
            {"image_size": 32, "epochs": 8, "batch_size": 32},
            profile(),
        )
        self.assertGreaterEqual(estimated / 215.4, 1.0)
        self.assertLessEqual(estimated / 215.4, 1.20)

    def test_resnet18_is_never_underestimated(self):
        # The P1-C run of this configuration was killed at its 600s timeout, so
        # anything at or above 600s is consistent with the evidence. At the
        # default 1800s timeout it should still be accepted, because it would
        # finish in roughly 700-900s.
        estimated = estimate_image_training_seconds(
            "image_resnet18",
            {"image_size": 32, "epochs": 12, "batch_size": 32},
            profile(),
        )
        self.assertGreaterEqual(estimated, 600.0)
        self.assertLess(estimated, 1440.0)

    def test_small_dataset_plan_is_not_over_rejected(self):
        # Regression: modelling the cold-cache cost as a multiplier made this
        # plan cost 356s and rejected it, although 1200 images need about 70s.
        estimated = estimate_image_training_seconds(
            "image_resnet18",
            {"image_size": 32, "epochs": 12, "batch_size": 32},
            profile(train=1200, validation=400, timeout=300),
        )
        self.assertLess(estimated, 240.0)

    def test_monotonic_in_size_and_epochs(self):
        sizes = [
            estimate_image_training_seconds(
                "image_resnet18",
                {"image_size": size, "epochs": 4, "batch_size": 32},
                profile(),
            )
            for size in (32, 64, 96, 128)
        ]
        self.assertEqual(sizes, sorted(sizes))
        epochs = [
            estimate_image_training_seconds(
                "image_resnet18",
                {"image_size": 64, "epochs": count, "batch_size": 32},
                profile(),
            )
            for count in (1, 2, 4, 8)
        ]
        self.assertEqual(epochs, sorted(epochs))

    def test_scales_with_sample_count(self):
        small = estimate_image_training_seconds(
            "image_small_cnn",
            {"image_size": 32, "epochs": 5, "batch_size": 32},
            profile(train=2000, validation=400),
        )
        large = estimate_image_training_seconds(
            "image_small_cnn",
            {"image_size": 32, "epochs": 5, "batch_size": 32},
            profile(train=8000, validation=1600),
        )
        self.assertLess(small, large)

    def test_unknown_adapter_or_device_returns_none(self):
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_not_allow_listed",
                {"image_size": 32, "epochs": 1},
                profile(),
            )
        )
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_small_cnn",
                {"image_size": 32, "epochs": 1},
                profile(cuda=True),
            )
        )

    def test_incomplete_hyperparameters_return_none(self):
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_small_cnn",
                {"image_size": 32},
                profile(),
            )
        )
        self.assertIsNone(
            estimate_image_training_seconds(
                "image_small_cnn",
                {"image_size": 32, "epochs": 1},
                profile(train=0, validation=0),
            )
        )

    def test_recommended_image_size_uses_native_detail(self):
        self.assertEqual(recommended_image_size(profile()), 32)
        wide = profile()
        wide["image_summary"]["train"] = {
            "median_width": 200,
            "median_height": 150,
        }
        self.assertEqual(recommended_image_size(wide), 160)
        tiny = profile()
        tiny["image_summary"]["train"] = {
            "median_width": 8,
            "median_height": 8,
        }
        self.assertEqual(recommended_image_size(tiny), 32)
        huge = profile()
        huge["image_summary"]["train"] = {
            "median_width": 800,
            "median_height": 600,
        }
        self.assertEqual(recommended_image_size(huge), 224)


class ImageRuntimeGateTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator()

    def test_rejects_plan_that_cannot_finish(self):
        with self.assertRaises(PlanValidationError) as context:
            self.validator.validate(
                plan("image_resnet18", 96, 12),
                profile(),
            )
        message = str(context.exception)
        self.assertIn("Estimated image training time", message)
        self.assertIn("exceeds 80%", message)

    def test_rejection_lists_feasible_reference_points(self):
        with self.assertRaises(PlanValidationError) as context:
            self.validator.validate(
                plan("image_resnet18", 224, 20),
                profile(),
            )
        message = str(context.exception)
        self.assertIn("Fitting configurations", message)
        self.assertIn("image_small_cnn@32px", message)

    def test_accepts_plan_within_limit(self):
        validated = self.validator.validate(
            plan("image_small_cnn", 32, 15),
            profile(),
        )
        self.assertEqual(validated.adapter_name, "image_small_cnn")

    def test_gate_scales_with_timeout(self):
        # 297s fits a 1800s timeout but not a 300s one.
        self.validator.validate(plan("image_small_cnn", 32, 15), profile())
        with self.assertRaises(PlanValidationError):
            self.validator.validate(
                plan("image_small_cnn", 32, 15),
                profile(timeout=300),
            )


class FakeLLMClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def chat(self, messages, purpose, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "purpose": purpose,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        content = self.contents.pop(0)
        return {
            "call_id": f"fake-call-{len(self.calls)}",
            "content": content,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
            },
            "cost": {"total_cost_rmb": "0.00022000"},
        }


class PlannerPlanRepairTests(unittest.TestCase):
    def setUp(self):
        self.data_profile = profile()
        self.existing = [
            ExperimentPlan(
                experiment_id="exp_001",
                adapter_name="image_small_cnn",
                hyperparameters=hyperparameters(32, 15),
                reason="Baseline.",
            )
        ]

    @staticmethod
    def _content(adapter, size, epochs):
        values = (
            hyperparameters(size, epochs)
            if adapter == "image_small_cnn"
            else pretrained_hyperparameters(size, epochs)
        )
        return json.dumps(
            {
                "action": "run_experiment",
                "adapter_name": adapter,
                "hyperparameters": values,
                "reason": "Test proposal.",
            }
        )

    def test_rejected_plan_is_retried_with_reason_in_context(self):
        client = FakeLLMClient(
            [
                self._content("image_resnet18", 224, 20),
                self._content("image_small_cnn", 64, 5),
            ]
        )
        planner = LLMExperimentPlanner(
            llm_client=client,
            max_plan_repairs=2,
        )
        decision = planner.plan_next(
            data_profile=self.data_profile,
            experiment_history=[],
            existing_plans=self.existing,
            remaining_budget_rmb="19.9",
            next_experiment_id="exp_002",
        )

        self.assertEqual(decision.action, "run_experiment")
        self.assertEqual(decision.plan.adapter_name, "image_small_cnn")
        self.assertEqual(len(client.calls), 2)

        repair_turn = client.calls[1]["messages"][-1]
        self.assertEqual(repair_turn["role"], "user")
        self.assertIn("rejected", repair_turn["content"])
        self.assertIn("exceeds 80%", repair_turn["content"])
        self.assertIn("image_small_cnn@32px", repair_turn["content"])

    def test_repair_budget_is_bounded(self):
        client = FakeLLMClient(
            [
                self._content("image_resnet18", 224, 20),
                self._content("image_resnet18", 224, 30),
            ]
        )
        planner = LLMExperimentPlanner(
            llm_client=client,
            max_plan_repairs=2,
        )
        with self.assertRaises(PlanValidationError):
            planner.plan_next(
                data_profile=self.data_profile,
                experiment_history=[],
                existing_plans=self.existing,
                remaining_budget_rmb="19.9",
                next_experiment_id="exp_002",
            )
        self.assertEqual(len(client.calls), 2)

    def test_repair_disabled_raises_immediately(self):
        client = FakeLLMClient([self._content("image_resnet18", 224, 20)])
        planner = LLMExperimentPlanner(
            llm_client=client,
            max_plan_repairs=1,
        )
        with self.assertRaises(PlanValidationError):
            planner.plan_next(
                data_profile=self.data_profile,
                experiment_history=[],
                existing_plans=self.existing,
                remaining_budget_rmb="19.9",
                next_experiment_id="exp_002",
            )
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
