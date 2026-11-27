import json
import unittest

from superagent.experiment_plan import ExperimentPlan
from superagent.llm.experiment_planner import (
    LLMExperimentPlanner,
    PlannerResponseError,
)
from superagent.plan_validator import PlanValidationError


class FakeLLMClient:
    def __init__(self, content):
        self.content = content
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
        return {
            "call_id": "fake-call-id",
            "content": self.content,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
            },
            "cost": {"total_cost_rmb": "0.00022000"},
        }


class ExperimentPlannerTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "modality": "tabular",
            "task_type": "binary_classification",
            "train_samples": 8117,
            "validation_samples": 2030,
            "n_features": 21,
            # macro_f1 keeps class_weight="balanced" selectable: the validator
            # only offers that option for metrics that average over classes.
            "metric": "macro_f1",
        }
        self.existing_plans = [
            ExperimentPlan(
                experiment_id="exp_001",
                adapter_name="tabular_one_hot_logistic",
                hyperparameters={"max_iter": 2000, "C": 1.0},
                reason="Baseline.",
            ),
            ExperimentPlan(
                experiment_id="exp_002",
                adapter_name="tabular_extra_trees",
                hyperparameters={
                    "n_estimators": 400,
                    "max_features": "sqrt",
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                },
                reason="Nonlinear baseline.",
            ),
        ]
        self.history = [
            {
                "experiment_id": "exp_001",
                "model_name": "one_hot_logistic_regression",
                "status": "success",
                "validation_accuracy": 0.664039,
                "training_time_seconds": 0.5,
                "model_config": {"max_iter": 2000, "C": 1.0},
            },
            {
                "experiment_id": "exp_002",
                "model_name": "one_hot_extra_trees",
                "status": "success",
                "validation_accuracy": 0.756158,
                "training_time_seconds": 1.2,
                "model_config": {
                    "n_estimators": 400,
                    "min_samples_leaf": 2,
                },
            },
        ]

    def call_planner(self, content):
        client = FakeLLMClient(content)
        planner = LLMExperimentPlanner(client)
        decision = planner.plan_next(
            data_profile=self.profile,
            experiment_history=self.history,
            existing_plans=self.existing_plans,
            remaining_budget_rmb="17.25",
            next_experiment_id="exp_003",
        )
        return decision, client

    def test_accepts_valid_experiment(self):
        response = {
            "action": "run_experiment",
            "adapter_name": "tabular_extra_trees",
            "hyperparameters": {
                "n_estimators": 600,
                "max_features": "sqrt",
                "min_samples_leaf": 3,
                "class_weight": "balanced",
            },
            "reason": "Try stronger regularization with more trees.",
        }
        decision, client = self.call_planner(json.dumps(response))

        self.assertEqual(decision.action, "run_experiment")
        self.assertEqual(decision.plan.experiment_id, "exp_003")
        self.assertEqual(decision.plan.hyperparameters["n_estimators"], 600)
        self.assertEqual(decision.call_id, "fake-call-id")
        self.assertEqual(client.calls[0]["purpose"], "exploration")

    def test_accepts_stop_decision(self):
        decision, _ = self.call_planner(
            json.dumps(
                {
                    "action": "stop",
                    "reason": "Further exploration is not justified.",
                }
            )
        )
        self.assertEqual(decision.action, "stop")
        self.assertIsNone(decision.plan)

    def test_parses_json_code_fence(self):
        content = """```json
        {
          "action": "stop",
          "reason": "Budget should be preserved."
        }
        ```"""
        decision, _ = self.call_planner(content)
        self.assertEqual(decision.action, "stop")

    def test_rejects_duplicate_configuration(self):
        duplicate = {
            "action": "run_experiment",
            "adapter_name": "tabular_extra_trees",
            "hyperparameters": {
                "n_estimators": 400,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            "reason": "Repeat the best experiment.",
        }
        with self.assertRaises(PlanValidationError):
            self.call_planner(json.dumps(duplicate))

    def test_rejects_invalid_parameter(self):
        invalid = {
            "action": "run_experiment",
            "adapter_name": "tabular_extra_trees",
            "hyperparameters": {
                "n_estimators": 50000,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            "reason": "Use too many trees.",
        }
        with self.assertRaises(PlanValidationError):
            self.call_planner(json.dumps(invalid))

    def test_rejects_unknown_response_key(self):
        response = {
            "action": "stop",
            "reason": "Stop.",
            "test_accuracy": 0.99,
        }
        with self.assertRaises(PlannerResponseError):
            self.call_planner(json.dumps(response))

    def test_prompt_contains_budget_history_and_no_task_identity(self):
        _, client = self.call_planner(
            json.dumps(
                {
                    "action": "stop",
                    "reason": "No additional experiment is needed.",
                }
            )
        )
        prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn('"remaining_budget_rmb": "17.25"', prompt)
        self.assertIn('"experiment_id": "exp_002"', prompt)
        self.assertIn("tabular_extra_trees", prompt)
        self.assertNotIn("released_task_1", prompt)

    def test_image_history_exposes_metrics_and_oom_requires_reduction(self):
        image_profile = {
            "modality": "image",
            "task_type": "multiclass_classification",
            "number_of_classes": 3,
        }
        failed_plan = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="image_resnet18",
            hyperparameters={
                "image_size": 224,
                "batch_size": 32,
                "learning_rate": 0.0003,
                "epochs": 10,
                "data_augmentation": True,
                "early_stopping_patience": 3,
                "weight_decay": 0.0001,
                "pretrained": True,
                "freeze_backbone_epochs": 2,
            },
            reason="Initial transfer-learning experiment.",
        )
        history = [
            {
                "experiment_id": "exp_001",
                "model_name": "resnet18_classifier",
                "status": "failed",
                "train_accuracy": 0.90,
                "validation_accuracy": 0.72,
                "f1_macro": 0.70,
                "generalization_gap": 0.18,
                "epoch_count": 2,
                "best_epoch": 1,
                "peak_gpu_memory_mb": 7900.0,
                "training_time_seconds": 20.0,
                "early_stopped": False,
                "error_type": "gpu_out_of_memory",
                "error_message": "CUDA out of memory",
            }
        ]
        response = {
            "action": "run_experiment",
            "adapter_name": "image_resnet18",
            "hyperparameters": {
                **failed_plan.hyperparameters,
                "batch_size": 16,
            },
            "reason": "Reduce batch size after GPU OOM.",
        }
        client = FakeLLMClient(json.dumps(response))
        decision = LLMExperimentPlanner(client).plan_next(
            data_profile=image_profile,
            experiment_history=history,
            existing_plans=[failed_plan],
            remaining_budget_rmb="18.0",
            next_experiment_id="exp_002",
        )

        self.assertEqual(decision.plan.hyperparameters["batch_size"], 16)
        system_prompt = client.calls[0]["messages"][0]["content"]
        user_prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn("gpu_out_of_memory", system_prompt)
        for field in (
            "train_accuracy",
            "validation_accuracy",
            "macro_f1",
            "generalization_gap",
            "epoch_count",
            "best_epoch",
            "peak_gpu_memory_mb",
            "training_time_seconds",
            "training_time",
            "early_stopped",
        ):
            self.assertIn(f'"{field}"', user_prompt)

    def test_rejects_image_plan_that_does_not_reduce_after_gpu_oom(self):
        image_profile = {"modality": "image"}
        failed_plan = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="image_small_cnn",
            hyperparameters={
                "image_size": 128,
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10,
                "data_augmentation": True,
                "early_stopping_patience": 3,
                "weight_decay": 0.0001,
            },
            reason="Initial CNN experiment.",
        )
        response = {
            "action": "run_experiment",
            "adapter_name": "image_small_cnn",
            "hyperparameters": {
                **failed_plan.hyperparameters,
                "learning_rate": 0.0005,
            },
            "reason": "Only change learning rate.",
        }
        client = FakeLLMClient(json.dumps(response))
        with self.assertRaisesRegex(
            PlannerResponseError, "reduce batch_size"
        ):
            LLMExperimentPlanner(client).plan_next(
                data_profile=image_profile,
                experiment_history=[
                    {
                        "experiment_id": "exp_001",
                        "status": "failed",
                        "error_type": "gpu_out_of_memory",
                    }
                ],
                existing_plans=[failed_plan],
                remaining_budget_rmb="18.0",
                next_experiment_id="exp_002",
            )


if __name__ == "__main__":
    unittest.main()
