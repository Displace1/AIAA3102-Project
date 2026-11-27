import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from adapters.base import ExperimentResult
from superagent.error_classifier import classify_exception
from superagent.experiment_plan import ExperimentPlan
from superagent.experiment_registry import ExperimentRegistry
from superagent.llm.client import LLMClient, LLMClientError
from superagent.plan_validator import PlanValidator
from superagent.result_analyzer import ResultAnalyzer


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(
                f"HTTP {self.status_code}", response=response
            )

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


class QualityExtensionTests(unittest.TestCase):
    def test_result_analyzer_detects_imbalance_without_gap_flag(self):
        """A large generalization gap must NOT raise an overfitting flag.

        The gap heuristic was removed after measurement showed it was
        direction-blind (see ResultAnalyzer's docstring). Class imbalance
        detection is unaffected and must still fire.
        """
        result = ExperimentResult(
            experiment_id="exp_001",
            model_name="model",
            status="success",
            validation_accuracy=0.70,
            training_time_seconds=1.0,
            artifact_path="model.joblib",
            train_accuracy=0.90,
            generalization_gap=0.20,
            class_recall={"0": 0.90, "1": 0.60},
        )
        diagnosis = ResultAnalyzer().analyze(result)
        self.assertNotIn("possible_overfitting", diagnosis)
        self.assertIn("class_recall_imbalance", diagnosis)

    def test_large_gap_alone_yields_no_warning(self):
        """gap=0.30 with balanced recalls and a healthy margin is not flagged."""
        result = ExperimentResult(
            experiment_id="exp_002",
            model_name="model",
            status="success",
            validation_accuracy=0.80,
            training_time_seconds=1.0,
            artifact_path="model.joblib",
            train_accuracy=1.00,
            generalization_gap=0.30,
            class_recall={"0": 0.79, "1": 0.81},
            predicted_class_distribution={"0": 40, "1": 40},
            majority_class_accuracy=0.55,
        )
        diagnosis = ResultAnalyzer().analyze(result)
        self.assertEqual(diagnosis, ["no_obvious_metric_warning"])

    def test_error_classifier_uses_stable_categories(self):
        self.assertEqual(classify_exception(MemoryError()), "out_of_memory")
        self.assertEqual(
            classify_exception(FileNotFoundError()), "data_file_error"
        )
        self.assertEqual(
            classify_exception(ValueError("feature shape mismatch")),
            "data_validation_error",
        )
        self.assertEqual(
            classify_exception(RuntimeError("CUDA out of memory")),
            "gpu_out_of_memory",
        )

    def test_new_adapters_are_allow_listed(self):
        registry = ExperimentRegistry(random_seed=42)
        self.assertEqual(
            registry.adapter_for("tabular_random_forest").adapter_name,
            "tabular_random_forest",
        )
        self.assertEqual(
            registry.adapter_for("text_pair_char_tfidf_svm").adapter_name,
            "text_pair_char_tfidf_svm",
        )
        plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="tabular_random_forest",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 12,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            reason="Test a nonlinear bagged tree candidate.",
        )
        PlanValidator().validate(plan, {"modality": "tabular"})

    def test_llm_client_retries_429_but_not_400(self):
        success = {
            "id": "response-1",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"TEST_API_KEY": "fake"}
        ):
            client = LLMClient(
                endpoint="https://example.invalid/chat",
                model="fake-model",
                log_path=Path(directory) / "calls.jsonl",
                api_key_environment_variable="TEST_API_KEY",
                max_retries=3,
                retry_base_seconds=0,
            )
            retry_session = FakeSession(
                [FakeResponse(429), FakeResponse(200, success)]
            )
            client.session = retry_session
            self.assertEqual(
                client.chat([], purpose="test")["content"], "ok"
            )
            self.assertEqual(retry_session.calls, 2)

            bad_request_session = FakeSession([FakeResponse(400)])
            client.session = bad_request_session
            with self.assertRaises(LLMClientError):
                client.chat([], purpose="test")
            self.assertEqual(bad_request_session.calls, 1)


if __name__ == "__main__":
    unittest.main()
