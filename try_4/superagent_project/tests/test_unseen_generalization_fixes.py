import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from adapters.base import ExperimentResult
from adapters.tabular_baseline import TabularBaselineAdapter
from adapters.text_pair_baseline import TextWordTfidfAdapter
from superagent.controller import SuperAgentController
from superagent.data_inspector import DataInspector
from superagent.experiment_plan import ExperimentPlan
from superagent.llm.experiment_planner import LLMExperimentPlanner
from superagent.final_evaluator import FinalEvaluator
from superagent.predictor import ArtifactPredictor
from superagent.model_selector import select_best_model
from superagent.plan_validator import PlanValidationError, PlanValidator
from superagent.result_analyzer import ResultAnalyzer
from superagent.tabular_data import build_tabular_preprocessor
from superagent.task_config import load_task_config


class SequenceLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, purpose, temperature, max_tokens):
        self.calls.append(max_tokens)
        response = self.responses.pop(0)
        return {
            "call_id": f"call-{len(self.calls)}",
            "content": response.get("content", ""),
            "finish_reason": response.get("finish_reason"),
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": max_tokens + 10,
            },
            "cost": {"total_cost_rmb": "0.001"},
        }


class UnseenGeneralizationFixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_npy_config(self, metric="accuracy", planner_mode="deterministic"):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 4))
        labels = (features[:, 0] > 0).astype(int)
        paths = {}
        for name, value in {
            "train_features": features[:24],
            "train_labels": labels[:24],
            "validation_features": features[24:32],
            "validation_labels": labels[24:32],
            "test_features": features[32:],
            "test_labels": labels[32:],
        }.items():
            path = self.root / f"{name}.npy"
            np.save(path, value)
            paths[name] = str(path)
        config = {
            "run_type": "development_run",
            "task": {"name": "fix_probe", "metric": metric},
            "data": {"format": "npy_split", **paths},
            "execution": {
                "planner_mode": planner_mode,
                "initial_deterministic_experiments": 1,
                "max_experiments": 1,
                "experiment_timeout_seconds": 30,
                "maximum_memory_mb": 1024,
                "maximum_cpu_threads": 1,
            },
        }
        path = self.root / f"config_{metric}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _write_csv(self, name, fieldnames, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_task_config_accepts_all_supported_metrics(self):
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            with self.subTest(metric=metric):
                self.assertEqual(
                    load_task_config(self._write_npy_config(metric)).metric,
                    metric,
                )

    def test_model_selector_uses_configured_metric(self):
        accurate = ExperimentResult(
            "exp_001", "accurate", "success", 0.95, 2.0, "a.joblib",
            f1_macro=0.50, balanced_accuracy=0.55,
        )
        balanced = ExperimentResult(
            "exp_002", "balanced", "success", 0.90, 2.0, "b.joblib",
            f1_macro=0.80, balanced_accuracy=0.82,
        )
        self.assertIs(select_best_model([accurate, balanced]), accurate)
        self.assertIs(
            select_best_model([accurate, balanced], metric="macro_f1"),
            balanced,
        )
        self.assertIs(
            select_best_model(
                [accurate, balanced], metric="balanced_accuracy"
            ),
            balanced,
        )

    def test_macro_f1_flows_through_lock_and_final_evaluation(self):
        config_path = self._write_npy_config(metric="macro_f1")
        config = load_task_config(config_path)
        summary = SuperAgentController(config_path, self.root / "metric_runs").run()
        self.assertEqual(summary["selection_metric"], "macro_f1")
        self.assertIsNotNone(summary["validation_metric_value"])
        run_directory = Path(summary["run_directory"])
        lock = json.loads(
            (run_directory / "solution_lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(lock["selection_metric"], "macro_f1")
        final = FinalEvaluator().evaluate(run_directory, config)
        metrics = json.loads(Path(final.metrics_path).read_text(encoding="utf-8"))
        self.assertEqual(final.selection_metric, "macro_f1")
        self.assertAlmostEqual(final.test_metric_value, metrics["f1_macro"])

    def test_planner_recovers_from_length_truncated_empty_response(self):
        client = SequenceLLMClient(
            [
                {"content": "", "finish_reason": "length"},
                {
                    "content": json.dumps(
                        {"action": "stop", "reason": "Enough evidence."}
                    ),
                    "finish_reason": "stop",
                },
            ]
        )
        planner = LLMExperimentPlanner(
            client, max_tokens=100, max_response_attempts=3
        )
        decision = planner.plan_next(
            data_profile={"modality": "tabular", "metric": "accuracy"},
            experiment_history=[],
            existing_plans=[],
            remaining_budget_rmb="20",
            next_experiment_id="exp_001",
        )
        self.assertEqual(decision.action, "stop")
        self.assertEqual(client.calls, [100, 2048])

    def test_all_failed_run_writes_structured_partial_result(self):
        config_path = self._write_npy_config()

        def fail_execute(**kwargs):
            return ExperimentResult(
                experiment_id=kwargs["experiment_id"],
                model_name=kwargs["adapter"].model_name,
                status="failed",
                validation_accuracy=None,
                training_time_seconds=0.01,
                artifact_path=None,
                error_type="experiment_timeout",
                raw_error_type="TimeoutError",
                error_message="injected timeout",
            )

        runs = self.root / "runs"
        with patch(
            "superagent.controller.ExperimentExecutor.execute",
            side_effect=fail_execute,
        ):
            summary = SuperAgentController(config_path, runs).run()
        run_directory = Path(summary["run_directory"])
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stop_reason"], "all_experiments_failed")
        self.assertTrue((run_directory / "summary.json").is_file())
        self.assertTrue((run_directory / "stop_reason.json").is_file())
        self.assertTrue((run_directory / "final").is_dir())
        self.assertFalse((run_directory / "solution_lock.json").exists())
        state = json.loads((run_directory / "run_state.json").read_text())
        self.assertEqual(state["status"], "failed")

    def test_wine_numeric_features_keep_compact_representation(self):
        features, labels = load_wine(return_X_y=True)
        train_x, val_x, train_y, val_y = train_test_split(
            features,
            labels,
            test_size=0.30,
            random_state=42,
            stratify=labels,
        )
        paths = {}
        for name, value in {
            "train_features": train_x,
            "train_labels": train_y,
            "validation_features": val_x,
            "validation_labels": val_y,
        }.items():
            path = self.root / f"wine_{name}.npy"
            np.save(path, value)
            paths[name] = str(path)
        config = SimpleNamespace(
            data={"format": "npy_split", **paths},
            metric="accuracy",
            maximum_cpu_threads=1,
        )
        profile = DataInspector().inspect(config)
        wine_output = self.root / "wine_output"
        wine_output.mkdir()
        result = TabularBaselineAdapter(42).run(
            "exp_001",
            config,
            profile,
            wine_output,
            {
                "C": 1.0,
                "max_iter": 3000,
                "numeric_preprocessing": "standard",
                "categorical_preprocessing": "onehot",
                "class_weight": None,
            },
        )
        self.assertEqual(profile["numeric_feature_count"], 13)
        self.assertLessEqual(result.feature_count, 20)
        self.assertGreaterEqual(result.validation_accuracy, 0.90)
        self.assertGreater(result.training_time_seconds, 0.0)

    def test_tree_one_hot_preprocessor_stays_sparse(self):
        rows = 200
        data = np.asarray(
            [[float(index), f"category_{index}"] for index in range(rows)],
            dtype=object,
        )
        profile = {
            "numeric_column_indices": [0],
            "categorical_column_indices": [1],
        }
        preprocessor = build_tabular_preprocessor(
            profile,
            numeric_preprocessing="none",
            categorical_preprocessing="onehot",
        )
        transformed = preprocessor.fit_transform(data)
        self.assertTrue(sparse.issparse(transformed))
        self.assertLess(transformed.nnz, rows * 3)

    def test_csv_contract_supports_tabular_and_single_text(self):
        tabular_rows = [
            {"age": str(20 + index), "group": "a" if index % 2 else "b", "y": str(index % 2)}
            for index in range(30)
        ]
        text_rows = [
            {
                "headline": f"This is a sufficiently descriptive headline number {index}",
                "y": str(index % 2),
            }
            for index in range(30)
        ]
        tabular_paths = [
            self._write_csv(f"tabular_{split}.csv", ["age", "group", "y"], tabular_rows)
            for split in ("train", "validation", "test")
        ]
        text_paths = [
            self._write_csv(f"text_{split}.csv", ["headline", "y"], text_rows)
            for split in ("train", "validation", "test")
        ]
        tabular_config = SimpleNamespace(
            data={
                "format": "csv_split",
                "train": str(tabular_paths[0]),
                "validation": str(tabular_paths[1]),
                "test": str(tabular_paths[2]),
                "target_column": "y",
                "columns": {
                    "numeric": ["age"],
                    "categorical": ["group"],
                },
            },
            metric="accuracy",
        )
        text_config = SimpleNamespace(
            data={
                "format": "csv_split",
                "train": str(text_paths[0]),
                "validation": str(text_paths[1]),
                "test": str(text_paths[2]),
                "target_column": "y",
                "columns": {"text": ["headline"]},
            },
            metric="accuracy",
        )
        tabular_profile = DataInspector().inspect(tabular_config)
        text_profile = DataInspector().inspect(text_config)
        self.assertEqual(tabular_profile["modality"], "tabular")
        self.assertEqual(text_profile["modality"], "text")
        text_output = self.root / "text_output"
        text_output.mkdir()
        result = TextWordTfidfAdapter(42).run(
            "exp_001",
            text_config,
            text_profile,
            text_output,
            {
                "ngram_min": 1,
                "ngram_max": 2,
                "min_df": 1,
                "max_df": 1.0,
                "max_features": 1000,
                "sublinear_tf": True,
                "C": 1.0,
                "max_iter": 1000,
            },
        )
        self.assertEqual(result.status, "success")

        tabular_output = self.root / "tabular_output"
        tabular_output.mkdir()
        tabular_config.maximum_cpu_threads = 1
        tabular_result = TabularBaselineAdapter(42).run(
            "exp_002",
            tabular_config,
            tabular_profile,
            tabular_output,
            {
                "C": 1.0,
                "max_iter": 1000,
                "numeric_preprocessing": "standard",
                "categorical_preprocessing": "onehot",
                "class_weight": None,
            },
        )
        predictions = ArtifactPredictor().predict_input(
            tabular_result.artifact_path,
            "tabular_one_hot_logistic",
            tabular_paths[2],
        )
        self.assertEqual(len(predictions), len(tabular_rows))

    def test_repeat_error_key_is_scoped_to_adapter(self):
        first = ExperimentResult(
            "exp_001", "small_cnn", "failed", None, 1.0, None,
            error_type="experiment_timeout",
        )
        second = ExperimentResult(
            "exp_002", "resnet18", "failed", None, 1.0, None,
            error_type="experiment_timeout",
        )
        plans = [
            ExperimentPlan("exp_001", "image_small_cnn", {}, "first"),
            ExperimentPlan("exp_002", "image_resnet18", {}, "second"),
        ]
        first_key = SuperAgentController._experiment_error_key(first, plans)
        second_key = SuperAgentController._experiment_error_key(second, plans)
        self.assertNotEqual(first_key, second_key)
        self.assertEqual(first_key[1], second_key[1])

    def test_degenerate_model_is_explicitly_diagnosed(self):
        result = ExperimentResult(
            "exp_001",
            "majority_predictor",
            "success",
            0.992,
            1.0,
            "model.joblib",
            class_recall={"0": 0.999, "1": 0.05},
            predicted_class_distribution={"0": 999, "1": 1},
            majority_class_accuracy=0.991,
        )
        diagnosis = ResultAnalyzer().analyze(result)
        self.assertIn("degenerate_model", diagnosis)
        self.assertIn("near_majority_baseline", diagnosis)

    def test_image_plan_over_budget_is_rejected(self):
        profile = {
            "modality": "image",
            "train_samples": 5000,
            "execution_context": {
                "cuda_available": False,
                "experiment_timeout_seconds": 120,
            },
        }
        plan = ExperimentPlan(
            "exp_001",
            "image_resnet18",
            {
                "image_size": 224,
                "batch_size": 32,
                "learning_rate": 0.0003,
                "epochs": 12,
                "data_augmentation": True,
                "early_stopping_patience": 3,
                "weight_decay": 0.0001,
                "pretrained": False,
                "freeze_backbone_epochs": 0,
            },
            "Budget probe.",
        )
        with self.assertRaisesRegex(PlanValidationError, "Estimated image"):
            PlanValidator().validate(plan, profile)


if __name__ == "__main__":
    unittest.main()
