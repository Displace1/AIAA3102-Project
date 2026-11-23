import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from superagent.controller import SuperAgentController
from superagent.experiment_plan import ExperimentPlan
from superagent.llm.experiment_planner import PlannerDecision
from superagent.logger import RunLogger
from superagent.recovery import RunRecovery, RunState
from tests import absolutize_config_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResumeBudgetManager:
    def spent_rmb(self):
        return Decimal("0.00000000")

    def remaining_rmb(self):
        return Decimal("20.00000000")


class StopAfterRecoveryPlanner:
    def __init__(self):
        self.calls = []

    def plan_next(
        self,
        data_profile,
        experiment_history,
        existing_plans,
        remaining_budget_rmb,
        next_experiment_id,
    ):
        self.calls.append(
            {
                "history": list(experiment_history),
                "plans": list(existing_plans),
                "next_experiment_id": next_experiment_id,
            }
        )
        return PlannerDecision(
            action="stop",
            reason="Recovery integration test completed.",
            plan=None,
            call_id="resume-stop-call",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            cost={"total_cost_rmb": "0.00000000"},
        )


class ControllerResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "task1_resume.json"

        with (PROJECT_ROOT / "configs" / "task1.json").open(
            "r", encoding="utf-8"
        ) as file:
            config = json.load(file)
        config = absolutize_config_paths(
            config, PROJECT_ROOT / "configs"
        )
        config["execution"]["max_experiments"] = 4
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        self.logger = RunLogger(
            runs_dir=self.root / "runs",
            task_name="released_task_1",
            config_path=self.config_path,
        )
        self.run_directory = self.logger.run_dir
        self._create_interrupted_run()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_interrupted_run(self):
        exp_001 = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="tabular_one_hot_logistic",
            hyperparameters={"max_iter": 2000, "C": 1.0},
            reason="Existing completed baseline.",
        )
        exp_002 = ExperimentPlan(
            experiment_id="exp_002",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 400,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            reason="Interrupted deterministic experiment.",
        )

        for plan in (exp_001, exp_002):
            self.logger.write_json(
                f"experiments/{plan.experiment_id}/plan.json",
                {**plan.to_dict(), "plan_source": "deterministic"},
            )

        self.logger.write_json(
            "experiments/exp_001/experiment.json",
            {
                "experiment_id": "exp_001",
                "model_name": "existing_model",
                "status": "success",
                "validation_accuracy": 0.70,
                "training_time_seconds": 123.0,
                "artifact_path": "existing-model.joblib",
            },
        )
        RunRecovery().write_checkpoint(
            self.run_directory,
            RunState(
                task_name="released_task_1",
                status="interrupted",
                completed_experiment_ids=["exp_001"],
                in_progress_experiment_id="exp_002",
                next_experiment_number=3,
                accumulated_runtime_seconds=4.0,
                resume_count=0,
            ),
        )

    def test_controller_resumes_incomplete_experiment_without_rerunning_done(self):
        planner = StopAfterRecoveryPlanner()
        budget = ResumeBudgetManager()

        def planner_factory(project_root, run_directory):
            self.assertEqual(project_root, PROJECT_ROOT)
            self.assertEqual(Path(run_directory), self.run_directory)
            return planner, budget

        controller = SuperAgentController(
            config_path=self.config_path,
            runs_dir=self.root / "unused-runs-directory",
            planner_factory=planner_factory,
            resume_run_directory=self.run_directory,
        )

        with patch.dict(os.environ, {"AIGC_API_KEY": "fake-test-key"}):
            summary = controller.run()

        self.assertEqual(Path(summary["run_directory"]), self.run_directory)
        self.assertEqual(summary["completed_experiments"], 2)
        self.assertEqual(summary["stop_reason"], "planner_requested_stop")

        with (
            self.run_directory
            / "experiments"
            / "exp_001"
            / "experiment.json"
        ).open("r", encoding="utf-8") as file:
            exp_001_result = json.load(file)
        self.assertEqual(exp_001_result["training_time_seconds"], 123.0)

        with (
            self.run_directory
            / "experiments"
            / "exp_002"
            / "experiment.json"
        ).open("r", encoding="utf-8") as file:
            exp_002_result = json.load(file)
        self.assertEqual(exp_002_result["status"], "success")

        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(planner.calls[0]["history"]), 2)
        self.assertEqual(planner.calls[0]["next_experiment_id"], "exp_003")

        with (self.run_directory / "run_state.json").open(
            "r", encoding="utf-8"
        ) as file:
            state = json.load(file)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["resume_count"], 1)
        self.assertGreaterEqual(state["accumulated_runtime_seconds"], 4.0)

        self.assertTrue((self.run_directory / "resume_events.jsonl").is_file())
        self.assertTrue((self.run_directory / "solution_lock.json").is_file())


if __name__ == "__main__":
    unittest.main()
