import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from superagent.controller import (
    SuperAgentController,
)
from superagent.experiment_plan import (
    ExperimentPlan,
)
from superagent.llm.experiment_planner import (
    PlannerDecision,
)


from tests import absolutize_config_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeBudgetManager:
    def __init__(self):
        self._spent = Decimal("0.00000000")
        self._remaining = Decimal("20.00000000")

    def spent_rmb(self):
        return self._spent

    def remaining_rmb(self):
        return self._remaining


class FakePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
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
                "data_profile": data_profile,
                "experiment_history": list(
                    experiment_history
                ),
                "existing_plans": list(
                    existing_plans
                ),
                "remaining_budget_rmb": (
                    remaining_budget_rmb
                ),
                "next_experiment_id": (
                    next_experiment_id
                ),
            }
        )

        if not self.decisions:
            raise AssertionError(
                "FakePlanner has no remaining decisions."
            )

        return self.decisions.pop(0)


class ControllerAutonomousLoopTests(
    unittest.TestCase
):
    def setUp(self):
        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.temp_root = Path(
            self.temp_directory.name
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def create_temporary_config(self):
        source_config_path = (
            PROJECT_ROOT
            / "configs"
            / "task1.json"
        )

        with source_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        config = absolutize_config_paths(
            config, source_config_path.parent
        )

        config["execution"][
            "planner_mode"
        ] = "llm"

        config["execution"][
            "initial_deterministic_experiments"
        ] = 2

        # Use 4 here so the Fake Planner can:
        # 1. propose Exp 3
        # 2. receive Exp 3 feedback
        # 3. return stop
        config["execution"][
            "max_experiments"
        ] = 4

        config["execution"][
            "max_runtime_minutes"
        ] = 30

        config["execution"][
            "no_improvement_patience"
        ] = 3

        config["execution"][
            "minimum_improvement"
        ] = 0.001

        config["execution"][
            "maximum_planner_errors"
        ] = 2

        temporary_config_path = (
            self.temp_root
            / "task1_test.json"
        )

        with temporary_config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return temporary_config_path

    def build_fake_planner(self):
        exp_003_plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name=(
                "tabular_extra_trees"
            ),
            hyperparameters={
                "n_estimators": 300,
                "max_features": "log2",
                "min_samples_leaf": 3,
                "class_weight": "balanced",
            },
            reason=(
                "Test a different Extra Trees "
                "regularization configuration."
            ),
        )

        run_decision = PlannerDecision(
            action="run_experiment",
            reason=exp_003_plan.reason,
            plan=exp_003_plan,
            call_id="fake-call-001",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            cost={
                "total_cost_rmb": "0.00000000"
            },
        )

        stop_decision = PlannerDecision(
            action="stop",
            reason=(
                "The feedback from Exp 3 has "
                "been observed; stop the test."
            ),
            plan=None,
            call_id="fake-call-002",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            cost={
                "total_cost_rmb": "0.00000000"
            },
        )

        return FakePlanner(
            decisions=[
                run_decision,
                stop_decision,
            ]
        )

    def test_controller_feedback_loop(self):
        config_path = (
            self.create_temporary_config()
        )

        fake_planner = (
            self.build_fake_planner()
        )

        fake_budget_manager = (
            FakeBudgetManager()
        )

        def fake_planner_factory(
            project_root,
            run_directory,
        ):
            self.assertEqual(
                project_root,
                PROJECT_ROOT,
            )

            self.assertTrue(
                Path(run_directory).is_dir()
            )

            return (
                fake_planner,
                fake_budget_manager,
            )

        controller = SuperAgentController(
            config_path=config_path,
            runs_dir=(
                self.temp_root
                / "runs"
            ),
            planner_factory=(
                fake_planner_factory
            ),
        )

        # Controller checks that the environment
        # variable exists when planner_mode=llm.
        # The fake value is never sent anywhere.
        with patch.dict(
            os.environ,
            {
                "AIGC_API_KEY": (
                    "fake-test-key"
                )
            },
        ):
            summary = controller.run()

        # Two deterministic experiments plus
        # one Fake Planner experiment.
        self.assertEqual(
            summary["completed_experiments"],
            3,
        )

        self.assertEqual(
            summary["stop_reason"],
            "planner_requested_stop",
        )

        self.assertEqual(
            summary["planner_mode"],
            "llm",
        )

        self.assertFalse(
            summary["test_evaluated"]
        )

        self.assertEqual(
            summary["llm_cost_rmb"],
            "0E-8",
        )

        # The planner should have been called twice:
        # first to create Exp 3, then to stop.
        self.assertEqual(
            len(fake_planner.calls),
            2,
        )

        first_call = fake_planner.calls[0]
        second_call = fake_planner.calls[1]

        # First call sees Exp 1 and Exp 2.
        self.assertEqual(
            len(
                first_call[
                    "experiment_history"
                ]
            ),
            2,
        )

        self.assertEqual(
            first_call[
                "next_experiment_id"
            ],
            "exp_003",
        )

        # Second call sees Exp 1, Exp 2 and Exp 3.
        self.assertEqual(
            len(
                second_call[
                    "experiment_history"
                ]
            ),
            3,
        )

        self.assertEqual(
            second_call[
                "next_experiment_id"
            ],
            "exp_004",
        )

        run_directory = Path(
            summary["run_directory"]
        )

        self.assertTrue(
            (
                run_directory
                / "experiments"
                / "exp_003"
                / "experiment.json"
            ).is_file()
        )

        self.assertTrue(
            (
                run_directory
                / "planner_decisions.jsonl"
            ).is_file()
        )

        self.assertTrue(
            (
                run_directory
                / "stop_reason.json"
            ).is_file()
        )

        self.assertTrue(
            (
                run_directory
                / "solution_lock.json"
            ).is_file()
        )

        with (
            run_directory
            / "solution_lock.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:
            solution_lock = json.load(file)

        self.assertTrue(
            solution_lock[
                "selected_automatically"
            ]
        )

        self.assertFalse(
            solution_lock[
                "test_used_for_selection"
            ]
        )

        self.assertFalse(
            solution_lock[
                "test_evaluation_completed"
            ]
        )


if __name__ == "__main__":
    unittest.main()