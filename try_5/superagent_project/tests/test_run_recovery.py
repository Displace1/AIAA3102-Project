import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from superagent.recovery import (
    RecoveryConfigurationError,
    RunAlreadyCompletedError,
    RunRecovery,
    RunState,
    configuration_sha256,
)


class RunRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.run_directory = self.root / "run"
        self.run_directory.mkdir()
        self.config_path = self.root / "task.json"
        self.config_path.write_text("{}", encoding="utf-8")
        self.config = SimpleNamespace(
            name="test_task",
            source_path=self.config_path,
        )
        self.write_json(
            self.run_directory / "run_manifest.json",
            {
                "run_id": "run",
                "task_name": "test_task",
                "config_path": str(self.config_path.resolve()),
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_plan(self, experiment_id):
        self.write_json(
            self.run_directory / "experiments" / experiment_id / "plan.json",
            {
                "experiment_id": experiment_id,
                "adapter_name": "fake_adapter",
                "hyperparameters": {"value": 1},
                "reason": "Recovery test plan.",
                "plan_source": "llm_planner",
            },
        )

    def write_result(self, experiment_id):
        self.write_json(
            self.run_directory
            / "experiments"
            / experiment_id
            / "experiment.json",
            {
                "experiment_id": experiment_id,
                "model_name": "fake_model",
                "status": "success",
                "validation_accuracy": 0.75,
                "training_time_seconds": 1.0,
                "artifact_path": "model.joblib",
            },
        )

    def test_loads_completed_and_incomplete_experiments(self):
        self.write_plan("exp_001")
        self.write_result("exp_001")
        self.write_plan("exp_002")

        recovered = RunRecovery().load(self.run_directory, self.config)

        self.assertEqual(len(recovered.plans), 1)
        self.assertEqual(len(recovered.results), 1)
        self.assertEqual(recovered.incomplete_plan.experiment_id, "exp_002")
        self.assertEqual(recovered.incomplete_plan_source, "llm_planner")
        self.assertEqual(recovered.next_experiment_number, 3)

    def test_checkpoint_round_trip_restores_runtime(self):
        recovery = RunRecovery()
        recovery.write_checkpoint(
            self.run_directory,
            RunState(
                task_name="test_task",
                status="interrupted",
                completed_experiment_ids=["exp_001"],
                in_progress_experiment_id="exp_002",
                next_experiment_number=3,
                accumulated_runtime_seconds=12.5,
                resume_count=2,
            ),
        )

        recovered = recovery.load(self.run_directory, self.config)
        self.assertEqual(recovered.accumulated_runtime_seconds, 12.5)
        self.assertEqual(recovered.resume_count, 2)

    def test_rejects_completed_run(self):
        self.write_json(self.run_directory / "solution_lock.json", {})
        with self.assertRaises(RunAlreadyCompletedError):
            RunRecovery().load(self.run_directory, self.config)

    def test_rejects_corrupt_experiment_json(self):
        self.write_plan("exp_001")
        result_path = (
            self.run_directory
            / "experiments"
            / "exp_001"
            / "experiment.json"
        )
        result_path.write_text("{not-json", encoding="utf-8")

        with self.assertRaisesRegex(Exception, "Invalid JSON"):
            RunRecovery().load(self.run_directory, self.config)

    def test_rejects_changed_configuration_when_hash_is_available(self):
        self.write_json(
            self.run_directory / "run_manifest.json",
            {
                "run_id": "run",
                "task_name": "test_task",
                "config_path": str(self.config_path.resolve()),
                "config_sha256": configuration_sha256(self.config_path),
            },
        )
        self.config_path.write_text('{"changed": true}', encoding="utf-8")

        with self.assertRaises(RecoveryConfigurationError):
            RunRecovery().load(self.run_directory, self.config)


if __name__ == "__main__":
    unittest.main()
