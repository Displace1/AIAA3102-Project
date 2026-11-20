import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.image_small_cnn import SmallCNNAdapter
from scripts.create_synthetic_image_dataset import (
    create_synthetic_image_dataset,
    write_smoke_config,
)
from superagent.controller import SuperAgentController
from superagent.experiment_plan import ExperimentPlan


class SmokeImageRegistry:
    """Use production adapters with a one-epoch integration-test plan."""

    def __init__(self, random_seed):
        self.random_seed = random_seed

    def adapter_for(self, adapter_name):
        if adapter_name != "image_small_cnn":
            raise ValueError(f"Unknown smoke adapter: {adapter_name}")
        return SmallCNNAdapter(random_seed=self.random_seed)

    @staticmethod
    def default_plans_for(data_profile):
        if data_profile["modality"] != "image":
            raise ValueError("Smoke registry only supports image data.")
        return [
            ExperimentPlan(
                experiment_id="exp_001",
                adapter_name="image_small_cnn",
                hyperparameters={
                    "image_size": 16,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "data_augmentation": False,
                    "early_stopping_patience": 1,
                    "weight_decay": 0.0,
                },
                reason="Run a one-epoch image controller smoke test.",
            )
        ]


class ImageControllerIntegrationTests(unittest.TestCase):
    def test_controller_trains_selects_and_locks_image_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_directory = create_synthetic_image_dataset(
                root / "dataset"
            )
            config_path = write_smoke_config(
                root / "image_smoke.json",
                dataset_directory,
            )

            with patch(
                "superagent.controller.ExperimentRegistry",
                SmokeImageRegistry,
            ):
                summary = SuperAgentController(
                    config_path=config_path,
                    runs_dir=root / "runs",
                ).run()

            self.assertEqual(summary["modality"], "image")
            self.assertEqual(summary["run_type"], "development_run")
            self.assertEqual(summary["planner_mode"], "deterministic")
            self.assertEqual(summary["completed_experiments"], 1)
            self.assertEqual(summary["selected_experiment"], "exp_001")
            self.assertEqual(summary["selected_model"], "small_cnn")
            self.assertEqual(
                summary["stop_reason"], "deterministic_mode_completed"
            )
            self.assertFalse(summary["test_evaluated"])

            run_directory = Path(summary["run_directory"])
            profile = self._read_json(run_directory / "data_profile.json")
            self.assertEqual(profile["modality"], "image")
            self.assertFalse(profile["test_labels_inspected"])

            manifest = self._read_json(run_directory / "run_manifest.json")
            self.assertEqual(manifest["run_type"], "development_run")

            lock = self._read_json(run_directory / "solution_lock.json")
            self.assertTrue(lock["solution_locked"])
            self.assertEqual(lock["run_type"], "development_run")
            self.assertTrue(lock["selected_automatically"])
            self.assertFalse(lock["test_used_for_selection"])
            self.assertFalse(lock["test_evaluation_completed"])
            self.assertTrue(Path(lock["artifact_path"]).is_file())

            experiment_directory = run_directory / "experiments" / "exp_001"
            self.assertTrue(
                (experiment_directory / "model_package.pt").is_file()
            )
            self.assertTrue(
                (experiment_directory / "artifact_manifest.json").is_file()
            )

    @staticmethod
    def _read_json(path):
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)


if __name__ == "__main__":
    unittest.main()
