import json
import tempfile
import unittest
from pathlib import Path

from superagent.task_config import (
    load_task_config,
)


class TaskConfigTests(unittest.TestCase):
    def test_loads_task1_execution_settings(self):
        root = Path(__file__).resolve().parents[1]

        config = load_task_config(
            root / "configs" / "task1.json"
        )

        self.assertEqual(
            config.planner_mode,
            "llm",
        )
        self.assertEqual(
            config.run_type,
            "official_report_run",
        )
        self.assertEqual(
            config.initial_deterministic_experiments,
            2,
        )
        self.assertEqual(
            config.max_experiments,
            8,
        )
        self.assertEqual(
            config.max_runtime_minutes,
            180,
        )
        self.assertEqual(
            config.no_improvement_patience,
            3,
        )
        self.assertEqual(
            config.minimum_improvement,
            0.001,
        )
        self.assertEqual(
            config.maximum_planner_errors,
            2,
        )
        self.assertEqual(
            config.maximum_same_experiment_errors,
            2,
        )
        self.assertEqual(
            config.experiment_timeout_seconds,
            1800,
        )
        self.assertEqual(
            config.maximum_memory_mb,
            4096,
        )
        self.assertEqual(
            config.maximum_cpu_threads,
            2,
        )
        self.assertEqual(
            config.resource_poll_interval_seconds,
            0.2,
        )

    def test_loads_image_directory_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_directory = root / "train"
            validation_directory = root / "validation"
            test_directory = root / "test"
            for split_directory in (
                train_directory,
                validation_directory,
                test_directory,
            ):
                split_directory.mkdir()

            config_path = root / "image_task.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": {
                            "name": "synthetic_image_task",
                            "metric": "accuracy",
                        },
                        "data": {
                            "format": "image_directory_split",
                            "train_directory": str(train_directory),
                            "validation_directory": str(
                                validation_directory
                            ),
                            "test_directory": str(test_directory),
                        },
                        "execution": {
                            "planner_mode": "deterministic",
                            "initial_deterministic_experiments": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_task_config(config_path)

            self.assertEqual(
                config.data["format"],
                "image_directory_split",
            )
            self.assertEqual(config.run_type, "development_run")
            self.assertEqual(
                config.data["train_directory"],
                str(train_directory.resolve()),
            )
            self.assertEqual(
                config.data["validation_directory"],
                str(validation_directory.resolve()),
            )
            self.assertEqual(
                config.data["test_directory"],
                str(test_directory.resolve()),
            )

    def test_rejects_missing_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_directory = root / "train"
            validation_directory = root / "validation"
            train_directory.mkdir()
            validation_directory.mkdir()
            config_path = root / "missing_test_directory.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": {"name": "image_task"},
                        "data": {
                            "format": "image_directory_split",
                            "train_directory": str(train_directory),
                            "validation_directory": str(
                                validation_directory
                            ),
                            "test_directory": str(root / "missing_test"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Configured directory does not exist",
            ):
                load_task_config(config_path)

    def test_rejects_unknown_run_type(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "task1.json").open(
            "r", encoding="utf-8"
        ) as file:
            raw = json.load(file)
        raw["run_type"] = "manual_best_result"

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid_run_type.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_type"):
                load_task_config(config_path)


if __name__ == "__main__":
    unittest.main()
