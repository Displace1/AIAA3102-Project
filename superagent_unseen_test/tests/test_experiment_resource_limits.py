import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.base import ExperimentResult, ModelAdapter
from superagent.executor import ExperimentExecutor


class FastAdapter(ModelAdapter):
    model_name = "fake_fast_model"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        thread_environment = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "LOKY_MAX_CPU_COUNT",
            )
        }
        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=self.model_name,
            status="success",
            validation_accuracy=1.0,
            training_time_seconds=0.0,
            artifact_path=None,
            model_config={"thread_environment": thread_environment},
        )


class SlowAdapter(ModelAdapter):
    model_name = "fake_slow_model"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        time.sleep(5)
        raise AssertionError("SlowAdapter should have been terminated.")


class MemoryAdapter(ModelAdapter):
    model_name = "fake_memory_model"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        allocations = []
        while True:
            allocations.append(bytearray(8 * 1024 * 1024))
            time.sleep(0.03)


class CrashAdapter(ModelAdapter):
    model_name = "fake_crash_model"

    def run(
        self,
        experiment_id,
        config,
        data_profile,
        output_dir,
        hyperparameters,
    ):
        os._exit(7)


class ExperimentResourceLimitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.plan = SimpleNamespace(hyperparameters={})
        self.executor = ExperimentExecutor()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def make_config(timeout_seconds=10, maximum_memory_mb=1024):
        return SimpleNamespace(
            experiment_timeout_seconds=timeout_seconds,
            maximum_memory_mb=maximum_memory_mb,
            maximum_cpu_threads=1,
            resource_poll_interval_seconds=0.02,
        )

    def execute(self, adapter, config):
        return self.executor.execute(
            experiment_id="resource_test",
            adapter=adapter,
            plan=self.plan,
            config=config,
            data_profile={},
            experiment_dir=self.output_directory,
        )

    def test_fast_adapter_succeeds_and_receives_cpu_limit(self):
        result = self.execute(FastAdapter(), self.make_config())

        self.assertEqual(result.status, "success")
        self.assertEqual(result.worker_exit_code, 0)
        self.assertIsNotNone(result.peak_memory_mb)
        self.assertGreater(result.peak_memory_mb, 0)
        thread_environment = result.model_config["thread_environment"]
        self.assertTrue(
            all(value == "1" for value in thread_environment.values())
        )

    def test_slow_adapter_is_stopped_at_timeout(self):
        result = self.execute(
            SlowAdapter(),
            self.make_config(timeout_seconds=0.5),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "experiment_timeout")
        self.assertLess(result.training_time_seconds, 4)

    def test_memory_adapter_is_stopped_at_limit(self):
        result = self.execute(
            MemoryAdapter(),
            self.make_config(timeout_seconds=15, maximum_memory_mb=256),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "memory_limit_exceeded")
        self.assertGreaterEqual(result.peak_memory_mb, 256)

    def test_crash_adapter_reports_worker_crash(self):
        result = self.execute(CrashAdapter(), self.make_config())

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "worker_crashed")
        self.assertEqual(result.worker_exit_code, 7)


if __name__ == "__main__":
    unittest.main()
