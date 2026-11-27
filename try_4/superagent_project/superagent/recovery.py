import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

from adapters.base import ExperimentResult
from superagent.experiment_plan import ExperimentPlan


STATE_SCHEMA_VERSION = 1
EXPERIMENT_ID_PATTERN = re.compile(r"^exp_(\d+)$")
VALID_RUN_STATUSES = {"running", "interrupted", "completed", "failed"}


class RecoveryError(RuntimeError):
    """Base error for unsafe or inconsistent run recovery."""


class RunAlreadyCompletedError(RecoveryError):
    """Raised when a locked run is selected for recovery."""


class RecoveryConfigurationError(RecoveryError):
    """Raised when the requested configuration does not match the run."""


class CorruptRunError(RecoveryError):
    """Raised when persisted run artifacts are missing or invalid."""


@dataclass(frozen=True)
class RunState:
    task_name: str
    status: str
    completed_experiment_ids: list[str]
    in_progress_experiment_id: Optional[str]
    next_experiment_number: int
    accumulated_runtime_seconds: float
    resume_count: int
    stop_reason: Optional[str] = None
    schema_version: int = STATE_SCHEMA_VERSION
    updated_at: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data["updated_at"] = (
            self.updated_at
            or datetime.now().astimezone().isoformat()
        )
        return data


@dataclass(frozen=True)
class RecoveredRun:
    run_directory: Path
    plans: list[ExperimentPlan]
    results: list[ExperimentResult]
    incomplete_plan: Optional[ExperimentPlan]
    incomplete_plan_source: Optional[str]
    accumulated_runtime_seconds: float
    resume_count: int
    next_experiment_number: int


def configuration_sha256(config_path):
    """Return a stable hash of the exact task configuration bytes."""
    path = Path(config_path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRecovery:
    def load(self, run_directory, config):
        run_directory = Path(run_directory).expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(
                f"Run directory does not exist: {run_directory}"
            )

        if (run_directory / "solution_lock.json").is_file():
            raise RunAlreadyCompletedError(
                "The selected run already contains solution_lock.json and "
                "is complete. Start a new run instead of resuming it."
            )

        manifest = self._read_json(
            run_directory / "run_manifest.json",
            description="run manifest",
        )
        self._validate_configuration(manifest, config)

        plans, results, incomplete_plan, incomplete_source = (
            self._load_experiments(run_directory)
        )
        state = self._load_state(run_directory, config.name)
        next_number = self._next_experiment_number(
            plans, incomplete_plan
        )

        return RecoveredRun(
            run_directory=run_directory,
            plans=plans,
            results=results,
            incomplete_plan=incomplete_plan,
            incomplete_plan_source=incomplete_source,
            accumulated_runtime_seconds=(
                state.accumulated_runtime_seconds
            ),
            resume_count=state.resume_count,
            next_experiment_number=next_number,
        )

    def write_checkpoint(self, run_directory, state):
        if not isinstance(state, RunState):
            raise TypeError("state must be a RunState instance.")
        if state.status not in VALID_RUN_STATUSES:
            raise ValueError(
                f"Invalid run status: {state.status!r}."
            )
        if state.accumulated_runtime_seconds < 0:
            raise ValueError(
                "accumulated_runtime_seconds cannot be negative."
            )
        if state.resume_count < 0:
            raise ValueError("resume_count cannot be negative.")
        if state.next_experiment_number <= 0:
            raise ValueError("next_experiment_number must be positive.")

        path = Path(run_directory).expanduser().resolve() / "run_state.json"
        self._atomic_write_json(path, state.to_dict())
        return path

    def _load_experiments(self, run_directory):
        experiments_directory = run_directory / "experiments"
        if not experiments_directory.exists():
            return [], [], None, None
        if not experiments_directory.is_dir():
            raise CorruptRunError(
                f"Experiments path is not a directory: {experiments_directory}"
            )

        directories = []
        for path in experiments_directory.iterdir():
            if not path.is_dir():
                continue
            match = EXPERIMENT_ID_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            directories.append((int(match.group(1)), path))
        directories.sort(key=lambda item: item[0])

        plans = []
        results = []
        incomplete_plan = None
        incomplete_source = None

        for _, experiment_directory in directories:
            experiment_id = experiment_directory.name
            plan_path = experiment_directory / "plan.json"
            result_path = experiment_directory / "experiment.json"

            if not plan_path.is_file():
                if result_path.exists():
                    raise CorruptRunError(
                        f"{experiment_id} has experiment.json but no plan.json."
                    )
                continue

            plan_record = self._read_json(
                plan_path,
                description=f"plan for {experiment_id}",
            )
            plan = self._parse_plan(plan_record, experiment_id, plan_path)

            if not result_path.is_file():
                if incomplete_plan is not None:
                    raise CorruptRunError(
                        "Multiple incomplete experiments were found: "
                        f"{incomplete_plan.experiment_id} and {experiment_id}."
                    )
                incomplete_plan = plan
                incomplete_source = plan_record.get("plan_source")
                continue

            if incomplete_plan is not None:
                raise CorruptRunError(
                    f"Completed experiment {experiment_id} appears after "
                    f"incomplete experiment {incomplete_plan.experiment_id}."
                )

            result_record = self._read_json(
                result_path,
                description=f"result for {experiment_id}",
            )
            result = self._parse_result(
                result_record, experiment_id, result_path
            )
            plans.append(plan)
            results.append(result)

        return plans, results, incomplete_plan, incomplete_source

    def _load_state(self, run_directory, task_name):
        state_path = run_directory / "run_state.json"
        if not state_path.is_file():
            return RunState(
                task_name=task_name,
                status="interrupted",
                completed_experiment_ids=[],
                in_progress_experiment_id=None,
                next_experiment_number=1,
                accumulated_runtime_seconds=0.0,
                resume_count=0,
            )

        record = self._read_json(state_path, description="run state")
        if record.get("schema_version") != STATE_SCHEMA_VERSION:
            raise CorruptRunError(
                "Unsupported run_state.json schema_version: "
                f"{record.get('schema_version')!r}."
            )
        if record.get("task_name") != task_name:
            raise RecoveryConfigurationError(
                "run_state.json task name does not match the configuration."
            )

        try:
            state = RunState(
                task_name=record["task_name"],
                status=record["status"],
                completed_experiment_ids=list(
                    record.get("completed_experiment_ids", [])
                ),
                in_progress_experiment_id=record.get(
                    "in_progress_experiment_id"
                ),
                next_experiment_number=int(
                    record.get("next_experiment_number", 1)
                ),
                accumulated_runtime_seconds=float(
                    record.get("accumulated_runtime_seconds", 0.0)
                ),
                resume_count=int(record.get("resume_count", 0)),
                stop_reason=record.get("stop_reason"),
                schema_version=int(record["schema_version"]),
                updated_at=record.get("updated_at"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CorruptRunError(
                f"Invalid run_state.json: {error}"
            ) from error

        if state.status not in VALID_RUN_STATUSES:
            raise CorruptRunError(
                f"Invalid run state status: {state.status!r}."
            )
        if state.accumulated_runtime_seconds < 0 or state.resume_count < 0:
            raise CorruptRunError(
                "Run state runtime and resume count cannot be negative."
            )
        return state

    @staticmethod
    def _parse_plan(record, expected_id, path):
        try:
            plan = ExperimentPlan(
                experiment_id=record["experiment_id"],
                adapter_name=record["adapter_name"],
                hyperparameters=dict(record["hyperparameters"]),
                reason=record["reason"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CorruptRunError(f"Invalid plan file {path}: {error}") from error
        if plan.experiment_id != expected_id:
            raise CorruptRunError(
                f"Plan ID {plan.experiment_id!r} does not match directory "
                f"{expected_id!r}."
            )
        return plan

    @staticmethod
    def _parse_result(record, expected_id, path):
        allowed_fields = {field.name for field in fields(ExperimentResult)}
        filtered = {
            key: value for key, value in record.items() if key in allowed_fields
        }
        try:
            result = ExperimentResult(**filtered)
        except (TypeError, ValueError) as error:
            raise CorruptRunError(
                f"Invalid experiment result {path}: {error}"
            ) from error
        if result.experiment_id != expected_id:
            raise CorruptRunError(
                f"Result ID {result.experiment_id!r} does not match directory "
                f"{expected_id!r}."
            )
        return result

    @staticmethod
    def _next_experiment_number(plans, incomplete_plan):
        experiment_ids = [plan.experiment_id for plan in plans]
        if incomplete_plan is not None:
            experiment_ids.append(incomplete_plan.experiment_id)
        numbers = []
        for experiment_id in experiment_ids:
            match = EXPERIMENT_ID_PATTERN.fullmatch(experiment_id)
            if match is None:
                raise CorruptRunError(
                    f"Invalid experiment ID: {experiment_id!r}."
                )
            numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    @staticmethod
    def _validate_configuration(manifest, config):
        if manifest.get("task_name") != config.name:
            raise RecoveryConfigurationError(
                "Run task name does not match the selected configuration."
            )

        current_path = Path(config.source_path).expanduser().resolve()
        manifest_path_value = manifest.get("config_path")
        if not manifest_path_value:
            raise CorruptRunError("run_manifest.json has no config_path.")
        manifest_path = Path(manifest_path_value).expanduser().resolve()

        expected_hash = manifest.get("config_sha256")
        if expected_hash is not None:
            actual_hash = configuration_sha256(current_path)
            if actual_hash != expected_hash:
                raise RecoveryConfigurationError(
                    "Task configuration content has changed since the run began."
                )
        elif manifest_path != current_path:
            raise RecoveryConfigurationError(
                "Task configuration path does not match the original run."
            )

    @staticmethod
    def _read_json(path, description):
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as error:
            raise CorruptRunError(
                f"Missing {description}: {path}"
            ) from error
        except json.JSONDecodeError as error:
            raise CorruptRunError(
                f"Invalid JSON in {description} {path}: {error}"
            ) from error
        if not isinstance(data, dict):
            raise CorruptRunError(
                f"Expected a JSON object in {description}: {path}"
            )
        return data

    @staticmethod
    def _atomic_write_json(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
