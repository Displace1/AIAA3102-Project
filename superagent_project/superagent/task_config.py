import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from superagent.metrics import SUPPORTED_METRICS


VALID_PLANNER_MODES = {
    "deterministic",
    "llm",
}

VALID_RUN_TYPES = {
    "development_run",
    "official_report_run",
}


@dataclass(frozen=True)
class TaskConfig:
    # Basic task information.
    name: str
    metric: str
    data: Dict[str, Any]
    source_path: Path
    run_type: str

    # Basic execution settings.
    random_seed: int
    evaluate_test: bool

    # Autonomous experiment settings.
    planner_mode: str
    initial_deterministic_experiments: int
    max_experiments: int
    max_runtime_minutes: int
    no_improvement_patience: int
    minimum_improvement: float
    maximum_planner_errors: int
    maximum_same_experiment_errors: int

    # Per-experiment resource limits. These are consumed by the isolated
    # experiment executor; max_runtime_minutes still limits the whole run.
    experiment_timeout_seconds: int
    maximum_memory_mb: int
    maximum_cpu_threads: int
    resource_poll_interval_seconds: float


def _required_path(data, key):
    if key not in data:
        raise ValueError(
            f"Missing data configuration field: {key}"
        )

    path = Path(
        data[key]
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Configured file does not exist: {path}"
        )

    return str(path)


def _required_directory(data, key):
    if key not in data:
        raise ValueError(
            f"Missing data configuration field: {key}"
        )

    path = Path(
        data[key]
    ).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Configured directory does not exist: {path}"
        )

    return str(path)


def _read_positive_integer(
    execution,
    key,
    default,
):
    value = execution.get(key, default)

    if isinstance(value, bool):
        raise ValueError(
            f"execution.{key} must be an integer."
        )

    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"execution.{key} must be an integer."
        ) from error

    if value <= 0:
        raise ValueError(
            f"execution.{key} must be positive."
        )

    return value


def _read_nonnegative_integer(
    execution,
    key,
    default,
):
    value = execution.get(key, default)

    if isinstance(value, bool):
        raise ValueError(
            f"execution.{key} must be an integer."
        )

    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"execution.{key} must be an integer."
        ) from error

    if value < 0:
        raise ValueError(
            f"execution.{key} cannot be negative."
        )

    return value


def _read_nonnegative_float(
    execution,
    key,
    default,
):
    value = execution.get(key, default)

    if isinstance(value, bool):
        raise ValueError(
            f"execution.{key} must be a number."
        )

    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"execution.{key} must be a number."
        ) from error

    if value < 0:
        raise ValueError(
            f"execution.{key} cannot be negative."
        )

    return value


def load_task_config(path):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Task configuration does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    task = raw.get("task", {})
    data = dict(raw.get("data", {}))
    execution = raw.get("execution", {})

    if not isinstance(task, dict):
        raise ValueError(
            "task configuration must be an object."
        )

    if not isinstance(data, dict):
        raise ValueError(
            "data configuration must be an object."
        )

    if not isinstance(execution, dict):
        raise ValueError(
            "execution configuration must be an object."
        )

    name = task.get("name")
    metric = task.get("metric", "accuracy")
    data_format = data.get("format")
    run_type = raw.get("run_type", "development_run")

    if not name:
        raise ValueError("task.name is required.")

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            "task.metric must be one of: "
            f"{sorted(SUPPORTED_METRICS)}"
        )

    if run_type not in VALID_RUN_TYPES:
        raise ValueError(
            "run_type must be one of: "
            f"{sorted(VALID_RUN_TYPES)}"
        )

    # Validate data files without reading test labels.
    if data_format == "npy_split":
        for key in (
            "train_features",
            "train_labels",
            "validation_features",
            "validation_labels",
            "test_features",
            "test_labels",
        ):
            data[key] = _required_path(data, key)

    elif data_format == "csv_split":
        for key in (
            "train",
            "validation",
            "test",
        ):
            data[key] = _required_path(data, key)

        columns = data.get("columns")
        if columns is not None:
            if not isinstance(columns, dict):
                raise ValueError("data.columns must be an object.")
            unknown_column_groups = set(columns) - {
                "text", "numeric", "categorical", "target"
            }
            if unknown_column_groups:
                raise ValueError(
                    "Unknown data.columns fields: "
                    f"{sorted(unknown_column_groups)}"
                )
            declared_target = columns.get("target")
            if declared_target:
                if data.get("target_column") not in (None, declared_target):
                    raise ValueError(
                        "data.target_column and data.columns.target disagree."
                    )
                data["target_column"] = declared_target
            for group in ("text", "numeric", "categorical"):
                value = columns.get(group, [])
                if not isinstance(value, list) or not all(
                    isinstance(column, str) and column for column in value
                ):
                    raise ValueError(
                        f"data.columns.{group} must be a list of column names."
                    )
        if not data.get("target_column"):
            raise ValueError(
                "data.target_column is required "
                "for CSV tasks."
            )

    elif data_format == "image_directory_split":
        for key in (
            "train_directory",
            "validation_directory",
            "test_directory",
        ):
            data[key] = _required_directory(data, key)

    else:
        raise ValueError(
            "data.format must be "
            "'npy_split', 'csv_split', or "
            "'image_directory_split'."
        )

    random_seed = _read_nonnegative_integer(
        execution=execution,
        key="random_seed",
        default=42,
    )

    evaluate_test = bool(
        execution.get("evaluate_test", False)
    )

    # Keep test evaluation disabled until the final
    # evaluator and test gate are implemented.
    if evaluate_test:
        raise ValueError(
            "Test evaluation is disabled during "
            "experiment development. Lock a final "
            "solution before using the test set."
        )

    planner_mode = execution.get(
        "planner_mode",
        "deterministic",
    )

    if planner_mode not in VALID_PLANNER_MODES:
        raise ValueError(
            "execution.planner_mode must be one of: "
            f"{sorted(VALID_PLANNER_MODES)}"
        )

    initial_deterministic_experiments = (
        _read_nonnegative_integer(
            execution=execution,
            key="initial_deterministic_experiments",
            default=2,
        )
    )

    max_experiments = _read_positive_integer(
        execution=execution,
        key="max_experiments",
        default=8,
    )

    max_runtime_minutes = _read_positive_integer(
        execution=execution,
        key="max_runtime_minutes",
        default=180,
    )

    no_improvement_patience = (
        _read_positive_integer(
            execution=execution,
            key="no_improvement_patience",
            default=3,
        )
    )

    minimum_improvement = _read_nonnegative_float(
        execution=execution,
        key="minimum_improvement",
        default=0.001,
    )

    maximum_planner_errors = (
        _read_positive_integer(
            execution=execution,
            key="maximum_planner_errors",
            default=2,
        )
    )
    maximum_same_experiment_errors = _read_positive_integer(
        execution=execution,
        key="maximum_same_experiment_errors",
        default=2,
    )

    experiment_timeout_seconds = _read_positive_integer(
        execution=execution,
        key="experiment_timeout_seconds",
        default=1800,
    )

    maximum_memory_mb = _read_positive_integer(
        execution=execution,
        key="maximum_memory_mb",
        default=4096,
    )

    maximum_cpu_threads = _read_positive_integer(
        execution=execution,
        key="maximum_cpu_threads",
        default=2,
    )

    resource_poll_interval_seconds = _read_nonnegative_float(
        execution=execution,
        key="resource_poll_interval_seconds",
        default=0.2,
    )
    if resource_poll_interval_seconds == 0:
        raise ValueError(
            "execution.resource_poll_interval_seconds must be positive."
        )

    if (
        initial_deterministic_experiments
        > max_experiments
    ):
        raise ValueError(
            "execution.initial_deterministic_experiments "
            "cannot exceed execution.max_experiments."
        )

    return TaskConfig(
        name=name,
        metric=metric,
        data=data,
        source_path=path,
        run_type=run_type,
        random_seed=random_seed,
        evaluate_test=False,
        planner_mode=planner_mode,
        initial_deterministic_experiments=(
            initial_deterministic_experiments
        ),
        max_experiments=max_experiments,
        max_runtime_minutes=max_runtime_minutes,
        no_improvement_patience=(
            no_improvement_patience
        ),
        minimum_improvement=minimum_improvement,
        maximum_planner_errors=(
            maximum_planner_errors
        ),
        maximum_same_experiment_errors=maximum_same_experiment_errors,
        experiment_timeout_seconds=experiment_timeout_seconds,
        maximum_memory_mb=maximum_memory_mb,
        maximum_cpu_threads=maximum_cpu_threads,
        resource_poll_interval_seconds=resource_poll_interval_seconds,
    )
