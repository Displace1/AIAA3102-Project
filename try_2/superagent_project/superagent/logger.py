import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy
import sklearn

from superagent.recovery import configuration_sha256


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class RunLogger:
    def __init__(
        self,
        runs_dir,
        task_name,
        config_path,
        run_type="development_run",
        resume_run_directory=None,
    ):
        if resume_run_directory is not None:
            self.run_dir = Path(resume_run_directory).expanduser().resolve()
            if not self.run_dir.is_dir():
                raise FileNotFoundError(
                    f"Run directory does not exist: {self.run_dir}"
                )
            self.append_jsonl(
                "resume_events.jsonl",
                {
                    "event": "run_resumed",
                    "resumed_at": datetime.now().astimezone().isoformat(),
                },
            )
            return

        safe_task_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)
        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.run_dir = Path(runs_dir).resolve() / (
            f"{safe_task_name}_{timestamp}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_dir.name,
                "task_name": task_name,
                "run_type": run_type,
                "config_path": str(Path(config_path).resolve()),
                "config_sha256": configuration_sha256(config_path),
                "started_at": datetime.now().astimezone().isoformat(),
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "package_versions": {
                    "numpy": numpy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__,
                },
                "test_evaluation_enabled": False,
            },
        )

    def experiment_dir(self, experiment_id, allow_existing=False):
        path = self.run_dir / "experiments" / experiment_id
        path.mkdir(parents=True, exist_ok=allow_existing)
        return path

    def write_json(self, relative_path, data):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    def append_jsonl(self, relative_path, data):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            line = json.dumps(
                data,
                ensure_ascii=False,
                default=_json_default,
            )
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
        return path
