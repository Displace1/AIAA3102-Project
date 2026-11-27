import time
from dataclasses import dataclass

import psutil


BYTES_PER_MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class ResourceCheck:
    elapsed_seconds: float
    memory_mb: float
    peak_memory_mb: float
    violation: str | None = None


def process_tree_memory_bytes(process_id):
    """
    Return total RSS memory used by a process and all its descendants.
    """

    try:
        parent = psutil.Process(process_id)
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        return 0

    processes = [parent]

    try:
        processes.extend(
            parent.children(recursive=True)
        )
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        pass

    total_memory = 0

    for process in processes:
        try:
            total_memory += (
                process.memory_info().rss
            )
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    return total_memory


def terminate_process_tree(
    process_id,
    grace_period_seconds=3.0,
):
    """
    Terminate a process and all descendants.

    First request graceful termination. Processes that are still alive
    after the grace period are forcefully killed.
    """

    try:
        parent = psutil.Process(process_id)
    except psutil.NoSuchProcess:
        return []

    try:
        children = parent.children(
            recursive=True
        )
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        children = []

    processes = children + [parent]
    affected_process_ids = []

    # Terminate children before the parent.
    for process in children:
        try:
            affected_process_ids.append(
                process.pid
            )
            process.terminate()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            pass

    try:
        affected_process_ids.append(
            parent.pid
        )
        parent.terminate()
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        pass

    _, survivors = psutil.wait_procs(
        processes,
        timeout=grace_period_seconds,
    )

    for process in survivors:
        try:
            process.kill()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            pass

    if survivors:
        psutil.wait_procs(
            survivors,
            timeout=1.0,
        )

    return affected_process_ids


class ResourceMonitor:
    """
    Monitor elapsed time and process-tree memory for one experiment.
    """

    def __init__(
        self,
        process_id,
        timeout_seconds,
        maximum_memory_mb,
    ):
        self.process_id = process_id
        self.timeout_seconds = (
            timeout_seconds
        )
        self.maximum_memory_mb = (
            maximum_memory_mb
        )

        self.started_at = (
            time.perf_counter()
        )
        self.peak_memory_mb = 0.0

    def sample(self):
        elapsed_seconds = (
            time.perf_counter()
            - self.started_at
        )

        memory_bytes = (
            process_tree_memory_bytes(
                self.process_id
            )
        )

        memory_mb = (
            memory_bytes
            / BYTES_PER_MEGABYTE
        )

        self.peak_memory_mb = max(
            self.peak_memory_mb,
            memory_mb,
        )

        violation = None

        if (
            elapsed_seconds
            >= self.timeout_seconds
        ):
            violation = "experiment_timeout"

        elif (
            memory_mb
            >= self.maximum_memory_mb
        ):
            violation = (
                "memory_limit_exceeded"
            )

        return ResourceCheck(
            elapsed_seconds=elapsed_seconds,
            memory_mb=memory_mb,
            peak_memory_mb=(
                self.peak_memory_mb
            ),
            violation=violation,
        )