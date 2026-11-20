import multiprocessing
import time

from adapters.base import ExperimentResult
from superagent.error_classifier import (
    classify_exception,
)
from superagent.isolated_worker import (
    run_adapter_worker,
)
from superagent.resource_monitor import (
    ResourceMonitor,
    terminate_process_tree,
)
from superagent.result_analyzer import (
    ResultAnalyzer,
)


class ExperimentExecutor:
    def execute(
        self,
        experiment_id,
        adapter,
        plan,
        config,
        data_profile,
        experiment_dir,
    ):
        started_at = time.perf_counter()

        process = None
        receive_connection = None
        send_connection = None

        try:
            # Always use spawn so Windows and Linux
            # follow the same isolated-process path.
            context = (
                multiprocessing.get_context(
                    "spawn"
                )
            )

            (
                receive_connection,
                send_connection,
            ) = context.Pipe(
                duplex=False
            )

            process = context.Process(
                target=run_adapter_worker,
                args=(
                    send_connection,
                    adapter,
                    experiment_id,
                    plan,
                    config,
                    data_profile,
                    experiment_dir,
                ),
                name=(
                    f"experiment-{experiment_id}"
                ),
            )

            # Do not set daemon=True. sklearn/joblib
            # may need to create child processes.
            process.start()

            # The parent only receives. Closing its
            # copy of the send side is important for
            # correct EOF handling.
            send_connection.close()
            send_connection = None

            monitor = ResourceMonitor(
                process_id=process.pid,
                timeout_seconds=(
                    config
                    .experiment_timeout_seconds
                ),
                maximum_memory_mb=(
                    config.maximum_memory_mb
                ),
            )

            payload = None
            resource_check = monitor.sample()

            while process.is_alive():
                resource_check = (
                    monitor.sample()
                )

                if (
                    resource_check.violation
                    is not None
                ):
                    terminate_process_tree(
                        process.pid
                    )

                    process.join(timeout=5)

                    return self._failure_result(
                        experiment_id=(
                            experiment_id
                        ),
                        model_name=(
                            adapter.model_name
                        ),
                        started_at=started_at,
                        error_type=(
                            resource_check
                            .violation
                        ),
                        raw_error_type=(
                            "ResourceLimitExceeded"
                        ),
                        error_message=(
                            self
                            ._resource_error_message(
                                resource_check,
                                config,
                            )
                        ),
                        traceback_text=None,
                        peak_memory_mb=(
                            resource_check
                            .peak_memory_mb
                        ),
                        worker_exit_code=(
                            process.exitcode
                        ),
                    )

                # poll() also provides the wait
                # interval, so no extra sleep is
                # needed here.
                if receive_connection.poll(
                    config
                    .resource_poll_interval_seconds
                ):
                    try:
                        payload = (
                            receive_connection
                            .recv()
                        )
                    except (EOFError, OSError):
                        payload = None

                    break

            # The process may exit immediately after
            # sending. Give it time to clean up.
            process.join(timeout=5)

            if process.is_alive():
                terminate_process_tree(
                    process.pid
                )
                process.join(timeout=5)

                return self._failure_result(
                    experiment_id=experiment_id,
                    model_name=(
                        adapter.model_name
                    ),
                    started_at=started_at,
                    error_type=(
                        "worker_shutdown_timeout"
                    ),
                    raw_error_type=(
                        "WorkerShutdownTimeout"
                    ),
                    error_message=(
                        "The experiment worker sent "
                        "a result but did not exit."
                    ),
                    traceback_text=None,
                    peak_memory_mb=(
                        resource_check
                        .peak_memory_mb
                    ),
                    worker_exit_code=(
                        process.exitcode
                    ),
                )

            # The worker may have exited between the
            # last poll and the loop condition.
            if (
                payload is None
                and receive_connection.poll(1.0)
            ):
                try:
                    payload = (
                        receive_connection.recv()
                    )
                except (EOFError, OSError):
                    payload = None

            if payload is None:
                return self._failure_result(
                    experiment_id=experiment_id,
                    model_name=(
                        adapter.model_name
                    ),
                    started_at=started_at,
                    error_type=(
                        "worker_crashed"
                    ),
                    raw_error_type=(
                        "WorkerProcessExit"
                    ),
                    error_message=(
                        "Experiment worker exited "
                        "without returning a result. "
                        f"Exit code: "
                        f"{process.exitcode}."
                    ),
                    traceback_text=None,
                    peak_memory_mb=(
                        resource_check
                        .peak_memory_mb
                    ),
                    worker_exit_code=(
                        process.exitcode
                    ),
                )

            if payload.get("status") == "failed":
                return self._failure_result(
                    experiment_id=experiment_id,
                    model_name=(
                        adapter.model_name
                    ),
                    started_at=started_at,
                    error_type=payload.get(
                        "error_type",
                        "training_error",
                    ),
                    raw_error_type=payload.get(
                        "raw_error_type",
                        "UnknownWorkerError",
                    ),
                    error_message=payload.get(
                        "error_message",
                        "Experiment worker failed.",
                    ),
                    traceback_text=payload.get(
                        "traceback"
                    ),
                    peak_memory_mb=(
                        resource_check
                        .peak_memory_mb
                    ),
                    worker_exit_code=(
                        process.exitcode
                    ),
                )

            if payload.get("status") != "success":
                return self._failure_result(
                    experiment_id=experiment_id,
                    model_name=(
                        adapter.model_name
                    ),
                    started_at=started_at,
                    error_type=(
                        "invalid_worker_response"
                    ),
                    raw_error_type=(
                        "InvalidWorkerResponse"
                    ),
                    error_message=(
                        "Experiment worker returned "
                        "an invalid response."
                    ),
                    traceback_text=None,
                    peak_memory_mb=(
                        resource_check
                        .peak_memory_mb
                    ),
                    worker_exit_code=(
                        process.exitcode
                    ),
                )

            result = payload.get("result")

            if not isinstance(
                result,
                ExperimentResult,
            ):
                return self._failure_result(
                    experiment_id=experiment_id,
                    model_name=(
                        adapter.model_name
                    ),
                    started_at=started_at,
                    error_type=(
                        "invalid_worker_response"
                    ),
                    raw_error_type=(
                        "InvalidExperimentResult"
                    ),
                    error_message=(
                        "Experiment worker did not "
                        "return ExperimentResult."
                    ),
                    traceback_text=None,
                    peak_memory_mb=(
                        resource_check
                        .peak_memory_mb
                    ),
                    worker_exit_code=(
                        process.exitcode
                    ),
                )

            if result.training_time_seconds <= 0:
                result.training_time_seconds = (
                    time.perf_counter() - started_at
                )
            result.peak_memory_mb = (
                resource_check
                .peak_memory_mb
            )
            result.worker_exit_code = (
                process.exitcode
            )
            result.diagnosis = (
                ResultAnalyzer().analyze(
                    result
                )
            )

            return result

        except Exception as error:
            if (
                process is not None
                and process.is_alive()
            ):
                terminate_process_tree(
                    process.pid
                )
                process.join(timeout=5)

            if process is not None:
                process.join(timeout=0.2)

            worker_crashed = (
                isinstance(error, OSError)
                and process is not None
                and process.exitcode not in (None, 0)
            )

            return self._failure_result(
                experiment_id=experiment_id,
                model_name=adapter.model_name,
                started_at=started_at,
                error_type=(
                    "worker_crashed"
                    if worker_crashed
                    else classify_exception(error)
                ),
                raw_error_type=(
                    "WorkerProcessExit"
                    if worker_crashed
                    else type(error).__name__
                ),
                error_message=(
                    "Experiment worker exited without returning a result. "
                    f"Exit code: {process.exitcode}. Pipe error: {error}"
                    if worker_crashed
                    else str(error)
                ),
                traceback_text=None,
                peak_memory_mb=None,
                worker_exit_code=(
                    process.exitcode
                    if process is not None
                    else None
                ),
            )

        finally:
            if receive_connection is not None:
                try:
                    receive_connection.close()
                except OSError:
                    pass

            if send_connection is not None:
                try:
                    send_connection.close()
                except OSError:
                    pass

    @staticmethod
    def _resource_error_message(
        resource_check,
        config,
    ):
        if (
            resource_check.violation
            == "experiment_timeout"
        ):
            return (
                "Experiment exceeded timeout: "
                f"{config.experiment_timeout_seconds} "
                "seconds. "
                f"Peak memory: "
                f"{resource_check.peak_memory_mb:.1f} MB."
            )

        return (
            "Experiment exceeded memory limit: "
            f"{config.maximum_memory_mb} MB. "
            f"Observed memory: "
            f"{resource_check.memory_mb:.1f} MB; "
            f"peak memory: "
            f"{resource_check.peak_memory_mb:.1f} MB."
        )

    @staticmethod
    def _failure_result(
        experiment_id,
        model_name,
        started_at,
        error_type,
        raw_error_type,
        error_message,
        traceback_text,
        peak_memory_mb,
        worker_exit_code,
    ):
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            status="failed",
            validation_accuracy=None,
            training_time_seconds=(
                time.perf_counter()
                - started_at
            ),
            artifact_path=None,
            peak_memory_mb=peak_memory_mb,
            worker_exit_code=(
                worker_exit_code
            ),
            error_type=error_type,
            raw_error_type=raw_error_type,
            error_message=error_message,
            traceback=traceback_text,
        )

        result.diagnosis = (
            ResultAnalyzer().analyze(result)
        )

        return result
