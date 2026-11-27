import os
import traceback
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from superagent.error_classifier import (
    classify_exception,
)


def _configure_cpu_threads(maximum_cpu_threads):
    """Limit common numerical and joblib thread pools."""

    thread_count = str(maximum_cpu_threads)

    environment_variables = {
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "VECLIB_MAXIMUM_THREADS": thread_count,
        "LOKY_MAX_CPU_COUNT": thread_count,
    }

    for name, value in environment_variables.items():
        os.environ[name] = value


def run_adapter_worker(
    send_connection,
    adapter,
    experiment_id,
    plan,
    config,
    data_profile,
    experiment_dir,
):
    """
    Run one adapter inside an isolated child process.

    The result or exception information is sent to the parent process
    through a multiprocessing Pipe.
    """

    try:
        _configure_cpu_threads(
            config.maximum_cpu_threads
        )

        if threadpool_limits is None:
            thread_context = nullcontext()
        else:
            thread_context = threadpool_limits(
                limits=config.maximum_cpu_threads
            )

        with thread_context:
            result = adapter.run(
                experiment_id=experiment_id,
                config=config,
                data_profile=data_profile,
                output_dir=experiment_dir,
                hyperparameters=(
                    plan.hyperparameters
                ),
            )

        send_connection.send(
            {
                "status": "success",
                "result": result,
            }
        )

    except BaseException as error:
        try:
            send_connection.send(
                {
                    "status": "failed",
                    "error_type": (
                        classify_exception(error)
                    ),
                    "raw_error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(error),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )
        except (
            BrokenPipeError,
            EOFError,
            OSError,
        ):
            # The parent process may already have
            # terminated the experiment.
            pass

    finally:
        try:
            send_connection.close()
        except OSError:
            pass
