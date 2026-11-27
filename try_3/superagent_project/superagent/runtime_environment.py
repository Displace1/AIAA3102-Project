"""Describe the machine an experiment runs on, and how much time it has left.

Two problems are solved here, both of which used to be guessed:

1. **Which machine is this?**  ``IMAGE_BATCH_COST`` in
   :mod:`superagent.image_runtime` is calibrated on one machine (Windows, CPU,
   one antivirus policy).  Without a fingerprint the estimate is applied on any
   machine, silently, with coefficients that may be 2-10x off.  Every data
   profile now carries ``execution_context["os_family"]`` and
   ``execution_context["training_device"]`` so the cost model can refuse to
   predict where it has no calibration.

2. **How much wall clock is left?**  The executor used to only *kill* a worker
   at the timeout, which wastes the whole budget.  It now publishes an absolute
   ``wall_clock_deadline`` into the profile; the image trainer stops at an
   epoch boundary before that deadline and keeps its best weights.

Absolute wall-clock time is used on purpose: it survives ``spawn`` (unlike
``time.perf_counter``, whose origin is per-process) and needs no protocol
change through the worker pipe.
"""

import os
import platform
import sys
import time

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency of images
    torch = None


# The parent must still receive the result, join the worker and clean up after
# the deadline it publishes.  Keeping the margin here (instead of in each
# adapter) means a worker that ignores its deadline cannot make the parent's
# hard timeout fire first.
EXECUTOR_DEADLINE_RESERVE_SECONDS = 10.0

# After training the adapter still has to run one validation inference pass,
# save the artifact, write the manifest and send the result back.  That work is
# independent of ``epochs``, so it is subtracted from the training budget once.
TRAINING_TAIL_RESERVE_SECONDS = 20.0


def os_family():
    """Return a coarse OS name that decides dataloader behaviour."""
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _cpu_model_name():
    try:
        name = platform.processor()
    except Exception:  # pragma: no cover - platform specific
        return None
    name = (name or "").strip()
    return name or platform.machine() or None


def _npu_available():
    """Detect Huawei Ascend without importing torch_npu eagerly.

    ``torch.cuda.is_available()`` is False on Ascend, so without this flag an
    NPU node looks like a plain CPU node in every recorded result.
    """
    if torch is None:
        return False
    try:
        npu = getattr(torch, "npu", None)
        if npu is None:
            return False
        return bool(npu.is_available())
    except Exception:  # pragma: no cover - driver specific
        return False


def describe_runtime():
    """Return the runtime facts that decide how fast image training is."""
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    device_name = None
    memory_mb = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            memory_mb = (
                torch.cuda.get_device_properties(0).total_memory
                / (1024 * 1024)
            )
        except Exception:  # pragma: no cover - driver specific
            device_name = None
            memory_mb = None

    return {
        # The OS decides ``num_workers`` (0 on Windows, 4 elsewhere), which is
        # the single largest structural difference in per-batch cost.
        "os_family": os_family(),
        # What ``select_training_device("auto")`` will actually pick.
        "training_device": "cuda" if cuda_available else "cpu",
        "device_name": device_name or _cpu_model_name(),
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
        "npu_available": _npu_available(),
        "accelerator_memory_mb": memory_mb,
        "torch_threads": (
            int(torch.get_num_threads()) if torch is not None else 0
        ),
        "cpu_count": os.cpu_count() or 1,
    }


def attach_experiment_deadline(data_profile, timeout_seconds):
    """Return a copy of ``data_profile`` carrying an absolute deadline."""
    if not isinstance(data_profile, dict) or timeout_seconds is None:
        return data_profile
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return data_profile

    deadline = time.time() + timeout - EXECUTOR_DEADLINE_RESERVE_SECONDS
    context = dict(data_profile.get("execution_context") or {})
    context["wall_clock_deadline"] = deadline
    context["experiment_timeout_seconds"] = timeout
    updated = dict(data_profile)
    updated["execution_context"] = context
    return updated


def remaining_training_budget(
    deadline_epoch,
    now=None,
    tail_reserve=TRAINING_TAIL_RESERVE_SECONDS,
):
    """Seconds available for the training loop, or ``None`` when unbudgeted."""
    if deadline_epoch is None:
        return None
    try:
        deadline = float(deadline_epoch)
        reserve = float(tail_reserve)
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else float(now)
    return max(0.0, deadline - current - reserve)
