import subprocess


def classify_exception(error):
    """Map implementation-specific exceptions to stable recovery categories."""
    error_name = type(error).__name__.lower()
    error_message = str(error).lower()
    if (
        error_name in {"gpuoutofmemoryerror", "outofmemoryerror"}
        or "cuda out of memory" in error_message
        or "cuda error: out of memory" in error_message
    ):
        return "gpu_out_of_memory"
    if isinstance(error, MemoryError):
        return "out_of_memory"
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "dependency_error"
    if isinstance(error, FileNotFoundError):
        return "data_file_error"
    if isinstance(error, (KeyError, TypeError)):
        return "invalid_configuration"
    if isinstance(error, ValueError):
        message = str(error).lower()
        data_markers = ("sample", "feature", "label", "column", "shape")
        if any(marker in message for marker in data_markers):
            return "data_validation_error"
        return "invalid_configuration"
    if isinstance(error, OSError):
        return "system_resource_error"
    return "training_error"
