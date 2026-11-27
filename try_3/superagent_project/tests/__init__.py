"""Automated tests for the deterministic SuperAgent."""

from pathlib import Path

# Data keys that hold filesystem paths in a task configuration.
CONFIG_PATH_KEYS = (
    "train_features",
    "train_labels",
    "validation_features",
    "validation_labels",
    "test_features",
    "test_labels",
    "train",
    "validation",
    "test",
    "train_directory",
    "validation_directory",
    "test_directory",
)


def absolutize_config_paths(config, config_directory):
    """Freeze config-relative data paths before copying a config elsewhere.

    ``load_task_config`` resolves relative paths against the config file's own
    directory.  Tests that copy ``configs/task1.json`` into a temporary folder
    must therefore rewrite the data paths, or the copy would look for data next
    to the temporary folder.
    """
    data = config.get("data")
    if not isinstance(data, dict):
        return config
    for key in CONFIG_PATH_KEYS:
        value = data.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            data[key] = str((Path(config_directory) / value).resolve())
    return config
