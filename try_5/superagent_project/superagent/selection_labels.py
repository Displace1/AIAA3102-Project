"""Load the validation labels a run is scored against, for every data format.

Selection's paired test needs the labels that produced each experiment's
``validation_accuracy``. Those are the same rows every adapter used, so exactly
one loader per format is needed rather than one per adapter.

Kept separate from ``tabular_data.load_tabular_split`` because that helper also
loads and prepares features, which selection does not need.
"""
from __future__ import annotations

import csv

import numpy as np

from superagent.label_encoding import encode_labels


class LabelLoadError(RuntimeError):
    """Raised when validation labels cannot be recovered for a run."""


def load_validation_labels(config, data_profile=None):
    """Return the validation labels as int64 codes, or raise LabelLoadError.

    Callers that only want a best-effort annotation should catch this and record
    the skip reason rather than failing the whole run.
    """
    data_profile = data_profile or {}
    mapping = data_profile.get("label_mapping")
    data = getattr(config, "data", None)
    if not isinstance(data, dict):
        raise LabelLoadError("task configuration has no data section")

    fmt = data.get("format")
    try:
        if fmt == "npy_split":
            raw = np.load(data["validation_labels"], allow_pickle=True)
            return encode_labels(np.asarray(raw).ravel().tolist(), mapping)

        if fmt == "csv_split":
            target = data.get("target_column")
            if not target:
                raise LabelLoadError("csv_split task declares no target_column")
            values = []
            with open(data["validation"], "r", encoding="utf-8-sig",
                      newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or target not in reader.fieldnames:
                    raise LabelLoadError(
                        f"validation CSV is missing target column {target!r}")
                for row in reader:
                    values.append(row[target])
            return encode_labels(values, mapping)

        if fmt == "image_directory_split":
            # The image pipeline derives labels from the folder names; the
            # class_to_index map in the profile is what the adapters used.
            class_to_index = data_profile.get("class_to_index")
            if not class_to_index:
                raise LabelLoadError(
                    "image task has no class_to_index in its data profile")
            root = data["validation_directory"]
            import os
            labels = []
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):
                    continue
                code = class_to_index.get(name)
                if code is None:
                    continue
                for fn in sorted(os.listdir(folder)):
                    if os.path.isfile(os.path.join(folder, fn)):
                        labels.append(int(code))
            if not labels:
                raise LabelLoadError(
                    f"no labelled images found under {root!r}")
            return np.asarray(labels, dtype=np.int64)

        raise LabelLoadError(f"unsupported data format: {fmt!r}")
    except LabelLoadError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise LabelLoadError(
            f"could not load validation labels for format {fmt!r}: {error}"
        ) from error
