"""Reuse measured per-epoch cost instead of predicting it from constants.

The analytic model in :mod:`superagent.image_runtime` answers "will this fit in
the timeout" only on the machine that produced its coefficients.  But every
successful image experiment already *measures* the answer: it records
``epoch_seconds`` for each epoch it ran, on the real data, with the real
dataloader, thread count, device and cold page cache.  That number needs no
calibration and is valid on any machine.

This module turns those measurements into a keyed cache:

    key = (adapter, image_size, batch_size, train_samples, validation_samples,
           os_family, training_device, torch_threads)

Keying on the dataset shape (not just the adapter) is deliberate: a per-epoch
cost measured on 5,000 samples says nothing about 50,000.  A key miss simply
falls back to the analytic model, which is the old behaviour.

The cache is injected into ``data_profile["execution_context"]`` so it travels
through ``spawn`` with no I/O in the worker and no module-level state.  The
controller owns persistence (``<runs_dir>/throughput_cache.json``).
"""

import json
from datetime import datetime, timezone

CACHE_SCHEMA_VERSION = 1
MAX_ENTRIES = 200

KEY_FIELDS = (
    "adapter_name",
    "image_size",
    "batch_size",
    "train_samples",
    "validation_samples",
    "os_family",
    "training_device",
    "torch_threads",
)


def measured_throughput_key(adapter_name, hyperparameters, data_profile):
    """Build the cache key for one (plan, dataset, machine) combination."""
    if not isinstance(hyperparameters, dict):
        return None
    context = data_profile.get("execution_context") or {}
    try:
        values = {
            "adapter_name": adapter_name,
            "image_size": int(hyperparameters["image_size"]),
            "batch_size": int(hyperparameters.get("batch_size", 32)),
            "train_samples": int(data_profile.get("train_samples") or 0),
            "validation_samples": int(
                data_profile.get("validation_samples") or 0
            ),
            "os_family": context.get("os_family"),
            "training_device": context.get("training_device"),
            "torch_threads": context.get("torch_threads"),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if values["train_samples"] <= 0 or values["image_size"] <= 0:
        return None
    return json.dumps(values, sort_keys=True)


def measurement_from_result(plan, data_profile, result):
    """Build a cache entry from a finished experiment, or ``None``.

    ``max_epoch_seconds`` is kept next to the mean because the first epoch pays
    the cold page cache penalty; predicting a longer run from the mean would
    repeat the original under-estimate that let a 12-epoch plan burn 600s.
    """
    if result is None or getattr(result, "status", None) != "success":
        return None
    epoch_count = int(getattr(result, "epoch_count", 0) or 0)
    if epoch_count <= 0:
        return None

    model_config = getattr(result, "model_config", None) or {}
    history = model_config.get("history") or []
    raw_epoch_seconds = getattr(result, "epoch_seconds", None) or [
        record.get("epoch_seconds")
        for record in history
        if isinstance(record, dict)
    ]
    epoch_seconds = [
        float(value)
        for value in raw_epoch_seconds
        if isinstance(value, (int, float)) and float(value) > 0
    ]
    if epoch_seconds:
        mean_epoch_seconds = sum(epoch_seconds) / len(epoch_seconds)
        max_epoch_seconds = max(epoch_seconds)
    else:
        average = float(getattr(result, "training_time_seconds", 0.0) or 0.0)
        if average <= 0:
            return None
        mean_epoch_seconds = average / epoch_count
        max_epoch_seconds = mean_epoch_seconds

    key = measured_throughput_key(
        plan.adapter_name,
        plan.hyperparameters,
        data_profile,
    )
    if key is None:
        return None
    return {
        "key": key,
        "adapter_name": plan.adapter_name,
        "image_size": int(plan.hyperparameters["image_size"]),
        "batch_size": int(plan.hyperparameters.get("batch_size", 32)),
        "train_samples": int(data_profile.get("train_samples") or 0),
        "validation_samples": int(data_profile.get("validation_samples") or 0),
        "mean_epoch_seconds": round(mean_epoch_seconds, 4),
        "max_epoch_seconds": round(max_epoch_seconds, 4),
        "epochs_completed": epoch_count,
        "stopped_reason": getattr(result, "stopped_reason", None),
        "experiment_id": getattr(result, "experiment_id", None),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def attach_measured_throughput(data_profile, entries):
    """Return a profile copy carrying the cache entries for the validator."""
    if not isinstance(data_profile, dict):
        return data_profile
    updated = dict(data_profile)
    context = dict(updated.get("execution_context") or {})
    context["measured_throughput"] = dict(entries or {})
    updated["execution_context"] = context
    return updated


def lookup_measured_epoch_seconds(execution_context, key):
    """Return ``(mean, max)`` per-epoch seconds for ``key``, or ``None``."""
    if not key:
        return None
    entries = (execution_context or {}).get("measured_throughput") or {}
    entry = entries.get(key)
    if not isinstance(entry, dict):
        return None
    try:
        mean_epoch_seconds = float(entry["mean_epoch_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    if mean_epoch_seconds <= 0:
        return None
    try:
        max_epoch_seconds = float(entry["max_epoch_seconds"])
    except (KeyError, TypeError, ValueError):
        max_epoch_seconds = mean_epoch_seconds
    return mean_epoch_seconds, max(max_epoch_seconds, mean_epoch_seconds)


class ThroughputCache:
    """Small JSON-backed cache of measured per-epoch costs."""

    def __init__(self, path):
        self.path = path
        self._entries = {}
        self._load()

    def _load(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        entries = payload.get("entries")
        if isinstance(entries, dict):
            self._entries = {
                key: value
                for key, value in entries.items()
                if isinstance(value, dict)
            }

    def entries(self):
        return dict(self._entries)

    def get(self, key):
        return self._entries.get(key)

    def record(self, entry):
        if not isinstance(entry, dict) or not entry.get("key"):
            return False
        key = entry["key"]
        previous = self._entries.get(key)
        self._entries[key] = entry
        self._trim()
        if previous == entry:
            return False
        return self.save()

    def _trim(self):
        if len(self._entries) <= MAX_ENTRIES:
            return
        ordered = sorted(
            self._entries.items(),
            key=lambda item: item[1].get("measured_at") or "",
        )
        for key, _ in ordered[: len(self._entries) - MAX_ENTRIES]:
            del self._entries[key]

    def save(self):
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "entries": self._entries,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError:
            # A cache that cannot be written must never fail an experiment.
            return False
        return True
