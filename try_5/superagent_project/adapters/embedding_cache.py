"""Disk cache for sentence embeddings, shared across isolated experiments.

Why this is not just a dict
---------------------------
Every experiment runs in a *spawned* child process (``ExperimentExecutor`` ->
``run_adapter_worker``). The adapter object is pickled across that boundary, so
an in-memory ``self._cache = {}`` is useless: each experiment would start with
an empty cache and re-encode the corpus.

Measured cost of getting this wrong: encoding 20 Newsgroups takes 213s on the
cluster. A run spends 8 experiments, so without a cache the same 213s is paid
eight times (~28 min) out of a 30-60 minute budget. With the cache it is paid
once and every later experiment reads it in under a second.

What the cache key must include
-------------------------------
Enough to make a stale entry impossible rather than unlikely:

  * the absolute data file path AND its size/mtime -- editing a CSV invalidates
  * which text columns were used, and in what order -- a pair task and a
    single-text task over the same file are different problems
  * the encoder model name AND its dimension -- swapping encoders must not hit
  * whether vectors were L2-normalised -- changes the numbers

The text *content* is not hashed: hashing 8k documents per lookup would cost
more than the encode it is protecting against. Size+mtime is the cheaper guard
and is what a user actually changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

CACHE_DIRECTORY = "embedding_cache"
SCHEMA_VERSION = 1


def _stat_signature(path):
    """(absolute path, size, mtime_ns) -- cheap but detects an edited file."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_key(data_paths, text_columns, model_name, dimension, normalised=True):
    """Stable hash identifying one encoded corpus.

    ``data_paths`` may be a single path (single-text task) or a pair of paths for
    two separate files; both are folded in so a split swap cannot collide.
    """
    if isinstance(data_paths, (str, os.PathLike)):
        data_paths = [data_paths]
    payload = {
        "schema": SCHEMA_VERSION,
        "files": [_stat_signature(p) for p in data_paths],
        "columns": list(text_columns),
        "model": model_name,
        "dimension": int(dimension),
        "normalised": bool(normalised),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def cache_path(cache_directory, key, split):
    return Path(cache_directory) / f"{key}.{split}.npy"


def load(cache_directory, key, split, expected_rows=None):
    """Return the cached array, or None when absent or unusable.

    A corrupt or truncated ``.npy`` is treated as a miss, not an error: a cache
    must never be able to fail an experiment.
    """
    path = cache_path(cache_directory, key, split)
    if not path.is_file():
        return None
    try:
        array = np.load(path)
    except (OSError, ValueError):
        return None
    if array.ndim != 2:
        return None
    if expected_rows is not None and array.shape[0] != expected_rows:
        return None
    return array


def store(cache_directory, key, split, array):
    """Write atomically, so a killed experiment cannot leave a partial file.

    The temp name carries the pid: two experiments in a run are sequential, but
    a resumed run could in principle race, and a shared ``.tmp`` would corrupt.
    """
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = cache_path(directory, key, split)
    temp = directory / f"{key}.{split}.{os.getpid()}.tmp.npy"
    np.save(temp, np.asarray(array))
    os.replace(temp, final)
    return final


def describe(cache_directory, key, split):
    path = cache_path(cache_directory, key, split)
    if not path.is_file():
        return {"hit": False, "path": str(path)}
    stat = path.stat()
    return {"hit": True, "path": str(path),
            "bytes": int(stat.st_size), "mtime": stat.st_mtime}


class TimedLookup:
    """Small helper that records whether a lookup hit, for the metrics block."""

    def __init__(self):
        self.hit = False
        self.seconds = 0.0
        self.rows = 0

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._started
        return False

    def as_dict(self):
        return {"hit": self.hit, "lookup_seconds": round(self.seconds, 4),
                "rows": self.rows}
