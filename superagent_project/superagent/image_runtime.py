"""Runtime cost model for image experiments.

The cost model answers one question that the LLM planner cannot answer on its
own: "will this (adapter, image_size, epochs) plan finish inside the experiment
timeout?"  It is used as a gate in :mod:`superagent.plan_validator`, so it must
be *slightly pessimistic* rather than unbiased: underestimating lets a doomed
plan through and burns the whole timeout, while overestimating only costs one
planner retry.

Model
-----
    T = SETUP + epochs * (warm_per_epoch + COLD_IO_PER_IMAGE * train_samples)
    warm_per_epoch = n_train * (a_tr + b_tr * size^2)
                   + n_val   * (a_val + b_val * size^2)

Two details matter more than the exact coefficients:

* The per-batch cost has an intercept, because ``num_workers=0`` on Windows
  puts PNG decoding and resizing on the critical path.  That part scales with
  pixels but not with model FLOPs, so a pure ``size^2`` scaling of a single
  reference number cannot fit both 32px and 224px (measured: 16x more area
  costs 7.4x per batch for ``small_cnn``, 4.4x for ``resnet18``).
* The cold-cache penalty is *additive per file*, not a multiplier on the model
  cost: it is the cost of opening freshly written images, so it depends on how
  many files are read per epoch, not on how heavy the network is.  Modelling it
  as a multiplier (``x2.1``) made estimates for small datasets 3-5x too high
  and for heavy models too low, which is how an earlier revision rejected a
  resnet18 plan that would have finished in a tenth of the timeout.

Provenance
----------
``IMAGE_BATCH_COST`` was fitted from 12 (adapter, image_size) measurements taken
through the project's own dataloaders, three repeats each, median kept
(``test_tasks/calibrate_image_runtime.py``).  Fit residuals: 1.9% (small_cnn),
3.0% (resnet18), 13.4% (mobilenet -- hence the low-confidence uplift).
``COLD_IO_SECONDS_PER_IMAGE`` comes from real runs: small_cnn over 5000
Fashion-MNIST PNGs costs 17.5s/epoch with a cold page cache against 8.25s/epoch
warm, i.e. 9.25s of file reload per epoch, or 0.00185s per image.

Runtimes are machine specific.  ``COLD_IO_SECONDS_PER_IMAGE`` and the batch
coefficients were fitted on one machine, so the estimate is only served when
the data profile describes that machine.  ``CALIBRATED_OS_FAMILY`` and
``CALIBRATED_TRAINING_DEVICE`` are the fingerprint; anything else (a Linux node,
a GPU, an NPU node that falls back to CPU) gets ``None`` instead of a number
that looks authoritative and is not.  Re-run ``calibrate_image_runtime.py`` on
the target machine and record its fingerprint here to re-enable the gate.
"""

import math

from superagent.throughput_cache import (
    lookup_measured_epoch_seconds,
    measured_throughput_key,
)


# Fingerprint of the machine the coefficients were fitted on.  ``os_family``
# matters because ``image_dataset.py`` picks ``num_workers=0`` on Windows and
# ``min(threads, 4)`` elsewhere, which moves PNG decoding off/onto the critical
# path and changes the intercept term by a factor of several.
CALIBRATED_OS_FAMILY = "windows"
CALIBRATED_TRAINING_DEVICE = "cpu"


# cost per batch = intercept + area_coefficient * image_size**2   (seconds)
IMAGE_BATCH_COST = {
    "cpu": {
        "image_small_cnn": {
            "train": (0.0288, 2.087e-05),
            "validation": (0.0141, 9.314e-06),
            "low_confidence": False,
        },
        "image_resnet18": {
            "train": (0.2565, 7.547e-05),
            "validation": (0.0473, 2.749e-05),
            "low_confidence": False,
        },
        "image_mobilenet_v3_small": {
            "train": (0.0832, 1.355e-05),
            "validation": (0.0302, 2.271e-06),
            "low_confidence": True,
        },
    },
    # GPU decode/compute ratios differ by orders of magnitude from the CPU ones
    # (no PNG contention, far larger compute share), so this table is
    # deliberately empty: an unknown device must fall back to the hard timeout
    # instead of silently reusing CPU coefficients.
    "cuda": {},
}

# Process spawn + torch import + dataloader scan + weight load, measured as
# (wall clock - training wall clock) of a spawned worker (6.8s) plus margin.
FIXED_SETUP_SECONDS = 12.0

# Re-reading freshly written images every epoch costs about 1.85ms per image
# (9.25s per epoch over 5000 PNGs). unseen data is always freshly written, so
# this is the normal case rather than an outlier.
COLD_IO_SECONDS_PER_IMAGE = 0.00185

# mobilenet's area coefficient has 46.6% spread at 128px, so its extrapolation
# beyond the calibrated range gets an explicit safety uplift.
LOW_CONFIDENCE_UPLIFT = 1.10

# Applied on top of a *measured* slowest epoch.  The measurement is exact for
# the run that produced it, but epochs vary within a run (measured 3.0-11.6s at
# 64px on this machine when the page cache and memory pressure changed), so the
# prediction keeps a small margin instead of trusting a single sample.
MEASURED_THROUGHPUT_UPLIFT = 1.10

DEFAULT_BATCH_SIZE = 32


def recommended_image_size(data_profile):
    """Pick a default resolution that keeps native detail without upscaling."""
    summary = data_profile.get("image_summary", {}).get("train", {})
    width = int(summary.get("median_width", summary.get("minimum_width", 128)))
    height = int(
        summary.get("median_height", summary.get("minimum_height", 128))
    )
    native = max(16, min(width, height))
    rounded = int(math.ceil(native / 32) * 32)
    return max(32, min(224, rounded))


def _training_device(execution_context):
    device = execution_context.get("training_device")
    if device:
        return str(device)
    # Profiles recorded before the fingerprint existed only carry the boolean.
    return "cuda" if execution_context.get("cuda_available") else "cpu"


def _runtime_matches_calibration(execution_context):
    """True when the estimate applies to the machine that produced it.

    An absent ``os_family`` is accepted so that profiles recorded before the
    field existed still work; a *present but different* value disables the
    estimate, which is the case that used to fail silently (Linux nodes running
    with Windows coefficients).
    """
    os_family = execution_context.get("os_family")
    if os_family is not None and str(os_family) != CALIBRATED_OS_FAMILY:
        return False
    return _training_device(execution_context) == CALIBRATED_TRAINING_DEVICE


def estimate_image_training_seconds(adapter_name, hyperparameters, data_profile):
    """Estimated wall-clock seconds for one image experiment.

    Returns ``None`` when the combination is unknown (unsupported adapter, a
    device without a calibrated cost table, or a machine whose fingerprint does
    not match the calibration).  Callers must treat ``None`` as "no estimate
    available" and fall back to the hard timeout rather than assuming the plan
    is cheap.
    """
    seconds, _ = estimate_image_training_seconds_with_source(
        adapter_name, hyperparameters, data_profile
    )
    return seconds


def estimate_image_training_seconds_with_source(
    adapter_name,
    hyperparameters,
    data_profile,
):
    """Like :func:`estimate_image_training_seconds`, plus where it came from.

    Source is ``"measured"`` when the same (adapter, size, batch, dataset,
    machine) combination has already run and its per-epoch cost was recorded,
    and ``"model"`` when the analytic coefficients were used.  The measured
    branch is what makes the gate usable off the calibration machine: it needs
    no coefficients at all.
    """
    context = data_profile.get("execution_context") or {}
    if not isinstance(hyperparameters, dict):
        return None, None
    try:
        image_size = int(hyperparameters["image_size"])
        epochs = int(hyperparameters["epochs"])
        batch_size = int(
            hyperparameters.get("batch_size", DEFAULT_BATCH_SIZE)
        )
    except (KeyError, TypeError, ValueError):
        return None, None
    if image_size <= 0 or epochs <= 0:
        return None, None
    batch_size = max(1, batch_size)

    measured = lookup_measured_epoch_seconds(
        context,
        measured_throughput_key(adapter_name, hyperparameters, data_profile),
    )
    if measured is not None:
        _, slowest_epoch = measured
        # The slowest observed epoch already contains the cold page cache
        # penalty and the validation pass, so it replaces the whole analytic
        # per-epoch term; SETUP is still fixed.  The uplift covers run-to-run
        # variance (measured at 4x within a single run on this machine).
        return (
            float(
                FIXED_SETUP_SECONDS
                + epochs * slowest_epoch * MEASURED_THROUGHPUT_UPLIFT
            ),
            "measured",
        )

    if not _runtime_matches_calibration(context):
        return None, None
    device = _training_device(context)
    coefficients = IMAGE_BATCH_COST.get(device, {}).get(adapter_name)
    if coefficients is None:
        return None, None

    train_samples = int(data_profile.get("train_samples", 0) or 0)
    validation_samples = int(data_profile.get("validation_samples", 0) or 0)
    if train_samples <= 0:
        return None, None

    area = float(image_size) ** 2
    train_batches = math.ceil(train_samples / batch_size)
    validation_batches = math.ceil(validation_samples / batch_size)

    a_train, b_train = coefficients["train"]
    a_validation, b_validation = coefficients["validation"]

    per_epoch = (
        train_batches * (a_train + b_train * area)
        + validation_batches * (a_validation + b_validation * area)
        + COLD_IO_SECONDS_PER_IMAGE * train_samples
    )
    estimate = FIXED_SETUP_SECONDS + epochs * per_epoch
    if coefficients.get("low_confidence"):
        estimate *= LOW_CONFIDENCE_UPLIFT
    return float(estimate), "model"
