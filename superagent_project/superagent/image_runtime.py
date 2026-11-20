import math


REFERENCE_SECONDS_PER_IMAGE = {
    "cpu": {
        "image_small_cnn": (32, 0.0016),
        "image_mobilenet_v3_small": (128, 0.0320),
        "image_resnet18": (128, 0.0411),
    },
    "cuda": {
        "image_small_cnn": (128, 0.00030),
        "image_mobilenet_v3_small": (224, 0.00080),
        "image_resnet18": (224, 0.00100),
    },
}


def recommended_image_size(data_profile):
    summary = data_profile.get("image_summary", {}).get("train", {})
    width = int(summary.get("median_width", summary.get("minimum_width", 128)))
    height = int(
        summary.get("median_height", summary.get("minimum_height", 128))
    )
    native = max(16, min(width, height))
    rounded = int(math.ceil(native / 32) * 32)
    return max(32, min(224, rounded))


def estimate_image_training_seconds(adapter_name, hyperparameters, data_profile):
    context = data_profile.get("execution_context") or {}
    device = "cuda" if context.get("cuda_available") else "cpu"
    reference = REFERENCE_SECONDS_PER_IMAGE.get(device, {}).get(adapter_name)
    if reference is None:
        return None
    reference_size, seconds_per_image = reference
    image_size = int(hyperparameters["image_size"])
    epochs = int(hyperparameters["epochs"])
    samples = int(data_profile.get("train_samples", 0))
    scale = (image_size / reference_size) ** 2
    # Include validation and fixed loader/model setup overhead.
    return float(10.0 + samples * epochs * seconds_per_image * scale * 1.25)
