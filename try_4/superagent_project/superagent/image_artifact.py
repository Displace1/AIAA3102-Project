import json
import os
from pathlib import Path

import torch

from superagent.image_dataset import IMAGENET_MEAN, IMAGENET_STD


ARTIFACT_FORMAT_VERSION = 1


class ImageArtifactError(RuntimeError):
    pass


def save_image_artifact(
    path,
    model,
    adapter_name,
    architecture,
    model_config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    image_size = int(model_config["image_size"])

    package = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": "image_classifier",
        "adapter_name": adapter_name,
        "architecture": architecture,
        "model_config": model_config,
        "preprocessing": {
            "color_mode": "RGB",
            "image_size": image_size,
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
        "state_dict": state_dict,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(package, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return path


def load_image_artifact(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")

    try:
        package = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ImageArtifactError(
            f"Cannot load image artifact {path}: {error}"
        ) from error

    if not isinstance(package, dict):
        raise ImageArtifactError("Image artifact must contain a dictionary.")

    required_fields = {
        "format_version",
        "artifact_type",
        "adapter_name",
        "architecture",
        "model_config",
        "preprocessing",
        "state_dict",
    }

    missing_fields = required_fields - set(package)

    if missing_fields:
        raise ImageArtifactError(
            f"Artifact is missing fields: {sorted(missing_fields)}"
        )

    if package["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ImageArtifactError(
            "Unsupported image artifact format version: "
            f"{package['format_version']}"
        )

    if package["artifact_type"] != "image_classifier":
        raise ImageArtifactError(
            f"Unexpected artifact type: {package['artifact_type']}"
        )

    model_config = package["model_config"]
    if not isinstance(model_config, dict):
        raise ImageArtifactError("model_config must be a dictionary.")
    if "number_of_classes" not in model_config:
        raise ImageArtifactError(
            "model_config is missing number_of_classes."
        )
    if int(model_config["number_of_classes"]) < 2:
        raise ImageArtifactError(
            "number_of_classes must be at least two."
        )
    if not isinstance(package["state_dict"], dict):
        raise ImageArtifactError("state_dict must be a dictionary.")

    return package


def build_model_from_artifact(package):
    """Reconstruct an image model without downloading pretrained weights."""

    architecture = package.get("architecture")
    model_config = package.get("model_config")
    state_dict = package.get("state_dict")
    if not isinstance(model_config, dict):
        raise ImageArtifactError("model_config must be a dictionary.")
    if "number_of_classes" not in model_config:
        raise ImageArtifactError(
            "model_config is missing number_of_classes."
        )
    if not isinstance(state_dict, dict):
        raise ImageArtifactError("state_dict must be a dictionary.")

    number_of_classes = int(model_config["number_of_classes"])
    if architecture == "small_cnn":
        # Local imports avoid a module cycle with adapters.image_base.
        from adapters.image_small_cnn import SmallCNN

        model = SmallCNN(number_of_classes)
    elif architecture == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            number_of_classes,
        )
    elif architecture == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(
            input_features,
            number_of_classes,
        )
    else:
        raise ImageArtifactError(
            f"Unsupported image architecture: {architecture}"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ImageArtifactError(
            "The saved weights do not match architecture "
            f"{architecture}: {error}"
        ) from error

    model.eval()
    return model


def write_artifact_manifest(package, output_path):
    manifest = {
        key: value
        for key, value in package.items()
        if key != "state_dict"
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path
