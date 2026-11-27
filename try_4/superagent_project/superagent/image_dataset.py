import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from superagent.image_inspector import SUPPORTED_IMAGE_EXTENSIONS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageDataLoaders:
    train: DataLoader
    validation: DataLoader
    class_to_index: dict
    index_to_class: dict


def convert_image_to_rgb(image):
    """Pickle-safe RGB conversion for multiprocessing DataLoader workers."""

    return image.convert("RGB")


def build_image_transform(image_size, training, data_augmentation):
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    operations = []
    if training and data_augmentation:
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.85, 1.15),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )
    else:
        resize_size = max(image_size, int(round(image_size * 1.14)))
        operations.extend(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
            ]
        )

    operations.extend(
        [
            transforms.Lambda(convert_image_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_image_dataloaders(
    config,
    data_profile,
    image_size,
    batch_size,
    data_augmentation,
    num_workers=None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if data_profile.get("modality") != "image":
        raise ValueError("Image data loaders require an image data profile.")

    if num_workers is None:
        # The experiment already runs in an isolated spawned process.
        # Spawning nested DataLoader workers on Windows loads PyTorch again
        # in every worker and can consume several extra gigabytes of RAM.
        num_workers = (
            0
            if os.name == "nt"
            else min(config.maximum_cpu_threads, 4)
        )
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_dataset = datasets.ImageFolder(
        config.data["train_directory"],
        transform=build_image_transform(
            image_size=image_size,
            training=True,
            data_augmentation=data_augmentation,
        ),
    )
    validation_dataset = datasets.ImageFolder(
        config.data["validation_directory"],
        transform=build_image_transform(
            image_size=image_size,
            training=False,
            data_augmentation=False,
        ),
    )

    expected_mapping = data_profile["class_to_index"]
    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Training class mapping differs from the inspected data profile."
        )
    if validation_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Validation class mapping differs from the training mapping."
        )

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    common_loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_arguments,
    )
    return ImageDataLoaders(
        train=train_loader,
        validation=validation_loader,
        class_to_index=dict(expected_mapping),
        index_to_class={
            index: name for name, index in expected_mapping.items()
        },
    )


class UnlabeledImageDataset(Dataset):
    def __init__(self, input_path, image_size):
        self.input_path = Path(input_path).expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Image inference input does not exist: {self.input_path}"
            )
        if self.input_path.is_file():
            candidates = [self.input_path]
        else:
            candidates = sorted(
                path for path in self.input_path.rglob("*") if path.is_file()
            )
        self.paths = [
            path
            for path in candidates
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(
                f"No supported images found for inference: {self.input_path}"
            )
        self.transform = build_image_transform(
            image_size=image_size,
            training=False,
            data_augmentation=False,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image)
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Cannot load image {path}: {error}") from error
        return tensor, str(path)


def build_image_inference_loader(
    input_path,
    image_size,
    batch_size,
    num_workers=0,
):
    dataset = UnlabeledImageDataset(input_path, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
