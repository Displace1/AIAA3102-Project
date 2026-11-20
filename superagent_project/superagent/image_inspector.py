from collections import Counter
from pathlib import Path
from statistics import median

from PIL import Image, UnidentifiedImageError
import torch


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageInspectionError(ValueError):
    pass


class ImageInspector:
    """Inspect directory-based image classification data without test leakage."""

    def inspect(self, config):
        train_directory = Path(config.data["train_directory"])
        validation_directory = Path(config.data["validation_directory"])
        test_directory = Path(config.data["test_directory"])

        train_classes = self._class_directories(train_directory, "train")
        validation_classes = self._class_directories(
            validation_directory, "validation"
        )
        if set(train_classes) != set(validation_classes):
            missing = sorted(set(train_classes) - set(validation_classes))
            unexpected = sorted(set(validation_classes) - set(train_classes))
            raise ImageInspectionError(
                "Train and validation class directories differ. "
                f"Missing in validation: {missing}; "
                f"unexpected in validation: {unexpected}."
            )
        if len(train_classes) < 2:
            raise ImageInspectionError(
                "Image classification requires at least two train classes."
            )

        train_distribution, train_paths = self._labeled_split(
            train_directory, train_classes, "train"
        )
        validation_distribution, validation_paths = self._labeled_split(
            validation_directory, train_classes, "validation"
        )
        test_paths = self._image_paths(test_directory)
        if not test_paths:
            raise ImageInspectionError(
                f"Test directory contains no supported images: {test_directory}"
            )

        train_summary = self._inspect_paths(
            train_paths, train_directory, "train", reveal_paths=True
        )
        validation_summary = self._inspect_paths(
            validation_paths,
            validation_directory,
            "validation",
            reveal_paths=True,
        )
        test_summary = self._inspect_paths(
            test_paths, test_directory, "test", reveal_paths=False
        )

        class_names = sorted(train_classes)
        return {
            "modality": "image",
            "task_type": (
                "binary_classification"
                if len(class_names) == 2
                else "multiclass_classification"
            ),
            "data_format": "image_directory_split",
            "train_samples": len(train_paths),
            "validation_samples": len(validation_paths),
            "test_samples": len(test_paths),
            "number_of_classes": len(class_names),
            "class_names": class_names,
            "class_to_index": {
                name: index for index, name in enumerate(class_names)
            },
            "class_distribution": train_distribution,
            "train_class_distribution": train_distribution,
            "validation_class_distribution": validation_distribution,
            "supported_extensions": sorted(SUPPORTED_IMAGE_EXTENSIONS),
            "image_summary": {
                "train": train_summary,
                "validation": validation_summary,
                "test": test_summary,
            },
            "test_labels_inspected": False,
            "metric": config.metric,
            "execution_context": {
                "experiment_timeout_seconds": (
                    getattr(config, "experiment_timeout_seconds", 1800)
                ),
                "maximum_cpu_threads": getattr(
                    config, "maximum_cpu_threads", 2
                ),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        }

    @staticmethod
    def _class_directories(directory, split_name):
        classes = {
            path.name: path
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if not classes:
            raise ImageInspectionError(
                f"{split_name} directory has no class subdirectories: "
                f"{directory}"
            )
        return classes

    def _labeled_split(self, root, classes, split_name):
        distribution = {}
        paths = []
        for class_name in sorted(classes):
            class_paths = self._image_paths(root / class_name)
            if not class_paths:
                raise ImageInspectionError(
                    f"{split_name} class contains no supported images: "
                    f"{class_name}"
                )
            distribution[class_name] = len(class_paths)
            paths.extend(class_paths)
        return distribution, sorted(paths)

    @staticmethod
    def _image_paths(directory):
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    @staticmethod
    def _inspect_paths(paths, root, split_name, reveal_paths):
        extension_counts = Counter()
        mode_counts = Counter()
        size_counts = Counter()
        widths = []
        heights = []
        corrupt = []

        for path in paths:
            extension_counts[path.suffix.lower()] += 1
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                    image.verify()
                widths.append(int(width))
                heights.append(int(height))
                mode_counts[mode] += 1
                size_counts[f"{width}x{height}"] += 1
            except (OSError, SyntaxError, UnidentifiedImageError) as error:
                identifier = (
                    str(path.relative_to(root))
                    if reveal_paths
                    else path.name
                )
                corrupt.append(f"{identifier}: {type(error).__name__}")

        if corrupt:
            preview = corrupt[:5]
            raise ImageInspectionError(
                f"Detected {len(corrupt)} corrupt {split_name} images: {preview}"
            )
        if not widths:
            raise ImageInspectionError(
                f"No readable images were found in the {split_name} split."
            )

        common_sizes = [
            {"size": size, "count": count}
            for size, count in size_counts.most_common(10)
        ]
        grayscale_modes = {"1", "I", "I;16", "F", "L"}
        grayscale_samples = sum(
            count
            for mode, count in mode_counts.items()
            if mode in grayscale_modes
        )
        return {
            "samples": len(paths),
            "extension_counts": dict(sorted(extension_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
            "rgb_samples": int(mode_counts.get("RGB", 0)),
            "rgba_samples": int(mode_counts.get("RGBA", 0)),
            "grayscale_samples": int(grayscale_samples),
            "minimum_width": min(widths),
            "maximum_width": max(widths),
            "minimum_height": min(heights),
            "maximum_height": max(heights),
            "median_width": int(median(widths)),
            "median_height": int(median(heights)),
            "common_sizes": common_sizes,
            "corrupt_images": 0,
        }
