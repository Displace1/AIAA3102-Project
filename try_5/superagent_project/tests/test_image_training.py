import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from superagent.image_dataset import (
    UnlabeledImageDataset,
    build_image_dataloaders,
)
from superagent.image_inspector import ImageInspector
from superagent.image_training import (
    select_training_device,
    train_image_classifier,
)


class TinyImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, inputs):
        return self.network(inputs)


class ImageTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train = self.root / "train"
        self.validation = self.root / "validation"
        self.test = self.root / "test"
        for split in (self.train, self.validation):
            for class_name in ("dark", "light"):
                (split / class_name).mkdir(parents=True)
        self.test.mkdir()
        for index in range(4):
            self._image(self.train / "dark" / f"dark_{index}.png", index)
            self._image(
                self.train / "light" / f"light_{index}.png", 255 - index
            )
        for index in range(2):
            self._image(self.validation / "dark" / f"dark_{index}.png", 0)
            self._image(
                self.validation / "light" / f"light_{index}.png", 255
            )
        self._image(self.test / "gray.png", 128, mode="L")
        self.config = SimpleNamespace(
            random_seed=42,
            maximum_cpu_threads=1,
            metric="accuracy",
            data={
                "train_directory": str(self.train),
                "validation_directory": str(self.validation),
                "test_directory": str(self.test),
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _image(path, value, mode="RGB"):
        color = value if mode == "L" else (value, value, value)
        Image.new(mode, (16, 16), color=color).save(path)

    def test_builds_deterministic_loaders_and_converts_grayscale(self):
        profile = ImageInspector().inspect(self.config)
        loaders = build_image_dataloaders(
            config=self.config,
            data_profile=profile,
            image_size=16,
            batch_size=2,
            data_augmentation=False,
            num_workers=0,
        )
        images, labels = next(iter(loaders.train))
        self.assertEqual(tuple(images.shape[1:]), (3, 16, 16))
        self.assertEqual(len(labels), 2)

        unlabeled = UnlabeledImageDataset(self.test, image_size=16)
        tensor, path = unlabeled[0]
        self.assertEqual(tuple(tensor.shape), (3, 16, 16))
        self.assertTrue(path.endswith("gray.png"))

    def test_trains_one_cpu_epoch_and_reloads_best_weights(self):
        profile = ImageInspector().inspect(self.config)
        loaders = build_image_dataloaders(
            config=self.config,
            data_profile=profile,
            image_size=16,
            batch_size=4,
            data_augmentation=False,
            num_workers=0,
        )
        model = TinyImageClassifier()
        result = train_image_classifier(
            model=model,
            data_loaders=loaders,
            output_directory=self.root / "training_output",
            number_of_classes=2,
            learning_rate=0.01,
            epochs=1,
            weight_decay=0.0,
            early_stopping_patience=1,
            random_seed=42,
            requested_device="cpu",
            use_mixed_precision=False,
        )

        self.assertEqual(result.epochs_completed, 1)
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.peak_gpu_memory_mb, 0.0)
        self.assertTrue(Path(result.best_weights_path).is_file())
        self.assertEqual(len(result.confusion_matrix), 2)

    def test_early_stopping_records_completed_epochs(self):
        profile = ImageInspector().inspect(self.config)
        loaders = build_image_dataloaders(
            config=self.config,
            data_profile=profile,
            image_size=16,
            batch_size=4,
            data_augmentation=False,
            num_workers=0,
        )
        fixed_metrics = {
            "loss": 0.5,
            "accuracy": 0.75,
            "precision_macro": 0.75,
            "recall_macro": 0.75,
            "f1_macro": 0.75,
            "confusion_matrix": [[1, 1], [0, 2]],
            "class_recall": {"0": 0.5, "1": 1.0},
        }
        with patch(
            "superagent.image_training.evaluate_image_model",
            return_value=fixed_metrics,
        ):
            result = train_image_classifier(
                model=TinyImageClassifier(),
                data_loaders=loaders,
                output_directory=self.root / "early_stopping_output",
                number_of_classes=2,
                learning_rate=0.01,
                epochs=5,
                weight_decay=0.0,
                early_stopping_patience=1,
                random_seed=42,
                requested_device="cpu",
                use_mixed_precision=False,
            )

        self.assertTrue(result.early_stopped)
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.epochs_completed, 2)

    def test_auto_device_falls_back_to_cpu_without_cuda(self):
        with patch(
            "superagent.image_training.torch.cuda.is_available",
            return_value=False,
        ):
            device = select_training_device("auto")
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
