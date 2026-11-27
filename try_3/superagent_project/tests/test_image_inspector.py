import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from superagent.data_inspector import DataInspector
from superagent.image_inspector import ImageInspectionError, ImageInspector


class ImageInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train = self.root / "train"
        self.validation = self.root / "validation"
        self.test = self.root / "test"
        for split in (self.train, self.validation):
            for class_name in ("cat", "dog"):
                (split / class_name).mkdir(parents=True)
        (self.test / "private_test_folder").mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def create_image(path, mode="RGB", size=(12, 10), image_format=None):
        color = 128 if mode == "L" else (10, 20, 30)
        Image.new(mode, size, color=color).save(path, format=image_format)

    def config(self):
        return SimpleNamespace(
            metric="accuracy",
            data={
                "format": "image_directory_split",
                "train_directory": str(self.train),
                "validation_directory": str(self.validation),
                "test_directory": str(self.test),
            },
        )

    def populate_valid_dataset(self):
        self.create_image(self.train / "cat" / "cat_rgb.PNG")
        self.create_image(self.train / "cat" / "cat_gray.png", mode="L")
        self.create_image(self.train / "dog" / "dog.jpg", size=(20, 18))
        self.create_image(self.validation / "cat" / "cat.png")
        self.create_image(self.validation / "dog" / "dog.webp")
        self.create_image(
            self.test / "private_test_folder" / "unknown.png", mode="L"
        )

    def test_inspects_image_splits_without_test_label_leakage(self):
        self.populate_valid_dataset()

        profile = ImageInspector().inspect(self.config())

        self.assertEqual(profile["modality"], "image")
        self.assertEqual(profile["task_type"], "binary_classification")
        self.assertEqual(profile["number_of_classes"], 2)
        self.assertEqual(profile["class_names"], ["cat", "dog"])
        self.assertEqual(
            profile["train_class_distribution"], {"cat": 2, "dog": 1}
        )
        self.assertEqual(
            profile["validation_class_distribution"], {"cat": 1, "dog": 1}
        )
        self.assertEqual(profile["test_samples"], 1)
        self.assertEqual(
            profile["image_summary"]["train"]["grayscale_samples"], 1
        )
        self.assertEqual(
            profile["image_summary"]["train"]["maximum_width"], 20
        )
        self.assertFalse(profile["test_labels_inspected"])
        self.assertNotIn("private_test_folder", json.dumps(profile))

    def test_data_inspector_routes_image_format(self):
        self.populate_valid_dataset()
        profile = DataInspector().inspect(self.config())
        self.assertEqual(profile["modality"], "image")

    def test_rejects_train_validation_class_mismatch(self):
        (self.validation / "dog").rmdir()
        (self.validation / "bird").mkdir()

        with self.assertRaisesRegex(
            ImageInspectionError, "class directories differ"
        ):
            ImageInspector().inspect(self.config())

    def test_rejects_corrupt_images(self):
        self.populate_valid_dataset()
        (self.train / "cat" / "broken.png").write_bytes(b"not an image")

        with self.assertRaisesRegex(
            ImageInspectionError, "corrupt train images"
        ):
            ImageInspector().inspect(self.config())


if __name__ == "__main__":
    unittest.main()
