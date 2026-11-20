import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from adapters.image_mobilenet_v3 import MobileNetV3Adapter
from adapters.image_resnet18 import ResNet18Adapter
from adapters.image_small_cnn import SmallCNNAdapter
from superagent.experiment_plan import ExperimentPlan
from superagent.experiment_registry import ExperimentRegistry
from superagent.image_inspector import ImageInspector
from superagent.plan_validator import PlanValidator


class ImageAdapterTests(unittest.TestCase):
    def test_registry_builds_all_three_image_adapters(self):
        registry = ExperimentRegistry(random_seed=7)
        expected = {
            "image_small_cnn": SmallCNNAdapter,
            "image_resnet18": ResNet18Adapter,
            "image_mobilenet_v3_small": MobileNetV3Adapter,
        }
        for adapter_name, adapter_type in expected.items():
            self.assertIsInstance(
                registry.adapter_for(adapter_name), adapter_type
            )

    def test_image_models_produce_one_logit_per_class(self):
        specifications = [
            (
                SmallCNNAdapter(),
                {
                    "pretrained": False,
                },
            ),
            (
                ResNet18Adapter(),
                {
                    "pretrained": False,
                },
            ),
            (
                MobileNetV3Adapter(),
                {
                    "pretrained": False,
                },
            ),
        ]
        inputs = torch.zeros(2, 3, 64, 64)
        for adapter, parameters in specifications:
            model, _ = adapter.build_model(3, parameters)
            model.eval()
            with torch.inference_mode():
                outputs = model(inputs)
            self.assertEqual(tuple(outputs.shape), (2, 3))

    def test_default_image_plans_pass_validation(self):
        profile = {"modality": "image"}
        plans = ExperimentRegistry(random_seed=42).default_plans_for(profile)
        self.assertEqual(len(plans), 2)
        validated = PlanValidator().validate_many(plans, profile)
        self.assertEqual(validated, plans)

        mobile_plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="image_mobilenet_v3_small",
            hyperparameters={
                "image_size": 160,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "epochs": 8,
                "data_augmentation": True,
                "early_stopping_patience": 3,
                "weight_decay": 0.0001,
                "pretrained": True,
                "freeze_backbone_epochs": 2,
            },
            reason="Try an efficient pretrained image model.",
        )
        self.assertEqual(
            PlanValidator().validate(mobile_plan, profile), mobile_plan
        )

    def test_small_cnn_adapter_trains_and_packages_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            validation = root / "validation"
            test = root / "test"
            for split in (train, validation):
                for class_name in ("dark", "light"):
                    (split / class_name).mkdir(parents=True)
            test.mkdir()
            for index in range(4):
                self._image(train / "dark" / f"{index}.png", index)
                self._image(train / "light" / f"{index}.png", 255 - index)
            for index in range(2):
                self._image(validation / "dark" / f"{index}.png", 0)
                self._image(validation / "light" / f"{index}.png", 255)
            self._image(test / "unlabeled.png", 128)

            config = SimpleNamespace(
                random_seed=42,
                maximum_cpu_threads=0,
                metric="accuracy",
                data={
                    "train_directory": str(train),
                    "validation_directory": str(validation),
                    "test_directory": str(test),
                },
            )
            profile = ImageInspector().inspect(config)
            output = root / "experiment"
            result = SmallCNNAdapter(random_seed=42).run(
                experiment_id="exp_001",
                config=config,
                data_profile=profile,
                output_dir=output,
                hyperparameters={
                    "image_size": 16,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "data_augmentation": False,
                    "early_stopping_patience": 1,
                    "weight_decay": 0.0,
                },
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.epoch_count, 1)
            self.assertEqual(result.best_epoch, 1)
            self.assertGreaterEqual(result.peak_gpu_memory_mb, 0.0)
            self.assertFalse(result.early_stopped)
            self.assertTrue(Path(result.artifact_path).is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "artifact_manifest.json").is_file())
            self.assertTrue((output / "val_predictions.npy").is_file())
            package = torch.load(
                result.artifact_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(package["format_version"], 1)
            self.assertEqual(package["adapter_name"], "image_small_cnn")
            self.assertEqual(
                package["model_config"]["class_to_index"],
                {"dark": 0, "light": 1},
            )

    @staticmethod
    def _image(path, value):
        Image.new("RGB", (16, 16), (value, value, value)).save(path)


if __name__ == "__main__":
    unittest.main()
