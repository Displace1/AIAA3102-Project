import json
import tempfile
import unittest
from pathlib import Path

import torch

from adapters.image_mobilenet_v3 import MobileNetV3Adapter
from adapters.image_resnet18 import ResNet18Adapter
from adapters.image_small_cnn import SmallCNNAdapter
from superagent.image_artifact import (
    ImageArtifactError,
    build_model_from_artifact,
    load_image_artifact,
    save_image_artifact,
    write_artifact_manifest,
)


class ImageArtifactTests(unittest.TestCase):
    def test_round_trip_reconstructs_all_supported_architectures(self):
        specifications = [
            (SmallCNNAdapter(), "small_cnn"),
            (ResNet18Adapter(), "resnet18"),
            (MobileNetV3Adapter(), "mobilenet_v3_small"),
        ]
        inputs = torch.randn(2, 3, 64, 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for adapter, architecture in specifications:
                original, _ = adapter.build_model(
                    number_of_classes=3,
                    hyperparameters={"pretrained": False},
                )
                original.eval()
                artifact_path = root / f"{architecture}.pt"
                save_image_artifact(
                    path=artifact_path,
                    model=original,
                    adapter_name=adapter.adapter_name,
                    architecture=architecture,
                    model_config=self._model_config(),
                )

                package = load_image_artifact(artifact_path)
                reconstructed = build_model_from_artifact(package)
                with torch.inference_mode():
                    expected = original(inputs)
                    actual = reconstructed(inputs)
                self.assertTrue(torch.equal(expected, actual))

    def test_writes_json_manifest_without_weights(self):
        model, _ = SmallCNNAdapter().build_model(3, {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "model.pt"
            manifest_path = root / "artifact_manifest.json"
            save_image_artifact(
                path=artifact_path,
                model=model,
                adapter_name="image_small_cnn",
                architecture="small_cnn",
                model_config=self._model_config(),
            )
            package = load_image_artifact(artifact_path)
            write_artifact_manifest(package, manifest_path)

            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertNotIn("state_dict", manifest)
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(
                manifest["preprocessing"]["color_mode"], "RGB"
            )

    def test_rejects_unknown_architecture(self):
        package = {
            "architecture": "unknown_network",
            "model_config": {"number_of_classes": 2},
            "state_dict": {},
        }
        with self.assertRaisesRegex(
            ImageArtifactError, "Unsupported image architecture"
        ):
            build_model_from_artifact(package)

    @staticmethod
    def _model_config():
        return {
            "number_of_classes": 3,
            "image_size": 64,
            "class_to_index": {"a": 0, "b": 1, "c": 2},
            "index_to_class": {0: "a", 1: "b", 2: "c"},
            "pretrained": False,
            "data_augmentation": False,
            "random_seed": 42,
        }


if __name__ == "__main__":
    unittest.main()
