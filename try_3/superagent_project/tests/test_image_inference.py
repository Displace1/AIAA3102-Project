import tempfile
import unittest
from pathlib import Path

from PIL import Image

from adapters.image_small_cnn import SmallCNN
from superagent.image_artifact import save_image_artifact
from superagent.predictor import ArtifactPredictor


class ImageInferenceTests(unittest.TestCase):
    def test_unlabeled_directory_returns_paths_predictions_and_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "model.pt"
            input_directory = root / "images"
            input_directory.mkdir()
            Image.new("RGB", (16, 16), (10, 20, 30)).save(
                input_directory / "rgb.png"
            )
            Image.new("L", (16, 16), 128).save(
                input_directory / "gray.png"
            )
            save_image_artifact(
                path=artifact_path,
                model=SmallCNN(number_of_classes=2),
                adapter_name="image_small_cnn",
                architecture="small_cnn",
                model_config={
                    "number_of_classes": 2,
                    "image_size": 16,
                    "batch_size": 2,
                    "class_to_index": {"cat": 0, "dog": 1},
                    "index_to_class": {0: "cat", 1: "dog"},
                },
            )

            batch = ArtifactPredictor().predict_input_with_metadata(
                artifact_path=artifact_path,
                adapter_name="image_small_cnn",
                input_path=input_directory,
            )

            self.assertEqual(len(batch.predictions), 2)
            self.assertEqual(len(batch.input_paths), 2)
            self.assertEqual(len(batch.class_names), 2)
            self.assertEqual(
                {Path(path).name for path in batch.input_paths},
                {"gray.png", "rgb.png"},
            )
            for prediction, class_name in zip(
                batch.predictions, batch.class_names
            ):
                self.assertEqual(
                    class_name,
                    "cat" if int(prediction) == 0 else "dog",
                )


if __name__ == "__main__":
    unittest.main()
