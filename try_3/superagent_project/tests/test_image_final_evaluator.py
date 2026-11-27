import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from superagent.final_evaluator import FinalEvaluationError, FinalEvaluator


class ImageFinalEvaluatorTests(unittest.TestCase):
    def test_reads_image_test_labels_only_during_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            test = root / "test"
            for split in (train, test):
                for class_name in ("cat", "dog"):
                    (split / class_name).mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(test / "cat" / "a.png")
            Image.new("RGB", (8, 8)).save(test / "dog" / "b.png")
            config = SimpleNamespace(
                data={
                    "format": "image_directory_split",
                    "train_directory": str(train),
                    "test_directory": str(test),
                }
            )

            labels = FinalEvaluator._load_test_labels(config)

            self.assertEqual(labels.tolist(), [0, 1])

    def test_rejects_second_final_evaluation(self):
        lock = {
            "selected_experiment": "exp_001",
            "selected_model": "small_cnn",
            "selection_metric": "accuracy",
            "validation_accuracy": 0.8,
            "artifact_path": "model_package.pt",
            "selected_automatically": True,
            "test_used_for_selection": False,
            "solution_locked": True,
            "test_evaluation_completed": True,
        }
        with self.assertRaisesRegex(
            FinalEvaluationError, "already been completed"
        ):
            FinalEvaluator._validate_lock(lock, Path("unused"))


if __name__ == "__main__":
    unittest.main()
