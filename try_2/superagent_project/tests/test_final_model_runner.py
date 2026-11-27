import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from adapters.tabular_baseline import normalize_categorical_array
from adapters.text_pair_baseline import build_pair_features
from adapters.image_small_cnn import SmallCNN
from superagent.final_model_runner import (
    FinalModelInferenceError,
    FinalModelRunner,
)
from superagent.image_artifact import save_image_artifact
from superagent.predictor import PredictionError


class FinalModelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def prepare_package(self, name, adapter_name, artifact):
        run_directory = self.root / name
        final_directory = run_directory / "final"
        final_directory.mkdir(parents=True)
        model_path = final_directory / "final_model.joblib"
        joblib.dump(artifact, model_path)
        self.write_json(
            final_directory / "final_solution.json",
            {
                "selected_experiment": "exp_001",
                "adapter_name": adapter_name,
            },
        )
        self.write_json(
            final_directory / "artifact_hashes.json",
            {"final_model_sha256": self.sha256(model_path)},
        )
        self.write_json(
            run_directory / "solution_lock.json",
            {
                "selected_experiment": "exp_001",
                "solution_locked": True,
                "test_evaluation_completed": True,
                "test_used_for_selection": False,
            },
        )
        return run_directory

    def make_tabular_artifact(self):
        features = np.asarray(
            [["red"], ["red"], ["blue"], ["blue"]], dtype=object
        )
        labels = np.asarray([1, 1, 0, 0], dtype=int)
        pipeline = Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )
        pipeline.fit(normalize_categorical_array(features), labels)
        return pipeline

    def make_text_artifact(self):
        first = ["very good", "very bad", "great", "awful"]
        second = ["very bad", "very good", "awful", "great"]
        labels = np.asarray([1, 0, 1, 0], dtype=int)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(first + second)
        classifier = LogisticRegression(random_state=42)
        classifier.fit(build_pair_features(vectorizer, first, second), labels)
        return {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "feature_columns": ["headline_A", "headline_B"],
        }

    def prepare_image_package(self):
        run_directory = self.root / "image_run"
        final_directory = run_directory / "final"
        final_directory.mkdir(parents=True)
        model_path = final_directory / "final_model.pt"
        model = SmallCNN(number_of_classes=2)
        save_image_artifact(
            path=model_path,
            model=model,
            adapter_name="image_small_cnn",
            architecture="small_cnn",
            model_config={
                "number_of_classes": 2,
                "image_size": 16,
                "batch_size": 2,
                "class_to_index": {"dark": 0, "light": 1},
                "index_to_class": {0: "dark", 1: "light"},
            },
        )
        self.write_json(
            final_directory / "final_solution.json",
            {
                "selected_experiment": "exp_001",
                "adapter_name": "image_small_cnn",
                "model_filename": "final_model.pt",
            },
        )
        self.write_json(
            final_directory / "artifact_hashes.json",
            {"final_model_sha256": self.sha256(model_path)},
        )
        self.write_json(
            run_directory / "solution_lock.json",
            {
                "selected_experiment": "exp_001",
                "solution_locked": True,
                "test_evaluation_completed": True,
                "test_used_for_selection": False,
            },
        )
        return run_directory

    def test_tabular_inference_is_portable_and_writes_consistent_outputs(self):
        run_directory = self.prepare_package(
            "tabular_run",
            "tabular_one_hot_logistic",
            self.make_tabular_artifact(),
        )
        moved_directory = self.root / "moved_tabular_run"
        shutil.move(str(run_directory), str(moved_directory))
        input_path = self.root / "new_features.npy"
        np.save(
            input_path,
            np.asarray([["red"], ["blue"], ["red"]], dtype=object),
        )
        output_directory = self.root / "tabular_predictions"

        result = FinalModelRunner().predict(
            moved_directory, input_path, output_directory
        )

        self.assertEqual(result.prediction_count, 3)
        npy_predictions = np.load(result.predictions_npy_path)
        with Path(result.predictions_csv_path).open(
            "r", encoding="utf-8", newline=""
        ) as file:
            csv_predictions = [
                int(row["prediction"]) for row in csv.DictReader(file)
            ]
        self.assertEqual(npy_predictions.tolist(), csv_predictions)
        manifest = json.loads(
            Path(result.manifest_path).read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["labels_read"])
        self.assertFalse(manifest["llm_called"])

    def test_text_inference_accepts_unlabeled_csv(self):
        run_directory = self.prepare_package(
            "text_run",
            "text_pair_word_tfidf_logistic",
            self.make_text_artifact(),
        )
        input_path = self.root / "headline_pairs.csv"
        with input_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["headline_A", "headline_B"])
            writer.writerow(["very good", "very bad"])
            writer.writerow(["awful", "great"])

        result = FinalModelRunner().predict(
            run_directory, input_path, self.root / "text_predictions"
        )

        self.assertEqual(result.prediction_count, 2)

    def test_image_inference_accepts_image_directory(self):
        run_directory = self.prepare_image_package()
        input_directory = self.root / "new_images"
        input_directory.mkdir()
        Image.new("RGB", (16, 16), (0, 0, 0)).save(
            input_directory / "a.png"
        )
        Image.new("L", (20, 20), 255).save(
            input_directory / "b.png"
        )

        result = FinalModelRunner().predict(
            run_directory,
            input_directory,
            self.root / "image_predictions",
        )

        self.assertEqual(result.prediction_count, 2)
        predictions = np.load(result.predictions_npy_path)
        self.assertEqual(tuple(predictions.shape), (2,))
        with Path(result.predictions_csv_path).open(
            "r", encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(
            list(rows[0]),
            ["row_id", "input_path", "prediction", "class_name"],
        )
        self.assertEqual(
            [Path(row["input_path"]).name for row in rows],
            ["a.png", "b.png"],
        )
        self.assertEqual(
            [row["class_name"] for row in rows],
            [
                "dark" if int(row["prediction"]) == 0 else "light"
                for row in rows
            ],
        )
        manifest = json.loads(
            Path(result.manifest_path).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["input_kind"], "directory")
        self.assertTrue(manifest["input_paths_included"])
        self.assertTrue(manifest["class_names_included"])
        self.assertFalse(manifest["labels_read"])

    def test_rejects_tampered_model(self):
        run_directory = self.prepare_package(
            "tampered_run",
            "tabular_one_hot_logistic",
            self.make_tabular_artifact(),
        )
        with (run_directory / "final" / "final_model.joblib").open("ab") as file:
            file.write(b"tampered")
        input_path = self.root / "tampered_input.npy"
        np.save(input_path, np.asarray([["red"]], dtype=object))

        with self.assertRaisesRegex(
            FinalModelInferenceError, "SHA-256"
        ):
            FinalModelRunner().predict(
                run_directory, input_path, self.root / "tampered_output"
            )

    def test_rejects_nonempty_output_directory(self):
        run_directory = self.prepare_package(
            "existing_output_run",
            "tabular_one_hot_logistic",
            self.make_tabular_artifact(),
        )
        input_path = self.root / "existing_output_input.npy"
        np.save(input_path, np.asarray([["red"]], dtype=object))
        output_directory = self.root / "existing_output"
        output_directory.mkdir()
        (output_directory / "keep.txt").write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(
            FinalModelInferenceError, "not empty"
        ):
            FinalModelRunner().predict(
                run_directory, input_path, output_directory
            )

    def test_rejects_text_input_missing_required_column(self):
        run_directory = self.prepare_package(
            "missing_column_run",
            "text_pair_word_tfidf_logistic",
            self.make_text_artifact(),
        )
        input_path = self.root / "missing_column.csv"
        input_path.write_text("headline_A\nonly one column\n", encoding="utf-8")

        with self.assertRaisesRegex(PredictionError, "headline_B"):
            FinalModelRunner().predict(
                run_directory, input_path, self.root / "missing_column_output"
            )


if __name__ == "__main__":
    unittest.main()
