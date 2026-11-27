import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from adapters.tabular_baseline import normalize_categorical_array
from adapters.image_small_cnn import SmallCNN
from adapters.text_pair_baseline import build_pair_features
from scripts.verify_final_artifacts import verify_final_artifacts
from superagent.final_evaluator import FinalEvaluationError, FinalEvaluator
from superagent.image_artifact import save_image_artifact
from superagent.recovery import configuration_sha256


class FinalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def prepare_run(self, name, config, adapter_name, artifact):
        run_directory = self.root / name
        experiment_directory = run_directory / "experiments" / "exp_001"
        experiment_directory.mkdir(parents=True)
        if isinstance(artifact, Path):
            artifact_path = experiment_directory / "model_package.pt"
            shutil.copy2(artifact, artifact_path)
        else:
            artifact_path = experiment_directory / "model.joblib"
            joblib.dump(artifact, artifact_path)

        self.write_json(
            run_directory / "run_manifest.json",
            {
                "run_id": name,
                "task_name": config.name,
                "config_path": str(config.source_path),
                "config_sha256": configuration_sha256(config.source_path),
            },
        )
        self.write_json(
            experiment_directory / "plan.json",
            {
                "experiment_id": "exp_001",
                "adapter_name": adapter_name,
                "hyperparameters": {},
                "reason": "Synthetic final evaluation test.",
            },
        )
        self.write_json(
            run_directory / "solution_lock.json",
            {
                "selected_experiment": "exp_001",
                "selected_model": "synthetic_model",
                "selection_metric": "accuracy",
                "validation_accuracy": 0.75,
                "artifact_path": str(artifact_path.resolve()),
                "selected_automatically": True,
                "test_used_for_selection": False,
                "solution_locked": True,
                "test_evaluation_completed": False,
            },
        )
        return run_directory

    def make_tabular_fixture(self):
        config_path = self.root / "tabular_config.json"
        config_path.write_text('{"task": "tabular"}', encoding="utf-8")
        X_train = np.asarray(
            [["red"], ["red"], ["blue"], ["blue"], ["red"], ["blue"]],
            dtype=object,
        )
        y_train = np.asarray([1, 1, 0, 0, 1, 0], dtype=int)
        X_test = np.asarray(
            [["red"], ["blue"], ["red"], ["blue"]], dtype=object
        )
        y_test = np.asarray([1, 0, 1, 0], dtype=int)
        test_features = self.root / "tabular_test_features.npy"
        test_labels = self.root / "tabular_test_labels.npy"
        np.save(test_features, X_test)
        np.save(test_labels, y_test)

        pipeline = Pipeline(
            [
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )
        pipeline.fit(normalize_categorical_array(X_train), y_train)
        config = SimpleNamespace(
            name="synthetic_tabular",
            source_path=config_path,
            data={
                "format": "npy_split",
                "test_features": str(test_features),
                "test_labels": str(test_labels),
            },
        )
        return config, pipeline

    def make_text_fixture(self):
        config_path = self.root / "text_config.json"
        config_path.write_text('{"task": "text"}', encoding="utf-8")
        train_a = ["very good", "very bad", "great story", "awful story"]
        train_b = ["very bad", "very good", "awful story", "great story"]
        labels = np.asarray([1, 0, 1, 0], dtype=int)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectorizer.fit(train_a + train_b)
        classifier = LogisticRegression(random_state=42)
        classifier.fit(
            build_pair_features(vectorizer, train_a, train_b), labels
        )
        test_path = self.root / "text_test.csv"
        with test_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["headline_A", "headline_B", "label_pairwise"])
            writer.writerow(["very good", "very bad", 1])
            writer.writerow(["awful story", "great story", 0])

        config = SimpleNamespace(
            name="synthetic_text",
            source_path=config_path,
            data={
                "format": "csv_split",
                "test": str(test_path),
                "target_column": "label_pairwise",
            },
        )
        artifact = {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "feature_columns": ["headline_A", "headline_B"],
            "target_column": "label_pairwise",
        }
        return config, artifact

    def make_image_fixture(self):
        config_path = self.root / "image_config.json"
        config_path.write_text('{"task": "image"}', encoding="utf-8")
        train_directory = self.root / "image_train"
        test_directory = self.root / "image_test"
        for root in (train_directory, test_directory):
            for class_name in ("dark", "light"):
                (root / class_name).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (16, 16), (index, index, index)).save(
                test_directory / "dark" / f"dark_{index}.png"
            )
            Image.new("RGB", (16, 16), (255, 255, 255)).save(
                test_directory / "light" / f"light_{index}.png"
            )

        artifact_path = self.root / "source_image_model.pt"
        save_image_artifact(
            path=artifact_path,
            model=SmallCNN(number_of_classes=2),
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
        config = SimpleNamespace(
            name="synthetic_image",
            source_path=config_path,
            data={
                "format": "image_directory_split",
                "train_directory": str(train_directory),
                "test_directory": str(test_directory),
            },
        )
        return config, artifact_path

    def test_tabular_final_evaluation_exports_and_locks_once(self):
        config, artifact = self.make_tabular_fixture()
        run_directory = self.prepare_run(
            "tabular_run", config, "tabular_one_hot_logistic", artifact
        )

        result = FinalEvaluator().evaluate(run_directory, config)

        self.assertEqual(result.test_samples, 4)
        self.assertEqual(result.test_accuracy, 1.0)
        self.assertTrue(Path(result.final_model_path).is_file())
        self.assertTrue(Path(result.predictions_path).is_file())
        loaded = joblib.load(result.final_model_path)
        self.assertTrue(hasattr(loaded, "predict"))
        lock = json.loads(
            (run_directory / "solution_lock.json").read_text(encoding="utf-8")
        )
        self.assertTrue(lock["test_evaluation_completed"])
        self.assertFalse(lock["test_used_for_selection"])
        verification = verify_final_artifacts(run_directory)
        self.assertTrue(verification["verified"])
        self.assertEqual(verification["test_samples"], 4)
        with self.assertRaisesRegex(
            FinalEvaluationError, "already been completed"
        ):
            FinalEvaluator().evaluate(run_directory, config)

    def test_text_pair_final_evaluation(self):
        config, artifact = self.make_text_fixture()
        run_directory = self.prepare_run(
            "text_run", config, "text_pair_word_tfidf_logistic", artifact
        )

        result = FinalEvaluator().evaluate(run_directory, config)

        self.assertEqual(result.test_samples, 2)
        self.assertGreaterEqual(result.test_accuracy, 0.5)
        predictions = np.load(
            run_directory / "final" / "test_predictions.npy"
        )
        self.assertEqual(len(predictions), 2)
        hashes = json.loads(
            (run_directory / "final" / "artifact_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hashes["source_model_sha256"], hashes["final_model_sha256"]
        )

    def test_image_final_evaluation_and_artifact_verification(self):
        config, artifact_path = self.make_image_fixture()
        run_directory = self.prepare_run(
            "image_run", config, "image_small_cnn", artifact_path
        )

        result = FinalEvaluator().evaluate(run_directory, config)

        self.assertEqual(result.test_samples, 4)
        self.assertTrue(result.final_model_path.endswith("final_model.pt"))
        verification = verify_final_artifacts(run_directory)
        self.assertTrue(verification["verified"])

    def test_rejects_missing_solution_lock(self):
        config, _ = self.make_tabular_fixture()
        run_directory = self.root / "unlocked_run"
        run_directory.mkdir()
        self.write_json(
            run_directory / "run_manifest.json",
            {
                "task_name": config.name,
                "config_path": str(config.source_path),
                "config_sha256": configuration_sha256(config.source_path),
            },
        )
        with self.assertRaisesRegex(FinalEvaluationError, "Missing solution"):
            FinalEvaluator().evaluate(run_directory, config)

    def test_rejects_changed_configuration(self):
        config, artifact = self.make_tabular_fixture()
        run_directory = self.prepare_run(
            "changed_config_run",
            config,
            "tabular_one_hot_logistic",
            artifact,
        )
        config.source_path.write_text('{"changed": true}', encoding="utf-8")
        with self.assertRaisesRegex(FinalEvaluationError, "changed"):
            FinalEvaluator().evaluate(run_directory, config)


if __name__ == "__main__":
    unittest.main()
