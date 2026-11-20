import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "raw" / "extracted" / "bank" / "bank-full.csv"
OUTPUT_DIRECTORY = ROOT / "data"
CONFIG_PATH = ROOT / "unseen_config.json"
MANIFEST_PATH = ROOT / "split_manifest.json"
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/222/"
    "bank%2Bmarketing.zip"
)
DATASET_DOI = "10.24432/C5K306"
RANDOM_SEED = 42
LABEL_MAPPING = {"no": 0, "yes": 1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source():
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"Missing official source CSV: {RAW_CSV}")

    with RAW_CSV.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)
        if header[-1] != "y":
            raise ValueError("Expected final target column named y.")
        rows = list(reader)

    if len(rows) != 45211:
        raise ValueError(f"Expected 45211 rows, found {len(rows)}.")
    features = np.asarray([row[:-1] for row in rows], dtype=object)
    try:
        labels = np.asarray(
            [LABEL_MAPPING[row[-1]] for row in rows], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"Unexpected target label: {error.args[0]}") from error
    return header[:-1], features, labels


def split_data(features, labels):
    # First reserve the final 15% test split. The remaining 85% is split so
    # validation is also 15% of the original dataset.
    X_development, X_test, y_development, y_test = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    validation_fraction_of_development = 0.15 / 0.85
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=validation_fraction_of_development,
        random_state=RANDOM_SEED,
        stratify=y_development,
    )
    return {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
    }


def save_splits(splits):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": ("C_train.npy", "y_train.npy"),
        "validation": ("C_val.npy", "y_val.npy"),
        "test": ("C_test.npy", "y_test.npy"),
    }
    output_paths = {}
    for split_name, (features, labels) in splits.items():
        feature_name, label_name = filenames[split_name]
        feature_path = OUTPUT_DIRECTORY / feature_name
        label_path = OUTPUT_DIRECTORY / label_name
        np.save(feature_path, features, allow_pickle=True)
        np.save(label_path, labels, allow_pickle=False)
        output_paths[split_name] = {
            "features": feature_path,
            "labels": label_path,
        }
    return output_paths


def write_config(output_paths):
    config = {
        "run_type": "development_run",
        "task": {
            "name": "blind_bank_subscription_trial",
            "metric": "accuracy",
        },
        "data": {
            "format": "npy_split",
            "train_features": str(output_paths["train"]["features"]),
            "train_labels": str(output_paths["train"]["labels"]),
            "validation_features": str(
                output_paths["validation"]["features"]
            ),
            "validation_labels": str(
                output_paths["validation"]["labels"]
            ),
            "test_features": str(output_paths["test"]["features"]),
            "test_labels": str(output_paths["test"]["labels"]),
        },
        "execution": {
            "random_seed": RANDOM_SEED,
            "evaluate_test": False,
            "planner_mode": "llm",
            "initial_deterministic_experiments": 2,
            "max_experiments": 8,
            "max_runtime_minutes": 180,
            "no_improvement_patience": 3,
            "minimum_improvement": 0.001,
            "maximum_planner_errors": 2,
            "maximum_same_experiment_errors": 2,
            "experiment_timeout_seconds": 1800,
            "maximum_memory_mb": 4096,
            "maximum_cpu_threads": 2,
            "resource_poll_interval_seconds": 0.2,
        },
    }
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def distribution(labels):
    counts = Counter(int(value) for value in labels)
    return {
        "no_0": counts.get(0, 0),
        "yes_1": counts.get(1, 0),
    }


def write_manifest(feature_names, splits, output_paths):
    outer_zip = ROOT / "raw" / "bank+marketing.zip"
    manifest = {
        "dataset": "Bank Marketing",
        "provider": "UCI Machine Learning Repository",
        "source_url": SOURCE_URL,
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "raw_archive_sha256": sha256(outer_zip),
        "raw_csv_sha256": sha256(RAW_CSV),
        "classification_task": (
            "Predict whether a bank client subscribes to a term deposit."
        ),
        "target_column": "y",
        "label_mapping": LABEL_MAPPING,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "total_samples": sum(len(labels) for _, labels in splits.values()),
        "split_method": "stratified random 70/15/15",
        "random_seed": RANDOM_SEED,
        "test_policy": (
            "Test labels are prepared for the final evaluator only and must "
            "not be inspected during autonomous model selection."
        ),
        "splits": {},
    }
    for split_name, (features, labels) in splits.items():
        split_details = {
            "samples": int(len(labels)),
            "feature_shape": list(features.shape),
            "features_path": str(output_paths[split_name]["features"]),
            "labels_path": str(output_paths[split_name]["labels"]),
            "features_sha256": sha256(output_paths[split_name]["features"]),
            "labels_sha256": sha256(output_paths[split_name]["labels"]),
        }
        if split_name == "test":
            split_details["class_distribution"] = "withheld_until_final_evaluation"
        else:
            split_details["class_distribution"] = distribution(labels)
        manifest["splits"][split_name] = split_details
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main():
    feature_names, features, labels = load_source()
    splits = split_data(features, labels)
    output_paths = save_splits(splits)
    write_config(output_paths)
    manifest = write_manifest(feature_names, splits, output_paths)
    print(f"Prepared dataset: {ROOT}")
    print(f"Samples: {manifest['total_samples']}")
    for split_name, details in manifest["splits"].items():
        if split_name == "test":
            print(
                f"{split_name}: {details['samples']} samples, "
                "class distribution withheld"
            )
        else:
            print(
                f"{split_name}: {details['samples']} samples, "
                f"distribution={details['class_distribution']}"
            )
    print(f"Configuration: {CONFIG_PATH}")
    print("No SuperAgent run was started.")


if __name__ == "__main__":
    main()
