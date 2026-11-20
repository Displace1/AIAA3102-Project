import csv
from collections import Counter

import numpy as np

from superagent.image_inspector import ImageInspector


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "?"}


class DataInspector:
    def inspect(self, config):
        data_format = config.data["format"]
        if data_format == "npy_split":
            return self._inspect_npy(config)
        if data_format == "csv_split":
            return self._inspect_csv(config)
        if data_format == "image_directory_split":
            return ImageInspector().inspect(config)
        raise ValueError(f"Unsupported data format: {data_format}")

    def _inspect_npy(self, config):
        X_train = np.load(
            config.data["train_features"], allow_pickle=True
        )
        y_train = np.load(
            config.data["train_labels"], allow_pickle=True
        )
        X_val = np.load(
            config.data["validation_features"], allow_pickle=True
        )
        y_val = np.load(
            config.data["validation_labels"], allow_pickle=True
        )

        if X_train.ndim != 2:
            raise ValueError("NPY feature data must be a 2D array.")
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Feature and label row counts do not match.")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                "Train and validation feature counts do not match."
            )

        unique_counts = [
            len({str(value) for value in X_train[:, index]})
            for index in range(X_train.shape[1])
        ]
        missing_counts = [
            sum(_is_missing(value) for value in X_train[:, index])
            for index in range(X_train.shape[1])
        ]
        constant_columns = [
            index
            for index, count in enumerate(unique_counts)
            if count <= 1
        ]
        labels, counts = np.unique(y_train, return_counts=True)

        # Object arrays with low-cardinality columns are treated as tabular
        # categorical data. This decision does not use a task identity.
        categorical_columns = sum(
            count <= max(50, int(len(X_train) * 0.05))
            for count in unique_counts
        )

        return {
            "modality": "tabular",
            "task_type": "binary_classification"
            if len(labels) == 2
            else "multiclass_classification",
            "data_format": "npy_split",
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "n_features": int(X_train.shape[1]),
            "categorical_feature_count": int(categorical_columns),
            "unique_counts": unique_counts,
            "missing_counts": missing_counts,
            "constant_column_indices": constant_columns,
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(labels, counts)
            },
            "metric": config.metric,
        }

    def _inspect_csv(self, config):
        train_rows, fieldnames = self._read_csv(config.data["train"])
        val_rows, val_fieldnames = self._read_csv(
            config.data["validation"]
        )
        target = config.data["target_column"]

        if fieldnames != val_fieldnames:
            raise ValueError("Train and validation CSV schemas differ.")
        if target not in fieldnames:
            raise ValueError(f"Target column not found: {target}")

        feature_columns = [
            column for column in fieldnames if column != target
        ]
        if len(feature_columns) != 2:
            raise ValueError(
                "This MVP recognizes a text-pair task only when the CSV "
                "contains exactly two feature columns."
            )

        labels = Counter(row[target] for row in train_rows)
        length_summary = {}
        for column in feature_columns:
            word_counts = [
                len(row[column].split()) for row in train_rows
            ]
            length_summary[column] = {
                "minimum_words": min(word_counts),
                "average_words": sum(word_counts) / len(word_counts),
                "maximum_words": max(word_counts),
                "empty_count": sum(
                    not row[column].strip() for row in train_rows
                ),
            }

        return {
            "modality": "text_pair",
            "task_type": "binary_classification"
            if len(labels) == 2
            else "multiclass_classification",
            "data_format": "csv_split",
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "feature_columns": feature_columns,
            "target_column": target,
            "text_length_summary": length_summary,
            "class_distribution": dict(sorted(labels.items())),
            "metric": config.metric,
        }

    @staticmethod
    def _read_csv(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        if not rows:
            raise ValueError(f"CSV file contains no data rows: {path}")
        return rows, fieldnames
