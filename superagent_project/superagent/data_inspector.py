import csv
from collections import Counter

import numpy as np

from superagent.image_inspector import ImageInspector
from superagent.tabular_data import infer_tabular_column_types, is_missing


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
        numeric_indices, categorical_indices = infer_tabular_column_types(
            X_train
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
            "numeric_feature_count": len(numeric_indices),
            "categorical_feature_count": len(categorical_indices),
            "numeric_column_indices": numeric_indices,
            "categorical_column_indices": categorical_indices,
            "unique_counts": unique_counts,
            "missing_counts": missing_counts,
            "constant_column_indices": constant_columns,
            "class_distribution": {
                str(label): int(count)
                for label, count in zip(labels, counts)
            },
            "minority_rate": float(counts.min() / counts.sum()),
            "majority_class_accuracy": float(counts.max() / counts.sum()),
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
        labels = Counter(row[target] for row in train_rows)
        declared = config.data.get("columns") or {}
        modality, selected_features = self._csv_modality(
            train_rows=train_rows,
            feature_columns=feature_columns,
            declared=declared,
        )
        feature_columns = selected_features

        if modality == "tabular":
            matrix = np.asarray(
                [
                    [row[column] for column in feature_columns]
                    for row in train_rows
                ],
                dtype=object,
            )
            numeric_names = list(declared.get("numeric", []))
            categorical_names = list(declared.get("categorical", []))
            if numeric_names or categorical_names:
                explicit_numeric = [
                    feature_columns.index(name) for name in numeric_names
                ]
                explicit_categorical = [
                    feature_columns.index(name) for name in categorical_names
                ]
            else:
                explicit_numeric = None
                explicit_categorical = None
            numeric_indices, categorical_indices = infer_tabular_column_types(
                matrix,
                explicit_numeric=explicit_numeric,
                explicit_categorical=explicit_categorical,
            )
            counts = list(labels.values())
            return {
                "modality": "tabular",
                "task_type": (
                    "binary_classification"
                    if len(labels) == 2
                    else "multiclass_classification"
                ),
                "data_format": "csv_split",
                "train_samples": len(train_rows),
                "validation_samples": len(val_rows),
                "n_features": len(feature_columns),
                "feature_columns": feature_columns,
                "target_column": target,
                "numeric_feature_count": len(numeric_indices),
                "categorical_feature_count": len(categorical_indices),
                "numeric_column_indices": numeric_indices,
                "categorical_column_indices": categorical_indices,
                "numeric_columns": [
                    feature_columns[index] for index in numeric_indices
                ],
                "categorical_columns": [
                    feature_columns[index] for index in categorical_indices
                ],
                "unique_counts": [
                    len({str(row[column]) for row in train_rows})
                    for column in feature_columns
                ],
                "missing_counts": [
                    sum(is_missing(row[column]) for row in train_rows)
                    for column in feature_columns
                ],
                "class_distribution": dict(sorted(labels.items())),
                "minority_rate": min(counts) / sum(counts),
                "majority_class_accuracy": max(counts) / sum(counts),
                "metric": config.metric,
                "modality_detection": (
                    "explicit_columns" if declared else "content_heuristic"
                ),
            }

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
            "modality": modality,
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
            "minority_rate": min(labels.values()) / sum(labels.values()),
            "majority_class_accuracy": max(labels.values()) / sum(labels.values()),
            "modality_detection": (
                "explicit_columns" if declared else "content_heuristic"
            ),
        }

    @staticmethod
    def _csv_modality(train_rows, feature_columns, declared):
        text_columns = list(declared.get("text", []))
        numeric_columns = list(declared.get("numeric", []))
        categorical_columns = list(declared.get("categorical", []))
        declared_features = text_columns + numeric_columns + categorical_columns
        if declared_features:
            if len(set(declared_features)) != len(declared_features):
                raise ValueError("A CSV feature column was declared more than once.")
            missing = set(declared_features) - set(feature_columns)
            if missing:
                raise ValueError(
                    f"Declared CSV columns do not exist: {sorted(missing)}"
                )
            undeclared = set(feature_columns) - set(declared_features)
            if undeclared:
                raise ValueError(
                    "Explicit data.columns must classify every feature; "
                    f"undeclared: {sorted(undeclared)}"
                )
            if text_columns and (numeric_columns or categorical_columns):
                raise ValueError(
                    "Mixed text and structured CSV features are not yet supported."
                )
            if text_columns:
                if len(text_columns) not in {1, 2}:
                    raise ValueError(
                        "Text CSV tasks require one or two text columns."
                    )
                return ("text" if len(text_columns) == 1 else "text_pair"), text_columns
            return "tabular", numeric_columns + categorical_columns

        def looks_like_text(column):
            values = [str(row[column]).strip() for row in train_rows]
            nonempty = [value for value in values if value]
            if not nonempty:
                return False
            numeric = 0
            for value in nonempty:
                try:
                    float(value)
                    numeric += 1
                except ValueError:
                    pass
            numeric_rate = numeric / len(nonempty)
            average_words = sum(len(value.split()) for value in nonempty) / len(
                nonempty
            )
            average_characters = sum(map(len, nonempty)) / len(nonempty)
            unique_rate = len(set(nonempty)) / len(nonempty)
            return (
                numeric_rate < 0.1
                and unique_rate >= 0.1
                and (average_words >= 3 or average_characters >= 20)
            )

        if len(feature_columns) in {1, 2} and all(
            looks_like_text(column) for column in feature_columns
        ):
            return (
                "text" if len(feature_columns) == 1 else "text_pair"
            ), feature_columns
        return "tabular", feature_columns

    @staticmethod
    def _read_csv(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        if not rows:
            raise ValueError(f"CSV file contains no data rows: {path}")
        return rows, fieldnames
