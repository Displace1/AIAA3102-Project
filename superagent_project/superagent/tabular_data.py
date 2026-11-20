import csv
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


MISSING_TEXT = {"", "nan", "none", "null", "?"}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_TEXT


def infer_tabular_column_types(array, explicit_numeric=None, explicit_categorical=None):
    """Infer stable numeric/categorical column indices without task identity."""
    values = np.asarray(array, dtype=object)
    if values.ndim != 2:
        raise ValueError("Tabular features must be a 2D array.")
    if explicit_numeric is not None or explicit_categorical is not None:
        numeric = sorted(set(explicit_numeric or []))
        categorical = sorted(set(explicit_categorical or []))
        if set(numeric) & set(categorical):
            raise ValueError("Numeric and categorical columns overlap.")
        if sorted(numeric + categorical) != list(range(values.shape[1])):
            raise ValueError(
                "Explicit numeric/categorical columns must cover every feature."
            )
        return numeric, categorical

    numeric = []
    categorical = []
    sample_count = max(1, values.shape[0])
    numeric_cardinality_threshold = max(
        10,
        min(50, int(sample_count * 0.005)),
    )

    for index in range(values.shape[1]):
        nonmissing = [
            value for value in values[:, index] if not is_missing(value)
        ]
        parseable = 0
        for value in nonmissing:
            try:
                float(value)
                parseable += 1
            except (TypeError, ValueError):
                pass
        unique_count = len({str(value) for value in nonmissing})
        parse_rate = parseable / max(1, len(nonmissing))
        is_continuous = unique_count >= numeric_cardinality_threshold
        if parse_rate >= 0.98 and is_continuous:
            numeric.append(index)
        else:
            categorical.append(index)
    return numeric, categorical


def load_tabular_split(config, data_profile, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split!r}.")
    if config.data["format"] == "npy_split":
        feature_key = {
            "train": "train_features",
            "validation": "validation_features",
            "test": "test_features",
        }[split]
        label_key = {
            "train": "train_labels",
            "validation": "validation_labels",
            "test": "test_labels",
        }[split]
        features = np.load(config.data[feature_key], allow_pickle=True)
        labels = np.load(config.data[label_key], allow_pickle=True)
        return np.asarray(features), np.asarray(labels, dtype=int)

    if config.data["format"] != "csv_split":
        raise ValueError("Tabular loader supports only NPY or CSV splits.")
    return load_tabular_csv(
        config.data[split],
        data_profile["feature_columns"],
        data_profile["target_column"],
        require_target=True,
    )


def load_tabular_csv(path, feature_columns, target_column=None, require_target=False):
    rows = []
    labels = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        missing = set(feature_columns) - set(fieldnames)
        if missing:
            raise ValueError(f"CSV is missing feature columns: {sorted(missing)}")
        if require_target and target_column not in fieldnames:
            raise ValueError(f"CSV is missing target column: {target_column}")
        for row in reader:
            rows.append([row[column] for column in feature_columns])
            if require_target:
                labels.append(int(row[target_column]))
    features = np.asarray(rows, dtype=object)
    if features.ndim != 2:
        features = features.reshape(len(rows), len(feature_columns))
    if require_target:
        return features, np.asarray(labels, dtype=int)
    return features


def build_tabular_preprocessor(
    data_profile,
    numeric_preprocessing="standard",
    categorical_preprocessing="onehot",
):
    numeric_indices = list(data_profile.get("numeric_column_indices", []))
    categorical_indices = list(
        data_profile.get("categorical_column_indices", [])
    )
    transformers = []

    if numeric_indices:
        numeric_steps = [
            (
                "coerce_numeric",
                FunctionTransformer(
                    _coerce_numeric,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
        ]
        if numeric_preprocessing == "standard":
            numeric_steps.append(("scale", StandardScaler()))
        elif numeric_preprocessing == "minmax":
            numeric_steps.append(("scale", MinMaxScaler()))
        elif numeric_preprocessing != "none":
            raise ValueError(
                f"Unknown numeric preprocessing: {numeric_preprocessing}"
            )
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_indices)
        )

    if categorical_indices:
        categorical_steps = [
            (
                "normalize",
                FunctionTransformer(
                    _normalize_categorical,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
        if categorical_preprocessing == "onehot":
            categorical_steps.append(
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                )
            )
        elif categorical_preprocessing == "ordinal":
            categorical_steps.append(
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            )
        else:
            raise ValueError(
                "Unknown categorical preprocessing: "
                f"{categorical_preprocessing}"
            )
        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps),
                categorical_indices,
            )
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical columns were found.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def transformed_feature_count(preprocessor, fallback):
    try:
        return int(len(preprocessor.get_feature_names_out()))
    except (AttributeError, ValueError):
        return int(fallback)


def package_tabular_model(model, data_profile, data_format):
    model.superagent_type_aware = True
    model.superagent_data_format = data_format
    if data_format == "npy_split":
        return model
    return {
        "artifact_type": "tabular_classifier",
        "model": model,
        "data_format": data_format,
        "feature_columns": list(data_profile["feature_columns"]),
        "target_column": data_profile.get("target_column"),
    }


def _coerce_numeric(values):
    array = np.asarray(values, dtype=object)
    output = np.empty(array.shape, dtype=float)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            output[row, column] = np.nan if is_missing(value) else float(value)
    return output


def _normalize_categorical(values):
    array = np.asarray(values, dtype=object)
    output = np.empty(array.shape, dtype=object)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            output[row, column] = (
                "__MISSING__" if is_missing(value) else str(value)
            )
    return output
