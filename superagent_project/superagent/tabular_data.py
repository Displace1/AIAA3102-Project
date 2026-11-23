import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from superagent.label_encoding import encode_labels
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


MISSING_TEXT = {"", "nan", "none", "null", "?"}

# Ordinal coding is only safe when the class proportion moves in one direction
# across the levels, so two independent checks must both pass.  A single
# |Spearman| number is not enough: an alternating effect such as
# [high, 0, low, 0, high, 0, low, 0] still correlates with the code because its
# two peaks sit in the lower half of the rank space.
#
# 1. Significance.  Compare a flat proportion against its isotonic (monotone)
#    fit using the binomial deviance drop and df = blocks - 1.  This scales
#    itself with the level count and the sample size, so one threshold holds
#    for every dataset (measured false positive rate ~1% for k in 4..30).
ORDINAL_SIGNIFICANCE_LEVEL = 0.01
# 2. Shape.  The isotonic fit can win a large deviance drop from a single
#    extreme endpoint while the curve reverses violently elsewhere, so the
#    share of variance the monotone fit actually explains must also be high.
ORDINAL_MIN_SHAPE_SCORE = 0.80
# Ordinal coding assumes consecutive levels, not a sparse code set.
ORDINAL_DENSITY_THRESHOLD = 0.8


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_TEXT


@dataclass(frozen=True)
class ColumnTypeDecision:
    """Per-column outcome of type inference, kept for audit and prompting."""

    column: int
    kind: str
    levels: int
    monotonicity: float
    reason: str
    shape_score: float = 0.0
    monotone_p_value: float = 1.0

    def to_dict(self):
        return {
            "column": self.column,
            "type": self.kind,
            "levels": self.levels,
            "monotonicity": round(float(self.monotonicity), 3),
            "shape_score": round(float(self.shape_score), 3),
            "monotone_p_value": float(f"{self.monotone_p_value:.3g}"),
            "reason": self.reason,
        }


def infer_tabular_column_types(
    array,
    explicit_numeric=None,
    explicit_categorical=None,
    explicit_ordinal=None,
    target=None,
):
    """Split feature columns into numeric, ordinal and categorical indices.

    ``ordinal`` is the integer-level channel: columns that are numeric but have
    too few distinct values to be treated as continuous, and whose ordering is
    either provable (two levels) or supported by a monotone association with the
    target.  Keeping it separate stops "is it a number" and "is it ordered"
    from being collapsed into a single decision.
    """
    decisions = classify_tabular_columns(
        array,
        explicit_numeric=explicit_numeric,
        explicit_categorical=explicit_categorical,
        explicit_ordinal=explicit_ordinal,
        target=target,
    )
    return split_column_indices(decisions)


def split_column_indices(decisions):
    numeric = []
    ordinal = []
    categorical = []
    for decision in decisions:
        if decision.kind == "numeric":
            numeric.append(decision.column)
        elif decision.kind == "ordinal":
            ordinal.append(decision.column)
        else:
            categorical.append(decision.column)
    return numeric, ordinal, categorical


def classify_tabular_columns(
    array,
    explicit_numeric=None,
    explicit_categorical=None,
    explicit_ordinal=None,
    target=None,
):
    """Classify every feature column, keeping the reason for each decision."""
    values = np.asarray(array, dtype=object)
    if values.ndim != 2:
        raise ValueError("Tabular features must be a 2D array.")
    column_count = values.shape[1]
    unique_counts = [
        len({str(value) for value in values[:, index]})
        for index in range(column_count)
    ]

    declared = (
        explicit_numeric is not None
        or explicit_categorical is not None
        or explicit_ordinal is not None
    )
    if declared:
        groups = {
            "numeric": sorted(set(explicit_numeric or [])),
            "ordinal": sorted(set(explicit_ordinal or [])),
            "categorical": sorted(set(explicit_categorical or [])),
        }
        assigned = set()
        for indices in groups.values():
            overlap = assigned & set(indices)
            if overlap:
                raise ValueError(
                    "Numeric, ordinal and categorical columns overlap."
                )
            assigned |= set(indices)
        if sorted(assigned) != list(range(column_count)):
            raise ValueError(
                "Explicit numeric/ordinal/categorical columns must cover "
                "every feature."
            )
        decisions = []
        for kind, indices in groups.items():
            for index in indices:
                decisions.append(
                    ColumnTypeDecision(
                        column=index,
                        kind=kind,
                        levels=unique_counts[index],
                        monotonicity=0.0,
                        reason="declared",
                    )
                )
        return sorted(decisions, key=lambda decision: decision.column)

    sample_count = max(1, values.shape[0])
    numeric_cardinality_threshold = max(
        10,
        min(50, int(sample_count * 0.005)),
    )

    decisions = []
    for index in range(column_count):
        column = values[:, index]
        mask = np.asarray(
            [not is_missing(value) for value in column], dtype=bool
        )
        present = [value for value, keep in zip(column, mask) if keep]
        levels = unique_counts[index]

        if not present:
            decisions.append(
                ColumnTypeDecision(index, "categorical", levels, 0.0, "empty")
            )
            continue

        numeric_values = []
        unparsable = 0
        for value in present:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                unparsable += 1
        parse_rate = (len(present) - unparsable) / len(present)

        if parse_rate < 0.98:
            decisions.append(
                ColumnTypeDecision(
                    index, "categorical", levels, 0.0, "not_numeric"
                )
            )
            continue

        integers = all(value.is_integer() for value in numeric_values)
        if not integers:
            if levels >= numeric_cardinality_threshold:
                decisions.append(
                    ColumnTypeDecision(
                        index, "numeric", levels, 0.0, "continuous"
                    )
                )
            else:
                decisions.append(
                    ColumnTypeDecision(
                        index,
                        "categorical",
                        levels,
                        0.0,
                        "low_cardinality_numeric",
                    )
                )
            continue

        if levels <= 1:
            decisions.append(
                ColumnTypeDecision(
                    index, "categorical", levels, 0.0, "constant"
                )
            )
            continue
        if levels == 2:
            # With two levels ordinal coding only fixes a sign, so no ordering
            # assumption can be wrong, and one column beats a rank-deficient
            # two-column one-hot block.
            decisions.append(
                ColumnTypeDecision(index, "ordinal", levels, 0.0, "binary")
            )
            continue
        if levels >= numeric_cardinality_threshold:
            decisions.append(
                ColumnTypeDecision(
                    index,
                    "numeric",
                    levels,
                    0.0,
                    "high_cardinality_integer",
                )
            )
            continue

        span = int(max(numeric_values) - min(numeric_values)) + 1
        density = levels / span if span > 0 else 0.0
        if density < ORDINAL_DENSITY_THRESHOLD:
            decisions.append(
                ColumnTypeDecision(
                    index, "categorical", levels, 0.0, "sparse_integer_levels"
                )
            )
            continue

        labels = None
        if target is not None and len(target) == len(column):
            labels = np.asarray(target)[mask]
        monotonicity = ordinal_monotonicity(numeric_values, labels)
        shape_score = ordinal_monotone_shape(numeric_values, labels)
        p_value = ordinal_monotone_p_value(numeric_values, labels)
        is_ordinal = (
            p_value < ORDINAL_SIGNIFICANCE_LEVEL
            and shape_score >= ORDINAL_MIN_SHAPE_SCORE
        )
        if is_ordinal:
            reason = "monotone_levels"
        elif p_value >= ORDINAL_SIGNIFICANCE_LEVEL:
            reason = "no_monotone_association"
        else:
            reason = "non_monotone_shape"
        decisions.append(
            ColumnTypeDecision(
                index,
                "ordinal" if is_ordinal else "categorical",
                levels,
                monotonicity,
                reason,
                shape_score,
                p_value,
            )
        )

    return decisions


def ordinal_monotonicity(values, labels):
    """``|Spearman|`` between integer levels and the target.

    Reported as evidence only.  It is *not* the decision rule: a rare positive
    class attenuates the correlation, and an alternating effect can still
    correlate with the code.  ``ordinal_monotone_p_value`` and
    ``ordinal_monotone_shape`` are what gate the ordinal channel.
    """
    if labels is None or len(labels) != len(values) or len(values) < 8:
        return 0.0
    target = np.asarray(labels)
    classes = np.unique(target)
    if len(classes) < 2:
        return 0.0
    ranks = _average_ranks(np.asarray(values, dtype=float))
    if len(classes) == 2:
        return abs(_pearson(ranks, (target == classes[-1]).astype(float)))
    return max(
        abs(_pearson(ranks, (target == klass).astype(float)))
        for klass in classes
    )


def _level_counts(values, labels):
    """Per-level totals and per-class positives for one integer column."""
    levels = np.unique(values)
    totals = np.asarray([(values == level).sum() for level in levels], dtype=float)
    classes = np.unique(labels)
    positives = {
        klass: np.asarray(
            [((values == level) & (labels == klass)).sum() for level in levels],
            dtype=float,
        )
        for klass in classes
    }
    return levels, totals, positives


def _binomial_deviance(positive, total, proportion):
    expected = np.clip(total * proportion, 1e-12, None)
    expected_negative = np.clip(total * (1.0 - proportion), 1e-12, None)
    negative = total - positive
    terms = np.where(
        positive > 0,
        positive * np.log(np.clip(positive, 1e-12, None) / expected),
        0.0,
    )
    terms = terms + np.where(
        negative > 0,
        negative * np.log(np.clip(negative, 1e-12, None) / expected_negative),
        0.0,
    )
    return 2.0 * float(terms.sum())


def ordinal_monotone_p_value(values, labels):
    """p-value for "flat proportion" against a monotone alternative.

    The isotonic (monotone) fit is compared with the constant fit through the
    binomial deviance drop, using ``df = distinct blocks - 1``.  Both the level
    count and the sample size enter the scale of the statistic, so the same
    significance level is valid for every dataset, and a rare positive class
    (which deflates :func:`ordinal_monotonicity`) does not hide a real trend.

    Returns ``1.0`` when it cannot be measured, which keeps the caller on the
    conservative one-hot side.
    """
    if labels is None or len(labels) != len(values) or len(values) < 8:
        return 1.0
    target = np.asarray(labels)
    if len(np.unique(target)) < 2:
        return 1.0
    levels, totals, positives = _level_counts(
        np.asarray(values, dtype=float), target
    )
    if len(levels) < 3 or totals.sum() <= 0:
        return 1.0
    class_count = len(positives)
    best = 1.0
    for positive in positives.values():
        overall = positive.sum() / totals.sum()
        if overall <= 0.0 or overall >= 1.0:
            continue
        flat = _binomial_deviance(
            positive, totals, np.full(len(levels), overall)
        )
        fitted = IsotonicRegression(
            increasing=True, out_of_bounds="clip"
        ).fit_transform(
            levels, positive / totals, sample_weight=totals
        )
        monotone = _binomial_deviance(positive, totals, fitted)
        degrees = max(1, len(np.unique(np.round(fitted, 12))) - 1)
        p_value = float(chi2.sf(max(0.0, flat - monotone), degrees))
        # One-vs-rest is evaluated for every class, so correct for multiplicity.
        best = min(best, min(1.0, p_value * class_count))
    return best


def ordinal_monotone_shape(values, labels):
    """Share of the level-proportion variance a monotone curve can explain.

    A monotone trend scores close to ``1.0``.  An alternating or U-shaped
    effect scores low, which is what stops a single extreme endpoint from
    making a wildly non-monotone curve look ordinal.
    """
    if labels is None or len(labels) != len(values) or len(values) < 8:
        return 0.0
    target = np.asarray(labels)
    if len(np.unique(target)) < 2:
        return 0.0
    levels, totals, positives = _level_counts(
        np.asarray(values, dtype=float), target
    )
    if len(levels) < 3 or totals.sum() <= 0:
        return 0.0
    best = 0.0
    for positive in positives.values():
        proportion = positive / np.maximum(totals, 1.0)
        mean = np.average(proportion, weights=totals)
        total_variance = float((totals * (proportion - mean) ** 2).sum())
        if total_variance <= 0.0:
            continue
        fitted = IsotonicRegression(
            increasing=True, out_of_bounds="clip"
        ).fit_transform(
            levels, proportion, sample_weight=totals
        )
        residual = float((totals * (proportion - fitted) ** 2).sum())
        best = max(best, 1.0 - residual / total_variance)
    return best


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    total = values.shape[0]
    while start < total:
        stop = start
        while stop + 1 < total and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def _pearson(left, right):
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt((left_centered**2).sum() * (right_centered**2).sum())
    )
    if denominator <= 0.0:
        return 0.0
    return float((left_centered * right_centered).sum() / denominator)


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
        return np.asarray(features), encode_labels(
            labels.tolist(),
            (data_profile or {}).get("label_mapping"),
        )

    if config.data["format"] != "csv_split":
        raise ValueError("Tabular loader supports only NPY or CSV splits.")
    return load_tabular_csv(
        config.data[split],
        data_profile["feature_columns"],
        data_profile["target_column"],
        require_target=True,
        label_mapping=(data_profile or {}).get("label_mapping"),
    )


def load_tabular_csv(
    path,
    feature_columns,
    target_column=None,
    require_target=False,
    label_mapping=None,
):
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
                labels.append(row[target_column])
    features = np.asarray(rows, dtype=object)
    if features.ndim != 2:
        features = features.reshape(len(rows), len(feature_columns))
    if require_target:
        return features, encode_labels(labels, label_mapping)
    return features


def build_tabular_preprocessor(
    data_profile,
    numeric_preprocessing="standard",
    categorical_preprocessing="onehot",
    ordinal_preprocessing="ordinal",
):
    numeric_indices = list(data_profile.get("numeric_column_indices", []))
    ordinal_indices = list(data_profile.get("ordinal_column_indices", []))
    categorical_indices = list(
        data_profile.get("categorical_column_indices", [])
    )

    if ordinal_preprocessing == "onehot":
        categorical_indices = sorted(
            set(categorical_indices) | set(ordinal_indices)
        )
        ordinal_indices = []
    elif ordinal_preprocessing != "ordinal":
        raise ValueError(
            f"Unknown ordinal preprocessing: {ordinal_preprocessing}"
        )

    assigned = numeric_indices + ordinal_indices + categorical_indices
    if len(assigned) != len(set(assigned)):
        raise ValueError("Column type indices must be disjoint.")

    transformers = []

    if numeric_indices:
        transformers.append(
            (
                "numeric",
                Pipeline(_numeric_pipeline_steps(numeric_preprocessing)),
                numeric_indices,
            )
        )

    if ordinal_indices:
        # Ordinal levels keep their numeric order through float coercion; an
        # OrdinalEncoder would sort the raw strings and place "10" before "2".
        # The median is the correct imputation for an ordered scale and is
        # already what the numeric branch uses, so the steps are shared.
        transformers.append(
            (
                "ordinal",
                Pipeline(_numeric_pipeline_steps(numeric_preprocessing)),
                ordinal_indices,
            )
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


def _numeric_pipeline_steps(numeric_preprocessing):
    steps = [
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
        steps.append(("scale", StandardScaler()))
    elif numeric_preprocessing == "minmax":
        steps.append(("scale", MinMaxScaler()))
    elif numeric_preprocessing != "none":
        raise ValueError(
            f"Unknown numeric preprocessing: {numeric_preprocessing}"
        )
    return steps


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
