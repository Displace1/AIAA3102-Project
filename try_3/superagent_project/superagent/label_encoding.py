"""Map arbitrary class labels onto the integer codes the adapters train on.

Every adapter in this project predicts integer class indices: the image path
builds ``index_to_class`` from sorted directory names, and the tabular/text
paths hand ``y`` straight to scikit-learn estimators.  Real task files often
carry readable labels instead (``"yes"``/``"no"``, ``">50K"``, ``"setosa"``), so
the split loaders need one deterministic, split-independent encoding.

The mapping is derived from the training split only and stored in
``data_profile``:

    "label_encoding": "identity" | "string_mapping",
    "label_mapping": {"no": 0, "yes": 1},   # only when a remap is needed
    "class_labels": ["no", "yes"],          # code -> original label

``identity`` means the labels were already integral numbers, so no remap is
applied and the original values are kept (a model trained on labels 1..3 stays
on 1..3 rather than being silently shifted).
"""

import numpy as np


def _integer_codes(values):
    """Return integer codes when every label is an integral number, else None."""
    codes = []
    for value in values:
        text = str(value).strip()
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number) or not number.is_integer():
            return None
        codes.append(int(number))
    return np.asarray(codes, dtype=int)


def describe_labels(values):
    """Return the profile fields that describe how labels are encoded."""
    values = list(values)
    if not values:
        raise ValueError("Cannot describe an empty label column.")
    distinct_text = {str(value).strip() for value in values}
    direct = _integer_codes(values)
    if direct is not None and len(set(direct.tolist())) == len(distinct_text):
        return {
            "label_encoding": "identity",
            "label_mapping": None,
            "class_labels": None,
        }
    # Sorted order keeps the mapping independent of row order, so a rerun on the
    # same data always produces the same codes.
    labels = sorted(distinct_text)
    return {
        "label_encoding": "string_mapping",
        "label_mapping": {label: index for index, label in enumerate(labels)},
        "class_labels": labels,
    }


def encode_labels(values, mapping=None):
    """Encode raw labels into integer codes using ``mapping`` when present."""
    if not mapping:
        codes = _integer_codes(values)
        if codes is None:
            raise ValueError(
                "Labels are not integral numbers and no mapping was provided. "
                "The data profile must be rebuilt so the label encoding is "
                "recorded before the split is loaded."
            )
        return codes
    encoded = []
    unknown = set()
    for value in values:
        key = str(value).strip()
        index = mapping.get(key)
        if index is None:
            unknown.add(key)
            continue
        encoded.append(index)
    if unknown:
        raise ValueError(
            "This split contains class labels that never appear in the "
            f"training split: {sorted(unknown)}. The label set must match "
            "across splits; fix the data or declare the classes explicitly."
        )
    return np.asarray(encoded, dtype=int)


def decode_labels(codes, class_labels):
    """Map integer codes back to the original labels, when a mapping exists."""
    if not class_labels:
        return codes
    return [class_labels[int(code)] for code in codes]
