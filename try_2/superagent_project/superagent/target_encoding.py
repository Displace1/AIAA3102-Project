"""Cross-fitted target encoding that is safe inside a scikit-learn Pipeline.

Why this exists
---------------
``sklearn.preprocessing.TargetEncoder`` only cross-fits inside
``fit_transform(X, y)``. A ``Pipeline`` calls ``fit_transform(X)`` without the
labels (verified by measurement, not by reading the source): with pure-noise
labels, encoding through a Pipeline gave train AUC 0.674 while validation sat at
0.517, i.e. the training rows were encoded with their own labels. Encoding
manually with ``fit_transform(X, y)`` gave 0.507 / 0.517, which is the expected
no-signal result.

This transformer therefore takes the labels at ``fit`` time and performs the
cross-fitting itself, so it can be dropped into a Pipeline without leaking.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder


class CrossFittedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-encode categorical columns with out-of-fold statistics.

    ``fit_transform(X, y)`` encodes each training row using statistics from the
    other folds only. ``transform(X)`` (used on validation/test) uses the
    statistics fit on the whole training split, which is correct: those rows
    were never part of the fit.

    The columns are expected to be object arrays of strings; the caller is
    responsible for normalising missing values first.
    """

    def __init__(self, target_type="binary", n_splits=5, smooth="auto",
                 random_state=42):
        self.target_type = target_type
        self.n_splits = n_splits
        self.smooth = smooth
        self.random_state = random_state

    # -- internals ---------------------------------------------------------
    def _encoder(self):
        return TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            random_state=self.random_state,
        )

    def _n_output_columns(self):
        if self.target_type == "binary":
            return 1
        return int(self.n_classes_)

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        if y is None:
            raise ValueError(
                "CrossFittedTargetEncoder requires labels at fit time; "
                "use fit_transform(X, y) inside a Pipeline."
            )
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y))
        self.encoder_ = self._encoder()
        self.encoder_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        return np.asarray(self.encoder_.transform(X), dtype=np.float64)

    def fit_transform(self, X, y=None, **fit_params):
        """Encode training rows out-of-fold so no row sees its own label."""
        if y is None:
            raise ValueError(
                "CrossFittedTargetEncoder requires labels; a Pipeline that "
                "drops y before this step cannot use it safely."
            )
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_in_ = X.shape[1]

        # Fall back to plain fitting when there are too few rows per class for
        # a stratified split; the caller has already been warned by the
        # minimum-sample guards in the threshold module.
        min_class = int(np.bincount(y).min()) if len(y) else 0
        n_splits = min(self.n_splits, min_class)
        if n_splits < 2:
            self.encoder_ = self._encoder()
            out = self.encoder_.fit_transform(X, y)
            return np.asarray(out, dtype=np.float64)

        out = np.zeros((X.shape[0], self._n_output_columns()), dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, holdout_idx in splitter.split(X, y):
            fold_encoder = self._encoder()
            fold_encoder.fit(X[train_idx], y[train_idx])
            out[holdout_idx] = np.asarray(
                fold_encoder.transform(X[holdout_idx]), dtype=np.float64)

        # The full-data encoder handles validation/test in transform().
        self.encoder_ = self._encoder()
        self.encoder_.fit(X, y)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"target_enc_{i}" for i in range(self._n_output_columns())],
            dtype=object,
        )
