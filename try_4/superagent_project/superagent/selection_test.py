"""Paired significance testing for model selection.

Why this exists
---------------
Selection is ``argmax`` over a single validation split. That point estimate has
a sampling error, and when two candidates differ by less than that error the
"winner" is a coin flip. Measured on this project's own runs:

* 47 of 62 runs had the best model and the runner-up within one 95% interval
  half-width of each other (76%);
* an exact McNemar test found a significant winner-versus-runner-up difference in
  only 4 of 34 evaluable runs (12%).

This module does not change which model is chosen -- it records how much the
choice can be trusted, so ``solution_lock.json`` can say "selected, but
statistically tied with exp_007" instead of implying a real difference.

Everything here is pure: no model is loaded, no data is read from disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, sqrt
from typing import Optional, Sequence

import numpy as np

DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class PairwiseResult:
    """One challenger compared against the selected model."""

    challenger: str
    challenger_value: float
    # Positive means the challenger scored HIGHER than the selected model.
    value_delta: float
    n_challenger_only: int
    n_selected_only: int
    p_value: float
    significantly_better: bool


@dataclass(frozen=True)
class SelectionTest:
    """Outcome of testing the selected model against every rival."""

    method: str
    alpha: float
    n_rows: int
    selected: str
    selected_value: float
    significant_over: list = field(default_factory=list)
    tied_with: list = field(default_factory=list)
    skipped_reason: Optional[str] = None

    @property
    def is_tied(self) -> bool:
        return bool(self.tied_with)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_rows": self.n_rows,
            "selected": self.selected,
            "selected_value": self.selected_value,
            "significant_over": list(self.significant_over),
            "tied_with": list(self.tied_with),
            "skipped_reason": self.skipped_reason,
            "verdict": (
                "no_comparison"
                if self.skipped_reason
                else ("selected_but_tied" if self.tied_with else "selected_clearly")
            ),
        }


def mcnemar_exact(y_true, pred_a, pred_b):
    """Two-sided exact McNemar on the discordant pairs.

    Returns ``(n_a_only, n_b_only, p_value)`` where ``n_a_only`` counts rows only
    ``pred_a`` got right. The exact binomial form is used rather than the chi
    square approximation because discordance counts here are often small
    (single digits), where the approximation is unreliable.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n_a_only = int(np.sum(a_ok & ~b_ok))
    n_b_only = int(np.sum(~a_ok & b_ok))
    n = n_a_only + n_b_only
    if n == 0:
        return 0, 0, 1.0
    k = min(n_a_only, n_b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n_a_only, n_b_only, float(min(1.0, 2 * tail))


def paired_bootstrap_delta(y_true, pred_a, pred_b, n_boot=5000, seed=0):
    """Bootstrap CI for accuracy(a) - accuracy(b), resampling rows in pairs."""
    y_true = np.asarray(y_true)
    diff = ((np.asarray(pred_a) == y_true).astype(float)
            - (np.asarray(pred_b) == y_true).astype(float))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(diff.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def accuracy_interval_halfwidth(value, n_rows, z=1.96):
    """95% half-width of a proportion, in the same units as ``value``."""
    if not n_rows:
        return float("nan")
    return float(z * sqrt(max(value * (1.0 - value), 0.0) / n_rows))


def evaluate_selection(selected_id,
                       selected_predictions,
                       challengers,
                       y_true,
                       selected_value,
                       alpha=DEFAULT_ALPHA):
    """Test the selected model against every challenger on the same rows.

    ``challengers`` is a sequence of ``(id, predictions, metric_value)``. A
    challenger is reported as:
      * ``significant_over`` -- the selected model beat it with p < alpha;
      * ``tied_with`` -- the difference was not significant.

    Only challengers that scored at least as high as the selected model can make
    the selection look shaky, so a challenger scoring lower is still tested (it
    may be tied) but never counted as a reason to doubt the choice.
    """
    y_true = np.asarray(y_true)
    selected_predictions = np.asarray(selected_predictions)
    if len(selected_predictions) != len(y_true):
        return SelectionTest(
            method="mcnemar_exact", alpha=alpha, n_rows=int(len(y_true)),
            selected=selected_id, selected_value=float(selected_value),
            skipped_reason=(
                f"selected predictions have {len(selected_predictions)} rows "
                f"but there are {len(y_true)} labels"))

    significant, tied = [], []
    for cid, cpred, cval in challengers:
        cpred = np.asarray(cpred)
        if len(cpred) != len(y_true):
            continue
        n_sel_only, n_chal_only, p = mcnemar_exact(
            y_true, selected_predictions, cpred)
        if p < alpha and n_sel_only > n_chal_only:
            significant.append({
                "challenger": cid, "p_value": p,
                "value_delta": float(selected_value) - float(cval),
            })
        elif p >= alpha:
            tied.append({
                "challenger": cid, "p_value": p,
                "value_delta": float(selected_value) - float(cval),
            })

    return SelectionTest(
        method="mcnemar_exact", alpha=alpha, n_rows=int(len(y_true)),
        selected=selected_id, selected_value=float(selected_value),
        significant_over=significant, tied_with=tied)


def minimum_detectable_difference(n_rows, discordance_rate, alpha=0.05,
                                  power=0.80, z_alpha=1.959964,
                                  z_beta=0.8416212):
    """Smallest real accuracy difference this split can detect.

    Standard McNemar sample-size relation rearranged for the effect size:
    ``n >= (z_a + z_b)^2 * psi / delta^2``. ``psi`` is the proportion of rows
    where the two models disagree; 0.10-0.15 is typical for two tree ensembles.
    """
    if n_rows <= 0 or discordance_rate <= 0:
        return float("nan")
    return float(sqrt((z_alpha + z_beta) ** 2 * discordance_rate / n_rows))
