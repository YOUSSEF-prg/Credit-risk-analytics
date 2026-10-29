"""Threshold selection and illustrative decision-cost analysis.

The default threshold of 0.5 has no special economic meaning. Here the threshold is
chosen on the **validation** set, by maximising an F-beta score. ``beta = 2`` weights
recall (catching defaulters) more heavily than precision, as the project owner requested.
The test set is never used for this choice.

Costs in :func:`expected_cost` are hypothetical inputs. The dataset contains no
exposure, loss-given-default or revenue information, so any monetary figure derived
here is illustrative, not an estimate of real bank losses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.evaluation import validate_probabilities

DEFAULT_GRID = np.round(np.arange(0.01, 0.991, 0.005), 3)


def threshold_table(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray = DEFAULT_GRID
) -> pd.DataFrame:
    """Confusion counts and decision metrics for each candidate threshold."""
    y = np.asarray(y).astype(int)
    p = validate_probabilities(p)
    rows = []
    n_pos = int(y.sum())
    for t in thresholds:
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = n_pos - tp
        tn = len(y) - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_pos if n_pos else 0.0
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f_beta(precision, recall, 1.0),
                "f2": f_beta(precision, recall, 2.0),
                "flagged_rate": float(flagged.mean()),
            }
        )
    return pd.DataFrame(rows)


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score from precision and recall (0 when both are 0)."""
    denom = beta**2 * precision + recall
    return float((1 + beta**2) * precision * recall / denom) if denom > 0 else 0.0


def select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    beta: float = 2.0,
    thresholds: np.ndarray = DEFAULT_GRID,
    max_flagged_rate: float | None = None,
) -> dict[str, float]:
    """Choose the threshold that maximises F-beta on the supplied (validation) data.

    ``max_flagged_rate`` adds an operational capacity constraint: only thresholds
    that flag at most this share of clients are eligible.

    Ties are broken by the lower threshold (higher recall). The search uses a fixed
    grid with step 0.005, not every observed score, to limit overfitting to the
    validation sample.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    table = threshold_table(y, p, thresholds)
    if max_flagged_rate is not None:
        if not 0 < max_flagged_rate <= 1:
            raise ValueError("max_flagged_rate must lie in (0, 1]")
        table = table[table["flagged_rate"] <= max_flagged_rate].reset_index(drop=True)
        if table.empty:
            raise ValueError(f"No threshold flags at most {max_flagged_rate:.0%} of clients")
    scores = np.array(
        [f_beta(pr, rc, beta) for pr, rc in zip(table.precision, table.recall, strict=True)]
    )
    best = int(np.flatnonzero(scores == scores.max())[0])
    row = table.iloc[best]
    objective = f"f_beta(beta={beta:g})"
    if max_flagged_rate is not None:
        objective += f" s.t. flagged_rate <= {max_flagged_rate:g}"
    return {
        "threshold": float(row["threshold"]),
        "objective": objective,
        "objective_value": float(scores[best]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "flagged_rate": float(row["flagged_rate"]),
    }


def expected_cost(
    table: pd.DataFrame, cost_false_negative: float, cost_false_positive: float
) -> pd.DataFrame:
    """Add illustrative total and per-client cost columns under user-specified unit costs."""
    if cost_false_negative < 0 or cost_false_positive < 0:
        raise ValueError("Costs must be non-negative")
    out = table.copy()
    n = out[["tp", "fp", "tn", "fn"]].sum(axis=1)
    out["illustrative_cost"] = cost_false_negative * out["fn"] + cost_false_positive * out["fp"]
    out["illustrative_cost_per_client"] = out["illustrative_cost"] / n
    return out
