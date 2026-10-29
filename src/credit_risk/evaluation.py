"""Evaluation metrics for probability-of-default models.

Threshold-free metrics (ROC-AUC, PR-AUC, Brier score, log loss) assess the PD
estimates themselves. Threshold metrics (precision, recall, F-scores, confusion
matrix) assess one decision rule, at a threshold chosen on the validation set
(see :mod:`credit_risk.decision_analysis`).

Confidence intervals use the non-parametric percentile bootstrap over observations,
holding the model and threshold fixed. They capture sampling variability of the
evaluation set, not of model training.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROBABILITY_EPS = 1e-15


def validate_probabilities(p: np.ndarray) -> np.ndarray:
    """Check that predictions are finite probabilities in [0, 1]."""
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Probabilities contain NaN or infinite values")
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError("Probabilities must lie in [0, 1]")
    return arr


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics of the PD estimates."""
    y = np.asarray(y).astype(int)
    p = validate_probabilities(p)
    base_rate = y.mean()
    brier = brier_score_loss(y, p)
    brier_reference = base_rate * (1 - base_rate)  # Brier score of always predicting the base rate
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier),
        "brier_skill_score": float(1 - brier / brier_reference),
        "log_loss": float(log_loss(y, np.clip(p, PROBABILITY_EPS, 1 - PROBABILITY_EPS))),
        "mean_predicted_pd": float(p.mean()),
        "observed_default_rate": float(base_rate),
        "expected_calibration_error": expected_calibration_error(y, p),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Decision metrics when predicting default for ``p >= threshold``."""
    y = np.asarray(y).astype(int)
    y_hat = (validate_probabilities(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(fbeta_score(y, y_hat, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, y_hat, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "flagged_rate": float(y_hat.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean |observed rate - mean predicted PD| over equal-frequency bins."""
    table = calibration_table(y, p, n_bins=n_bins)
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["observed_rate"] - table["mean_predicted"]).abs()).sum())


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table over equal-frequency (quantile) bins, with Wilson 95% intervals."""
    y = np.asarray(y).astype(int)
    p = validate_probabilities(p)
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    grouped = frame.groupby("bin").agg(
        count=("y", "size"), observed_rate=("y", "mean"), mean_predicted=("p", "mean")
    )
    z = 1.959964
    n, rate = grouped["count"], grouped["observed_rate"]
    centre = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    grouped["observed_low"] = centre - half
    grouped["observed_high"] = centre + half
    return grouped.reset_index()


def bootstrap_confidence_intervals(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals for the main metrics at a fixed threshold."""
    y = np.asarray(y).astype(int)
    p = validate_probabilities(p)
    rng = np.random.default_rng(random_seed)
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "log_loss": lambda yy, pp: log_loss(
            yy, np.clip(pp, PROBABILITY_EPS, 1 - PROBABILITY_EPS), labels=[0, 1]
        ),
        "precision": lambda yy, pp: precision_score(yy, pp >= threshold, zero_division=0),
        "recall": lambda yy, pp: recall_score(yy, pp >= threshold, zero_division=0),
        "f1": lambda yy, pp: fbeta_score(yy, pp >= threshold, beta=1, zero_division=0),
        "f2": lambda yy, pp: fbeta_score(yy, pp >= threshold, beta=2, zero_division=0),
    }
    samples: dict[str, list[float]] = {name: [] for name in metric_fns}
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():  # resample with one class only: AUC undefined
            continue
        for name, fn in metric_fns.items():
            samples[name].append(float(fn(yy, pp)))
    alpha = (1 - confidence) / 2
    return {
        name: {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1 - alpha)),
            "n_resamples": len(values),
        }
        for name, values in samples.items()
    }


def curve_points(y: np.ndarray, p: np.ndarray, max_points: int = 200) -> dict[str, list[float]]:
    """Down-sampled ROC and precision-recall curves for storage and plotting."""
    y = np.asarray(y).astype(int)
    p = validate_probabilities(p)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)

    def thin(*arrays: np.ndarray) -> list[list[float]]:
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, max_points).round().astype(int))
        return [np.round(a[idx], 5).tolist() for a in arrays]

    fpr_s, tpr_s = thin(fpr, tpr)
    prec_s, rec_s = thin(precision, recall)
    return {"roc_fpr": fpr_s, "roc_tpr": tpr_s, "pr_precision": prec_s, "pr_recall": rec_s}


def group_metrics(
    y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float, min_count: int = 50
) -> pd.DataFrame:
    """Per-group diagnostics (e.g. by SEX or age band): calibration-in-the-large and error rates.

    These are descriptive fairness diagnostics, not a legal fairness assessment.
    """
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p), "g": groups.values})
    rows = []
    for group, sub in frame.groupby("g", observed=True):
        if len(sub) < min_count:
            continue
        flagged = sub["p"] >= threshold
        positives, negatives = sub["y"] == 1, sub["y"] == 0
        rows.append(
            {
                "group": str(group),
                "count": len(sub),
                "observed_default_rate": sub["y"].mean(),
                "mean_predicted_pd": sub["p"].mean(),
                "flagged_rate": flagged.mean(),
                "recall": (flagged & positives).sum() / positives.sum()
                if positives.any()
                else np.nan,
                "false_positive_rate": (flagged & negatives).sum() / negatives.sum()
                if negatives.any()
                else np.nan,
                "roc_auc": roc_auc_score(sub["y"], sub["p"]) if sub["y"].nunique() == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str = "roc_auc",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the difference metric(A) - metric(B) on the same resampled observations.

    Pairing removes the shared sampling noise, so this test is far more sensitive than
    comparing two separate confidence intervals. ``p_value_two_sided`` is the bootstrap
    proportion of resamples on the "other side" of zero, doubled.
    """
    fns = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier": brier_score_loss}
    if metric not in fns:
        raise ValueError(f"metric must be one of {sorted(fns)}")
    fn = fns[metric]
    y = np.asarray(y).astype(int)
    p_a, p_b = validate_probabilities(p_a), validate_probabilities(p_b)
    rng = np.random.default_rng(random_seed)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    alpha = (1 - confidence) / 2
    observed = float(fn(y, p_a) - fn(y, p_b))
    tail = (diffs <= 0).mean() if observed > 0 else (diffs >= 0).mean()
    return {
        "metric": metric,
        "difference": observed,
        "low": float(np.quantile(diffs, alpha)),
        "high": float(np.quantile(diffs, 1 - alpha)),
        "p_value_two_sided": float(min(1.0, 2 * tail)),
        "n_resamples": len(diffs),
    }


RISK_BANDS = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0)
RISK_BAND_LABELS = ("< 5%", "5–10%", "10–20%", "20–40%", "≥ 40%")


def risk_band_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Clients, defaults, observed default rate and mean PD per fixed PD band."""
    p = np.asarray(p, dtype=float)
    labels = list(RISK_BAND_LABELS)
    bands = pd.cut(p, list(RISK_BANDS), right=False, include_lowest=True, labels=labels)
    frame = pd.DataFrame({"band": bands, "y": np.asarray(y).astype(int), "p": p})
    table = frame.groupby("band", observed=False).agg(
        clients=("y", "size"),
        defaults=("y", "sum"),
        observed_rate=("y", "mean"),
        mean_pd=("p", "mean"),
    )
    table["share_of_clients"] = table["clients"] / max(len(frame), 1)
    table["share_of_defaults"] = table["defaults"] / max(int(frame["y"].sum()), 1)
    table = table.reset_index()
    table["band"] = table["band"].astype(str)
    return table
