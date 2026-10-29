"""SHAP explanations for tree-based PD models.

Explanations use ``shap.TreeExplainer`` on the fitted estimator, applied to the
pipeline's transformed features. They are given in **log-odds** (the model's raw
margin): for each client, base value + Σ SHAP values = logit(PD). This additivity is
checked explicitly (:func:`check_additivity`).

Interpretation rules (also stated in the dashboard):

* SHAP values describe how the **model** uses a feature for a given client. They are
  predictive associations learned from historical data, **not causal effects**.
  Changing a client's feature would not necessarily change their real default risk.
* Correlated features (monthly bills, the repayment codes and their engineered counts)
  share credit, so individual attributions within such a group can be unstable. Group
  totals (:func:`grouped_importance`) are more robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

FEATURE_GROUPS = {
    "Repayment status (PAY_1..6)": lambda c: c.startswith("PAY_") and not c.startswith("PAY_AMT"),
    "Delay history (engineered)": lambda c: c in {"months_delayed", "max_delay", "recent_delays_3m",
                                                 "delay_trend", "months_paid_duly"},
    "Bill amounts (BILL_AMT1..6)": lambda c: c.startswith("BILL_AMT"),
    "Payment amounts (PAY_AMT1..6)": lambda c: c.startswith("PAY_AMT"),
    "Utilisation & ratios (engineered)": lambda c: c.startswith("utilization")
    or c in {"payment_to_bill_ratio", "payment_1_to_limit", "bill_trend"},
    "Balance patterns (engineered)": lambda c: c in {"months_negative_bill", "months_zero_bill",
                                                    "inactive_account"},
    "Credit limit": lambda c: c == "LIMIT_BAL",
    "Demographics (SEX, EDUCATION, MARRIAGE, AGE)": lambda c: c in {"SEX", "EDUCATION",
                                                                    "MARRIAGE", "AGE"},
}  # fmt: skip


@dataclass
class ShapResult:
    values: pd.DataFrame  # one row per client, one column per model feature (log-odds)
    features: pd.DataFrame  # the feature values the model actually saw
    base_value: float  # expected model output (log-odds) over the background data

    def logit(self) -> pd.Series:
        return self.base_value + self.values.sum(axis=1)


def transform_for_model(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply all pipeline steps before the estimator and return named features."""
    steps = pipeline[:-1]
    transformed = steps.transform(X)
    names = steps[-1].get_feature_names_out()
    return pd.DataFrame(np.asarray(transformed, dtype=float), columns=names, index=X.index)


def explain(pipeline: Pipeline, X: pd.DataFrame) -> ShapResult:
    """Exact TreeSHAP values (log-odds) for a fitted tree-model pipeline."""
    import shap

    features = transform_for_model(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer.shap_values(features)
    if isinstance(values, list):  # some versions return one array per class
        values = values[1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    return ShapResult(pd.DataFrame(values, columns=features.columns, index=X.index), features, base)


def check_additivity(result: ShapResult, pd_values: np.ndarray, atol: float = 1e-4) -> float:
    """Max |base + ΣSHAP − logit(PD)|; raises if explanations do not add up to predictions."""
    p = np.clip(np.asarray(pd_values, dtype=float), 1e-12, 1 - 1e-12)
    gap = float(np.max(np.abs(result.logit().to_numpy() - np.log(p / (1 - p)))))
    if gap > atol:
        raise AssertionError(f"SHAP values do not reproduce the model output (max gap {gap:.2e})")
    return gap


def global_importance(result: ShapResult) -> pd.DataFrame:
    """Mean |SHAP| per feature, plus the share of total attribution and the sign of the effect."""
    abs_mean = result.values.abs().mean()
    corr = {
        c: np.corrcoef(result.features[c], result.values[c])[0, 1]
        if result.features[c].std() > 0
        else np.nan
        for c in result.values.columns
    }
    table = pd.DataFrame(
        {
            "mean_abs_shap": abs_mean,
            "share": abs_mean / abs_mean.sum(),
            "value_shap_correlation": pd.Series(corr),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False).rename_axis("feature").reset_index()


def grouped_importance(result: ShapResult) -> pd.DataFrame:
    """Mean |Σ SHAP within group|: attribution per group of related features.

    Summing within a group before taking the absolute value lets opposite-signed
    attributions among correlated features cancel. That makes group totals more stable than
    individual feature values.
    """
    rows = []
    assigned: set[str] = set()
    for group, member in FEATURE_GROUPS.items():
        cols = [c for c in result.values.columns if member(c) and c not in assigned]
        assigned.update(cols)
        if cols:
            rows.append(
                {
                    "group": group,
                    "features": len(cols),
                    "mean_abs_shap": float(result.values[cols].sum(axis=1).abs().mean()),
                }
            )
    rest = [c for c in result.values.columns if c not in assigned]
    if rest:
        rows.append(
            {
                "group": "Other",
                "features": len(rest),
                "mean_abs_shap": float(result.values[rest].sum(axis=1).abs().mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(result: ShapResult, row: int, top: int = 10) -> pd.DataFrame:
    """Top contributions for one client (by position), largest |SHAP| first.

    A final row "All other features" holds the sum of the remaining contributions, so
    the rows always add up to the client's total SHAP value.
    """
    values = result.values.iloc[row]
    order = values.abs().sort_values(ascending=False).index[:top]
    table = pd.DataFrame(
        {
            "feature": order,
            "value": result.features.iloc[row][order].to_numpy(),
            "shap_log_odds": values[order].to_numpy(),
        }
    )
    rest = {"feature": "All other features", "value": np.nan,
            "shap_log_odds": float(values.sum() - table["shap_log_odds"].sum())}  # fmt: skip
    return pd.concat([table, pd.DataFrame([rest])], ignore_index=True)
