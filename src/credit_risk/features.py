"""Deterministic feature engineering.

:class:`CreditFeatureEngineer` is a scikit-learn transformer that derives
behavioural features from the raw columns. It learns no statistics from the data:
every output depends only on the row itself and fixed constants. That makes it safe
to apply before the train/validation/test split, though it is always used inside the
model pipeline anyway, so preprocessing has one code path.

Assumptions (documented in docs/methodology.md):

* Undocumented ``EDUCATION`` codes 0, 5 and 6 are merged into 4 ("others"), and
  ``MARRIAGE`` 0 into 3 ("others"). This can be switched off for comparison.
* ``PAY_k <= 0`` is treated as "no recorded delay" and ``PAY_k >= 1`` as a delay of
  ``PAY_k`` months. The -2 / 0 codes are undocumented, so the raw values are also kept.
* Ratios use aggregate six-month sums. That avoids assuming which bill statement a
  given monthly payment settles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.data import BILL_AMOUNTS, PAY_AMOUNTS, PAY_STATUS

RATIO_CAP = 10.0  # fixed cap for ratio features to limit the influence of tiny denominators

ENGINEERED_FEATURES = [
    "utilization_1",
    "utilization_mean",
    "utilization_max",
    "bill_trend",
    "payment_to_bill_ratio",
    "payment_1_to_limit",
    "months_delayed",
    "max_delay",
    "recent_delays_3m",
    "delay_trend",
    "months_paid_duly",
    "months_negative_bill",
    "months_zero_bill",
    "inactive_account",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio clipped to [-RATIO_CAP, RATIO_CAP]; 0 where the denominator is not positive."""
    denom = denominator.where(denominator > 0)
    return (numerator / denom).clip(-RATIO_CAP, RATIO_CAP).fillna(0.0)


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered behavioural features to a DataFrame of raw (standardised) features.

    Parameters
    ----------
    merge_undocumented_codes:
        Map EDUCATION {0, 5, 6} -> 4 and MARRIAGE 0 -> 3.
    add_features:
        If False, the transformer only applies the code merge (raw-feature baseline).
    """

    def __init__(self, merge_undocumented_codes: bool = True, add_features: bool = True):
        self.merge_undocumented_codes = merge_undocumented_codes
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - scikit-learn convention
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditFeatureEngineer expects a pandas DataFrame")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = X.copy()
        if self.merge_undocumented_codes:
            out["EDUCATION"] = out["EDUCATION"].where(~out["EDUCATION"].isin([0, 5, 6]), 4)
            out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"] != 0, 3)
        if not self.add_features:
            return out

        limit = out["LIMIT_BAL"].astype(float)
        bills = out[BILL_AMOUNTS].astype(float)
        payments = out[PAY_AMOUNTS].astype(float)
        status = out[PAY_STATUS]
        delay = status.clip(lower=0)

        utilization = bills.div(limit.where(limit > 0), axis=0).clip(-RATIO_CAP, RATIO_CAP)
        out["utilization_1"] = utilization["BILL_AMT1"].fillna(0.0)
        out["utilization_mean"] = utilization.mean(axis=1).fillna(0.0)
        out["utilization_max"] = utilization.max(axis=1).fillna(0.0)
        out["bill_trend"] = _safe_ratio(bills["BILL_AMT1"] - bills["BILL_AMT6"], limit)
        out["payment_to_bill_ratio"] = _safe_ratio(
            payments.sum(axis=1), bills.clip(lower=0).sum(axis=1)
        )
        out["payment_1_to_limit"] = _safe_ratio(payments["PAY_AMT1"], limit)
        out["months_delayed"] = (delay > 0).sum(axis=1)
        out["max_delay"] = delay.max(axis=1)
        out["recent_delays_3m"] = (delay[PAY_STATUS[:3]] > 0).sum(axis=1)
        out["delay_trend"] = delay["PAY_1"] - delay["PAY_6"]
        out["months_paid_duly"] = (status == -1).sum(axis=1)
        out["months_negative_bill"] = (bills < 0).sum(axis=1)
        out["months_zero_bill"] = (bills == 0).sum(axis=1)
        out["inactive_account"] = ((bills == 0).all(axis=1) & (payments == 0).all(axis=1)).astype(
            int
        )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.feature_names_in_)
        if self.add_features:
            names += ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)
