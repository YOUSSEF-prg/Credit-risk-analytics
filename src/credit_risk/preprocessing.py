"""Preprocessing (ColumnTransformer) definitions.

Every learned statistic (medians, scaling parameters, one-hot vocabularies) is fitted
inside the model pipeline, so it is computed from the training data only, and only from
each training fold during cross-validation.

Two variants:

* ``linear``: for Logistic Regression. Signed log transform of monetary amounts,
  standardisation, and one-hot encoding of nominal variables and of the repayment
  status codes. The default rate is not monotonic in the raw ``PAY_k`` code (-2, -1
  and 0 all carry low risk), so a single linear coefficient on it would be misspecified.
* ``tree``: for Random Forest / XGBoost. Raw numeric values, with median imputation
  only. Tree splits are invariant to monotonic transforms, so no scaling is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.data import BILL_AMOUNTS, CATEGORICAL, PAY_AMOUNTS, PAY_STATUS

AMOUNT_COLUMNS = ["LIMIT_BAL", *BILL_AMOUNTS, *PAY_AMOUNTS]
MIN_CATEGORY_FREQUENCY = 30  # rarer repayment-status codes are pooled into one level


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|): compresses heavy-tailed amounts and keeps negative balances."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(kind: str, feature_names: Sequence[str]) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for ``kind`` in {"linear", "tree"}."""
    names = list(feature_names)
    if kind == "linear":
        nominal = [c for c in CATEGORICAL if c in names]
        status = [c for c in PAY_STATUS if c in names]
        amounts = [c for c in AMOUNT_COLUMNS if c in names]
        numeric = [c for c in names if c not in {*nominal, *status, *amounts}]
        one_hot = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse_output=False,
        )
        return ColumnTransformer(
            [
                (
                    "nominal",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    nominal,
                ),
                (
                    "pay_status",
                    Pipeline(
                        [("impute", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot)]
                    ),
                    status,
                ),
                (
                    "amounts",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            (
                                "signed_log",
                                FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    amounts,
                ),
                (
                    "numeric",
                    Pipeline(
                        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                    ),
                    numeric,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
    if kind == "tree":
        return ColumnTransformer(
            [("all", SimpleImputer(strategy="median"), names)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    raise ValueError(f"Unknown preprocessor kind: {kind!r}")
