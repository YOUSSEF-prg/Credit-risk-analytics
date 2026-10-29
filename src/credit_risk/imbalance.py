"""Class-imbalance strategies: resampling of the training data.

Every sampler here runs **inside** an ``imblearn.pipeline.Pipeline``, so it is applied
only when the pipeline is fitted, i.e. to the training data or the training folds of
cross-validation. ``predict`` / ``predict_proba`` never resample, so the validation and
test sets keep their natural class distribution.

Design choices:

* **Resampling happens on the raw (standardised) variables, before feature engineering.**
  Engineered features such as ``months_delayed`` are then recomputed from each synthetic
  client, so they stay consistent with that client's repayment history.
* **Mixed data types.** ``SEX``, ``EDUCATION``, ``MARRIAGE`` and the repayment-status codes
  ``PAY_1..PAY_6`` are categorical. Plain SMOTE would create meaningless values such as
  ``EDUCATION = 2.3``, so SMOTE-NC is used.
* **Distances.** Nearest-neighbour methods (SMOTE-NC, Tomek links, ENN) are sensitive to
  scale. :class:`ScaledResampler` gives the inner sampler numeric columns that have been
  signed-log transformed and standardised, using statistics from the data being resampled
  (the training fold), then maps the result back to the original units.
* **Excluded methods.** ADASYN, Borderline-SMOTE, SVM-SMOTE and KMeans-SMOTE are
  not used: they interpolate every column, including the categorical codes. NearMiss is
  not used either, because it is known to be sensitive to noise and outliers.

Resampling and class weighting change the class prior the model learns. The resulting
scores are then not calibrated PDs, and must be recalibrated before use as
probabilities (see :mod:`credit_risk.calibration`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks

from credit_risk.data import CATEGORICAL, PAY_STATUS

RESAMPLE_CATEGORICAL = [*CATEGORICAL, *PAY_STATUS]


class ScaledResampler(BaseSampler):
    """Wrap an imbalanced-learn sampler so it sees scaled numeric columns.

    Parameters
    ----------
    sampler:
        Any imbalanced-learn sampler. For SMOTE-NC (alone or inside SMOTETomek /
        SMOTEENN) the categorical column indices are set automatically.
    categorical_columns:
        Columns treated as categorical. They are passed through unscaled.
    """

    _sampling_type = "bypass"
    _parameter_constraints: dict = {}

    def __init__(self, sampler: Any, categorical_columns: list[str] | None = None):
        super().__init__()
        self.sampler = sampler
        self.categorical_columns = categorical_columns

    def _fit_resample(self, X, y):  # noqa: N803 - imblearn convention
        raise NotImplementedError  # fit_resample is overridden below

    def fit_resample(self, X, y, **params):  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ScaledResampler expects a pandas DataFrame")
        columns = list(X.columns)
        categorical = [c for c in (self.categorical_columns or RESAMPLE_CATEGORICAL) if c in X]
        numeric = [c for c in columns if c not in categorical]
        cat_idx = [columns.index(c) for c in categorical]

        values = X.to_numpy(dtype=float, copy=True)
        num_idx = [columns.index(c) for c in numeric]
        transformed = np.sign(values[:, num_idx]) * np.log1p(np.abs(values[:, num_idx]))
        mean, std = transformed.mean(axis=0), transformed.std(axis=0)
        std[std == 0] = 1.0
        values[:, num_idx] = (transformed - mean) / std

        sampler = self._configure(self.sampler, cat_idx)
        X_res, y_res = sampler.fit_resample(values, np.asarray(y))

        back = X_res[:, num_idx] * std + mean
        X_res[:, num_idx] = np.sign(back) * np.expm1(np.abs(back))
        out = pd.DataFrame(np.rint(X_res), columns=columns).astype(X.dtypes.to_dict())
        self.sampling_summary_ = {
            "n_before": len(y),
            "n_after": len(y_res),
            "default_rate_before": float(np.mean(y)),
            "default_rate_after": float(np.mean(y_res)),
        }
        return out, pd.Series(y_res, name=getattr(y, "name", None))

    @staticmethod
    def _configure(sampler: Any, cat_idx: list[int]) -> Any:
        from sklearn.base import clone

        sampler = clone(sampler)
        if isinstance(sampler, SMOTENC):
            sampler.set_params(categorical_features=cat_idx)
        elif isinstance(sampler, SMOTETomek | SMOTEENN) and isinstance(sampler.smote, SMOTENC):
            sampler.set_params(smote__categorical_features=cat_idx)
        return sampler


def build_sampler(name: str, random_seed: int = 42, **params: Any) -> ScaledResampler:
    """Return a configured sampler by short name.

    ``sampling_strategy`` (minority:majority ratio after resampling) can be passed for
    the over-/under-samplers. Cleaning methods (Tomek, ENN) take no ratio.
    """
    ratio = params.pop("sampling_strategy", "auto")
    placeholder = [0]  # replaced with the real categorical indices at fit time
    smote_nc = SMOTENC(
        categorical_features=placeholder, sampling_strategy=ratio, random_state=random_seed
    )
    samplers = {
        "random_oversampling": lambda: RandomOverSampler(
            sampling_strategy=ratio, random_state=random_seed
        ),
        "smotenc": lambda: smote_nc,
        "random_undersampling": lambda: RandomUnderSampler(
            sampling_strategy=ratio, random_state=random_seed
        ),
        "tomek_links": lambda: TomekLinks(),
        "edited_nearest_neighbours": lambda: EditedNearestNeighbours(),
        "smotenc_tomek": lambda: SMOTETomek(
            smote=smote_nc, sampling_strategy=ratio, random_state=random_seed
        ),
        "smotenc_enn": lambda: SMOTEENN(
            smote=smote_nc, sampling_strategy=ratio, random_state=random_seed
        ),
    }
    if name not in samplers:
        raise ValueError(f"Unknown sampler {name!r}; options: {sorted(samplers)}")
    return ScaledResampler(samplers[name](), **params)
