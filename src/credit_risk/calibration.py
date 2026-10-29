"""Probability calibration without touching the validation or test sets.

``CalibratedClassifierCV(ensemble=False)`` with stratified K-fold on the **training
data**:

1. Out-of-fold scores are produced for every training row. Each fold model is a full
   pipeline, including any resampling, fitted on the other folds.
2. A calibration map (Platt / sigmoid, or isotonic) is fitted from those out-of-fold
   scores to the observed training outcomes.
3. The base pipeline is refitted on the full training set, and the map is applied to
   its scores.

Because the map is learned on held-out rows with the natural class distribution, it
also undoes the prior shift that resampling or class weighting introduces.
Which method to use (none / sigmoid / isotonic) is decided on the validation set,
never on the test set.

* **Sigmoid:** two parameters; robust with little data, but assumes the miscalibration is
  sigmoid-shaped.
* **Isotonic:** non-parametric and monotone; more flexible, but can overfit with small
  samples. Here it is fitted on 18,000 out-of-fold scores, which is ample.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from credit_risk.models import build_pipeline

CALIBRATION_METHODS = ("sigmoid", "isotonic")


def fit_calibrated(
    spec: dict[str, Any],
    method: str,
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> CalibratedClassifierCV:
    """Fit a calibrated version of the pipeline described by ``spec`` on training data."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"method must be one of {CALIBRATION_METHODS}")
    calibrated = CalibratedClassifierCV(
        build_pipeline(spec, random_seed=random_seed),
        method=method,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed),
        ensemble=False,
        n_jobs=n_jobs,
    )
    return calibrated.fit(X_train, y_train)


def prior_correction(p, train_rate: float, resampled_rate: float):
    """Analytic correction for a known shift of the training class prior.

    A model trained on data with default rate ``resampled_rate`` (e.g. 0.5 after random
    under-/oversampling or balanced class weights) instead of the population rate
    ``train_rate`` has its odds multiplied by
    [π_s / (1 − π_s)] / [π / (1 − π)]. This function divides that factor out:

        odds_corrected = odds_model × [π / (1 − π)] / [π_s / (1 − π_s)]

    The correction is exact for random resampling and class weighting, and only
    approximate for SMOTE, cleaning methods and ensembles. It fits no parameters, so it
    is used here as a diagnostic of *why* rebalanced models are miscalibrated.
    """
    import numpy as np

    for rate in (train_rate, resampled_rate):
        if not 0 < rate < 1:
            raise ValueError("rates must lie strictly between 0 and 1")
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    factor = (train_rate / (1 - train_rate)) / (resampled_rate / (1 - resampled_rate))
    odds = p / (1 - p) * factor
    return odds / (1 + odds)
