"""Model pipelines and hyperparameter search.

A model is fully described by a specification dictionary from ``configs/models.yaml``:

.. code-block:: yaml

    xgboost_smotenc:
      estimator: xgboost
      preprocessor: tree
      sampler: {name: smotenc}          # optional; resampling of training data only
      params_from: xgboost               # optional; reuse tuned params of another model
      params: {n_estimators: 400}
      search: {max_depth: [3, 4, 5]}     # grid (default) or random search
      search_type: random
      n_iter: 20

Every model is a single pipeline: [resampling] -> feature engineering ->
preprocessing -> estimator. With a sampler, an ``imblearn.pipeline.Pipeline`` is
used, so resampling happens only in ``fit`` (training data or CV training folds) and
never in ``predict_proba``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from credit_risk.data import FEATURES
from credit_risk.features import ENGINEERED_FEATURES, CreditFeatureEngineer
from credit_risk.imbalance import build_sampler
from credit_risk.preprocessing import build_preprocessor

CV_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier_score": "neg_brier_score",
}


def _xgb_classifier(**params: Any) -> BaseEstimator:
    from xgboost import XGBClassifier

    params.setdefault("tree_method", "hist")
    params.setdefault("eval_metric", "logloss")
    # device: "cpu" (default) or "cuda". With CV running in parallel processes, every
    # process would open its own CUDA context, so "cuda" is best combined with n_jobs: 1.
    params.setdefault("device", "cpu")
    return XGBClassifier(**params)


def _with_tree_base(cls):
    """Allow ``base_max_depth`` in YAML for boosting/bagging ensembles of decision trees."""

    def factory(**params: Any) -> BaseEstimator:
        depth = params.pop("base_max_depth", None)
        if depth is not None:
            params["estimator"] = DecisionTreeClassifier(
                max_depth=depth, random_state=params.get("random_state")
            )
        return cls(**params)

    return factory


ESTIMATORS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": _xgb_classifier,
    "balanced_random_forest": BalancedRandomForestClassifier,
    "easy_ensemble": _with_tree_base(EasyEnsembleClassifier),
    "rusboost": _with_tree_base(RUSBoostClassifier),
}


def build_pipeline(spec: dict[str, Any], random_seed: int = 42) -> Pipeline:
    """Build an unfitted pipeline from a model specification."""
    engineer = CreditFeatureEngineer(**spec.get("features", {}))
    names = FEATURES + (ENGINEERED_FEATURES if engineer.add_features else [])

    estimator_name = spec["estimator"]
    if estimator_name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator_name!r}; options: {sorted(ESTIMATORS)}")
    params = dict(spec.get("params", {}))
    params.setdefault("random_state", random_seed)  # every registered estimator accepts a seed
    estimator = ESTIMATORS[estimator_name](**params)

    steps = [
        ("features", engineer),
        ("preprocess", build_preprocessor(spec.get("preprocessor", "linear"), names)),
        ("model", estimator),
    ]
    sampler_spec = spec.get("sampler")
    if sampler_spec:
        sampler_params = {k: v for k, v in sampler_spec.items() if k != "name"}
        sampler = build_sampler(sampler_spec["name"], random_seed=random_seed, **sampler_params)
        return ImbPipeline([("resample", sampler), *steps])
    return Pipeline(steps)


def tune_pipeline(
    pipeline: Pipeline,
    search_space: dict[str, list[Any]],
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    scoring: str = "average_precision",
    cv_folds: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1,
    search_type: str = "grid",
    n_iter: int = 20,
) -> tuple[Pipeline, dict[str, Any]]:
    """Hyperparameter search with stratified K-fold CV on the training data only.

    Every configuration is scored on average precision, ROC-AUC and Brier score, and
    the configuration with the best ``scoring`` is refitted on the full training data.
    With an empty search space the fixed configuration is still cross-validated, so
    every model has comparable CV estimates.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    if scoring not in CV_SCORING:
        raise ValueError(f"scoring must be one of {sorted(CV_SCORING)}")
    grid = {f"model__{k}": v for k, v in search_space.items()}

    if not grid:
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=n_jobs
        )
        fitted = pipeline.fit(X_train, y_train)
        summary = {
            "best_params": {},
            "search_type": "none",
            "cv_scoring": scoring,
            "cv_folds": cv_folds,
            **_cv_block({m: scores[f"test_{m}"] for m in CV_SCORING}, scoring),
            "cv_results": [],
        }
        return fitted, summary

    common = dict(scoring=CV_SCORING, refit=scoring, cv=cv, n_jobs=n_jobs, error_score="raise")
    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline, grid, n_iter=n_iter, random_state=random_seed, **common
        )
    elif search_type == "grid":
        search = GridSearchCV(pipeline, grid, **common)
    else:
        raise ValueError("search_type must be 'grid' or 'random'")
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    best = search.best_index_
    fold_scores = {
        m: np.array([results.loc[best, f"split{k}_test_{m}"] for k in range(cv_folds)])
        for m in CV_SCORING
    }
    summary = {
        "best_params": {k.removeprefix("model__"): _py(v) for k, v in search.best_params_.items()},
        "search_type": search_type,
        "n_candidates": len(results),
        "cv_scoring": scoring,
        "cv_folds": cv_folds,
        **_cv_block(fold_scores, scoring),
        "cv_results": [
            {
                "params": {k.removeprefix("model__"): _py(v) for k, v in row["params"].items()},
                "mean": float(row[f"mean_test_{scoring}"]),
                "std": float(row[f"std_test_{scoring}"]),
            }
            for _, row in results.iterrows()
        ],
    }
    return search.best_estimator_, summary


def _cv_block(fold_scores: dict[str, np.ndarray], scoring: str) -> dict[str, Any]:
    means = {m: float(np.mean(v)) for m, v in fold_scores.items()}
    stds = {m: float(np.std(v)) for m, v in fold_scores.items()}
    means["brier"] = -means.pop("neg_brier_score")
    stds["brier"] = stds.pop("neg_brier_score")
    return {
        "best_cv_score_mean": means[scoring] if scoring in means else means["brier"],
        "best_cv_score_std": stds[scoring] if scoring in stds else stds["brier"],
        "cv_metrics_mean": means,
        "cv_metrics_std": stds,
    }


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON output."""
    return value.item() if isinstance(value, np.generic) else value


def predict_pd(model: Pipeline, X: pd.DataFrame) -> pd.Series:  # noqa: N803
    """Return the predicted probability of default (positive class) as a Series."""
    classes = list(model.classes_)
    proba = model.predict_proba(X)[:, classes.index(1)]
    return pd.Series(proba, index=X.index, name="pd")
