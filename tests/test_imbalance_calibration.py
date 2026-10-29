from __future__ import annotations

import json

import numpy as np
import pytest

from credit_risk.calibration import fit_calibrated
from credit_risk.data import split_features_target, standardize_columns
from credit_risk.decision_analysis import select_threshold
from credit_risk.experiment import resolve_spec
from credit_risk.imbalance import RESAMPLE_CATEGORICAL, build_sampler
from credit_risk.models import build_pipeline, predict_pd

SAMPLERS = [
    "random_oversampling",
    "smotenc",
    "random_undersampling",
    "tomek_links",
    "edited_nearest_neighbours",
    "smotenc_tomek",
    "smotenc_enn",
]


@pytest.fixture
def xy(synthetic_raw):
    X, y = split_features_target(standardize_columns(synthetic_raw))
    # make the target learnable so neighbourhood methods behave realistically
    y = ((X["PAY_1"] >= 7) | (np.random.default_rng(0).random(len(X)) < 0.05)).astype(int)
    return X, y


@pytest.mark.parametrize("name", SAMPLERS)
def test_samplers_keep_valid_categorical_codes_and_types(xy, name):
    X, y = xy
    sampler = build_sampler(name)
    X_res, y_res = sampler.fit_resample(X, y)
    assert list(X_res.columns) == list(X.columns)
    assert (X_res.dtypes == X.dtypes).all()
    for col in RESAMPLE_CATEGORICAL:
        assert set(X_res[col]) <= set(X[col]), f"{name} created a new {col} code"
    assert (X_res.filter(like="PAY_AMT") >= 0).all().all()
    assert set(np.unique(y_res)) <= {0, 1}


def test_balancing_samplers_reach_target_ratio(xy):
    X, y = xy
    for name in ["random_oversampling", "smotenc", "random_undersampling"]:
        _, y_res = build_sampler(name).fit_resample(X, y)
        assert y_res.mean() == pytest.approx(0.5, abs=1e-9)


def test_smotenc_creates_new_numeric_points(xy):
    X, y = xy
    X_res, _ = build_sampler("smotenc").fit_resample(X, y)
    synthetic = X_res.iloc[len(X) :]
    assert len(synthetic) > 0
    merged = synthetic.merge(X, how="left", indicator=True)
    assert (merged["_merge"] == "left_only").mean() > 0.9  # mostly genuinely new rows


def test_resampling_only_happens_during_fit(xy):
    X, y = xy
    spec = {
        "estimator": "logistic_regression",
        "features": {"add_features": True},
        "params": {"max_iter": 2000},
        "sampler": {"name": "smotenc"},
    }
    model = build_pipeline(spec).fit(X, y)
    p = predict_pd(model, X)
    assert len(p) == len(X)  # predict_proba is never resampled


def test_resampled_model_inflates_pd_and_calibration_corrects_it(xy):
    X, y = xy
    spec = {
        "estimator": "logistic_regression",
        "features": {"add_features": True},
        "params": {"max_iter": 2000},
        "sampler": {"name": "random_undersampling"},
    }
    raw = predict_pd(build_pipeline(spec).fit(X, y), X)
    calibrated = predict_pd(fit_calibrated(spec, "sigmoid", X, y, cv_folds=3, n_jobs=1), X)
    assert raw.mean() > y.mean() + 0.03  # balanced training prior pushes PDs up
    assert abs(calibrated.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_fit_calibrated_rejects_unknown_method(xy):
    X, y = xy
    with pytest.raises(ValueError):
        fit_calibrated({"estimator": "logistic_regression"}, "beta", X, y)


def test_resolve_spec_inherits_tuned_params(tmp_path):
    models = {
        "base": {
            "estimator": "xgboost",
            "preprocessor": "tree",
            "params": {"n_jobs": 1},
            "search": {"max_depth": [2, 3]},
            "search_type": "random",
        },
        "variant": {"inherit": "base", "sampler": {"name": "smotenc"}, "params": {"gamma": 1}},
    }
    (tmp_path / "base.json").write_text(json.dumps({"cv": {"best_params": {"max_depth": 3}}}))
    spec = resolve_spec("variant", models, tmp_path)
    assert spec["params"] == {"n_jobs": 1, "max_depth": 3, "gamma": 1}
    assert spec["search"] == {} and spec["sampler"]["name"] == "smotenc"
    assert spec["inherits_from"] == "base"
    with pytest.raises(FileNotFoundError):
        resolve_spec("variant", models, tmp_path / "missing")


def test_capacity_constrained_threshold_respects_cap():
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.22).astype(int)
    p = np.clip(0.3 * y + 0.7 * rng.random(3000), 0, 1)
    free = select_threshold(y, p, beta=2)
    capped = select_threshold(y, p, beta=2, max_flagged_rate=0.3)
    assert capped["flagged_rate"] <= 0.3
    assert capped["threshold"] >= free["threshold"]
    assert capped["objective_value"] <= free["objective_value"]
    with pytest.raises(ValueError):
        select_threshold(y, p, max_flagged_rate=0.0)


def test_prior_correction_inverts_a_known_prior_shift():
    from credit_risk.calibration import prior_correction

    rng = np.random.default_rng(8)
    true_p = rng.uniform(0.01, 0.9, 5000)
    pi, pi_s = 0.22, 0.5
    factor = (pi_s / (1 - pi_s)) / (pi / (1 - pi))
    shifted = true_p / (1 - true_p) * factor
    shifted = shifted / (1 + shifted)  # what a model trained on a balanced sample outputs
    assert np.allclose(prior_correction(shifted, pi, pi_s), true_p)
    assert np.allclose(prior_correction(true_p, pi, pi), true_p)  # no shift -> identity
    with pytest.raises(ValueError):
        prior_correction(true_p, 0.0, 0.5)
