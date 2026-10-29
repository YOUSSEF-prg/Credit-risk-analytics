from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_risk.data import FEATURES, split_features_target, standardize_columns
from credit_risk.decision_analysis import expected_cost, f_beta, select_threshold, threshold_table
from credit_risk.evaluation import (
    bootstrap_confidence_intervals,
    calibration_table,
    probability_metrics,
    threshold_metrics,
    validate_probabilities,
)
from credit_risk.features import ENGINEERED_FEATURES, CreditFeatureEngineer
from credit_risk.models import build_pipeline, predict_pd, tune_pipeline

LOGREG_SPEC = {
    "estimator": "logistic_regression",
    "preprocessor": "linear",
    "features": {"add_features": True},
    "params": {"max_iter": 2000},
}


@pytest.fixture
def xy(synthetic_raw):
    return split_features_target(standardize_columns(synthetic_raw))


# --- feature engineering ----------------------------------------------------------


def test_feature_engineer_is_row_wise(xy):
    """Output for a row must not depend on other rows (no learned statistics)."""
    X, _ = xy
    fe = CreditFeatureEngineer().fit(X)
    full = fe.transform(X)
    subset = fe.transform(X.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], subset)


def test_feature_engineer_adds_features_and_merges_codes(xy):
    X, _ = xy
    out = CreditFeatureEngineer().fit_transform(X)
    assert list(out.columns) == FEATURES + ENGINEERED_FEATURES
    assert set(out["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"].unique()) <= {1, 2, 3}
    assert np.isfinite(out[ENGINEERED_FEATURES].to_numpy(dtype=float)).all()
    assert (X["EDUCATION"] == CreditFeatureEngineer().fit(X).transform(X)["EDUCATION"]).sum() < len(
        X
    )


def test_feature_engineer_does_not_mutate_input(xy):
    X, _ = xy
    before = X.copy()
    CreditFeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_known_feature_values():
    row = pd.DataFrame(
        [
            {
                "LIMIT_BAL": 100_000, "SEX": 1, "EDUCATION": 5, "MARRIAGE": 0, "AGE": 30,
                "PAY_1": 2, "PAY_2": 1, "PAY_3": -1, "PAY_4": 0, "PAY_5": -2, "PAY_6": -1,
                "BILL_AMT1": 50_000, "BILL_AMT2": 0, "BILL_AMT3": 0, "BILL_AMT4": 0,
                "BILL_AMT5": 0, "BILL_AMT6": 10_000,
                "PAY_AMT1": 6_000, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0,
                "PAY_AMT5": 0, "PAY_AMT6": 0,
            }
        ]
    )  # fmt: skip
    out = CreditFeatureEngineer().fit_transform(row).iloc[0]
    assert out["EDUCATION"] == 4 and out["MARRIAGE"] == 3
    assert out["utilization_1"] == pytest.approx(0.5)
    assert out["bill_trend"] == pytest.approx(0.4)
    assert out["payment_to_bill_ratio"] == pytest.approx(0.1)
    assert out["months_delayed"] == 2 and out["max_delay"] == 2
    assert out["recent_delays_3m"] == 2 and out["months_paid_duly"] == 2
    assert out["inactive_account"] == 0


# --- pipelines and leakage --------------------------------------------------------


def test_preprocessing_statistics_come_from_training_data_only(xy):
    X, y = xy
    X_train, y_train = X.iloc[:300], y.iloc[:300]
    model = build_pipeline(LOGREG_SPEC).fit(X_train, y_train)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    engineered = CreditFeatureEngineer().fit_transform(X_train)
    assert scaler.mean_[0] == pytest.approx(engineered["AGE"].mean())
    # Refitting on different data must change the learned statistics.
    other = build_pipeline(LOGREG_SPEC).fit(X.iloc[300:], y.iloc[300:])
    other_scaler = (
        other.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    )
    assert other_scaler.mean_[0] != pytest.approx(scaler.mean_[0])


def test_pipeline_outputs_valid_probabilities_and_handles_missing(xy):
    X, y = xy
    model = build_pipeline(LOGREG_SPEC).fit(X, y)
    X_missing = X.iloc[:5].astype(float).copy()
    X_missing.iloc[0, 0] = np.nan  # e.g. incomplete user input in the Explorer page
    p = predict_pd(model, X_missing)
    assert ((p >= 0) & (p <= 1)).all() and p.notna().all()


def test_tuning_returns_best_params(xy):
    X, y = xy
    model, summary = tune_pipeline(
        build_pipeline(LOGREG_SPEC), {"C": [0.1, 1.0]}, X, y, cv_folds=3, n_jobs=1
    )
    assert summary["best_params"]["C"] in {0.1, 1.0}
    assert len(summary["cv_results"]) == 2
    assert hasattr(model, "predict_proba")


def test_dummy_prior_predicts_training_rate(xy):
    X, y = xy
    spec = {
        "estimator": "dummy",
        "features": {"add_features": False},
        "params": {"strategy": "prior"},
    }
    p = predict_pd(build_pipeline(spec).fit(X, y), X)
    assert np.allclose(p, y.mean())


# --- metrics ----------------------------------------------------------------------


def test_probability_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_probabilities(np.array([0.2, 1.2]))
    with pytest.raises(ValueError):
        validate_probabilities(np.array([0.2, np.nan]))


def test_probability_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 * y + rng.random(400) * 0.7, 0, 1)
    m = probability_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    base = y.mean()
    assert m["brier_skill_score"] == pytest.approx(1 - m["brier"] / (base * (1 - base)))


def test_threshold_metrics_confusion_matrix():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.7, 0.2])
    m = threshold_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_calibration_table_perfectly_calibrated_case():
    rng = np.random.default_rng(3)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)
    table = calibration_table(y, p, n_bins=10)
    assert (table["observed_rate"] - table["mean_predicted"]).abs().max() < 0.03
    assert (table["observed_low"] <= table["observed_rate"]).all()


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.4 * y + rng.random(500) * 0.6, 0, 1)
    ci = bootstrap_confidence_intervals(y, p, 0.5, n_resamples=200)
    auc = roc_auc_score(y, p)
    assert ci["roc_auc"]["low"] <= auc <= ci["roc_auc"]["high"]


# --- threshold selection and costs ------------------------------------------------


def test_f_beta_weights_recall():
    assert f_beta(0.5, 0.5, 2) == pytest.approx(0.5)
    assert f_beta(0.3, 0.9, 2) > f_beta(0.9, 0.3, 2)
    assert f_beta(0.0, 0.0, 2) == 0.0


def test_select_threshold_maximises_f_beta_on_grid():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.35 * y + rng.random(2000) * 0.65, 0, 1)
    chosen = select_threshold(y, p, beta=2)
    table = threshold_table(y, p)
    assert chosen["objective_value"] == pytest.approx(table["f2"].max())
    # Emphasising recall should never pick a higher threshold than F1 does.
    assert chosen["threshold"] <= select_threshold(y, p, beta=1)["threshold"]


def test_threshold_table_counts_are_consistent():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.4, 0.3, 0.9])
    table = threshold_table(y, p, np.array([0.0, 0.5, 1.0]))
    assert (table[["tp", "fp", "tn", "fn"]].sum(axis=1) == 5).all()
    assert table.loc[0, "recall"] == 1.0 and table.loc[2, "tp"] == 0


def test_expected_cost_is_illustrative_linear_combination():
    table = pd.DataFrame([{"tp": 5, "fp": 10, "tn": 80, "fn": 5}])
    out = expected_cost(table, cost_false_negative=5.0, cost_false_positive=1.0)
    assert out.loc[0, "illustrative_cost"] == pytest.approx(35.0)
    with pytest.raises(ValueError):
        expected_cost(table, -1.0, 1.0)


def test_paired_bootstrap_detects_better_model_and_null_difference():
    from credit_risk.evaluation import paired_bootstrap_difference

    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 1500)
    noise = rng.random(1500)
    good = np.clip(0.5 * y + 0.5 * noise, 0, 1)
    weak = np.clip(0.1 * y + 0.9 * noise, 0, 1)
    better = paired_bootstrap_difference(y, good, weak, n_resamples=300)
    assert better["difference"] > 0 and better["low"] > 0 and better["p_value_two_sided"] < 0.01
    same = paired_bootstrap_difference(y, good, good, n_resamples=100)
    assert same["difference"] == 0 and same["low"] == 0 == same["high"]


def test_experiment_partitions_exclude_test_set(real_raw_path):
    """The experiment loader must expose train and validation rows only."""
    from credit_risk.config import load_config, resolve_path
    from credit_risk.data import DataSplits
    from credit_risk.experiment import load_partitions

    cfg = load_config("data")
    data = load_partitions(cfg)
    splits = DataSplits.from_json(resolve_path(cfg["split"]["output_path"]))
    assert set(data.val_ids).isdisjoint(splits.test_ids)
    assert len(data.y_train) + len(data.y_val) == 24_000
    assert not hasattr(data, "X_test")


def test_risk_band_table_partitions_clients_and_defaults():
    from credit_risk.evaluation import RISK_BAND_LABELS, risk_band_table

    rng = np.random.default_rng(9)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)
    bands = risk_band_table(y, p)
    assert list(bands["band"]) == list(RISK_BAND_LABELS)
    assert bands["clients"].sum() == 20_000
    assert bands["share_of_defaults"].sum() == pytest.approx(1.0)
    assert bands["observed_rate"].is_monotonic_increasing
