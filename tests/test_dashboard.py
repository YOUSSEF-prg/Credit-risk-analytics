"""Dashboard tests: results interface, input validation, and page smoke tests (AppTest)."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.config import PROJECT_ROOT, resolve_path
from credit_risk.data import FEATURES
from credit_risk.results import ResultsNotFoundError, ResultsStore
from credit_risk.validation import validate_feature_input

REQUIRED = [
    "artifacts/metrics/test_evaluation.json",
    "artifacts/explainability/summary_xgboost.json",
    "data/raw/default of credit card clients.xls",
]
ARTIFACTS_READY = all(resolve_path(p).exists() for p in REQUIRED)
needs_artifacts = pytest.mark.skipif(not ARTIFACTS_READY, reason="experiment outputs not available")


def _valid_client() -> dict:
    client = dict.fromkeys(FEATURES, 0)
    client.update(LIMIT_BAL=100_000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=35)
    return client


def test_valid_input_is_accepted():
    frame, errors = validate_feature_input(_valid_client())
    assert errors == [] and list(frame.columns) == FEATURES and len(frame) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("AGE", 150, "outside"),
        ("SEX", 3, "valid code"),
        ("LIMIT_BAL", None, "missing"),
        ("PAY_1", 1.5, "whole number"),
        ("BILL_AMT1", "abc", "not a number"),
        ("EDUCATION", 0, "valid code"),
    ],
)
def test_invalid_inputs_are_rejected_with_messages(field, value, message):
    client = _valid_client()
    client[field] = value
    frame, errors = validate_feature_input(client)
    assert frame is None and any(field in e and message in e for e in errors)


def test_results_store_reports_missing_outputs(tmp_path):
    store = ResultsStore(root=tmp_path)
    with pytest.raises(ResultsNotFoundError):
        _ = store.test_evaluation
    with pytest.raises(ResultsNotFoundError):
        store.predictions("test", "xgboost")


@needs_artifacts
def test_results_store_reads_stored_metrics():
    store = ResultsStore()
    assert store.selected_model == "xgboost"
    auc = store.metric("xgboost", "roc_auc", "test")
    assert auc == store.test_evaluation["models"]["xgboost"]["probability_metrics"]["roc_auc"]
    assert 0.5 < auc < 1
    with pytest.raises(KeyError):
        store.metric("xgboost", "accuracy_made_up")
    with pytest.raises(KeyError):
        store.model_record("not_a_model")


@needs_artifacts
def test_stored_shap_values_add_up_to_stored_test_pds():
    store = ResultsStore()
    values, _ = store.shap_values()
    preds = store.predictions("test", "xgboost")
    assert values["ID"].equals(preds["ID"])
    logit = store.shap_summary()["base_value_log_odds"] + values.drop(columns="ID").sum(axis=1)
    p = np.clip(preds["pd"].to_numpy(), 1e-12, 1 - 1e-12)
    assert (
        np.max(np.abs(logit - np.log(p / (1 - p)))) < 1e-3
    )  # CSV rounding to 6 significant digits


PAGES = [
    "app.py",
    *sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "pages").glob("*.py")),
]


@needs_artifacts
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_pages_render_without_errors(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page should render a title"


@needs_artifacts
def test_decision_simulator_updates_with_custom_threshold():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(PROJECT_ROOT / "pages/3_Decision_Simulator.py"), default_timeout=120
    ).run()
    at.radio[1].set_value("Custom").run()
    at.slider[1].set_value(0.5).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Threshold"] == "0.500"


@needs_artifacts
def test_whatif_form_predicts_and_explains():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(PROJECT_ROOT / "pages/4_Explainability.py"), default_timeout=180
    ).run()
    submit = [b for b in at.button if b.label == "Predict PD"]
    assert submit, "what-if form should be present"
    submit[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label: m.value for m in at.metric}
    assert labels["Predicted PD (experimental)"].endswith("%")


def test_local_explanation_rows_add_up_to_total_shap():
    import pandas as pd

    from credit_risk.explainability import ShapResult, local_explanation

    values = pd.DataFrame([[0.5, -0.2, 0.1, 0.05]], columns=list("abcd"))
    result = ShapResult(values, values * 10, base_value=-1.0)
    table = local_explanation(result, 0, top=2)
    assert list(table["feature"]) == ["a", "b", "All other features"]
    assert table["shap_log_odds"].sum() == pytest.approx(values.iloc[0].sum())
    assert pd.isna(table["value"].iloc[-1])
