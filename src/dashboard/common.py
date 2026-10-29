"""Shared helpers for the Streamlit pages: cached data access, colours, disclaimers.

Every number displayed by the dashboard is read from stored experiment outputs through
:class:`credit_risk.results.ResultsStore` or recomputed deterministically from stored
predictions (e.g. confusion counts at a user-chosen threshold). Nothing is invented for
display purposes.
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from credit_risk.config import load_config, resolve_path
from credit_risk.data import (
    DataSplits,
    apply_splits,
    load_raw,
    split_features_target,
    standardize_columns,
)
from credit_risk.features import CreditFeatureEngineer
from credit_risk.results import ResultsNotFoundError, ResultsStore

# Fixed colour per model (identity never depends on rank or filter state)
MODEL_COLORS = {
    "xgboost": "#eb6834",
    "random_forest": "#1baf7a",
    "logreg": "#2a78d6",
    "dummy_prior": "#9a9a9a",
}
OBSERVED_COLOR = "#eb6834"
PREDICTED_COLOR = "#2a78d6"
NEUTRAL = "#9a9a9a"
MODEL_LABELS = {
    "xgboost": "XGBoost (frozen)",
    "random_forest": "Random Forest",
    "logreg": "Logistic Regression",
    "dummy_prior": "Dummy (base rate)",
}
CODE_LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate school", 2: "University", 3: "High school", 4: "Other/unknown"},
    "MARRIAGE": {1: "Married", 2: "Single", 3: "Other/unknown"},
}

DISCLAIMER = (
    "Experimental outputs of models trained on a **historical public dataset** "
    "(UCI *Default of Credit Card Clients*, Taiwan, 2005). They are **not credit decisions**, "
    "they are not an assessment of any real person, and they are not a production bank model."
)


def page_setup(title: str, icon: str = "📊") -> None:
    st.set_page_config(page_title=f"{title} · Credit Risk Analytics", page_icon=icon, layout="wide")
    st.title(title)
    with st.sidebar:
        st.caption("Credit Risk Analytics · UCI dataset 350 · experimental research dashboard")


def disclaimer() -> None:
    st.caption(f"⚠️ {DISCLAIMER}")


@st.cache_resource
def store() -> ResultsStore:
    return ResultsStore()


def require_results() -> ResultsStore:
    """Return the results store or stop the page with a clear message."""
    s = store()
    try:
        _ = s.frozen_decision, s.test_evaluation, s.summary
    except ResultsNotFoundError as exc:
        st.error(
            "Experiment outputs are missing. Run `python scripts/run_experiments.py` and "
            f"`python scripts/evaluate_test.py` first.\n\nDetails: {exc}"
        )
        st.stop()
    return s


@st.cache_data
def split_frame(split: str) -> pd.DataFrame:
    """Raw + engineered features and target for one partition (validation or test)."""
    cfg = load_config("data")
    ds = cfg["dataset"]
    df = standardize_columns(load_raw(resolve_path(ds["raw_path"]), ds["expected_sha256"]))
    _, val, test = apply_splits(df, DataSplits.from_json(resolve_path(cfg["split"]["output_path"])))
    part = {"validation": val, "test": test}[split]
    X, y = split_features_target(part)
    engineered = CreditFeatureEngineer(merge_undocumented_codes=False).fit_transform(X)
    engineered.insert(0, "ID", part["ID"].to_numpy())
    engineered["default_next_month"] = y.to_numpy()
    return engineered


@st.cache_data
def predictions_with_features(split: str, model: str) -> pd.DataFrame:
    preds = store().predictions(split, model)[["ID", "pd"]]
    return split_frame(split).merge(preds, on="ID", how="inner", validate="one_to_one")


@st.cache_resource
def frozen_model():
    s = store()
    path = resolve_path("artifacts/models") / f"{s.selected_model}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def label_codes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, mapping in CODE_LABELS.items():
        if column in out:
            out[column] = out[column].map(lambda v, m=mapping: m.get(v, f"Undocumented ({v})"))
    return out


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"
