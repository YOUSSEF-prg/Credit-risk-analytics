"""Home page: project summary and headline test results of the frozen model."""

from __future__ import annotations

import streamlit as st

from dashboard.common import MODEL_LABELS, disclaimer, page_setup, require_results

page_setup("Credit Risk Analytics Platform", "🏦")
disclaimer()
store = require_results()
test = store.test_evaluation
frozen = store.frozen_decision
chosen = frozen["selected_model"]
pm = test["models"][chosen]["probability_metrics"]
ci = test["models"][chosen]["confidence_intervals"]
cap = test["models"][chosen]["threshold_metrics"]["f2_capacity_30pct"]

st.markdown(
    "Probability-of-default (PD) modelling on the UCI *Default of Credit Card Clients* dataset "
    "(30,000 clients, 22.1% observed default rate). The project compares calibrated models, explains "
    "them with SHAP, analyses decision thresholds and documents its methodology and limitations."
)

st.subheader(f"Frozen model on the held-out test set: {MODEL_LABELS.get(chosen, chosen)}")
cols = st.columns(5)
cols[0].metric(
    "Test clients",
    f"{test['n_test']:,}",
    help="20% of the data, never used for training, tuning or selection.",
)
cols[1].metric(
    "ROC-AUC",
    f"{pm['roc_auc']:.3f}",
    help=f"95% bootstrap CI [{ci['roc_auc']['low']:.3f}, {ci['roc_auc']['high']:.3f}]. Ranking quality.",
)
cols[2].metric(
    "PR-AUC",
    f"{pm['pr_auc']:.3f}",
    help=f"95% CI [{ci['pr_auc']['low']:.3f}, {ci['pr_auc']['high']:.3f}]. No-skill value = default rate ({test['test_default_rate']:.3f}).",
)
cols[3].metric(
    "Brier score",
    f"{pm['brier']:.4f}",
    help="Mean squared error of PDs (lower is better). Base-rate reference: 0.172.",
)
cols[4].metric(
    "Calibration error (ECE)",
    f"{pm['expected_calibration_error']:.3f}",
    help="Mean |observed − predicted| over 10 PD bins.",
)
st.markdown(
    f"With the threshold frozen on validation for **at most 30% of clients flagged** "
    f"({frozen['thresholds']['f2_capacity_30pct']:.3f}), the model flags {cap['flagged_rate']:.1%} of test clients "
    f"and catches **{cap['recall']:.1%} of defaulters** at **{cap['precision']:.1%} precision**. "
    "That is a screening rule under an illustrative capacity assumption, not a credit decision."
)  # fmt: skip

st.subheader("Pages")
st.markdown(
    """
| Page | What it shows |
|---|---|
| **Portfolio Overview** | Test-set clients: PD distribution, risk bands, observed vs predicted default rates by segment, with filters |
| **Model Performance** | Frozen-model test results with references, plus all 34 validation experiments (class-imbalance strategies, calibration) |
| **Decision Simulator** | Move the PD threshold and see recall, precision, flagged share, confusion matrix and illustrative costs |
| **Explainability** | SHAP drivers of the model, individual explanations, false positives/negatives, and a what-if client explorer |
| **Methodology** | Data provenance, validation protocol, metrics, assumptions and limitations |
"""
)

st.subheader("Protocol in one paragraph")
st.markdown(
    f"""
- **Split.** Stratified 60/20/20 train/validation/test, seed 42.
- **Tuning.** Every model is tuned by 5-fold cross-validation on the training set.
- **Variants.** 34 variants were compared on validation: Logistic Regression, Random Forest, XGBoost, 13 class-imbalance strategies and sigmoid/isotonic calibration.
- **Selection.** A pre-declared rule selected `{chosen}`.
- **Freezing.** The model and both thresholds were frozen ({frozen['frozen_utc']}) **before** the test set was read once (test accesses recorded: {test['test_access_count']}).
- **Source of figures.** Every figure in this app is read from the stored experiment outputs in `artifacts/`.
"""
)  # fmt: skip
