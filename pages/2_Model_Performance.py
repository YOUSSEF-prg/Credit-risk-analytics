"""Model performance: frozen-model test results and all validation experiments."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import confusion_matrix, pr_curves, reliability, roc_curves
from dashboard.common import (
    MODEL_COLORS,
    MODEL_LABELS,
    disclaimer,
    page_setup,
    require_results,
)

page_setup("Model Performance", "🎯")
disclaimer()
store = require_results()
test = store.test_evaluation
frozen = store.frozen_decision

tab_test, tab_val, tab_strategies = st.tabs(
    [
        "Final test evaluation",
        "Validation: compare any models",
        "Class-imbalance strategies & calibration",
    ]
)

# ======================================================================= test
with tab_test:
    st.markdown(
        f"The model **`{frozen['selected_model']}`** and its thresholds were frozen on validation and evaluated **once** "
        f"on {test['n_test']:,} held-out clients. The other models are shown as references only: nothing was re-selected after this evaluation."
    )
    models = list(test["models"])
    chosen_models = st.multiselect(
        "Models",
        models,
        default=models,
        format_func=lambda m: MODEL_LABELS.get(m, m),
        key="test_models",
    )
    rows = []
    for m in chosen_models:
        e = test["models"][m]
        pm, vm, ci = (
            e["probability_metrics"],
            e["validation_probability_metrics"],
            e["confidence_intervals"],
        )
        rows.append({
            "Model": MODEL_LABELS.get(m, m),
            "ROC-AUC (test)": f"{pm['roc_auc']:.3f} [{ci['roc_auc']['low']:.3f}, {ci['roc_auc']['high']:.3f}]",
            "PR-AUC (test)": f"{pm['pr_auc']:.3f} [{ci['pr_auc']['low']:.3f}, {ci['pr_auc']['high']:.3f}]",
            "Brier (test)": f"{pm['brier']:.4f}", "Log loss (test)": f"{pm['log_loss']:.4f}",
            "ECE (test)": f"{pm['expected_calibration_error']:.3f}",
            "ROC-AUC (val)": f"{vm['roc_auc']:.3f}", "PR-AUC (val)": f"{vm['pr_auc']:.3f}", "Brier (val)": f"{vm['brier']:.4f}",
        })  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Brackets: 95% percentile bootstrap intervals (1,000 resamples of the test set, model fixed)."
    )

    if chosen_models:
        c1, c2, c3 = st.columns(3)
        curves = {m: test["models"][m]["curves"] for m in chosen_models}
        c1.plotly_chart(
            roc_curves(
                curves,
                {m: test["models"][m]["probability_metrics"]["roc_auc"] for m in chosen_models},
                "ROC (test)",
            ),
            use_container_width=True,
        )
        c2.plotly_chart(pr_curves(curves, {m: test["models"][m]["probability_metrics"]["pr_auc"] for m in chosen_models},
                                  test["test_default_rate"], "Precision–recall (test)"), use_container_width=True)  # fmt: skip
        cal_models = [m for m in chosen_models if m != "dummy_prior"]
        c3.plotly_chart(reliability({m: pd.DataFrame(test["models"][m]["calibration_table"]) for m in cal_models},
                                    {m: test["models"][m]["probability_metrics"]["expected_calibration_error"] for m in cal_models},
                                    "Calibration (test)"), use_container_width=True)  # fmt: skip

    st.subheader("Paired differences on the test set")
    comp = pd.DataFrame(test["paired_comparisons"])
    comp["95% CI"] = comp.apply(lambda r: f"[{r['low']:+.4f}, {r['high']:+.4f}]", axis=1)
    comp["significant at 5%"] = (comp["low"] > 0) | (comp["high"] < 0)
    st.dataframe(
        comp[["model_a", "model_b", "metric", "difference", "95% CI", "significant at 5%"]].round(
            4
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Difference = A − B on the same bootstrap resamples. For the Brier score, negative means A is better."
    )

    st.subheader("Confusion matrices at the frozen thresholds (test)")
    model_cm = st.selectbox(
        "Model", models, format_func=lambda m: MODEL_LABELS.get(m, m), key="cm_model"
    )
    rules = test["models"][model_cm]["threshold_metrics"]
    cols = st.columns(len(rules))
    names = {
        "f2_unconstrained": "F2 (recall-oriented)",
        "f2_capacity_30pct": "F2, ≤ 30% flagged",
        "reference_0_5": "0.5 (reference)",
    }
    for col, (rule, tm) in zip(cols, rules.items(), strict=True):
        col.plotly_chart(
            confusion_matrix(
                tm["confusion_matrix"], f"{names.get(rule, rule)}: threshold {tm['threshold']:.3f}"
            ),
            use_container_width=True,
        )
        col.caption(
            f"Recall {tm['recall']:.1%} · Precision {tm['precision']:.1%} · Flagged {tm['flagged_rate']:.1%}"
        )

# ======================================================================= validation
with tab_val:
    st.markdown(
        "All models were tuned by cross-validation on the training set and evaluated on the **validation** set (6,000 clients)."
    )
    summary = store.summary.copy()
    metric_options = {
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier", "log_loss": "Log loss",
        "expected_calibration_error": "ECE", "mean_predicted_pd": "Mean PD", "cv_average_precision": "CV PR-AUC (train)",
        "recall": "Recall @ F2 threshold", "precision": "Precision @ F2 threshold", "flagged_rate": "Flagged @ F2 threshold",
        "capacity_recall": "Recall @ 30% capacity", "capacity_precision": "Precision @ 30% capacity",
    }  # fmt: skip
    default_models = [
        "logreg",
        "random_forest",
        "xgboost",
        "xgboost_smotenc",
        "xgboost_weighted",
        "xgboost__isotonic",
    ]
    picked = st.multiselect(
        "Models", list(summary.index), default=[m for m in default_models if m in summary.index]
    )
    metrics = st.multiselect("Metrics", list(metric_options), default=["roc_auc", "pr_auc", "brier", "expected_calibration_error", "cv_average_precision"],
                             format_func=metric_options.get)  # fmt: skip
    if picked and metrics:
        st.dataframe(
            summary.loc[picked, metrics].rename(columns=metric_options).style.format("{:.4f}"),
            use_container_width=True,
        )
        records = {m: store.model_record(m) for m in picked}
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            roc_curves(
                {m: r["curves"] for m, r in records.items()},
                {m: r["probability_metrics"]["roc_auc"] for m, r in records.items()},
                "ROC (validation)",
            ),
            use_container_width=True,
        )
        c2.plotly_chart(reliability({m: pd.DataFrame(r["calibration_table"]) for m, r in records.items()},
                                    {m: r["probability_metrics"]["expected_calibration_error"] for m, r in records.items()}, "Calibration (validation)"),
                        use_container_width=True)  # fmt: skip
    st.caption(
        "34 variants were compared on the same validation set, so the best validation score is optimistic. The final test evaluation is the unbiased estimate."
    )

# ======================================================================= strategies
with tab_strategies:
    s = store.summary.copy()
    s = s[s.index != "dummy_prior"]
    s["family"] = s.index.map(lambda n: "XGBoost" if n.startswith("xgboost") else "Logistic" if n.startswith("logreg")
                              else "Random forest" if "random_forest" in n else "Imbalance ensemble")  # fmt: skip
    s["calibrated"] = s["calibration"] != "none"
    fig = go.Figure()
    family_colors = {"XGBoost": MODEL_COLORS["xgboost"], "Logistic": MODEL_COLORS["logreg"],
                     "Random forest": MODEL_COLORS["random_forest"], "Imbalance ensemble": "#eda100"}  # fmt: skip
    for fam, sub in s.groupby("family"):
        for calibrated, symbol in [(False, "circle"), (True, "diamond-open")]:
            part = sub[sub.calibrated == calibrated]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part["roc_auc"], y=part["brier"], mode="markers", marker=dict(size=11, color=family_colors[fam], symbol=symbol, line=dict(width=1.5, color=family_colors[fam])),
                name=f"{fam}{' (recalibrated)' if calibrated else ''}", text=part.index,
                customdata=part[["pr_auc", "expected_calibration_error", "mean_predicted_pd"]],
                hovertemplate="<b>%{text}</b><br>ROC-AUC %{x:.4f}<br>Brier %{y:.4f}<br>PR-AUC %{customdata[0]:.4f}<br>ECE %{customdata[1]:.3f}<br>Mean PD %{customdata[2]:.3f}<extra></extra>",
            ))  # fmt: skip
    fig.update_layout(title=dict(text="Ranking quality vs probability accuracy, all validation experiments", x=0), height=520,
                      legend=dict(orientation="h", yanchor="top", y=-0.32, x=0), margin=dict(l=10, r=10, t=50, b=70))  # fmt: skip
    fig.update_xaxes(title="ROC-AUC (higher is better)")
    fig.update_yaxes(title="Brier score (lower is better)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**What the experiments show (validation; details in `notebooks/04` and `05`)**
- **No gain in discrimination.** No tested imbalance method (class weights, random over-/undersampling, SMOTE-NC, Tomek, ENN, hybrids, balanced ensembles) improved discrimination over training on the original distribution.
- **Rebalancing raises every PD.** Rebalanced models predict a mean PD of about 0.34–0.49 against 0.22 observed (top cluster in the chart). The cause is the class-prior shift: the odds are multiplied by about 3.5, and an analytic correction with no fitted parameters removes most of the bias.
- **SMOTE-NC also loses ranking quality**, unlike random oversampling. The synthetic interpolation adds noise.
- **Recalibration (diamonds) restores probability accuracy, but not ranking.** No recalibrated variant beats the plain XGBoost on PR-AUC.
- **These are findings for this dataset and protocol**, not general rules about resampling.
"""
    )
    st.dataframe(
        s[["family", "sampler", "calibration", "roc_auc", "pr_auc", "brier", "expected_calibration_error", "mean_predicted_pd", "cv_average_precision"]]
        .sort_values("pr_auc", ascending=False).style.format({c: "{:.4f}" for c in ["roc_auc", "pr_auc", "brier", "expected_calibration_error", "mean_predicted_pd", "cv_average_precision"]}),
        use_container_width=True,
    )  # fmt: skip
