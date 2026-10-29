"""Explainability: SHAP drivers, individual explanations, error analysis, what-if explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_risk.data import FEATURES
from credit_risk.explainability import ShapResult, explain, local_explanation
from credit_risk.validation import INPUT_CODES, INPUT_RULES, validate_feature_input
from dashboard.charts import horizontal_bars, waterfall
from dashboard.common import (
    CODE_LABELS,
    OBSERVED_COLOR,
    PREDICTED_COLOR,
    disclaimer,
    frozen_model,
    page_setup,
    predictions_with_features,
    require_results,
)

page_setup("Explainability", "🔍")
disclaimer()
store = require_results()
model_name = store.selected_model
summary = store.shap_summary()
values, features = store.shap_values()
threshold = store.frozen_decision["thresholds"]["f2_capacity_30pct"]

st.markdown(
    f"""
SHAP values for the frozen **`{model_name}`** model on the {summary["n_clients"]:,} test clients, in **log-odds**:
for every client, *base value + Σ contributions = logit(PD)*. The largest gap found was {summary["additivity_max_abs_gap"]:.1e}.

> **How to read them.** SHAP describes how *the model* uses each variable. These are **predictive associations
> learned from 2005 data, not causal effects**. For example, a higher credit limit lowering the PD does not mean that raising
> someone's limit would make them safer. Correlated variables (monthly bills, repayment codes and their counts) share credit,
> so the group totals below are more stable than individual features.
"""
)

tab_global, tab_local, tab_errors, tab_whatif = st.tabs(
    ["Global drivers", "Individual explanation", "False positives / negatives", "What-if client"]
)

# ======================================================================= global
with tab_global:
    groups = pd.DataFrame(summary["grouped_importance"])
    imp = pd.DataFrame(summary["global_importance"])
    c1, c2 = st.columns(2)
    c1.plotly_chart(horizontal_bars(groups["group"], groups["mean_abs_shap"], "Importance by variable group",
                                    "Mean |Σ SHAP| within group (log-odds)"), use_container_width=True)  # fmt: skip
    top = imp.head(15)
    colors = [
        OBSERVED_COLOR if (c or 0) > 0 else PREDICTED_COLOR for c in top["value_shap_correlation"]
    ]
    c2.plotly_chart(horizontal_bars(top["feature"], top["mean_abs_shap"], "Top 15 features (mean |SHAP|)", "Mean |SHAP| (log-odds)", color=colors),
                    use_container_width=True)  # fmt: skip
    c2.caption(
        "Orange: higher values push the PD up on average. Blue: higher values push it down (sign of the value–SHAP correlation)."
    )

    st.subheader("How each feature's value moves the prediction")
    feature = st.selectbox("Feature", list(imp["feature"].head(20)))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(values), size=min(3000, len(values)), replace=False)
    fig = go.Figure(go.Scatter(
        x=features[feature].iloc[idx], y=values[feature].iloc[idx], mode="markers",
        marker=dict(size=6, color=values[feature].iloc[idx], colorscale=[[0, PREDICTED_COLOR], [0.5, "#d9d9d9"], [1, OBSERVED_COLOR]], cmid=0, opacity=0.6),
        hovertemplate=f"{feature} = %{{x:,.4g}}<br>SHAP %{{y:+.3f}}<extra></extra>",
    ))  # fmt: skip
    fig.add_hline(y=0, line=dict(color="#999999", width=1))
    fig.update_layout(
        title=dict(text=f"SHAP dependence: {feature} (3,000 test clients)", x=0),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(title=f"{feature} (value seen by the model)")
    fig.update_yaxes(title="SHAP value (log-odds)")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================= local
data = predictions_with_features("test", model_name)
data["outcome"] = np.select(
    [(data.pd >= threshold) & (data.default_next_month == 1), (data.pd >= threshold) & (data.default_next_month == 0),
     (data.pd < threshold) & (data.default_next_month == 1)], ["TP", "FP", "FN"], "TN",
)  # fmt: skip
outcome_names = {"TP": "Flagged and defaulted (TP)", "FP": "Flagged, did not default (FP)",
                 "FN": "Not flagged, defaulted (FN)", "TN": "Not flagged, did not default (TN)"}  # fmt: skip

with tab_local:
    st.caption(f"Outcome categories use the frozen 30%-capacity threshold ({threshold:.3f}).")
    c1, c2 = st.columns([1, 2])
    group = c1.selectbox("Pick a client from", list(outcome_names), format_func=outcome_names.get)
    pool = data[data.outcome == group].sort_values("pd", ascending=False)
    client_id = c2.selectbox(f"Client ID ({len(pool):,} clients, sorted by PD)", pool["ID"].tolist(),
                             format_func=lambda i: f"ID {i}: PD {pool.set_index('ID').loc[i, 'pd']:.1%}")  # fmt: skip
    pos = int(np.flatnonzero(values["ID"].to_numpy() == client_id)[0])
    shap_only = values.drop(columns="ID")
    res = ShapResult(shap_only, features.drop(columns="ID"), summary["base_value_log_odds"])
    contrib = local_explanation(res, pos, top=10)
    row = data.set_index("ID").loc[client_id]
    m = st.columns(4)
    m[0].metric("Predicted PD", f"{row.pd:.1%}")
    m[1].metric(
        "Observed outcome", "Defaulted" if row.default_next_month == 1 else "Did not default"
    )
    m[2].metric("Flagged at frozen threshold", "Yes" if row.pd >= threshold else "No")
    m[3].metric("Test-set average PD", f"{data.pd.mean():.1%}")
    fig = waterfall(
        contrib,
        summary["base_value_log_odds"],
        f"Why the model gives client {client_id} a PD of {row.pd:.1%}",
    )
    st.plotly_chart(fig, use_container_width=True)
    compare = [
        "LIMIT_BAL",
        "AGE",
        "PAY_1",
        "months_delayed",
        "max_delay",
        "utilization_mean",
        "payment_to_bill_ratio",
        "PAY_AMT1",
        "BILL_AMT1",
    ]
    st.dataframe(pd.DataFrame({"This client": row[compare], "Test-set median": data[compare].median(),
                               "Median of defaulters": data.loc[data.default_next_month == 1, compare].median()}).T.round(3),
                 use_container_width=True)  # fmt: skip

# ======================================================================= errors
with tab_errors:
    profile = pd.DataFrame(store.test_evaluation["error_profiles_capacity_threshold"]).set_index(
        "outcome"
    )
    st.markdown(
        f"Median characteristics of test clients by outcome at the frozen threshold {threshold:.3f} (stored in `test_evaluation.json`)."
    )
    st.dataframe(profile.rename(index=outcome_names).round(3), use_container_width=True)
    mean_abs = (
        shap_only.abs()
        .groupby(data.set_index("ID").loc[values["ID"], "outcome"].to_numpy())
        .mean()
        .T
    )
    top_err = mean_abs.loc[
        mean_abs[["FP", "FN"]].max(axis=1).sort_values(ascending=False).index[:8]
    ]
    fig = go.Figure()
    for oc, color in [
        ("TP", OBSERVED_COLOR),
        ("FP", "#eda100"),
        ("FN", "#1baf7a"),
        ("TN", PREDICTED_COLOR),
    ]:
        fig.add_trace(
            go.Bar(
                y=top_err.index,
                x=top_err[oc],
                orientation="h",
                name=outcome_names[oc],
                marker_color=color,
            )
        )
    fig.update_layout(barmode="group", title=dict(text="Mean |SHAP| by outcome group", x=0), height=480,
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0), margin=dict(l=10, r=10, t=50, b=70))  # fmt: skip
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Mean |SHAP| (log-odds)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
- **Missed defaulters (FN)** have no recorded delay, low utilisation and higher limits. On every variable in the dataset they look like clients who repaid.
  Their default is not preceded by a warning sign the data captures. That is a limit of the available information, which no threshold or resampling can fix.
- **False alarms (FP)** do show delay history and high utilisation. They resemble defaulters but did not default in the following month.
"""
    )

# ======================================================================= what-if
with tab_whatif:
    st.markdown(
        "Enter a **hypothetical** feature vector to see the frozen model's PD and its explanation. "
        "Inputs must be within the ranges observed in the dataset. "
        "**This is not an assessment of a real person**, and the model is a historical research model, not a production scoring system."
    )
    model = frozen_model()
    if model is None:
        st.error(
            "The frozen model file (artifacts/models/xgboost.joblib) is not available on this machine."
        )
        st.stop()
    template_id = st.selectbox(
        "Start from test client",
        data["ID"].head(200).tolist(),
        help="Pre-fills the form. Edit any value.",
    )
    base = data.set_index("ID").loc[template_id]
    with st.form("whatif"):
        cols = st.columns(4)
        entered = {}
        for i, f in enumerate(FEATURES):
            col = cols[i % 4]
            if f in INPUT_CODES:
                opts = sorted(INPUT_CODES[f])
                default = int(base[f]) if int(base[f]) in opts else opts[-1]
                entered[f] = col.selectbox(
                    f,
                    opts,
                    index=opts.index(default),
                    format_func=lambda v, f=f: f"{v} · {CODE_LABELS[f][v]}",
                )
            else:
                low, high = INPUT_RULES[f]
                entered[f] = col.number_input(
                    f,
                    value=int(np.clip(base[f], low, high)),
                    step=1,
                    help=f"Observed range [{low:,}, {high:,}]",
                )
        submitted = st.form_submit_button("Predict PD")
    if submitted:
        frame, errors = validate_feature_input(entered)
        if errors:
            st.error("Invalid input:\n- " + "\n- ".join(errors))
        else:
            pd_value = float(model.predict_proba(frame)[0, 1])
            result = explain(model, frame)
            contrib = local_explanation(result, 0, top=10)
            pct_rank = (data.pd < pd_value).mean()
            r = st.columns(3)
            r[0].metric("Predicted PD (experimental)", f"{pd_value:.1%}")
            r[1].metric("Flagged at frozen threshold", "Yes" if pd_value >= threshold else "No")
            r[2].metric("Higher than … of test clients", f"{pct_rank:.0%}")
            fig = waterfall(contrib, result.base_value, "Explanation of the hypothetical client")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The model's undocumented-code merge (EDUCATION 0/5/6 → 4, MARRIAGE 0 → 3) is applied inside the pipeline."
            )
