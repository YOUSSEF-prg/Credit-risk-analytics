"""Decision simulator: explore PD thresholds and their consequences."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from credit_risk.decision_analysis import expected_cost, select_threshold, threshold_table
from credit_risk.evaluation import threshold_metrics
from dashboard.charts import confusion_matrix, threshold_curves
from dashboard.common import MODEL_LABELS, disclaimer, page_setup, require_results

page_setup("Decision Simulator", "🎚️")
disclaimer()
store = require_results()
frozen = store.frozen_decision

st.markdown(
    """
A PD threshold turns probabilities into a **screening rule**: clients with PD ≥ threshold are flagged for review.
Lowering it catches more defaulters (recall ↑) but flags more clients and adds false alarms (precision ↓).
**0.5 has no special meaning**: with a 22% default rate, it flags only about 12% of clients.
"""
)

c1, c2 = st.columns([1, 1])
model = c1.selectbox(
    "Model", ["xgboost", "random_forest", "logreg"], format_func=lambda m: MODEL_LABELS.get(m, m)
)
split = c2.radio("Evaluate on", ["validation", "test"], horizontal=True,
                 help="The reported thresholds were chosen on validation. Exploring on the test set is for illustration and does not change the frozen results.")  # fmt: skip
preds = store.predictions(split, model)
y, p = preds["default_next_month"].to_numpy(), preds["pd"].to_numpy()
table = threshold_table(y, p)

val_preds = store.predictions("validation", model)
capacity = st.slider(
    "Review capacity: maximum share of clients that can be flagged",
    0.05,
    1.0,
    0.30,
    0.05,
    format="%.2f",
)
capacity_rule = select_threshold(
    val_preds["default_next_month"], val_preds["pd"], beta=2.0, max_flagged_rate=capacity
)
recall_rule = select_threshold(val_preds["default_next_month"], val_preds["pd"], beta=2.0)
presets = {
    f"F2 within {capacity:.0%} capacity (validation-optimal)": capacity_rule["threshold"],
    "F2, unconstrained (validation-optimal)": recall_rule["threshold"],
    "0.5 (reference)": 0.5,
    "Custom": None,
}
choice = st.radio("Threshold rule", list(presets), horizontal=True)
if presets[choice] is None:
    threshold = st.slider(
        "Custom threshold",
        0.01,
        0.90,
        float(frozen["thresholds"]["f2_capacity_30pct"]),
        0.005,
        format="%.3f",
    )
else:
    threshold = presets[choice]

m = threshold_metrics(y, p, threshold)
k = st.columns(5)
k[0].metric("Threshold", f"{threshold:.3f}")
k[1].metric("Recall", f"{m['recall']:.1%}", help="Share of actual defaulters who are flagged.")
k[2].metric(
    "Precision",
    f"{m['precision']:.1%}",
    help=f"Share of flagged clients who default. Random flagging gives {y.mean():.1%}.",
)
k[3].metric(
    "Clients flagged",
    f"{m['flagged_rate']:.1%}",
    f"{int(m['flagged_rate'] * len(y)):,} of {len(y):,}",
    delta_color="off",
)
k[4].metric("F2 / F1", f"{m['f2']:.3f} / {m['f1']:.3f}")
if m["flagged_rate"] > capacity + 1e-9:
    st.warning(
        f"This threshold flags {m['flagged_rate']:.1%} of clients, above the {capacity:.0%} review capacity."
    )

left, right = st.columns([3, 2])
left.plotly_chart(threshold_curves(table, threshold, {"F2": recall_rule["threshold"], "capacity": capacity_rule["threshold"]},
                                   f"{MODEL_LABELS[model]}: metrics vs threshold ({split})"), use_container_width=True)  # fmt: skip
right.plotly_chart(
    confusion_matrix(m["confusion_matrix"], f"Confusion matrix at {threshold:.3f} ({split})"),
    use_container_width=True,
)
cm = m["confusion_matrix"]
right.markdown(
    f"- **{cm['tp']:,}** defaulters caught, **{cm['fn']:,}** missed\n"
    f"- **{cm['fp']:,}** false alarms among {cm['fp'] + cm['tn']:,} non-defaulters"
)

st.subheader("Illustrative cost analysis")
st.info(
    "**Hypothetical costs, chosen by you.** The dataset contains no exposure, LGD, margin or recovery data, "
    "so these numbers are *illustrative units*, not bank losses. This is not an expected-credit-loss, IFRS 9 or capital calculation."
)
cc = st.columns(2)
cost_fn = cc[0].number_input("Cost of a missed defaulter (FN)", 0.0, 1000.0, 5.0, 0.5)
cost_fp = cc[1].number_input("Cost of a false alarm (FP)", 0.0, 1000.0, 1.0, 0.5)
costs = expected_cost(table, cost_fn, cost_fp)
best = costs.loc[costs["illustrative_cost_per_client"].idxmin()]
fig = go.Figure(go.Scatter(x=costs["threshold"], y=costs["illustrative_cost_per_client"], mode="lines", line=dict(color="#2a78d6", width=2.5),
                           hovertemplate="Threshold %{x:.3f}<br>%{y:.3f} units per client<extra></extra>", name="Cost per client"))  # fmt: skip
fig.add_vline(
    x=best["threshold"],
    line=dict(dash="dot", color="#eb6834"),
    annotation_text=f"minimum at {best['threshold']:.3f}",
)
fig.add_vline(
    x=threshold,
    line=dict(color="#1f1f1f", width=1),
    annotation_text="selected",
    annotation_position="bottom right",
)
fig.update_layout(
    title=dict(text="Illustrative cost per client vs threshold", x=0),
    height=360,
    margin=dict(l=10, r=10, t=50, b=10),
)
fig.update_xaxes(title="Threshold", range=[0, 0.9])
fig.update_yaxes(title="Cost units per client", rangemode="tozero")
st.plotly_chart(fig, use_container_width=True)
if cost_fn + cost_fp > 0:
    theory = cost_fp / (cost_fp + cost_fn)
    st.caption(
        f"For calibrated PDs, the cost-minimising threshold is about FP / (FP + FN) = {theory:.3f}. "
        f"The empirical minimum here is {best['threshold']:.3f} "
        f"(recall {best['recall']:.1%}, precision {best['precision']:.1%}, {best['flagged_rate']:.1%} flagged)."
    )
