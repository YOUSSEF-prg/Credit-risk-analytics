"""Portfolio overview: test-set clients and the frozen model's out-of-sample PDs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_risk.evaluation import risk_band_table
from dashboard.charts import observed_vs_predicted_bars
from dashboard.common import (
    CODE_LABELS,
    OBSERVED_COLOR,
    PREDICTED_COLOR,
    disclaimer,
    label_codes,
    page_setup,
    pct,
    predictions_with_features,
    require_results,
)

page_setup("Portfolio Overview", "📈")
disclaimer()
store = require_results()
model = store.selected_model
data = predictions_with_features("test", model)

st.markdown(
    f"**{len(data):,} held-out test clients** with out-of-sample PDs from the frozen `{model}` model. "
    "*Observed* default rates are what happened. *Predicted* PDs are model estimates. Both are shown side by side."
)

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    sex = st.multiselect("Sex", options=[1, 2], default=[1, 2], format_func=CODE_LABELS["SEX"].get)
    edu_options = sorted(data["EDUCATION"].unique())
    edu = st.multiselect("Education", options=edu_options, default=edu_options,
                         format_func=lambda v: CODE_LABELS["EDUCATION"].get(v, f"Undocumented ({v})"))  # fmt: skip
    mar_options = sorted(data["MARRIAGE"].unique())
    mar = st.multiselect("Marital status", options=mar_options, default=mar_options,
                         format_func=lambda v: CODE_LABELS["MARRIAGE"].get(v, f"Undocumented ({v})"))  # fmt: skip
    age = st.slider(
        "Age", int(data.AGE.min()), int(data.AGE.max()), (int(data.AGE.min()), int(data.AGE.max()))
    )
    limit = st.slider("Credit limit (NT$)", 0, 1_000_000, (0, 1_000_000), step=10_000)
    delay = st.radio(
        "Latest repayment status (PAY_1)", ["All", "No delay (≤ 0)", "Delayed (≥ 1 month)"]
    )

mask = (
    data.SEX.isin(sex) & data.EDUCATION.isin(edu) & data.MARRIAGE.isin(mar)
    & data.AGE.between(*age) & data.LIMIT_BAL.between(*limit)
)  # fmt: skip
if delay.startswith("No"):
    mask &= data.PAY_1 <= 0
elif delay.startswith("Delayed"):
    mask &= data.PAY_1 >= 1
view = data[mask]
if view.empty:
    st.warning("No clients match these filters.")
    st.stop()

# ---------------------------------------------------------------- KPIs
c = st.columns(4)
c[0].metric(
    "Clients in selection",
    f"{len(view):,}",
    f"{len(view) / len(data):.0%} of test set",
    delta_color="off",
)
c[1].metric(
    "Observed default rate",
    pct(view.default_next_month.mean()),
    help=f"Share of these clients who actually defaulted next month. Whole test set: {data.default_next_month.mean():.1%}.",
)
c[2].metric(
    "Mean predicted PD",
    pct(view.pd.mean()),
    help="Average model estimate for these clients. Close to the observed rate = calibrated in the large.",
)
c[3].metric("High-risk share (PD ≥ 40%)", pct((view.pd >= 0.4).mean()))
if len(view) < 200:
    st.info(f"Only {len(view)} clients selected: observed rates have wide sampling uncertainty.")

# ---------------------------------------------------------------- PD distribution
left, right = st.columns(2)
with left:
    fig = go.Figure()
    bins = dict(start=0, end=1, size=0.025)
    fig.add_trace(go.Histogram(x=view.loc[view.default_next_month == 0, "pd"], xbins=bins, name="Did not default",
                               marker_color=PREDICTED_COLOR, opacity=0.75, hovertemplate="PD %{x}<br>%{y} clients<extra></extra>"))  # fmt: skip
    fig.add_trace(go.Histogram(x=view.loc[view.default_next_month == 1, "pd"], xbins=bins, name="Defaulted",
                               marker_color=OBSERVED_COLOR, opacity=0.75, hovertemplate="PD %{x}<br>%{y} clients<extra></extra>"))  # fmt: skip
    fig.update_layout(barmode="overlay", title=dict(text="Distribution of predicted PD by observed outcome", x=0),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0), height=400, margin=dict(l=10, r=10, t=50, b=70))  # fmt: skip
    fig.update_xaxes(title="Predicted PD", tickformat=".0%")
    fig.update_yaxes(title="Clients")
    st.plotly_chart(fig, use_container_width=True)

with right:
    table = risk_band_table(view.default_next_month, view.pd)
    fig = observed_vs_predicted_bars(
        table, "band", "Risk segments: predicted PD vs observed default rate"
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    table.assign(**{c: table[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
                    for c in ["observed_rate", "mean_pd", "share_of_clients", "share_of_defaults"]}),
    hide_index=True, use_container_width=True,
)  # fmt: skip

# ---------------------------------------------------------------- segments
st.subheader("Default rates across customer groups")
group_by = st.selectbox(
    "Group clients by",
    ["Latest repayment status (PAY_1)", "Months with a delay (of 6)", "Credit limit band", "Age band", "Education", "Sex", "Marital status"],
)  # fmt: skip
labelled = label_codes(view)
if group_by.startswith("Latest"):
    key = view.PAY_1.clip(upper=4).map(
        lambda v: {-2: "-2 (undoc.)", -1: "-1 paid duly", 0: "0 (undoc.)", 4: "≥ 4 months"}.get(
            v, f"{v} month(s)"
        )
    )
elif group_by.startswith("Months"):
    key = view.months_delayed.astype(str)
elif group_by.startswith("Credit"):
    key = pd.cut(view.LIMIT_BAL, [0, 50_000, 100_000, 200_000, 300_000, 500_000, np.inf],
                 labels=["≤ 50k", "50–100k", "100–200k", "200–300k", "300–500k", "> 500k"]).astype(str)  # fmt: skip
elif group_by.startswith("Age"):
    key = pd.cut(
        view.AGE,
        [20, 30, 40, 50, 60, 80],
        right=False,
        labels=["20–29", "30–39", "40–49", "50–59", "60+"],
    ).astype(str)
else:
    key = labelled[{"Education": "EDUCATION", "Sex": "SEX", "Marital status": "MARRIAGE"}[group_by]]
seg = view.assign(segment=key.to_numpy()).groupby("segment", sort=True).agg(
    clients=("pd", "size"), observed_rate=("default_next_month", "mean"), mean_pd=("pd", "mean")
).reset_index()  # fmt: skip
st.plotly_chart(
    observed_vs_predicted_bars(seg, "segment", f"Observed vs predicted by {group_by.lower()}"),
    use_container_width=True,
)
small = seg[seg.clients < 50]
if not small.empty:
    st.caption(
        f"Groups with fewer than 50 clients ({', '.join(map(str, small.segment))}) have very uncertain observed rates."
    )
st.caption(
    "Differences between groups are **associations in historical data**, not causal effects. "
    "Protected attributes (sex, marital status, age) are model inputs in this project at the owner's request. "
    "Any real use would require legal and fair-lending review."
)
