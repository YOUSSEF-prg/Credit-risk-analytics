"""Plotly chart builders shared by the dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.common import MODEL_COLORS, MODEL_LABELS, NEUTRAL, OBSERVED_COLOR, PREDICTED_COLOR

BASE_LAYOUT = dict(
    margin=dict(l=10, r=10, t=50, b=70),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    hoverlabel=dict(namelength=-1),
)


def _layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), **BASE_LAYOUT, **kwargs)
    return fig


def roc_curves(curves: dict[str, dict], aucs: dict[str, float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dash", width=1),
                             name="No skill", hoverinfo="skip"))  # fmt: skip
    for name, c in curves.items():
        fig.add_trace(
            go.Scatter(
                x=c["roc_fpr"], y=c["roc_tpr"], mode="lines", line=dict(width=2.5, color=MODEL_COLORS.get(name)),
                name=f"{MODEL_LABELS.get(name, name)} (AUC {aucs[name]:.3f})",
                hovertemplate="FPR %{x:.2f}<br>Recall %{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    return _layout(fig, title, height=420)


def pr_curves(
    curves: dict[str, dict], aps: dict[str, float], base_rate: float, title: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=base_rate, line=dict(color=NEUTRAL, dash="dash", width=1),
                  annotation_text=f"No skill = default rate ({base_rate:.1%})", annotation_position="bottom left")  # fmt: skip
    for name, c in curves.items():
        if name == "dummy_prior":
            continue  # constant scores: its PR curve is the no-skill line above
        fig.add_trace(
            go.Scatter(
                x=c["pr_recall"], y=c["pr_precision"], mode="lines", line=dict(width=2.5, color=MODEL_COLORS.get(name)),
                name=f"{MODEL_LABELS.get(name, name)} (AP {aps[name]:.3f})",
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _layout(fig, title, height=420)


def reliability(tables: dict[str, pd.DataFrame], eces: dict[str, float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dash", width=1),
                             name="Perfect calibration", hoverinfo="skip"))  # fmt: skip
    for name, t in tables.items():
        fig.add_trace(
            go.Scatter(
                x=t["mean_predicted"], y=t["observed_rate"], mode="lines+markers",
                line=dict(width=2.5, color=MODEL_COLORS.get(name)), marker=dict(size=8),
                error_y=dict(type="data", symmetric=False, array=t["observed_high"] - t["observed_rate"],
                             arrayminus=t["observed_rate"] - t["observed_low"], thickness=1),
                name=f"{MODEL_LABELS.get(name, name)} (ECE {eces[name]:.3f})",
                customdata=np.stack([t["count"]], axis=-1),
                hovertemplate="Mean PD %{x:.1%}<br>Observed %{y:.1%}<br>n = %{customdata[0]}<extra>%{fullData.name}</extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title="Mean predicted PD (10 equal-size bins)", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Observed default rate (95% Wilson CI)", range=[0, 1], tickformat=".0%")
    return _layout(fig, title, height=420)


def confusion_matrix(cm: dict[str, int], title: str) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    total = sum(cm.values())
    text = [[f"{v:,}<br>({v / total:.1%})" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=["Predicted: no default", "Predicted: default"], y=["Observed: no default", "Observed: default"],
            text=text, texttemplate="%{text}", colorscale=[[0, "#eaf2fc"], [1, "#2a78d6"]], showscale=False,
            hovertemplate="%{y}<br>%{x}<br>%{z:,} clients<extra></extra>",
        )
    )  # fmt: skip
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=300)


def observed_vs_predicted_bars(frame: pd.DataFrame, category: str, title: str) -> go.Figure:
    """Grouped bars: observed default rate vs mean predicted PD per category, with counts."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frame[category], y=frame["mean_pd"], name="Mean predicted PD", marker_color=PREDICTED_COLOR,
                         customdata=frame[["clients"]], hovertemplate="%{x}<br>Mean PD %{y:.1%}<br>n = %{customdata[0]:,}<extra></extra>"))  # fmt: skip
    fig.add_trace(go.Bar(x=frame[category], y=frame["observed_rate"], name="Observed default rate", marker_color=OBSERVED_COLOR,
                         customdata=frame[["clients"]], hovertemplate="%{x}<br>Observed %{y:.1%}<br>n = %{customdata[0]:,}<extra></extra>"))  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title="Rate", tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(title=None, type="category")
    return _layout(fig, title, height=380)


def threshold_curves(
    table: pd.DataFrame, threshold: float, markers: dict[str, float], title: str
) -> go.Figure:
    table = table.copy()
    # Precision is undefined (nobody flagged) or meaningless (a handful flagged) at the
    # highest thresholds: leave a gap instead of plotting a jump between 0 and 1.
    table.loc[table["tp"] + table["fp"] < 20, "precision"] = np.nan
    fig = go.Figure()
    for col, color, label in [
        ("recall", "#2a78d6", "Recall"),
        ("precision", "#eb6834", "Precision"),
        ("f2", "#1baf7a", "F2"),
        ("flagged_rate", NEUTRAL, "Share of clients flagged"),
    ]:
        fig.add_trace(go.Scatter(x=table["threshold"], y=table[col], mode="lines", name=label, line=dict(color=color, width=2.5),
                                 hovertemplate="Threshold %{x:.3f}<br>" + label + " %{y:.1%}<extra></extra>"))  # fmt: skip
    fig.add_vline(x=threshold, line=dict(color="#1f1f1f", width=1.5), annotation_text=f"selected {threshold:.3f}",
                  annotation_position="top right")  # fmt: skip
    for name, t in markers.items():
        fig.add_vline(x=t, line=dict(color=NEUTRAL, width=1, dash="dot"), annotation_text=name,
                      annotation_position="top left", annotation_font_size=10)  # fmt: skip
    fig.update_xaxes(title="Threshold on predicted PD", range=[0, 0.9])
    fig.update_yaxes(title="Value", tickformat=".0%", range=[0, 1.05])
    return _layout(fig, title, height=420, hovermode="x unified")


def horizontal_bars(
    labels, values, title: str, x_title: str, color: str = PREDICTED_COLOR, fmt: str = ".3f"
) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=color,
                           hovertemplate="%{y}<br>%{x:" + fmt + "}<extra></extra>"))  # fmt: skip
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=x_title)
    return _layout(fig, title, height=max(300, 28 * len(labels) + 80))


def waterfall(contributions: pd.DataFrame, base_value: float, title: str) -> go.Figure:
    """SHAP waterfall in log-odds: base value -> top features -> 'all other features' -> final."""
    labels = [
        f if pd.isna(v) else f"{f} = {v:,.4g}"
        for f, v in zip(contributions["feature"], contributions["value"], strict=True)
    ]
    fig = go.Figure(
        go.Waterfall(
            orientation="h", measure=["absolute", *["relative"] * len(contributions), "total"],
            y=["Base value (average client)", *labels, "Model output"],
            x=[base_value, *contributions["shap_log_odds"], 0],
            increasing=dict(marker=dict(color=OBSERVED_COLOR)), decreasing=dict(marker=dict(color=PREDICTED_COLOR)),
            totals=dict(marker=dict(color=NEUTRAL)), connector=dict(line=dict(color="#cccccc", width=1)),
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        )
    )  # fmt: skip
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title="Contribution to log-odds of default (orange increases, blue decreases the PD)"
    )
    return _layout(fig, title, height=max(380, 30 * len(contributions) + 140))
