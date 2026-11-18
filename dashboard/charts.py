"""Plotly chart builders for the dashboard.

Colour roles (validated reference palette):
  * Categorical slots 1-4 -> models / segments / stay-vs-churn (fixed order, never cycled)
  * Status palette        -> risk tiers (always paired with a text label)
  * Diverging blue <-> red -> SHAP contributions (lowers vs. raises churn risk)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Dark-mode steps of the validated palette; all >= 3:1 contrast on the navy theme background.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]
TIER_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
TIER_ORDER = ["Low", "Medium", "High", "Critical"]
SEGMENT_ORDER = ["Priority Rescue", "Cost-Efficient Save", "Loyal VIP - Nurture", "Stable - Grow"]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, SERIES))
DECREASE, INCREASE = SERIES[0], "#e66767"
STAY, CHURN = SERIES[0], SERIES[1]
# Colour follows the model, never its rank (matches the training-report figures).
MODEL_COLORS = dict(zip(["logistic_regression", "decision_tree", "random_forest", "xgboost"], SERIES))

# Theme tokens (match .streamlit/config.toml).
TEXT = "#e6edf7"
MUTED = "#9fb0c8"
GRID = "rgba(230,237,247,0.10)"
AXIS_LINE = "rgba(230,237,247,0.25)"
PANEL = "#16243d"

TITLE_MARGIN = 72    # px: fixed top margin that holds only the title
LEGEND_MARGIN = 104  # px: bottom margin that holds tick labels, axis title and the legend
LEGEND_GAP = 64      # px: legend top sits this far below the plot, clear of the x-axis title


def wrap_label(text: str, width: int = 14) -> str:
    """Break long category labels onto several lines so they never need rotating."""
    words, lines, line = str(text).split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "<br>".join(lines)


def _base(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    """Shared layout for every dashboard figure.

    * Title: fixed top margin (72px), pinned to the top of the figure, nothing else up there.
    * Legend: horizontal, centred, top-anchored BELOW the plot area, with a bottom margin
      sized to hold it. Plotly's autoexpand grows that margin if the legend wraps to more rows.
    * Axes: automargin so tick labels and axis titles are never clipped.
    """
    has_legend = sum(1 for t in fig.data if t.showlegend is not False) > 1
    top = TITLE_MARGIN if title else 24
    bottom = LEGEND_MARGIN if has_legend else 12
    # y is in plot-height units: at least -0.2, lower on short plots so the legend
    # always starts LEGEND_GAP px below the axis, under the x-axis title.
    plot_height = max(height - top - bottom, 1)
    legend_y = -max(0.2, LEGEND_GAP / plot_height)
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", xref="container", y=1, yanchor="top",
                   yref="container", pad=dict(t=18, l=4), font=dict(size=15, color=TEXT, weight=600))
        if title else None,
        margin=dict(l=12, r=28, t=top, b=bottom, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TEXT),
        showlegend=has_legend,
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="center", x=0.5,
                    title=None, font=dict(size=12, color=TEXT), bgcolor="rgba(0,0,0,0)",
                    itemwidth=30, tracegroupgap=0),
        hoverlabel=dict(font_size=12, bgcolor=PANEL, bordercolor=AXIS_LINE, font_color=TEXT),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS_LINE, automargin=True,
                     tickfont=dict(color=MUTED), title_font=dict(color=TEXT), title_standoff=10)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS_LINE, automargin=True,
                     tickfont=dict(color=MUTED), title_font=dict(color=TEXT), title_standoff=10)
    # Value labels above/next to bars must not be cut off at the plot edge.
    fig.update_traces(cliponaxis=False, textfont_color=TEXT, selector=dict(type="bar"))
    return fig


def tier_distribution(scored: pd.DataFrame) -> go.Figure:
    counts = scored["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    share = counts / counts.sum()
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color=[TIER_COLORS[t] for t in counts.index], cornerradius=4),
            text=[f"{c:,} ({s:.0%})" for c, s in zip(counts.values, share.values)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,} customers<extra></extra>",
        )
    )
    fig.update_yaxes(title="Customers", range=[0, counts.max() * 1.18])
    return _base(fig, title="Customers by risk tier")


def segment_revenue(summary: pd.DataFrame) -> go.Figure:
    s = summary.set_index("segment").reindex(SEGMENT_ORDER).dropna(subset=["customers"]).reset_index()
    fig = go.Figure(
        go.Bar(
            y=s["segment"],
            x=s["annual_revenue_at_risk"],
            orientation="h",
            marker=dict(color=[SEGMENT_COLORS[x] for x in s["segment"]], cornerradius=4),
            text=[f"${v:,.0f}" for v in s["annual_revenue_at_risk"]],
            textposition="outside",
            customdata=s[["customers", "avg_churn_probability"]],
            hovertemplate="<b>%{y}</b><br>Revenue at risk: $%{x:,.0f}<br>"
            "Customers: %{customdata[0]:,}<br>Avg churn prob: %{customdata[1]:.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Expected annual revenue at risk (USD)", showgrid=True, gridcolor=GRID,
                     range=[0, s["annual_revenue_at_risk"].max() * 1.25])
    return _base(fig, title="Revenue at risk by segment")


def churn_rate_by(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    # Horizontal bars: long category names sit on the y-axis (automargin) and never rotate or collide.
    rates = df.groupby(col)["actual_churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    overall = df["actual_churn"].mean()
    fig = go.Figure(
        go.Bar(
            y=[wrap_label(v, width=16) for v in rates.index.astype(str)],
            x=rates["mean"],
            orientation="h",
            marker=dict(color=CHURN, cornerradius=4),
            text=[f"{v:.0%}" for v in rates["mean"]],
            textposition="outside",
            customdata=list(zip(rates.index.astype(str), rates["size"])),
            hovertemplate="<b>%{customdata[0]}</b><br>Churn rate: %{x:.1%}<br>"
            "Customers: %{customdata[1]:,}<extra></extra>",
        )
    )
    fig.add_vline(x=overall, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text=f"Overall {overall:.0%}", annotation_position="top",
                  annotation_font_color=MUTED)
    # Each bar carries its own % label and the dotted line marks the overall rate, so the x-axis
    # ticks are redundant; dropping them keeps narrow columns uncluttered.
    fig.update_xaxes(range=[0, rates["mean"].max() * 1.3], showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(bargap=0.3)
    return _base(fig, height=300, title=title)


def tenure_curve(df: pd.DataFrame) -> go.Figure:
    bins = pd.cut(df["tenure"], bins=range(0, 78, 6), right=False)
    g = df.groupby(bins, observed=True).agg(
        actual=("actual_churn", "mean"), predicted=("churn_probability", "mean"), n=("tenure", "size")
    )
    # Label each 6-month bin by its first month so ticks stay horizontal; the hover shows the range.
    x = [str(int(i.left)) for i in g.index]
    ranges = [f"{int(i.left)}-{int(i.right) - 1} months" for i in g.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g["actual"], name="Actual churn rate", mode="lines+markers",
                             line=dict(color=SERIES[1], width=2), marker=dict(size=8), customdata=ranges,
                             hovertemplate="%{customdata}: %{y:.1%}<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=x, y=g["predicted"], name="Mean predicted probability",
                             mode="lines+markers", line=dict(color=SERIES[0], width=2, dash="dash"),
                             marker=dict(size=8), hovertemplate="%{y:.1%}<extra>Predicted</extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", title="Churn")
    fig.update_xaxes(title="Tenure (months, start of 6-month bin)", tickangle=0)
    return _base(fig, height=390, title="Churn by tenure")


def probability_histogram(scored: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, value, color in (("Stayed", 0, STAY), ("Churned", 1, CHURN)):
        sub = scored.loc[scored["actual_churn"] == value, "churn_probability"]
        fig.add_trace(go.Histogram(x=sub, name=label, marker=dict(color=color, line=dict(width=1, color="rgba(0,0,0,0)")),
                                   xbins=dict(start=0, end=1, size=0.05), opacity=0.8,
                                   hovertemplate=f"{label}<br>p=%{{x}}<br>%{{y}} customers<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Predicted churn probability", tickformat=".0%")
    fig.update_yaxes(title="Customers")
    return _base(fig, height=390, title="Predicted probability by outcome")


def gauge(probability: float, tier: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", font=dict(size=44, color=TEXT)),
            gauge=dict(
                axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
                          tickcolor=MUTED, tickfont=dict(color=MUTED)),
                bar=dict(color=TIER_COLORS[tier], thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=AXIS_LINE,
                steps=[
                    dict(range=[0, 25], color="rgba(12,163,12,0.18)"),
                    dict(range=[25, 50], color="rgba(250,178,25,0.18)"),
                    dict(range=[50, 75], color="rgba(236,131,90,0.20)"),
                    dict(range=[75, 100], color="rgba(208,59,59,0.22)"),
                ],
            ),
            title=dict(text=f"Churn probability · <b>{tier} risk</b>", font=dict(size=14, color=TEXT)),
        )
    )
    fig.update_layout(height=260, margin=dict(l=28, r=28, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT))
    return fig


def shap_drivers(drivers: list[dict], units: str) -> go.Figure:
    d = pd.DataFrame(drivers).iloc[::-1]
    labels = [f"{r.label} = {r.value}" for r in d.itertuples()]
    colors = [INCREASE if v > 0 else DECREASE for v in d["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=d["shap_value"],
            y=labels,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:+.3f}" for v in d["shap_value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
        )
    )
    span = max(abs(d["shap_value"]).max(), 1e-6) * 1.35
    fig.update_xaxes(title=f"SHAP contribution ({units})<br>red raises risk, blue lowers it",
                     range=[-span, span], zeroline=True, zerolinecolor=MUTED, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=140 + 38 * len(d), title="Top churn drivers (SHAP)")


def global_importance(importance: pd.DataFrame, units: str, top_n: int = 15) -> go.Figure:
    d = importance.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=d["mean_abs_shap"],
            y=d["label"],
            orientation="h",
            marker=dict(color=SERIES[0], cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"Mean |SHAP value| ({units})", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=520, title="Global feature importance")


def roc_pr(curves: dict, display: dict, kind: str) -> go.Figure:
    fig = go.Figure()
    for key, c in curves.items():
        color = MODEL_COLORS.get(key, SERIES[0])
        if kind == "roc":
            x, y = c["roc"]["fpr"], c["roc"]["tpr"]
        else:
            x, y = c["pr"]["recall"], c["pr"]["precision"]
        fig.add_trace(go.Scatter(x=x, y=y, name=display.get(key, key), mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"{display.get(key, key)}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>"))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color=MUTED, dash="dot", width=1), hoverinfo="skip"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        title = "ROC (test set)"
    else:
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        title = "Precision-Recall (test set)"
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, 1.02])
    return _base(fig, height=450, title=title)


def metric_comparison(comparison: pd.DataFrame) -> go.Figure:
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]
    names = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision", "Accuracy"]
    fig = go.Figure()
    order = {k: i for i, k in enumerate(MODEL_COLORS)}
    for _, row in comparison.sort_values("key", key=lambda s: s.map(order)).iterrows():
        fig.add_trace(go.Bar(name=row["model"], x=names, y=[row[m] for m in metrics],
                             marker=dict(color=MODEL_COLORS[row["key"]], cornerradius=4,
                                         line=dict(width=2, color="rgba(0,0,0,0)")),
                             hovertemplate=f"<b>{row['model']}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1], title="Score")
    return _base(fig, height=380, title="Test-set metrics by model")


def dependence(scored: pd.DataFrame, feature: str, label: str) -> go.Figure:
    if pd.api.types.is_numeric_dtype(scored[feature]):
        sample = scored.sample(min(2500, len(scored)), random_state=0)
        fig = go.Figure(go.Scatter(
            x=sample[feature], y=sample["churn_probability"], mode="markers",
            marker=dict(size=8, color=SERIES[0], opacity=0.35, line=dict(width=0)),
            hovertemplate=f"{label}: %{{x}}<br>Churn prob: %{{y:.1%}}<extra></extra>",
        ))
        fig.update_xaxes(title=label, showgrid=True, gridcolor=GRID)
    else:
        g = scored.groupby(feature)["churn_probability"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=[wrap_label(v) for v in g.index.astype(str)], y=g.values,
                               marker=dict(color=SERIES[0], cornerradius=4),
                               text=[f"{v:.0%}" for v in g.values], textposition="outside",
                               hovertemplate="<b>%{x}</b><br>Mean churn prob: %{y:.1%}<extra></extra>"))
        fig.update_xaxes(title=label, tickangle=0)
    fig.update_yaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1.05])
    return _base(fig, height=380, title=f"Predicted churn vs. {label}")
