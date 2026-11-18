"""Plotly chart builders for the dashboard.

Colour roles (validated reference palette):
  * Categorical slots 1-4 -> models / segments / stay-vs-churn (fixed order, never cycled)
  * Status palette        -> risk tiers (always paired with a text label)
  * Diverging blue <-> red -> SHAP contributions (lowers vs. raises churn risk)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TIER_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
TIER_ORDER = ["Low", "Medium", "High", "Critical"]
SEGMENT_ORDER = ["Priority Rescue", "Cost-Efficient Save", "Loyal VIP - Nurture", "Stable - Grow"]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, SERIES))
DECREASE, INCREASE = "#2a78d6", "#e34948"
STAY, CHURN = SERIES[0], SERIES[1]
GRID = "rgba(128,128,128,0.18)"
# Colour follows the model, never its rank (matches the training-report figures).
MODEL_COLORS = dict(zip(["logistic_regression", "decision_tree", "random_forest", "xgboost"], SERIES))


def _base(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_size=12),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
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
    return _base(fig, title="Annual revenue at risk by segment")


def churn_rate_by(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    rates = df.groupby(col)["actual_churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    overall = df["actual_churn"].mean()
    fig = go.Figure(
        go.Bar(
            x=rates.index.astype(str),
            y=rates["mean"],
            marker=dict(color=CHURN, cornerradius=4),
            text=[f"{v:.0%}" for v in rates["mean"]],
            textposition="outside",
            customdata=rates["size"],
            hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1%}<br>Customers: %{customdata:,}<extra></extra>",
        )
    )
    fig.add_hline(y=overall, line=dict(color="gray", dash="dot", width=1),
                  annotation_text=f"Overall {overall:.0%}", annotation_position="top right")
    fig.update_yaxes(tickformat=".0%", range=[0, rates["mean"].max() * 1.25], title="Churn rate")
    return _base(fig, height=320, title=title)


def tenure_curve(df: pd.DataFrame) -> go.Figure:
    bins = pd.cut(df["tenure"], bins=range(0, 78, 6), right=False)
    g = df.groupby(bins, observed=True).agg(
        actual=("actual_churn", "mean"), predicted=("churn_probability", "mean"), n=("tenure", "size")
    )
    x = [f"{int(i.left)}-{int(i.right) - 1}" for i in g.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=g["actual"], name="Actual churn rate", mode="lines+markers",
                             line=dict(color=SERIES[1], width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=x, y=g["predicted"], name="Mean predicted probability",
                             mode="lines+markers", line=dict(color=SERIES[0], width=2, dash="dash"),
                             marker=dict(size=8)))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", title="Churn")
    fig.update_xaxes(title="Tenure (months)")
    return _base(fig, height=340, title="Churn falls sharply with tenure - and the model tracks it")


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
    return _base(fig, height=340, title="Predicted probability separates churners from stayers")


def gauge(probability: float, tier: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", font=dict(size=44)),
            gauge=dict(
                axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%"),
                bar=dict(color=TIER_COLORS[tier], thickness=0.3),
                steps=[
                    dict(range=[0, 25], color="rgba(12,163,12,0.12)"),
                    dict(range=[25, 50], color="rgba(250,178,25,0.14)"),
                    dict(range=[50, 75], color="rgba(236,131,90,0.16)"),
                    dict(range=[75, 100], color="rgba(208,59,59,0.16)"),
                ],
            ),
            title=dict(text=f"Churn probability · <b>{tier} risk</b>", font=dict(size=14)),
        )
    )
    fig.update_layout(height=260, margin=dict(l=24, r=24, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)")
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
    fig.update_xaxes(title=f"SHAP contribution ({units}) · red raises risk, blue lowers it",
                     range=[-span, span], zeroline=True, zerolinecolor="gray", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _base(fig, height=60 + 38 * len(d), title="Why this prediction? Top drivers (SHAP)")


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
                                 line=dict(color="gray", dash="dot", width=1), hoverinfo="skip"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        title = "ROC curve (test set)"
    else:
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        title = "Precision-Recall curve (test set)"
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(range=[0, 1.02])
    return _base(fig, height=400, title=title)


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
    return _base(fig, height=380, title="Test-set metrics at each model's tuned threshold")


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
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker=dict(color=SERIES[0], cornerradius=4),
                               text=[f"{v:.0%}" for v in g.values], textposition="outside",
                               hovertemplate="<b>%{x}</b><br>Mean churn prob: %{y:.1%}<extra></extra>"))
        fig.update_xaxes(title=label)
    fig.update_yaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1.05])
    return _base(fig, height=380, title=f"Predicted churn vs. {label}")
