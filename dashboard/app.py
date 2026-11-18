"""Streamlit dashboard for the Intelligent Customer Churn Prediction System.

Run:  streamlit run dashboard/app.py
If CHURN_API_URL is set (e.g. http://api:8000 in docker-compose), single-customer
predictions go through the FastAPI service; otherwise the model is loaded in-process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import charts  # noqa: E402

from churn import config  # noqa: E402
from churn.predict import ChurnService  # noqa: E402
from churn.segment import segment_customers, segment_summary  # noqa: E402

API_URL = os.getenv("CHURN_API_URL", "").rstrip("/")
MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
TABLE_ROW_HEIGHT = 40  # roomier rows than the 35px default
# Readable headers shared by the customer tables (display only; data is unchanged).
TABLE_COLUMNS = {
    config.ID_COL: st.column_config.TextColumn("Customer", pinned=True),
    "churn_probability": st.column_config.ProgressColumn("Churn prob.", format="percent",
                                                         min_value=0, max_value=1),
    "risk_tier": st.column_config.TextColumn("Risk tier", width="small"),
    "segment": st.column_config.TextColumn("Segment"),
    "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f", width="small"),
    "tenure": st.column_config.NumberColumn("Tenure", width="small"),
    "PaymentMethod": st.column_config.TextColumn("Payment method"),
    "InternetService": st.column_config.TextColumn("Internet"),
}
TIER_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

st.set_page_config(page_title="Churn Intelligence", page_icon="📉", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      .badge {display:inline-flex; align-items:center; gap:.4rem; padding:.2rem .65rem;
              border-radius:999px; border:1px solid #2a3b5c; background:#16243d; font-weight:600;}
      .dot {width:.6rem; height:.6rem; border-radius:50%; display:inline-block;}
      .muted {color:#9fb0c8; font-size:.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_service() -> ChurnService:
    return ChurnService()


@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(config.SCORED_PATH)


@st.cache_data
def load_artifacts() -> dict:
    return {
        "comparison": pd.read_csv(config.COMPARISON_PATH),
        "importance": pd.read_json(config.GLOBAL_IMPORTANCE_PATH),
        "curves": json.loads(config.CURVES_PATH.read_text()),
        "metadata": json.loads(config.METADATA_PATH.read_text()),
    }


def score_one(record: dict) -> tuple[dict, str]:
    """Score via the API when configured, else locally. Returns (result, source)."""
    if API_URL:
        try:
            import requests

            r = requests.post(f"{API_URL}/predict", json=record, timeout=15)
            r.raise_for_status()
            return r.json(), "FastAPI"
        except Exception as exc:  # noqa: BLE001 - surface and fall back
            st.warning(f"API unavailable ({exc}); scoring locally instead.")
    return load_service().score([record])[0], "local model"


def badge(tier: str) -> str:
    color = charts.TIER_COLORS[tier]
    return f'<span class="badge"><span class="dot" style="background:{color}"></span>{tier} risk</span>'


if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m churn.train` and reload this page.")
    st.stop()

service = load_service()
scored = load_scored()
art = load_artifacts()
meta = art["metadata"]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📉 Churn Intelligence")
    page = st.radio(
        "Navigate",
        [
            "Executive overview",
            "Customer predictor",
            "Retention planner",
            "Batch scoring",
            "Model performance",
            "Explainability",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Production model")
    st.markdown(f"**{meta['model_name']}** · v{meta['version']}")
    st.caption(
        f"Test ROC-AUC **{meta['test_metrics']['roc_auc']:.3f}** · "
        f"Recall **{meta['test_metrics']['recall']:.0%}** · "
        f"threshold {meta['decision_threshold']:.2f}"
    )
    st.caption(f"Scoring via: **{'FastAPI ' + API_URL if API_URL else 'in-process model'}**")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.header("Executive overview")
    st.caption(
        "Portfolio of 7,043 telecom customers (IBM Telco dataset) scored by the production model."
    )
    at_risk = scored["risk_tier"].isin(config.HIGH_RISK_TIERS)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    c2.metric("Historical churn rate", f"{scored['actual_churn'].mean():.1%}")
    c3.metric("High / critical risk", f"{at_risk.sum():,}", f"{at_risk.mean():.1%} of base",
              delta_color="off")
    c4.metric("Annual revenue at risk", f"${scored['annual_revenue_at_risk'].sum() / 1e6:.2f}M",
              f"${scored.loc[at_risk, 'annual_revenue_at_risk'].sum() / 1e6:.2f}M in high/critical",
              delta_color="off")

    left, right = st.columns(2)
    left.plotly_chart(charts.tier_distribution(scored), width="stretch")
    summary = segment_summary(scored)
    right.plotly_chart(charts.segment_revenue(summary), width="stretch")

    st.subheader("Segments")
    st.dataframe(
        summary.drop(columns="description").rename(columns={
            "segment": "Segment", "customers": "Customers", "avg_churn_probability": "Avg churn prob.",
            "avg_monthly_charges": "Avg monthly $", "annual_revenue_at_risk": "Revenue at risk $",
        }),
        hide_index=True, row_height=TABLE_ROW_HEIGHT,
        width="stretch",
        column_config={
            "Avg churn prob.": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Revenue at risk $": st.column_config.NumberColumn(format="$%.0f"),
            "Avg monthly $": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    # Long guidance text wraps here instead of being truncated inside a table cell.
    for col, row in zip(st.columns(len(summary)), summary.itertuples()):
        col.markdown(f"**{row.segment}**")
        col.caption(row.description)

    st.subheader("What drives churn")
    st.caption("Historical churn rate by customer attribute; dotted line = overall rate.")
    a, b, c = st.columns(3)
    a.plotly_chart(charts.churn_rate_by(scored, "Contract", "By contract"), width="stretch")
    b.plotly_chart(charts.churn_rate_by(scored, "InternetService", "By internet service"),
                   width="stretch")
    c.plotly_chart(charts.churn_rate_by(scored, "PaymentMethod", "By payment method"),
                   width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(charts.tenure_curve(scored), width="stretch")
    right.plotly_chart(charts.probability_histogram(scored[scored["split"] == "test"]), width="stretch")
    right.caption("Histogram uses held-out test customers only.")


def customer_form(defaults: dict, key: str) -> dict:
    """Input widgets for one customer; ``key`` resets widgets when the preset changes."""
    def opt(label, field, options):
        value = str(defaults.get(field, options[0]))
        return st.selectbox(label, options, index=options.index(value) if value in options else 0,
                            key=f"{key}-{field}")

    yes_no, net3 = ["No", "Yes"], ["No", "Yes", "No internet service"]
    rec: dict = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        rec["tenure"] = st.slider("Tenure (months)", 0, 72, int(defaults.get("tenure", 12)), key=f"{key}-tenure")
        rec["Contract"] = opt("Contract", "Contract", ["Month-to-month", "One year", "Two year"])
        rec["PaymentMethod"] = opt("Payment method", "PaymentMethod", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        rec["PaperlessBilling"] = opt("Paperless billing", "PaperlessBilling", yes_no)
        rec["MonthlyCharges"] = st.number_input("Monthly charges ($)", 0.0, 200.0,
                                                float(defaults.get("MonthlyCharges", 70.0)), 0.05,
                                                key=f"{key}-mc")
        rec["TotalCharges"] = st.number_input(
            "Total charges ($)", 0.0, 10000.0,
            float(defaults.get("TotalCharges", rec["MonthlyCharges"] * rec["tenure"])), 1.0,
            key=f"{key}-tc")
    with c2:
        st.markdown("**Services**")
        rec["PhoneService"] = opt("Phone service", "PhoneService", yes_no)
        rec["MultipleLines"] = opt("Multiple lines", "MultipleLines", ["No", "Yes", "No phone service"])
        rec["InternetService"] = opt("Internet service", "InternetService", ["Fiber optic", "DSL", "No"])
        for field, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                             ("DeviceProtection", "Device protection")]:
            rec[field] = opt(label, field, net3)
    with c3:
        st.markdown("**Support, streaming & household**")
        for field, label in [("TechSupport", "Tech support"), ("StreamingTV", "Streaming TV"),
                             ("StreamingMovies", "Streaming movies")]:
            rec[field] = opt(label, field, net3)
        rec["gender"] = opt("Gender", "gender", ["Female", "Male"])
        rec["SeniorCitizen"] = opt("Senior citizen", "SeniorCitizen", yes_no)
        rec["Partner"] = opt("Partner", "Partner", yes_no)
        rec["Dependents"] = opt("Dependents", "Dependents", yes_no)

    # Keep internet add-ons consistent with the chosen internet service.
    if rec["InternetService"] == "No":
        for f in config.PROTECTION_COLS + ["StreamingTV", "StreamingMovies"]:
            rec[f] = "No internet service"
    if rec["PhoneService"] == "No":
        rec["MultipleLines"] = "No phone service"
    return rec


def render_result(result: dict, source: str) -> None:
    tier = result["risk_tier"]
    g, info = st.columns([1, 1.4])
    g.plotly_chart(charts.gauge(result["churn_probability"], tier), width="stretch")
    with info:
        st.markdown(f"### {TIER_ICON[tier]} {badge(tier)}", unsafe_allow_html=True)
        verdict = "likely to churn" if result["will_churn"] else "likely to stay"
        st.markdown(
            f"Predicted **{verdict}** at the model's decision threshold "
            f"({service.threshold:.2f}).  \nSegment: **{result['segment']}** - "
            f"{result['segment_description']}"
        )
        plan = result.get("retention_plan")
        if plan:
            pb, econ = plan["playbook"], plan["economics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Revenue at risk (12m)", f"${econ['annual_revenue_at_risk']:,.0f}")
            m2.metric("Est. campaign cost", f"${econ['est_campaign_cost']:,.0f}")
            be = econ.get("break_even_save_rate")
            m3.metric("Break-even save rate", f"{be:.0%}" if be is not None else "n/a",
                      help="Share of customers like this one the campaign must retain to pay for itself.")
            st.markdown(f"**Playbook:** {pb['summary']}  \n"
                        f"<span class='muted'>SLA: {pb['sla']} · Owner: {pb['owner']} · "
                        f"Max discount: {pb['max_discount_pct']}%</span>", unsafe_allow_html=True)
        st.caption(f"Scored by {source}.")

    if result.get("top_drivers"):
        left, right = st.columns([1.1, 1])
        left.plotly_chart(charts.shap_drivers(result["top_drivers"], service.explainer.output_units),
                          width="stretch")
        with right:
            st.markdown("#### Recommended retention actions")
            for rec in result["retention_plan"]["recommendations"]:
                with st.container(border=True):
                    st.markdown(f"**{rec['rank']}. {rec['title']}** · `{rec['priority']} priority`")
                    st.markdown(rec["action"])
                    st.caption(f"Why: {rec['rationale']}  \nChannel: {rec['channel']} · "
                               f"Addresses: {', '.join(rec['addresses'][:3])} · "
                               f"~${rec['est_monthly_cost']:.2f}/month")


def page_predictor() -> None:
    st.header("Customer predictor")
    st.caption("Score a customer, see why the model thinks so, and get a personalised retention plan.")

    test = scored[scored["split"] == "test"].sort_values("annual_revenue_at_risk", ascending=False)
    options = ["New customer (blank profile)"] + test[config.ID_COL].head(300).tolist()
    choice = st.selectbox("Start from an existing test-set customer (sorted by revenue at risk) "
                          "or a blank profile", options, index=1)
    if choice.startswith("New"):
        defaults = {"tenure": 6, "Contract": "Month-to-month", "InternetService": "Fiber optic",
                    "MonthlyCharges": 85.0, "TotalCharges": 510.0}
    else:
        defaults = test.loc[test[config.ID_COL] == choice].iloc[0].to_dict()
        actual = "churned" if defaults["actual_churn"] == 1 else "stayed"
        st.caption(f"Ground truth for {choice}: customer **{actual}**.")

    with st.expander("Customer profile", expanded=True):
        record = customer_form(defaults, key=choice)
    result, source = score_one(record)
    render_result(result, source)


def page_planner() -> None:
    st.header("Retention planner")
    st.caption("Prioritise who to contact and simulate the return on a retention campaign.")

    active = scored[scored["actual_churn"] == 0] if st.toggle(
        "Only customers who have not churned yet", value=True,
        help="Historical churners are already gone; a real campaign targets the active base.") else scored

    c1, c2, c3, c4 = st.columns(4)
    tiers = c1.multiselect("Target risk tiers", charts.TIER_ORDER, default=["High", "Critical"])
    save_rate = c2.slider("Expected save rate", 0.05, 0.6, 0.25, 0.05,
                          help="Share of targeted at-risk customers the campaign retains.")
    cost = c3.number_input("Cost per targeted customer ($)", 0.0, 500.0, 40.0, 5.0)
    min_value = c4.number_input("Min. monthly charges ($)", 0.0, 120.0, 0.0, 5.0)

    target = active[active["risk_tier"].isin(tiers) & (active["MonthlyCharges"] >= min_value)]
    saved = target["annual_revenue_at_risk"].sum() * save_rate
    spend = len(target) * cost
    roi = (saved - spend) / spend if spend else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers targeted", f"{len(target):,}")
    m2.metric("Expected revenue retained", f"${saved:,.0f}")
    m3.metric("Campaign cost", f"${spend:,.0f}")
    m4.metric("ROI", f"{roi:.0%}", delta="profitable" if roi > 0 else "loss-making",
              delta_color="normal" if roi > 0 else "inverse")

    st.subheader("Priority call list")
    cols = [config.ID_COL, "churn_probability", "annual_revenue_at_risk", "risk_tier", "segment",
            "MonthlyCharges", "tenure", "Contract", "PaymentMethod", "InternetService"]
    priority = target.sort_values("annual_revenue_at_risk", ascending=False)[cols]
    st.dataframe(
        priority.head(500), hide_index=True, row_height=TABLE_ROW_HEIGHT, width="stretch", height=380,
        column_config={
            **TABLE_COLUMNS,
            "annual_revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.0f",
                                                                    width="small"),
        },
    )
    st.download_button("Download full call list (CSV)", priority.to_csv(index=False).encode(),
                       "priority_call_list.csv", "text/csv")

    if len(priority):
        st.subheader("Personalised plan")
        pick = st.selectbox("Customer", priority[config.ID_COL].head(200))
        row = scored.loc[scored[config.ID_COL] == pick, config.RAW_FEATURES].iloc[0].to_dict()
        render_result(service.score([row])[0], "local model")


def page_batch() -> None:
    st.header("Batch scoring")
    st.caption("Upload a CSV in the IBM Telco format (customerID optional; a Churn column is ignored). "
               "Try `data/sample/sample_customers.csv`.")
    up = st.file_uploader("Customer CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
    else:
        st.info("No file uploaded - showing the held-out test set as a sample.")
        raw = scored[scored["split"] == "test"][[config.ID_COL] + config.RAW_FEATURES]
    try:
        X = service.prepare(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    proba = service.pipeline.predict_proba(X)[:, 1]
    out = segment_customers(X, proba, service.high_value_threshold)
    if config.ID_COL in raw.columns:
        out.insert(0, config.ID_COL, raw[config.ID_COL].to_numpy())
    out["will_churn"] = proba >= service.threshold

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers scored", f"{len(out):,}")
    m2.metric("Predicted churners", f"{out['will_churn'].sum():,}", f"{out['will_churn'].mean():.1%}",
              delta_color="off")
    m3.metric("Annual revenue at risk", f"${out['annual_revenue_at_risk'].sum():,.0f}")
    left, right = st.columns(2)
    left.plotly_chart(charts.tier_distribution(out), width="stretch")
    right.plotly_chart(charts.segment_revenue(segment_summary(out)), width="stretch")

    tiers = st.multiselect("Filter tiers", charts.TIER_ORDER, default=charts.TIER_ORDER)
    view = out[out["risk_tier"].isin(tiers)].sort_values("churn_probability", ascending=False)
    # Show the scoring results first, then the customer's input fields.
    lead = [c for c in [config.ID_COL, "churn_probability", "risk_tier", "segment", "will_churn",
                        "annual_revenue_at_risk"] if c in view.columns]
    view = view[lead + [c for c in view.columns if c not in lead]]
    st.dataframe(view, hide_index=True, row_height=TABLE_ROW_HEIGHT, width="stretch", height=380,
                 column_config={**TABLE_COLUMNS,
                                "will_churn": st.column_config.CheckboxColumn("Predicted churn", width="small"),
                                "annual_revenue_at_risk": st.column_config.NumberColumn(
                                    "Revenue at risk", format="$%.0f", width="small")})
    st.download_button("Download scored CSV", view.to_csv(index=False).encode(), "scored_customers.csv",
                       "text/csv")


def page_models() -> None:
    st.header("Model performance")
    comp = art["comparison"]
    st.caption(
        f"Four models tuned with 5-fold stratified CV on {meta['n_train']:,} customers and evaluated on "
        f"{meta['n_test']:,} held-out customers. The production model is chosen by **cross-validated** "
        "ROC-AUC; each model's decision threshold is tuned on out-of-fold predictions to maximise F1."
    )
    table = comp[["model", "cv_roc_auc", "roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy",
                  "brier", "top20_capture", "threshold"]].rename(columns={
        "model": "Model", "cv_roc_auc": "CV ROC-AUC", "roc_auc": "Test ROC-AUC", "pr_auc": "PR-AUC",
        "f1": "F1", "recall": "Recall", "precision": "Precision", "accuracy": "Accuracy",
        "brier": "Brier ↓", "top20_capture": "Churners in top 20%", "threshold": "Threshold"})
    st.dataframe(table.style.format(precision=3).highlight_max(
        subset=["CV ROC-AUC", "Test ROC-AUC", "PR-AUC", "F1", "Recall", "Churners in top 20%"],
        props="font-weight:700"), hide_index=True, row_height=TABLE_ROW_HEIGHT, width="stretch")

    st.plotly_chart(charts.metric_comparison(comp), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(charts.roc_pr(art["curves"], MODEL_DISPLAY, "roc"), width="stretch")
    right.plotly_chart(charts.roc_pr(art["curves"], MODEL_DISPLAY, "pr"), width="stretch")

    left, right = st.columns([1, 1.3])
    cm = config.FIGURES_DIR / "confusion_matrix.png"
    if cm.exists():
        left.image(str(cm), caption="Production model on the test set")
    with right:
        st.markdown("#### How to read this")
        tm = meta["test_metrics"]
        st.markdown(
            f"- **Recall {tm['recall']:.0%}** - share of actual churners the model flags.\n"
            f"- **Precision {tm['precision']:.0%}** - share of flagged customers who really churn.\n"
            f"- **Top-20% capture {tm['top20_capture']:.0%}** - contacting only the riskiest fifth of "
            "customers reaches this share of all churners.\n"
            f"- **Brier {tm['brier']:.3f}** - probability calibration (lower is better); probabilities are "
            "used directly for revenue-at-risk, so calibration matters.\n"
            "- The IBM Telco dataset tops out around 0.84-0.86 ROC-AUC in published work; the models "
            "here are at that ceiling, so the gap between them is small."
        )
        with st.expander("Best hyper-parameters"):
            st.json(meta["best_params"])


def page_explain() -> None:
    st.header("Explainability")
    st.caption(f"SHAP values for the {meta['model_name']} model, computed on the held-out test set. "
               "One-hot columns are summed back into their original feature.")
    imp = art["importance"]
    left, right = st.columns([1, 1])
    left.plotly_chart(charts.global_importance(imp, meta["shap_output_units"]), width="stretch")
    bee = config.FIGURES_DIR / "shap_summary_beeswarm.png"
    if bee.exists():
        right.image(str(bee), caption="SHAP beeswarm: each dot is a customer; colour = feature value, "
                                      "x = impact on churn risk.")

    st.subheader("Feature effect explorer")
    candidates = [f for f in imp["feature"] if f in scored.columns]
    feat = st.selectbox("Feature", candidates, format_func=lambda f: config.FEATURE_LABELS.get(f, f))
    st.plotly_chart(charts.dependence(scored, feat, config.FEATURE_LABELS.get(feat, feat)), width="stretch")


{
    "Executive overview": page_overview,
    "Customer predictor": page_predictor,
    "Retention planner": page_planner,
    "Batch scoring": page_batch,
    "Model performance": page_models,
    "Explainability": page_explain,
}[page]()
