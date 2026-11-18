"""Personalised retention recommendations.

Each rule in ``RULES`` is an intervention that is *eligible* when its condition
matches the customer's profile. Eligible rules are then *ranked* by how much
the features they address are pushing this particular customer towards churn
(their positive SHAP contributions). The result is a short list of actions
that targets the real reasons a given customer is at risk, instead of a
generic offer for everyone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from churn import config

Customer = dict[str, Any]


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    action: str
    rationale: str
    channel: str
    est_monthly_cost: float  # cost to the business per customer per month (USD)
    features: tuple[str, ...]
    condition: Callable[[Customer], bool] = field(compare=False)


def _has_internet(c: Customer) -> bool:
    return c.get("InternetService") not in (None, "No")


RULES: list[Rule] = [
    Rule(
        id="contract_upgrade",
        title="Migrate to a 12-month contract",
        action="Offer a 12-month contract with a 15% discount for the first 3 months "
        "and a price-lock guarantee.",
        rationale="Month-to-month customers churn at ~43% vs. ~3% on two-year contracts; "
        "commitment is the single strongest retention lever in this data.",
        channel="Retention call / in-app offer",
        est_monthly_cost=9.0,
        features=("Contract", "tenure_group"),
        condition=lambda c: c.get("Contract") == "Month-to-month",
    ),
    Rule(
        id="autopay_incentive",
        title="Switch to automatic payment",
        action="Give a one-off $10 bill credit for enrolling in card or bank autopay.",
        rationale="Electronic-check payers churn at ~45% vs. ~16% on automatic payment; "
        "autopay removes a monthly decision point to leave.",
        channel="Email + SMS",
        est_monthly_cost=0.85,
        features=("PaymentMethod", "is_auto_payment", "PaperlessBilling"),
        condition=lambda c: c.get("is_auto_payment") == "No",
    ),
    Rule(
        id="tech_support_trial",
        title="Free Tech Support for 3 months",
        action="Activate complimentary Tech Support for 3 months with a proactive "
        "welcome call from a technician.",
        rationale="Internet customers without tech support churn at ~42% vs. ~15% with it, "
        "especially on fiber where service issues drive dissatisfaction.",
        channel="Outbound call",
        est_monthly_cost=5.0,
        features=("TechSupport", "InternetService", "num_protection_services"),
        condition=lambda c: _has_internet(c) and c.get("TechSupport") == "No",
    ),
    Rule(
        id="security_bundle",
        title="Bundle Online Security",
        action="Add Online Security + Online Backup free for 3 months, then at 50% off.",
        rationale="Customers without Online Security churn at ~42% vs. ~15% with it; "
        "each add-on also raises switching cost.",
        channel="In-app offer",
        est_monthly_cost=4.0,
        features=("OnlineSecurity", "OnlineBackup", "DeviceProtection", "num_protection_services"),
        condition=lambda c: _has_internet(c) and c.get("OnlineSecurity") == "No",
    ),
    Rule(
        id="early_life_care",
        title="Early-life onboarding programme",
        action="Enrol in the first-year care programme: 30/60/90-day check-ins, "
        "setup help and a dedicated contact.",
        rationale="Churn in the first 12 months is ~47% vs. ~17% afterwards; getting "
        "customers past year one is critical.",
        channel="Customer success manager",
        est_monthly_cost=3.0,
        features=("tenure", "tenure_group", "charge_to_tenure_ratio"),
        condition=lambda c: float(c.get("tenure", 0)) <= 12,
    ),
    Rule(
        id="plan_rightsizing",
        title="Plan right-sizing review",
        action="Run a bill review and propose a plan that matches actual usage, "
        "or a 10% loyalty discount if already optimal.",
        rationale="Customers paying $80+/month churn at ~34%; showing customers "
        "they are on the best plan reduces price-driven switching.",
        channel="Retention call",
        est_monthly_cost=8.0,
        features=("MonthlyCharges", "avg_monthly_spend", "TotalCharges"),
        condition=lambda c: float(c.get("MonthlyCharges", 0)) >= 80,
    ),
    Rule(
        id="price_lock",
        title="Explain recent bill increase and lock price",
        action="Proactively explain the bill change and offer a 12-month price lock.",
        rationale="This customer pays noticeably more now than their historical average; "
        "unexplained increases are a common trigger for leaving.",
        channel="Email",
        est_monthly_cost=2.0,
        features=("monthly_charge_delta",),
        condition=lambda c: float(c.get("monthly_charge_delta", 0)) > 5,
    ),
    Rule(
        id="fiber_experience",
        title="Fiber service quality check",
        action="Schedule a free line-quality check and router upgrade for the fiber connection.",
        rationale="Fiber customers churn at ~42%, more than twice the DSL rate, "
        "pointing to experience or value-for-money issues.",
        channel="Field technician",
        est_monthly_cost=6.0,
        features=("InternetService",),
        condition=lambda c: c.get("InternetService") == "Fiber optic",
    ),
    Rule(
        id="senior_support",
        title="Priority assisted support",
        action="Give access to the priority assisted-support line and a simplified bill.",
        rationale="Senior customers churn at ~42% vs. ~24% for non-seniors; extra support "
        "addresses complexity and service friction.",
        channel="Phone",
        est_monthly_cost=2.0,
        features=("SeniorCitizen",),
        condition=lambda c: c.get("SeniorCitizen") == "Yes",
    ),
    Rule(
        id="household_bundle",
        title="Household / multi-line bundle",
        action="Offer a discounted additional line or family bundle.",
        rationale="Accounts with no partner or dependents churn at ~34% vs. ~20%; "
        "household bundles create shared switching costs.",
        channel="Email",
        est_monthly_cost=5.0,
        features=("has_family", "Partner", "Dependents", "MultipleLines"),
        condition=lambda c: c.get("has_family") == "No",
    ),
    Rule(
        id="engagement_bundle",
        title="Streaming trial to deepen engagement",
        action="Offer a 2-month free trial of Streaming TV & Movies.",
        rationale="Customers with few services have little invested in the relationship; "
        "more services correlate with longer tenure.",
        channel="In-app offer",
        est_monthly_cost=4.0,
        features=("num_services", "StreamingTV", "StreamingMovies"),
        condition=lambda c: _has_internet(c) and int(c.get("num_services", 0)) <= 3,
    ),
]

TIER_PLAYBOOK = {
    "Critical": {
        "sla": "Contact within 48 hours",
        "owner": "Senior retention specialist",
        "max_discount_pct": 25,
        "summary": "Personal call with a tailored package; authorised to combine offers.",
    },
    "High": {
        "sla": "Contact within 7 days",
        "owner": "Retention team",
        "max_discount_pct": 15,
        "summary": "Targeted personalised offer via the customer's preferred channel.",
    },
    "Medium": {
        "sla": "Within 30 days",
        "owner": "Lifecycle marketing",
        "max_discount_pct": 5,
        "summary": "Automated nurture journey and satisfaction pulse survey.",
    },
    "Low": {
        "sla": "Routine",
        "owner": "Customer marketing",
        "max_discount_pct": 0,
        "summary": "Standard loyalty communications; eligible for upsell campaigns.",
    },
}

MAX_RECOMMENDATIONS = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def recommend(
    customer: Customer,
    shap_by_feature: dict[str, float],
    risk_tier: str,
    churn_probability: float,
) -> dict[str, Any]:
    """Build a retention plan for one customer.

    Parameters
    ----------
    customer:
        Engineered feature values for the customer (output of ``FeatureEngineer``).
    shap_by_feature:
        Aggregated SHAP value per model feature (positive = pushes towards churn).
    risk_tier, churn_probability:
        Model output for the customer.
    """
    scored = []
    for rule in RULES:
        try:
            eligible = rule.condition(customer)
        except (TypeError, ValueError):
            eligible = False
        if not eligible:
            continue
        # Only the churn-increasing part of each addressed feature counts.
        impact = sum(max(shap_by_feature.get(f, 0.0), 0.0) for f in rule.features)
        scored.append((impact, rule))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Prefer rules that address a real risk driver; fall back to eligible ones.
    relevant = [item for item in scored if item[0] > 0] or scored
    selected = relevant[: MAX_RECOMMENDATIONS.get(risk_tier, 2)]

    recs = []
    for rank, (impact, rule) in enumerate(selected, start=1):
        priority = "High" if rank == 1 and risk_tier in config.HIGH_RISK_TIERS else (
            "Medium" if risk_tier in config.HIGH_RISK_TIERS or rank == 1 else "Low"
        )
        recs.append(
            {
                "rank": rank,
                "id": rule.id,
                "title": rule.title,
                "action": rule.action,
                "rationale": rule.rationale,
                "channel": rule.channel,
                "priority": priority,
                "addresses": [config.FEATURE_LABELS.get(f, f) for f in rule.features],
                "driver_impact": round(impact, 4),
                "est_monthly_cost": rule.est_monthly_cost,
            }
        )

    monthly_charges = float(customer.get("MonthlyCharges", 0.0))
    revenue_at_risk = churn_probability * monthly_charges * 12
    total_cost = sum(r["est_monthly_cost"] for r in recs) * 3  # offers run ~3 months

    return {
        "playbook": TIER_PLAYBOOK[risk_tier],
        "recommendations": recs,
        "economics": {
            "annual_revenue_at_risk": round(revenue_at_risk, 2),
            "est_campaign_cost": round(total_cost, 2),
            "break_even_save_rate": round(total_cost / revenue_at_risk, 3)
            if revenue_at_risk > 0
            else None,
        },
    }
