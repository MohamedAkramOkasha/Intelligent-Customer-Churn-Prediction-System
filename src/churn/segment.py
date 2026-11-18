"""Customer risk segmentation.

Two dimensions:
  * Risk tier from the predicted churn probability (Low / Medium / High / Critical).
  * Customer value from MonthlyCharges (high value = at or above the training median).

Crossing them gives four actionable segments (see ``config.VALUE_SEGMENTS``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config


def risk_tier(probability: float) -> str:
    """Map a churn probability to a risk tier."""
    for tier, upper in config.RISK_TIERS:
        if probability < upper:
            return tier
    return config.RISK_TIERS[-1][0]


def value_segment(tier: str, monthly_charges: float, high_value_threshold: float) -> str:
    high_risk = tier in config.HIGH_RISK_TIERS
    high_value = monthly_charges >= high_value_threshold
    return config.VALUE_SEGMENTS[(high_risk, high_value)]


def segment_customers(
    customers: pd.DataFrame,
    probabilities: np.ndarray,
    high_value_threshold: float = config.DEFAULT_HIGH_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """Return ``customers`` with probability, tier, segment and revenue-at-risk columns."""
    out = customers.copy()
    out["churn_probability"] = np.round(probabilities, 4)
    out["risk_tier"] = [risk_tier(p) for p in probabilities]
    out["is_high_value"] = out["MonthlyCharges"].astype(float) >= high_value_threshold
    out["segment"] = [
        value_segment(t, m, high_value_threshold)
        for t, m in zip(out["risk_tier"], out["MonthlyCharges"].astype(float))
    ]
    # Expected revenue lost over the next 12 months if nothing is done.
    out["annual_revenue_at_risk"] = np.round(
        out["churn_probability"] * out["MonthlyCharges"].astype(float) * 12, 2
    )
    return out


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs per segment for reporting."""
    summary = (
        scored.groupby("segment")
        .agg(
            customers=("churn_probability", "size"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            annual_revenue_at_risk=("annual_revenue_at_risk", "sum"),
        )
        .sort_values("annual_revenue_at_risk", ascending=False)
        .round(3)
        .reset_index()
    )
    summary["description"] = summary["segment"].map(config.SEGMENT_DESCRIPTIONS)
    return summary
