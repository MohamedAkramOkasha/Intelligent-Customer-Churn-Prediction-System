import numpy as np
import pandas as pd
import pytest

from churn.features import FeatureEngineer
from churn.recommend import RULES, recommend
from churn.segment import risk_tier, segment_customers, segment_summary, value_segment


@pytest.mark.parametrize(
    "p,tier", [(0.0, "Low"), (0.249, "Low"), (0.25, "Medium"), (0.5, "High"), (0.75, "Critical"), (1.0, "Critical")]
)
def test_risk_tier_boundaries(p, tier):
    assert risk_tier(p) == tier


def test_value_segments():
    assert value_segment("Critical", 100, 70) == "Priority Rescue"
    assert value_segment("High", 30, 70) == "Cost-Efficient Save"
    assert value_segment("Low", 100, 70) == "Loyal VIP - Nurture"
    assert value_segment("Medium", 30, 70) == "Stable - Grow"


def test_segment_customers_revenue_at_risk():
    df = pd.DataFrame({"MonthlyCharges": [100.0, 20.0]})
    out = segment_customers(df, np.array([0.8, 0.1]), high_value_threshold=70)
    assert out["annual_revenue_at_risk"].tolist() == [960.0, 24.0]
    assert out["segment"].tolist() == ["Priority Rescue", "Stable - Grow"]
    assert segment_summary(out)["customers"].sum() == 2


def test_rule_ids_unique():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))


def test_recommendations_follow_shap_drivers(high_risk):
    customer = FeatureEngineer().transform(pd.DataFrame([high_risk])).iloc[0].to_dict()
    # Pretend payment method is by far the strongest driver.
    shap = {"PaymentMethod": 0.9, "Contract": 0.2, "tenure": 0.1}
    plan = recommend(customer, shap, "Critical", 0.85)
    recs = plan["recommendations"]
    assert recs[0]["id"] == "autopay_incentive"
    assert recs[0]["priority"] == "High"
    assert 1 <= len(recs) <= 4
    assert plan["economics"]["annual_revenue_at_risk"] == pytest.approx(0.85 * 99.5 * 12, rel=1e-3)


def test_ineligible_rules_are_never_recommended(low_risk):
    customer = FeatureEngineer().transform(pd.DataFrame([low_risk])).iloc[0].to_dict()
    plan = recommend(customer, {"Contract": 1.0, "PaymentMethod": 1.0}, "High", 0.6)
    ids = {r["id"] for r in plan["recommendations"]}
    # Two-year contract + autopay customer: those offers make no sense.
    assert "contract_upgrade" not in ids and "autopay_incentive" not in ids


def test_low_risk_gets_light_touch(low_risk):
    customer = FeatureEngineer().transform(pd.DataFrame([low_risk])).iloc[0].to_dict()
    plan = recommend(customer, {}, "Low", 0.05)
    assert len(plan["recommendations"]) <= 1
    assert plan["playbook"]["max_discount_pct"] == 0
