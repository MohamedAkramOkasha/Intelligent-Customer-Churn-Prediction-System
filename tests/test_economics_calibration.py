import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.calibration import calibration_report, calibration_table, wilson_interval
from churn.economics import best_profit_threshold, customer_value, net_profit, profit_curve

# --------------------------------------------------------------------------- #
# Economics / profit threshold
# --------------------------------------------------------------------------- #
Y = np.array([1, 1, 0, 0, 1, 0])
P = np.array([0.9, 0.7, 0.6, 0.3, 0.2, 0.1])
VALUE = np.array([1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0])


def test_customer_value_is_monthly_times_horizon():
    assert customer_value([10.0])[0] == 10.0 * config.REVENUE_HORIZON_MONTHS


def test_net_profit_by_hand():
    # threshold 0.65 -> contacts customers 0 and 1, both churners
    r = net_profit(Y, P, VALUE, 0.65, cost=40, save_rate=0.25)
    assert r["contacted"] == 2 and r["churners_reached"] == 2
    assert r["revenue_saved"] == pytest.approx(0.25 * 2400)
    assert r["net_profit"] == pytest.approx(600 - 80)


def test_best_threshold_stops_before_unprofitable_contacts():
    # Contacting customer 2 (p=0.6, not a churner) only adds cost; customer 4 (p=0.2) is a
    # churner worth 0.25*1200=300 > 40 but needs contacting 3 non-churners first (+300-120>0).
    grid = np.array([0.15, 0.25, 0.5, 0.65, 0.8, 0.95])
    t = best_profit_threshold(Y, P, VALUE, cost=40, save_rate=0.25, grid=grid)
    profits = {g: net_profit(Y, P, VALUE, g, 40, 0.25)["net_profit"] for g in grid}
    assert profits[t] == max(profits.values())
    assert t == 0.15  # reaching the third churner is still worth the extra contacts


def test_expensive_contacts_raise_the_threshold():
    cheap = best_profit_threshold(Y, P, VALUE, cost=1, save_rate=0.25)
    pricey = best_profit_threshold(Y, P, VALUE, cost=280, save_rate=0.25)
    assert pricey > cheap


def test_profit_curve_length_matches_grid():
    assert len(profit_curve(Y, P, VALUE, grid=np.array([0.1, 0.5]))) == 2


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_wilson_interval_contains_rate_and_narrows_with_n():
    lo_small, hi_small = wilson_interval(5, 20)
    lo_big, hi_big = wilson_interval(250, 1000)
    assert lo_small < 0.25 < hi_small and lo_big < 0.25 < hi_big
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def _scored(n, prob, churners, group="A", split="test"):
    return pd.DataFrame({"churn_probability": [prob] * n, "actual_churn": [1] * churners + [0] * (n - churners),
                         "risk_tier": group, "segment": group, "split": split})


def test_flags_large_gap_and_small_sample():
    big_gap = _scored(400, 0.60, 120, group="Off")        # predicts 60%, actual 30%
    fine = _scored(400, 0.30, 120, group="Good")          # predicts 30%, actual 30%
    tiny = _scored(20, 0.30, 6, group="Tiny")
    table = calibration_table(pd.concat([big_gap, fine, tiny]), "segment").set_index("group")
    assert table.loc["Off", "flag"] == "large gap"
    assert table.loc["Good", "flag"] == "ok"
    assert "small sample" in table.loc["Tiny", "flag"]
    assert table.loc["Off", "gap"] == pytest.approx(0.30)


def test_report_uses_test_split_only():
    df = pd.concat([_scored(200, 0.3, 60, "Low", "test"), _scored(5000, 0.9, 0, "Low", "train")])
    report = calibration_report(df)
    assert set(report["n"]) == {200}
    assert set(report["group_type"]) == {"risk_tier", "segment"}
