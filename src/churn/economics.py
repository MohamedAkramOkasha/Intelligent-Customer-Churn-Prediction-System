"""Retention-campaign economics and the profit-maximising decision threshold.

Assumptions (identical to the dashboard's Retention planner):

* Every customer at or above the threshold is contacted, at ``cost`` USD each.
* A contacted customer who would really have churned is retained with probability
  ``save_rate``; retaining them preserves ``MonthlyCharges x 12`` of revenue.
* Contacting a customer who would have stayed anyway costs money and gains nothing.

So for a threshold ``t`` evaluated on customers with known outcomes ``y``::

    net_profit(t) = save_rate * sum(y_i * value_i  for p_i >= t)  -  cost * #{p_i >= t}

The F1 threshold treats every churner (and every false alarm) as equally important.
The profit threshold weights them by money: it contacts more customers when saves are
valuable and contacts cheap, and fewer when contacts are expensive.
"""

from __future__ import annotations

import numpy as np

from churn import config

THRESHOLD_GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def customer_value(monthly_charges: np.ndarray) -> np.ndarray:
    """Revenue preserved by retaining a customer over the planning horizon."""
    return np.asarray(monthly_charges, dtype=float) * config.REVENUE_HORIZON_MONTHS


def net_profit(
    y_true: np.ndarray,
    proba: np.ndarray,
    value: np.ndarray,
    threshold: float,
    cost: float = config.DEFAULT_COST_PER_CUSTOMER,
    save_rate: float = config.DEFAULT_SAVE_RATE,
) -> dict[str, float]:
    """Campaign outcome if everyone with ``proba >= threshold`` is contacted."""
    y_true, proba, value = (np.asarray(a, dtype=float) for a in (y_true, proba, value))
    flagged = proba >= threshold
    contacted = int(flagged.sum())
    churners_reached = int((y_true * flagged).sum())
    revenue_saved = float(save_rate * (y_true * value)[flagged].sum())
    spend = float(cost * contacted)
    return {
        "threshold": float(threshold),
        "contacted": contacted,
        "churners_reached": churners_reached,
        "revenue_saved": revenue_saved,
        "cost": spend,
        "net_profit": revenue_saved - spend,
    }


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    value: np.ndarray,
    cost: float = config.DEFAULT_COST_PER_CUSTOMER,
    save_rate: float = config.DEFAULT_SAVE_RATE,
    grid: np.ndarray = THRESHOLD_GRID,
) -> list[dict[str, float]]:
    """``net_profit`` evaluated over a grid of thresholds."""
    return [net_profit(y_true, proba, value, t, cost, save_rate) for t in grid]


def best_profit_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    value: np.ndarray,
    cost: float = config.DEFAULT_COST_PER_CUSTOMER,
    save_rate: float = config.DEFAULT_SAVE_RATE,
    grid: np.ndarray = THRESHOLD_GRID,
) -> float:
    """Threshold on ``grid`` with the highest net profit (ties -> the higher threshold,
    i.e. the smaller, cheaper campaign)."""
    curve = profit_curve(y_true, proba, value, cost, save_rate, grid)
    profits = np.array([c["net_profit"] for c in curve])
    best = np.flatnonzero(profits == profits.max())[-1]
    return float(grid[best])
