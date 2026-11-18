"""Per-group probability calibration on the held-out test set.

For each risk tier and each customer segment, compare the model's mean predicted churn
probability with the churn rate actually observed, and attach a 95% Wilson confidence
interval to the observed rate so a reader can tell noise from real miscalibration.

Flags
-----
small sample  n < ``config.CALIBRATION_MIN_SAMPLE``: too few customers to trust the rate.
large gap     |predicted - actual| > ``config.CALIBRATION_MAX_GAP`` *and* the predicted
              rate falls outside the actual rate's 95% interval.

Usage:  python -m churn.calibration     (writes reports/calibration_report.csv)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config

Z95 = 1.959964


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (robust for small n / rare events)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_table(scored: pd.DataFrame, by: str) -> pd.DataFrame:
    """Calibration of ``churn_probability`` against ``actual_churn`` within each level of ``by``."""
    rows = []
    for group, g in scored.groupby(by, sort=False):
        n, churned = len(g), int(g["actual_churn"].sum())
        predicted, actual = float(g["churn_probability"].mean()), churned / len(g)
        lo, hi = wilson_interval(churned, n)
        gap = predicted - actual
        flags = []
        if n < config.CALIBRATION_MIN_SAMPLE:
            flags.append("small sample")
        if abs(gap) > config.CALIBRATION_MAX_GAP and not (lo <= predicted <= hi):
            flags.append("large gap")
        rows.append({
            "group_type": by, "group": group, "n": n,
            "predicted_rate": round(predicted, 4), "actual_rate": round(actual, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4), "gap": round(gap, 4),
            "flag": ", ".join(flags) or "ok",
        })
    return pd.DataFrame(rows)


def calibration_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Risk tiers and segments, test-set customers only (never seen in training)."""
    test = scored[scored["split"] == "test"]
    tier_order = [t for t, _ in config.RISK_TIERS]
    tiers = calibration_table(test, "risk_tier")
    tiers = tiers.set_index("group").reindex([t for t in tier_order if t in set(tiers["group"])]).reset_index()
    segments = calibration_table(test, "segment").sort_values("predicted_rate", ascending=False)
    return pd.concat([tiers, segments], ignore_index=True)[
        ["group_type", "group", "n", "predicted_rate", "actual_rate", "ci_low", "ci_high", "gap", "flag"]
    ]


def main() -> None:
    report = calibration_report(pd.read_csv(config.SCORED_PATH))
    config.CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.CALIBRATION_PATH, index=False)
    with pd.option_context("display.width", 140):
        print(report.to_string(index=False))
    print(f"\nwritten to {config.CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
