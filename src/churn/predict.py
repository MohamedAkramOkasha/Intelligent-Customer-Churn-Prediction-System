"""Inference service shared by the FastAPI backend and the Streamlit dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn import config
from churn.data import clean
from churn.explain import ChurnExplainer
from churn.recommend import recommend
from churn.segment import risk_tier, value_segment


class ChurnService:
    """Load the trained bundle once and serve predictions + explanations."""

    def __init__(self, model_path: Path = config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m churn.train` first."
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold: float = bundle["threshold"]
        self.high_value_threshold: float = bundle["high_value_threshold"]
        self.model_name: str = bundle["model_name"]
        self.version: str = bundle.get("version", "unknown")
        self.explainer = ChurnExplainer(self.pipeline, bundle["background"])
        self.metadata: dict[str, Any] = (
            json.loads(config.METADATA_PATH.read_text()) if config.METADATA_PATH.exists() else {}
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def prepare(records: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Clean raw customer records into the model's input schema."""
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
        missing = set(config.RAW_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        return clean(df, drop_duplicates=False)[config.RAW_FEATURES].reset_index(drop=True)

    def predict_proba(self, records) -> pd.Series:
        X = self.prepare(records)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], name="churn_probability")

    def score(
        self, records, explain: bool = True, top_n: int = 6
    ) -> list[dict[str, Any]]:
        """Full scoring: probability, tier, segment, drivers and retention plan."""
        X = self.prepare(records)
        proba = self.pipeline.predict_proba(X)[:, 1]

        drivers = self.explainer.explain(X, top_n=top_n) if explain else [None] * len(X)
        agg = self.explainer.aggregated_shap(X) if explain else None
        engineered = self.pipeline.named_steps["features"].transform(X).reset_index(drop=True)

        results = []
        for i, p in enumerate(proba):
            tier = risk_tier(float(p))
            monthly = float(X.at[i, "MonthlyCharges"])
            item: dict[str, Any] = {
                "churn_probability": round(float(p), 4),
                "will_churn": bool(p >= self.threshold),
                "risk_tier": tier,
                "segment": value_segment(tier, monthly, self.high_value_threshold),
                "segment_description": config.SEGMENT_DESCRIPTIONS[
                    value_segment(tier, monthly, self.high_value_threshold)
                ],
            }
            if explain:
                shap_map = agg.iloc[i].to_dict()
                item["top_drivers"] = drivers[i]
                item["retention_plan"] = recommend(
                    engineered.iloc[i].to_dict(), shap_map, tier, float(p)
                )
            results.append(item)
        return results

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "decision_threshold": round(self.threshold, 4),
            "high_value_threshold": self.high_value_threshold,
            "shap_output_units": self.explainer.output_units,
            "test_metrics": self.metadata.get("test_metrics", {}),
            "cv_roc_auc": self.metadata.get("cv_roc_auc"),
            "trained_at": self.metadata.get("trained_at"),
        }


@lru_cache(maxsize=1)
def get_service() -> ChurnService:
    return ChurnService()
