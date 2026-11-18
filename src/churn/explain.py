"""SHAP explainability for any model in the churn pipeline.

One-hot encoded columns are aggregated back to their source feature, so an
explanation reads "Contract type = Month-to-month (+0.82)" rather than listing
three separate dummy columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import transformed_to_source_map

TREE_MODEL_NAMES = (
    "XGBClassifier",
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
)


class ChurnExplainer:
    """Wrap a fitted churn ``Pipeline`` with a SHAP explainer."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, max_background: int = 200):
        self.feature_step = pipeline.named_steps["features"]
        self.preprocessor = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.transformed_names = list(self.preprocessor.get_feature_names_out())
        self.sources = transformed_to_source_map(self.preprocessor)
        self.source_features = list(dict.fromkeys(self.sources))

        bg = background.sample(min(max_background, len(background)), random_state=config.RANDOM_STATE)
        bg_t = self._transform(bg)

        model_name = type(self.model).__name__
        if isinstance(self.model, LogisticRegression):
            masker = shap.maskers.Independent(bg_t, max_samples=len(bg_t))
            self.explainer = shap.LinearExplainer(self.model, masker)
            self.output_units = "log-odds"
        elif model_name in TREE_MODEL_NAMES:
            self.explainer = shap.TreeExplainer(self.model)
            self.output_units = "log-odds" if model_name == "XGBClassifier" else "probability"
        else:  # model-agnostic fallback
            self.explainer = shap.Explainer(
                lambda a: self.model.predict_proba(a)[:, 1], bg_t, algorithm="permutation"
            )
            self.output_units = "probability"

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(self.feature_step.transform(X)), dtype=float)

    @property
    def expected_value(self) -> float:
        ev = getattr(self.explainer, "expected_value", 0.0)
        ev = np.atleast_1d(np.asarray(ev, dtype=float))
        return float(ev[-1])  # positive class for binary tree models

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Raw SHAP values for the positive (churn) class: shape (n, n_transformed)."""
        Xt = self._transform(X)
        if isinstance(self.explainer, (shap.LinearExplainer, shap.TreeExplainer)):
            values = self.explainer.shap_values(Xt)
        else:
            values = self.explainer(Xt).values
        if isinstance(values, list):  # older SHAP: one array per class
            values = values[-1]
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, -1]
        return values

    def aggregated_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per source feature: shape (n, n_source_features)."""
        values = self.shap_values(X)
        frame = pd.DataFrame(values, columns=self.sources)
        return frame.T.groupby(level=0, sort=False).sum().T[self.source_features]

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per source feature, sorted descending."""
        agg = self.aggregated_shap(X)
        engineered = self.feature_step.transform(X).reset_index(drop=True)
        imp = agg.abs().mean().sort_values(ascending=False)

        # For numeric features: does a higher value push churn up (+) or down (-)?
        direction = []
        for f in imp.index:
            if f in config.MODEL_NUMERIC and agg[f].std() > 0:
                values = engineered[f].to_numpy(dtype=float)
                direction.append(round(float(np.corrcoef(values, agg[f])[0, 1]), 3))
            else:
                direction.append(None)

        return pd.DataFrame(
            {
                "feature": imp.index,
                "label": [config.FEATURE_LABELS.get(f, f) for f in imp.index],
                "mean_abs_shap": imp.values,
                "value_shap_correlation": direction,
            }
        ).reset_index(drop=True)

    def explain(self, X: pd.DataFrame, top_n: int = 8) -> list[list[dict[str, Any]]]:
        """Per-customer top drivers, ordered by absolute contribution."""
        agg = self.aggregated_shap(X)
        engineered = self.feature_step.transform(X).reset_index(drop=True)
        results = []
        for i in range(len(agg)):
            row = agg.iloc[i]
            order = row.abs().sort_values(ascending=False).index[:top_n]
            drivers = []
            for feat in order:
                raw_value = engineered.at[i, feat]
                if isinstance(raw_value, (np.floating, float)):
                    raw_value = round(float(raw_value), 2)
                elif isinstance(raw_value, np.integer):
                    raw_value = int(raw_value)
                contribution = float(row[feat])
                drivers.append(
                    {
                        "feature": feat,
                        "label": config.FEATURE_LABELS.get(feat, feat),
                        "value": raw_value,
                        "shap_value": round(contribution, 4),
                        "direction": "increases churn risk"
                        if contribution > 0
                        else "decreases churn risk",
                    }
                )
            results.append(drivers)
        return results
