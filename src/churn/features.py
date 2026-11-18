"""Feature engineering and preprocessing.

Everything lives inside a scikit-learn ``Pipeline`` so the exact same
transformations run at training time, in the API and in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49m+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive behavioural features from the raw Telco columns.

    New features
    ------------
    num_services            Count of subscribed services (phone, internet & add-ons).
    num_protection_services Security/backup/device-protection/tech-support add-ons.
    avg_monthly_spend       TotalCharges / tenure: historical average bill.
    charge_to_tenure_ratio  MonthlyCharges / (tenure + 1): expensive *and* new.
    monthly_charge_delta    Current bill minus historical average (price increases).
    tenure_group            Lifecycle band (0-12m, 13-24m, 25-48m, 49m+).
    is_auto_payment         Bank transfer / credit card autopay vs. manual payment.
    has_family              Partner or dependents on the account.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D102 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102 - sklearn API
        df = X.copy()

        df["SeniorCitizen"] = (
            df["SeniorCitizen"]
            .astype(str)
            .map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
            .fillna("No")
        )
        for col in config.RAW_NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        has_internet = (df["InternetService"] != "No").astype(int)
        df["num_services"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1) + has_internet
        df["num_protection_services"] = (df[config.PROTECTION_COLS] == "Yes").sum(axis=1)

        tenure_safe = df["tenure"].clip(lower=1)
        df["avg_monthly_spend"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / tenure_safe, df["MonthlyCharges"]
        )
        df["charge_to_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        df["monthly_charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

        df["tenure_group"] = pd.cut(
            df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype(str)
        df["is_auto_payment"] = np.where(
            df["PaymentMethod"].astype(str).str.contains("automatic", case=False), "Yes", "No"
        )
        df["has_family"] = np.where(
            (df["Partner"] == "Yes") | (df["Dependents"] == "Yes"), "Yes", "No"
        )

        for col in config.MODEL_CATEGORICAL:
            df[col] = df[col].astype(str)

        return df[config.MODEL_CATEGORICAL + config.MODEL_NUMERIC]

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.array(config.MODEL_CATEGORICAL + config.MODEL_NUMERIC, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.MODEL_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.MODEL_CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=True,
    )


def transformed_to_source_map(preprocessor: ColumnTransformer) -> list[str]:
    """For each transformed column, return the model-input feature it came from.

    Used to aggregate one-hot SHAP values back to a single readable feature
    (e.g. ``cat__Contract_Month-to-month`` -> ``Contract``).
    """
    sources: list[str] = list(config.MODEL_NUMERIC)
    encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    for feature, categories in zip(config.MODEL_CATEGORICAL, encoder.categories_):
        sources.extend([feature] * len(categories))
    return sources
