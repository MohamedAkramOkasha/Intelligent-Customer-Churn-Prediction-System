"""Central configuration: paths, feature schema, and business rules."""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# CHURN_HOME lets Docker / CI point at a different project root.
PROJECT_ROOT = Path(os.getenv("CHURN_HOME", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
SCORED_PATH = PROCESSED_DIR / "scored_customers.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
GLOBAL_IMPORTANCE_PATH = MODELS_DIR / "global_shap_importance.json"
CURVES_PATH = MODELS_DIR / "evaluation_curves.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
ID_COL = "customerID"
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

# Features produced by churn.features.FeatureEngineer
ENGINEERED_NUMERIC = [
    "num_services",
    "num_protection_services",
    "avg_monthly_spend",
    "charge_to_tenure_ratio",
    "monthly_charge_delta",
]
ENGINEERED_CATEGORICAL = ["tenure_group", "is_auto_payment", "has_family"]

MODEL_NUMERIC = RAW_NUMERIC + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL

# Human-readable names used in explanations, the API and the dashboard.
FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "num_services": "Number of services",
    "num_protection_services": "Protection/support add-ons",
    "avg_monthly_spend": "Average historical monthly spend",
    "charge_to_tenure_ratio": "Charges relative to tenure",
    "monthly_charge_delta": "Current vs. historical price gap",
    "tenure_group": "Tenure band",
    "is_auto_payment": "Automatic payment",
    "has_family": "Has partner or dependents",
}

# --------------------------------------------------------------------------- #
# Risk segmentation
# --------------------------------------------------------------------------- #
# Upper bounds (exclusive) on churn probability for each tier.
RISK_TIERS = [
    ("Low", 0.25),
    ("Medium", 0.50),
    ("High", 0.75),
    ("Critical", 1.01),
]
HIGH_RISK_TIERS = {"High", "Critical"}

# Customers whose MonthlyCharges are >= this value are "high value".
# Set to the training-set median at train time and stored in model metadata;
# this is only a fallback.
DEFAULT_HIGH_VALUE_THRESHOLD = 70.0

VALUE_SEGMENTS = {
    (True, True): "Priority Rescue",
    (True, False): "Cost-Efficient Save",
    (False, True): "Loyal VIP - Nurture",
    (False, False): "Stable - Grow",
}
SEGMENT_DESCRIPTIONS = {
    "Priority Rescue": "High churn risk and high revenue. Act within 48 hours with a "
    "personal outreach and the strongest offers.",
    "Cost-Efficient Save": "High churn risk, lower revenue. Use low-cost, automated "
    "retention plays (contract migration, autopay incentives).",
    "Loyal VIP - Nurture": "Low risk, high revenue. Protect the relationship with "
    "loyalty perks and proactive service.",
    "Stable - Grow": "Low risk, lower revenue. Candidates for upsell and "
    "cross-sell campaigns.",
}
