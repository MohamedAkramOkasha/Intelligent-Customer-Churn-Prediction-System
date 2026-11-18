import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from churn import config  # noqa: E402

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": "Yes", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.5, "TotalCharges": 199.0,
}
LOW_RISK = {
    "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
    "tenure": 68, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.0, "TotalCharges": 4420.0,
}


@pytest.fixture
def high_risk():
    return dict(HIGH_RISK)


@pytest.fixture
def low_risk():
    return dict(LOW_RISK)


requires_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="trained model not found; run `python -m churn.train`"
)
