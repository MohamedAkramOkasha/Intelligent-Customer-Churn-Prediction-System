"""Request/response schemas for the churn API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """A single customer in the IBM Telco schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.1,
                "TotalCharges": 265.3,
            }
        }
    )

    customerID: str | None = Field(None, description="Optional identifier echoed back")
    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=1000)
    TotalCharges: float = Field(..., ge=0)

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def _senior_from_int(cls, v: Any) -> Any:
        # Accept the dataset's native 0/1 encoding too.
        return {0: "No", 1: "Yes", "0": "No", "1": "Yes"}.get(v, v)


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=5000)
    explain: bool = Field(False, description="Include SHAP drivers and retention plans")


class Driver(BaseModel):
    feature: str
    label: str
    value: Any
    shap_value: float
    direction: str


class Recommendation(BaseModel):
    rank: int
    id: str
    title: str
    action: str
    rationale: str
    channel: str
    priority: str
    addresses: list[str]
    driver_impact: float
    est_monthly_cost: float


class RetentionPlan(BaseModel):
    playbook: dict[str, Any]
    recommendations: list[Recommendation]
    economics: dict[str, Any]


class Prediction(BaseModel):
    customerID: str | None = None
    churn_probability: float
    will_churn: bool
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    segment: str
    segment_description: str
    top_drivers: list[Driver] | None = None
    retention_plan: RetentionPlan | None = None


class BatchResponse(BaseModel):
    count: int
    summary: dict[str, Any]
    predictions: list[Prediction]
