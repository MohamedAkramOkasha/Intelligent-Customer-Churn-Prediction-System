"""Tests that need the trained model artifacts."""

import numpy as np
import pytest
from conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def service():
    from churn.predict import ChurnService

    return ChurnService()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_model_ranks_risky_customer_higher(service, high_risk, low_risk):
    p = service.predict_proba([high_risk, low_risk])
    assert p[0] > 0.6 > 0.1 > p[1]


def test_score_contains_explanation(service, high_risk):
    r = service.score([high_risk])[0]
    assert r["risk_tier"] in {"High", "Critical"}
    assert r["segment"] == "Priority Rescue"
    assert len(r["top_drivers"]) == 6
    assert r["retention_plan"]["recommendations"]
    assert any(d["feature"] == "Contract" for d in r["top_drivers"])


def test_shap_values_are_additive(service, high_risk, low_risk):
    exp = service.explainer
    X = service.prepare([high_risk, low_risk])
    total = exp.shap_values(X).sum(axis=1) + exp.expected_value
    p = service.pipeline.predict_proba(X)[:, 1]
    ref = np.log(p / (1 - p)) if exp.output_units == "log-odds" else p
    np.testing.assert_allclose(total, ref, atol=1e-4)


def test_prepare_rejects_missing_fields(service):
    with pytest.raises(ValueError, match="Missing required fields"):
        service.prepare([{"tenure": 5}])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model/info").json()
    assert 0.8 < body["test_metrics"]["roc_auc"] <= 1
    assert 0 < body["decision_threshold"] < 1


def test_predict_endpoint(client, high_risk):
    r = client.post("/predict", json={**high_risk, "customerID": "TEST-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["customerID"] == "TEST-1"
    assert 0 <= body["churn_probability"] <= 1
    assert body["top_drivers"] and body["retention_plan"]["recommendations"]
    assert "X-Process-Time-ms" in r.headers


def test_predict_accepts_numeric_senior_flag(client, low_risk):
    r = client.post("/predict?explain=false", json={**low_risk, "SeniorCitizen": 0})
    assert r.status_code == 200
    assert r.json()["top_drivers"] is None


def test_predict_validation_error(client, high_risk):
    r = client.post("/predict", json={**high_risk, "Contract": "Lifetime"})
    assert r.status_code == 422


def test_batch_endpoint(client, high_risk, low_risk):
    r = client.post("/predict/batch", json={"customers": [high_risk, low_risk]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert sum(body["summary"]["risk_tiers"].values()) == 2


def test_feature_importance_endpoint(client):
    body = client.get("/model/feature-importance?top_n=5").json()
    assert len(body["features"]) == 5
    assert body["features"][0]["mean_abs_shap"] >= body["features"][-1]["mean_abs_shap"]
