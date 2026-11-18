"""Tests that need the trained model artifacts."""

import numpy as np
import pytest
from conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def service():
    from churn.predict import ChurnService

    return ChurnService()


TEST_KEY = "test-key-0123456789"


@pytest.fixture(scope="module")
def client():
    """Authenticated client: the server key is set and every request carries it."""
    from fastapi.testclient import TestClient

    from api.main import app

    mp = pytest.MonkeyPatch()
    mp.setenv("CHURN_API_KEY", TEST_KEY)
    with TestClient(app, headers={"X-API-Key": TEST_KEY}) as c:
        yield c
    mp.undo()


@pytest.fixture
def anon_client(monkeypatch):
    """Client that sends no API key (server key still configured)."""
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("CHURN_API_KEY", TEST_KEY)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
PROTECTED = [("get", "/model/info"), ("get", "/model/feature-importance"),
             ("post", "/predict"), ("post", "/predict/batch")]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_protected_endpoints_require_key(anon_client, method, path, high_risk):
    body = {"customers": [high_risk]} if path == "/predict/batch" else high_risk
    r = anon_client.request(method.upper(), path, json=body if method == "post" else None)
    assert r.status_code == 401
    assert r.json()["detail"] == "Missing or invalid API key"


def test_wrong_key_rejected(anon_client):
    r = anon_client.get("/model/info", headers={"X-API-Key": "not-the-key"})
    assert r.status_code == 401


def test_health_and_root_are_public(anon_client):
    assert anon_client.get("/health").status_code == 200
    assert anon_client.get("/").status_code == 200


def test_unconfigured_server_fails_closed(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.delenv("CHURN_API_KEY", raising=False)
    r = TestClient(app).get("/model/info", headers={"X-API-Key": "anything"})
    assert r.status_code == 503


def test_openapi_declares_api_key_scheme(anon_client):
    schemes = anon_client.get("/openapi.json").json()["components"]["securitySchemes"]
    assert any(s.get("type") == "apiKey" and s.get("name") == "X-API-Key" for s in schemes.values())


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
