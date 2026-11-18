"""FastAPI service for churn prediction, explanation and retention planning.

Run locally:
    export CHURN_API_KEY=<a long random secret>
    uvicorn api.main:app --reload
Docs:
    http://localhost:8000/docs   (click "Authorize" and paste the key)

Authentication: every endpoint except ``/`` and ``/health`` requires the key from the
``CHURN_API_KEY`` environment variable in an ``X-API-Key`` header. A missing or wrong key
returns 401. If the server has no key configured it fails closed with 503 instead of
silently running unprotected.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow `uvicorn api.main:app` without installing
    sys.path.insert(0, str(ROOT / "src"))

from api.schemas import BatchRequest, BatchResponse, Customer, Prediction  # noqa: E402
from churn import __version__, config  # noqa: E402
from churn.predict import ChurnService, get_service  # noqa: E402

logger = logging.getLogger("churn.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = get_service()
        logger.info("Loaded model: %s", service.model_name)
        if not os.getenv(config.API_KEY_ENV):
            logger.warning("%s is not set: protected endpoints will return 503", config.API_KEY_ENV)
    except FileNotFoundError as exc:  # keep /health up so the problem is visible
        logger.error("%s", exc)
    yield


app = FastAPI(
    title="Intelligent Customer Churn Prediction API",
    description=(
        "Predict churn probability for telecom customers, explain each prediction "
        "with SHAP, segment customers by risk and value, and generate personalised "
        "retention recommendations."
    ),
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description=f"API key (server reads it from ${config.API_KEY_ENV})")


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Reject requests without the correct X-API-Key header.

    The expected key is read from the environment on every request (never hard-coded,
    never logged) and compared in constant time to avoid timing attacks.
    """
    expected = os.getenv(config.API_KEY_ENV)
    if not expected:
        logger.error("%s is not set; refusing request", config.API_KEY_ENV)
        raise HTTPException(status_code=503, detail="API key is not configured on the server")
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "API-Key"},
        )


protected = [Depends(require_api_key)]


def _service() -> ChurnService:
    try:
        return get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _records(customers: list[Customer]) -> list[dict]:
    return [c.model_dump(exclude={"customerID"}) for c in customers]


# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root():
    return {"service": "churn-prediction-api", "version": __version__, "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    """Public liveness check (no API key required)."""
    model_ready = config.MODEL_PATH.exists()
    return {"status": "ok" if model_ready else "degraded", "model_loaded": model_ready}


@app.get("/model/info", tags=["model"], dependencies=protected)
def model_info():
    return _service().info()


@app.get("/model/feature-importance", tags=["model"], dependencies=protected)
def feature_importance(top_n: int = Query(15, ge=1, le=50)):
    if not config.GLOBAL_IMPORTANCE_PATH.exists():
        raise HTTPException(status_code=404, detail="Global importance not computed yet")
    data = json.loads(config.GLOBAL_IMPORTANCE_PATH.read_text())
    return {"units": _service().explainer.output_units, "features": data[:top_n]}


@app.post("/predict", response_model=Prediction, tags=["prediction"], dependencies=protected)
def predict(customer: Customer, explain: bool = Query(True, description="Include SHAP + plan")):
    """Score one customer, with SHAP drivers and a retention plan by default."""
    service = _service()
    try:
        result = service.score(_records([customer]), explain=explain)[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"customerID": customer.customerID, **result}


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"], dependencies=protected)
def predict_batch(request: BatchRequest):
    """Score up to 5,000 customers; explanations are optional (slower)."""
    service = _service()
    try:
        results = service.score(_records(request.customers), explain=request.explain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    predictions = [{"customerID": c.customerID, **r} for c, r in zip(request.customers, results)]
    probs = [r["churn_probability"] for r in results]
    summary = {
        "avg_churn_probability": round(sum(probs) / len(probs), 4),
        "predicted_churners": sum(r["will_churn"] for r in results),
        "risk_tiers": dict(Counter(r["risk_tier"] for r in results)),
        "segments": dict(Counter(r["segment"] for r in results)),
    }
    return {"count": len(results), "summary": summary, "predictions": predictions}
