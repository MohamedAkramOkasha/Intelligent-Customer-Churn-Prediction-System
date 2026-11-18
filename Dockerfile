# One image, two services (API and dashboard): the command is chosen in docker-compose.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    CHURN_HOME=/app \
    MPLBACKEND=Agg

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY data/raw ./data/raw
COPY models ./models
COPY data/processed ./data/processed
COPY reports ./reports
COPY .streamlit ./.streamlit

# Train inside the image if no model artifacts were shipped (e.g. a fresh clone).
RUN if [ ! -f models/churn_model.joblib ]; then python -m churn.train; fi

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
