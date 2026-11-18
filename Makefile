.PHONY: install train train-fast test lint api dashboard eda docker-up docker-down

export PYTHONPATH := src

install:
	pip install -r requirements-dev.txt

train:
	python -m churn.train

train-fast:
	python -m churn.train --fast

test:
	pytest

lint:
	ruff check src api dashboard tests

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

eda:
	python scripts/build_eda_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
