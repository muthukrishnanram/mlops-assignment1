.PHONY: setup data eda train test lint api docker-build docker-run clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -q -r requirements-dev.txt
	$(MAKE) data
	$(MAKE) eda
	$(MAKE) train

data:
	$(PY) data/download_data.py
	$(PY) src/data/preprocess.py

eda:
	$(PY) src/eda.py

train:
	$(PY) src/models/train.py

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .

api:
	$(VENV)/bin/uvicorn api.main:app --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

clean:
	rm -rf $(VENV) mlruns .pytest_cache .ruff_cache **/__pycache__
