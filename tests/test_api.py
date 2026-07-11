"""Tests for the FastAPI serving app. Uses TestClient's context manager so the
lifespan handler actually runs (loading the real packaged model) — these tests
therefore require models/final_model/ to exist (run src/models/train.py first,
same as the rest of the suite)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("No disease", "Disease present")
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_out_of_range_field(client):
    bad_patient = {**VALID_PATIENT, "sex": 5}  # sex must be 0 or 1
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID_PATIENT)  # generate at least one data point
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text
    assert "http_requests_total" in response.text
