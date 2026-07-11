"""FastAPI serving app for the heart disease risk classifier.

Endpoints:
    POST /predict  — patient features in, {prediction, label, confidence} out
    GET  /health   — liveness/readiness probe target
    GET  /metrics  — Prometheus scrape target

Usage:
    uvicorn api.main:app --reload
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.schemas import PatientFeatures, PredictionResponse  # noqa: E402
from src.models.predict import load_model, predict_one  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("heart_disease_api")

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds", ["method", "path"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions served, by predicted class", ["predicted_class"]
)

MODEL_STATE: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from models/final_model ...")
    MODEL_STATE["model"] = load_model()
    logger.info("Model loaded.")
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Heart Disease Risk Classifier API",
    description="Predicts heart disease presence/absence from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration)
    logger.info("%s %s -> %d (%.1fms)", request.method, path, response.status_code, duration * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_STATE["model"] is not None}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    model = MODEL_STATE["model"]
    prediction, confidence = predict_one(model, features.model_dump())
    label = "Disease present" if prediction == 1 else "No disease"

    PREDICTION_COUNT.labels(str(prediction)).inc()
    logger.info("Prediction served: class=%d confidence=%.3f", prediction, confidence)

    return PredictionResponse(prediction=prediction, label=label, confidence=confidence)
