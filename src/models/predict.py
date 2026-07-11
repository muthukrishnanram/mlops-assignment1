"""Load the packaged final model and run predictions on it.

Shared by the FastAPI serving app (api/main.py) and tests, so there is exactly
one code path for turning a patient's feature dict into a (prediction,
confidence) pair.
"""

import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS  # noqa: E402

FINAL_MODEL_DIR = ROOT / "models" / "final_model"


def load_model(model_dir: Path = FINAL_MODEL_DIR) -> Pipeline:
    """Loads the raw sklearn Pipeline (not a generic pyfunc wrapper) so callers
    get real .predict_proba() for confidence scores, not just the class label."""
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found — run src/models/train.py first")
    return mlflow.sklearn.load_model(str(model_dir))


def predict_one(model: Pipeline, features: dict) -> tuple[int, float]:
    """Returns (predicted_class, confidence) where confidence is the model's
    predicted probability of the predicted class."""
    row = pd.DataFrame([{col: float(features[col]) for col in FEATURE_COLUMNS}])
    p_disease = float(model.predict_proba(row)[0][1])
    prediction = int(p_disease >= 0.5)
    confidence = p_disease if prediction == 1 else 1 - p_disease
    return prediction, confidence
