"""Tests for src/models/evaluate.py and the training pipeline construction in
src/models/train.py. Uses tiny synthetic data and --fast-equivalent settings
throughout so the suite stays fast — full hyperparameter search is exercised
manually via `python src/models/train.py`, not in the test suite."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, TARGET_COL
from src.features.build_features import build_preprocessor
from src.models.evaluate import compute_metrics
from src.models.train import get_model_configs


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_all_wrong_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 0.0
    assert metrics["recall"] == 0.0


@pytest.mark.parametrize("fast", [True, False])
def test_get_model_configs_returns_all_three_models(fast):
    configs = get_model_configs(fast=fast)
    assert set(configs.keys()) == {"logistic_regression", "random_forest", "xgboost"}
    for _estimator, param_grid, cv in configs.values():
        assert isinstance(param_grid, dict) and len(param_grid) > 0
        assert cv.get_n_splits() == (2 if fast else 5)


def test_pipeline_fits_and_predicts_proba_on_synthetic_data(synthetic_clean_df):
    """End-to-end sanity check: preprocessor + classifier fit/predict without
    error, and predict_proba returns valid probabilities."""
    from sklearn.linear_model import LogisticRegression

    X = synthetic_clean_df[FEATURE_COLUMNS]
    y = synthetic_clean_df[TARGET_COL]

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()
