"""Shared pytest fixtures: a small synthetic heart-disease-shaped dataset so
tests never depend on network access or the real 303-row dataset."""

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, TARGET_COL


@pytest.fixture
def synthetic_clean_df() -> pd.DataFrame:
    """Mimics data/processed/heart_clean.csv's schema: already cleaned (no
    missing values), categorical columns as ints, binary target."""
    rng = np.random.default_rng(seed=0)
    n = 40
    data = {
        "age": rng.integers(29, 77, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "thalach": rng.integers(71, 202, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }
    assert set(data.keys()) == set(CONTINUOUS_FEATURES + CATEGORICAL_FEATURES)
    df = pd.DataFrame(data)
    df[TARGET_COL] = rng.integers(0, 2, n)
    return df
