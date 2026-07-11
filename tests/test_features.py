"""Tests for src/features/build_features.py's ColumnTransformer."""

import numpy as np

from src.config import CONTINUOUS_FEATURES, FEATURE_COLUMNS
from src.features.build_features import build_preprocessor


def test_preprocessor_output_has_no_missing_values(synthetic_clean_df):
    X = synthetic_clean_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert not np.isnan(transformed).any()


def test_preprocessor_scales_continuous_features(synthetic_clean_df):
    """After StandardScaler, each continuous column should have ~0 mean and ~1 std."""
    X = synthetic_clean_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()

    preprocessor.fit(X)
    scaled_continuous = preprocessor.named_transformers_["num"].transform(X[CONTINUOUS_FEATURES])

    assert np.allclose(scaled_continuous.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled_continuous.std(axis=0), 1, atol=1e-8)


def test_preprocessor_one_hot_encodes_categoricals(synthetic_clean_df):
    X = synthetic_clean_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    # output width = continuous cols + one-hot expansion of categoricals
    cat_encoder = preprocessor.named_transformers_["cat"]
    expected_cat_width = sum(len(cats) if len(cats) > 2 else 1 for cats in cat_encoder.categories_)
    assert transformed.shape[1] == len(CONTINUOUS_FEATURES) + expected_cat_width


def test_preprocessor_handles_unseen_category_at_transform_time(synthetic_clean_df):
    """handle_unknown='ignore' must not raise when serving-time data contains a
    category value never seen during training (e.g. a rare `thal` code)."""
    X = synthetic_clean_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    X_new = X.copy()
    X_new.loc[0, "thal"] = 999  # unseen category

    transformed = preprocessor.transform(X_new)
    assert transformed.shape[0] == len(X_new)
