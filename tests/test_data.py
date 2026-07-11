"""Tests for src/data/preprocess.py — the missing-value handling and target
binarization that clean() is responsible for."""

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, RAW_TARGET_COL, TARGET_COL
from src.data.preprocess import clean


def _raw_df(n=10) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {col: rng.uniform(1, 100, n) for col in CONTINUOUS_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 3, n).astype(float)
    data[RAW_TARGET_COL] = rng.integers(0, 5, n)  # multi-class 0-4, as in the raw UCI data
    return pd.DataFrame(data)


def test_clean_drops_rows_with_missing_ca_or_thal():
    """This is the exact edge case the real dataset has: a handful of rows with
    NaN in `ca`/`thal`. clean() must drop them rather than silently propagating
    NaNs into a model that can't handle them."""
    df = _raw_df(10)
    df.loc[2, "ca"] = np.nan
    df.loc[5, "thal"] = np.nan

    result = clean(df)

    assert len(result) == 8
    assert result.isna().sum().sum() == 0


def test_clean_binarizes_multiclass_target():
    df = _raw_df(20)
    df[RAW_TARGET_COL] = [0, 1, 2, 3, 4] * 4

    result = clean(df)

    assert set(result[TARGET_COL].unique()) <= {0, 1}
    # original num==0 rows must map to target==0, everything else to target==1
    assert (result[TARGET_COL] == (df.loc[result.index, RAW_TARGET_COL] > 0).astype(int)).all()
    assert RAW_TARGET_COL not in result.columns


def test_clean_casts_categorical_columns_to_int():
    df = _raw_df(10)
    result = clean(df)
    for col in CATEGORICAL_FEATURES:
        assert result[col].dtype == np.int64 or result[col].dtype == int


def test_clean_with_no_missing_values_keeps_all_rows():
    df = _raw_df(15)
    result = clean(df)
    assert len(result) == 15
