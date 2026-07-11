"""Clean the raw UCI Heart Disease data: handle missing values and binarize the target.

Reads data/raw/heart_disease.csv (see data/download_data.py) and writes
data/processed/heart_clean.csv. Feature scaling/one-hot encoding for modeling
happens later, in src/features/build_features.py — this stage only removes
missing values and fixes dtypes/target so the cleaned CSV is directly usable
for EDA and as the modeling input.

Usage:
    python src/data/preprocess.py
"""

import logging
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, RAW_TARGET_COL, TARGET_COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "heart_disease.csv"
PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "heart_clean.csv"
)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data/download_data.py first")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_before = df.isna().sum()
    missing_cols = missing_before[missing_before > 0]
    if not missing_cols.empty:
        logger.info("Missing values found:\n%s", missing_cols.to_string())

    # `ca` and `thal` carry a handful of missing values (~2% of rows) in this dataset.
    # With only 303 rows and two known-noisy categorical columns, dropping the affected
    # rows is more defensible than imputing a fabricated "number of vessels" or
    # "thalassemia type" for a clinical feature — those aren't safely interpolable.
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [RAW_TARGET_COL])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d/%d rows with missing values (%.1f%%)", n_dropped, n_before,
                     100 * n_dropped / n_before)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)

    # Binarize: 0 = no disease, 1-4 (increasing severity) -> 1 = disease present.
    df[TARGET_COL] = (df[RAW_TARGET_COL] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET_COL])

    df = df.reset_index(drop=True)
    return df


def main() -> pd.DataFrame:
    df = load_raw()
    df_clean = clean(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Saved cleaned data: %d rows x %d cols -> %s", *df_clean.shape,
                PROCESSED_DATA_PATH)
    logger.info("Class balance:\n%s", df_clean[TARGET_COL].value_counts().to_string())
    return df_clean


if __name__ == "__main__":
    main()
