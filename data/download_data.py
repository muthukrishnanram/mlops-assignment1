"""Download the UCI Heart Disease (Cleveland) dataset.

Fetches directly from the UCI ML Repository (dataset id=45) via the official
`ucimlrepo` client and writes the raw, unmodified features+target to
data/raw/heart_disease.csv. Run this before src/data/preprocess.py.

Usage:
    python data/download_data.py
"""

import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UCI_DATASET_ID = 45  # Heart Disease (Cleveland)
RAW_DATA_PATH = Path(__file__).parent / "raw" / "heart_disease.csv"


def download(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    logger.info("Fetching UCI Heart Disease dataset (id=%s)...", UCI_DATASET_ID)
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Saved %d rows x %d cols to %s", *df.shape, output_path)
    return df


if __name__ == "__main__":
    download()
