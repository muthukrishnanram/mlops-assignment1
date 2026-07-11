"""Exploratory Data Analysis for the cleaned Heart Disease dataset.

Reads data/processed/heart_clean.csv (see src/data/preprocess.py) and writes
professional EDA visualizations to report/figures/: per-feature histograms,
a correlation heatmap, and a class-balance bar chart.

Usage:
    python src/eda.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no X server in CI / this WSL2 shell

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, TARGET_COL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

PROCESSED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart_clean.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"

TARGET_LABELS = {0: "No disease", 1: "Disease present"}
HUE_ORDER = ["No disease", "Disease present"]
HUE_PALETTE = {"No disease": "#4C72B0", "Disease present": "#C44E52"}


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df[TARGET_COL].map(TARGET_LABELS).value_counts().reindex(HUE_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=HUE_PALETTE,
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Balance: Heart Disease Diagnosis")
    ax.set_ylabel("Number of patients")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_continuous_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, CONTINUOUS_FEATURES):
        sns.histplot(data=df, x=col, hue=df[TARGET_COL].map(TARGET_LABELS), hue_order=HUE_ORDER,
                     kde=True, palette=HUE_PALETTE, element="step", ax=ax)
        ax.set_title(col)
    for ax in axes.flat[len(CONTINUOUS_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Distributions of Continuous Features by Diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_histograms.png", dpi=150)
    plt.close(fig)


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=df[TARGET_COL].map(TARGET_LABELS), hue_order=HUE_ORDER,
                      palette=HUE_PALETTE, ax=ax)
        ax.set_title(col)
        ax.legend(fontsize=7, title=None)
    fig.suptitle("Categorical Feature Counts by Diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "categorical_counts.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_DATA_PATH} not found — run src/data/preprocess.py first")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_class_balance(df, FIGURES_DIR)
    plot_continuous_histograms(df, FIGURES_DIR)
    plot_categorical_counts(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)

    logger.info("Wrote EDA figures to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
