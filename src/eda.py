"""Exploratory Data Analysis for the Heart Disease dataset.

Reads data/raw/heart_disease.csv (see data/download_data.py) for the missing-
value analysis — the cleaned CSV has none left, by definition, so that
analysis has to run on the pre-cleaning data — and
data/processed/heart_clean.csv (see src/data/preprocess.py) for everything
else, writing professional EDA visualizations to report/figures/: missing
values, per-feature histograms, a correlation heatmap, and a class-balance
bar chart.

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

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_disease.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart_clean.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"

TARGET_LABELS = {0: "No disease", 1: "Disease present"}
HUE_ORDER = ["No disease", "Disease present"]
HUE_PALETTE = {"No disease": "#4C72B0", "Disease present": "#C44E52"}


def plot_missing_values(raw_df: pd.DataFrame, out_dir: Path) -> None:
    """Missing-value counts per column in the *raw*, pre-cleaning data — the
    cleaned CSV has none left by construction, so this has to run on the raw
    download to be meaningful."""
    missing = raw_df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#C44E52" if v > 0 else "#B0B0B0" for v in missing.values]
    sns.barplot(
        x=missing.index, y=missing.values, hue=missing.index, palette=colors, legend=False, ax=ax
    )
    for i, v in enumerate(missing.values):
        if v > 0:
            ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    ax.set_title(f"Missing Values per Column (raw data, n={len(raw_df)} rows)")
    ax.set_ylabel("Missing count")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df[TARGET_COL].map(TARGET_LABELS).value_counts().reindex(HUE_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        x=counts.index, y=counts.values, hue=counts.index, palette=HUE_PALETTE, legend=False, ax=ax
    )
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
        sns.histplot(
            data=df,
            x=col,
            hue=df[TARGET_COL].map(TARGET_LABELS),
            hue_order=HUE_ORDER,
            kde=True,
            palette=HUE_PALETTE,
            element="step",
            ax=ax,
        )
        ax.set_title(col)
    for ax in axes.flat[len(CONTINUOUS_FEATURES) :]:
        ax.axis("off")
    fig.suptitle("Distributions of Continuous Features by Diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_histograms.png", dpi=150)
    plt.close(fig)


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(
            data=df,
            x=col,
            hue=df[TARGET_COL].map(TARGET_LABELS),
            hue_order=HUE_ORDER,
            palette=HUE_PALETTE,
            ax=ax,
        )
        ax.set_title(col)
        ax.legend(fontsize=7, title=None)
    fig.suptitle("Categorical Feature Counts by Diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "categorical_counts.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_feature_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    """Pairwise scatter relationships between continuous features (diagonal:
    per-feature KDE), colored by diagnosis — complements the correlation
    heatmap's numeric summary with the actual pairwise shapes."""
    plot_df = df[CONTINUOUS_FEATURES + [TARGET_COL]].copy()
    plot_df["Diagnosis"] = plot_df[TARGET_COL].map(TARGET_LABELS)
    plot_df = plot_df.drop(columns=[TARGET_COL])

    grid = sns.pairplot(
        plot_df,
        hue="Diagnosis",
        hue_order=HUE_ORDER,
        palette=HUE_PALETTE,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 25},
        height=2.2,
    )
    grid.figure.suptitle(
        "Feature Relationships: Continuous Features by Diagnosis", y=1.02, fontsize=14
    )
    grid.savefig(out_dir / "feature_relationships.png", dpi=150)
    plt.close(grid.figure)


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"{RAW_DATA_PATH} not found — run data/download_data.py first")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found — run src/data/preprocess.py first"
        )
    raw_df = pd.read_csv(RAW_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_missing_values(raw_df, FIGURES_DIR)
    plot_class_balance(df, FIGURES_DIR)
    plot_continuous_histograms(df, FIGURES_DIR)
    plot_categorical_counts(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)
    plot_feature_relationships(df, FIGURES_DIR)

    logger.info("Wrote EDA figures to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
