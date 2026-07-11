"""One-off generator for the architecture diagram used in report/report.md.

Not part of the ML pipeline itself — run manually whenever the architecture
changes enough to need a new picture.

Usage:
    python scripts/generate_architecture_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).resolve().parents[1] / "report" / "figures" / "architecture_diagram.png"

BOX_STYLE = dict(boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.4)
COLORS = {
    "data": "#DDEBF7",
    "train": "#E2EFDA",
    "ci": "#FFF2CC",
    "deploy": "#FCE4D6",
    "monitor": "#EAD1DC",
}

# Grid: fixed column width + horizontal gap, fixed row height + vertical gap.
COL_W, COL_GAP = 2.1, 0.55
ROW_H, ROW_GAP = 1.0, 0.85


def col_x(col: int) -> float:
    return col * (COL_W + COL_GAP)


def row_y(row: int) -> float:
    # row 0 = top
    return (4 - row) * (ROW_H + ROW_GAP)


def box(ax, col, row, text, color, colspan=1, fontsize=9.5):
    x = col_x(col)
    y = row_y(row)
    w = COL_W * colspan + COL_GAP * (colspan - 1)
    h = ROW_H
    patch = FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor="#3a3a3a", **BOX_STYLE,
                            zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)
    return (x, y, w, h)


def arrow_h(ax, box_a, box_b, **kwargs):
    """Horizontal arrow: right edge of a -> left edge of b."""
    ax_, ay_, aw, ah = box_a
    bx_, by_, bw, bh = box_b
    start = (ax_ + aw, ay_ + ah / 2)
    end = (bx_, by_ + bh / 2)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                  color="#555555", linewidth=1.4, zorder=1,
                                  shrinkA=2, shrinkB=2, **kwargs))


def arrow_v(ax, box_a, box_b, x_frac=0.5, **kwargs):
    """Vertical arrow: bottom edge of a -> top edge of b."""
    ax_, ay_, aw, ah = box_a
    bx_, by_, bw, bh = box_b
    start = (ax_ + aw * x_frac, ay_)
    end = (bx_ + bw * x_frac, by_ + bh)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                  color="#555555", linewidth=1.4, zorder=1,
                                  shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.15",
                                  **kwargs))


def main():
    fig, ax = plt.subplots(figsize=(15, 10))

    # Row 0: data acquisition + cleaning
    b_uci = box(ax, 0, 0, "UCI Heart\nDisease dataset", COLORS["data"])
    b_download = box(ax, 1, 0, "download_data.py\n(ucimlrepo)", COLORS["data"])
    b_preprocess = box(ax, 2, 0, "preprocess.py\n(clean, binarize)", COLORS["data"])
    b_train = box(ax, 3, 0, "train.py\n(CV + tuning)", COLORS["train"])
    b_mlflow = box(ax, 4, 0, "MLflow tracking\n(SQLite + mlruns/)", COLORS["train"])

    arrow_h(ax, b_uci, b_download)
    arrow_h(ax, b_download, b_preprocess)
    arrow_h(ax, b_preprocess, b_train)
    arrow_h(ax, b_train, b_mlflow)

    # Row 1: EDA + packaged model (branch off row 0)
    b_eda = box(ax, 2, 1, "eda.py\n(EDA plots)", COLORS["data"])
    b_model = box(ax, 3, 1, "models/final_model\n(sklearn Pipeline)", COLORS["train"])

    arrow_v(ax, b_preprocess, b_eda, x_frac=0.3)
    arrow_v(ax, b_train, b_model, x_frac=0.3)

    # Row 2: CI/CD + containerization
    b_gh = box(ax, 0, 2, "GitHub Actions CI\nlint -> test -> fast-train", COLORS["ci"], colspan=2)
    b_dockerbuild = box(ax, 2, 2, "docker build/run\n+ health-check", COLORS["ci"])
    b_image = box(ax, 3, 2, "heart-disease-api\nDocker image", COLORS["deploy"])

    arrow_v(ax, b_model, b_gh, x_frac=0.85)
    arrow_h(ax, b_gh, b_dockerbuild)
    arrow_h(ax, b_dockerbuild, b_image)

    # Row 3: deployment
    b_minikube = box(ax, 0, 3, "Minikube Deployment\n(2 replicas)", COLORS["deploy"])
    b_svc = box(ax, 1, 3, "Service (LoadBalancer)\n/ Ingress", COLORS["deploy"])
    b_client = box(ax, 2, 3, "Client\ncurl / requests", COLORS["deploy"])

    arrow_v(ax, b_image, b_minikube, x_frac=0.5)
    arrow_h(ax, b_minikube, b_svc)
    arrow_h(ax, b_svc, b_client)

    # Row 4: monitoring
    b_api_metrics = box(ax, 0, 4, "API /metrics, /health\n+ structured logs", COLORS["monitor"])
    b_prom = box(ax, 1, 4, "Prometheus\n(scrapes /metrics)", COLORS["monitor"])
    b_grafana = box(ax, 2, 4, "Grafana dashboard\n(rate, latency, predictions)",
                     COLORS["monitor"])

    arrow_v(ax, b_minikube, b_api_metrics, x_frac=0.5)
    arrow_h(ax, b_api_metrics, b_prom)
    arrow_h(ax, b_prom, b_grafana)

    ax.set_title("Heart Disease Risk Classifier — End-to-End MLOps Architecture",
                 fontsize=14, fontweight="bold", pad=18)
    ax.set_xlim(-0.4, col_x(4) + COL_W + 0.4)
    ax.set_ylim(-0.4, row_y(0) + ROW_H + 0.6)
    ax.axis("off")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
