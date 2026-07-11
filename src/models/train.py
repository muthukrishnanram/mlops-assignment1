"""Train, cross-validate, and MLflow-track Logistic Regression + Random Forest
classifiers for heart disease risk, then export the better one as the final
packaged model.

Usage:
    python src/models/train.py            # full tuning run (reported results)
    python src/models/train.py --fast     # small grid, 2-fold CV, used only in CI
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COL  # noqa: E402
from src.features.build_features import build_preprocessor  # noqa: E402
from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MLFLOW_DB_PATH = ROOT / "mlflow.db"
FINAL_MODEL_DIR = ROOT / "models" / "final_model"
FIGURES_DIR = ROOT / "report" / "figures"
METRICS_SUMMARY_PATH = ROOT / "report" / "metrics_summary.json"
EXPERIMENT_NAME = "heart-disease-classification"


def get_model_configs(fast: bool) -> dict:
    """Estimator + hyperparameter grid + CV folds per model.

    --fast mode (used only by CI's smoke-test step) shrinks the grid to a single
    combination and drops CV to 2 folds purely to keep the pipeline job under a
    minute; the real, reported tuning run always uses the full grid below.
    """
    cv_folds = 2 if fast else 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    if fast:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                {"classifier__C": [1.0]},
                cv,
            ),
            "random_forest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {"classifier__n_estimators": [50], "classifier__max_depth": [5]},
                cv,
            ),
        }

    return {
        "logistic_regression": (
            # l2 penalty via lbfgs is already the default in this sklearn version;
            # only C (inverse regularization strength) is swept.
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            cv,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
            cv,
        ),
    }


def load_data():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found — run src/data/preprocess.py first"
        )
    df = pd.read_csv(PROCESSED_DATA_PATH)
    # float64 (not int) so MLflow's inferred schema tolerates missing values at
    # inference time and doesn't warn about integer columns being unable to hold NaN.
    X = df[FEATURE_COLUMNS].astype("float64")
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def train_and_log_model(name: str, estimator, param_grid: dict, cv, X_train, y_train,
                         X_test, y_test) -> dict:
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring=scoring, refit="roc_auc", n_jobs=-1
    )

    with mlflow.start_run(run_name=name):
        logger.info("Training %s (cv folds=%d)...", name, cv.get_n_splits())
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_idx = search.best_index_

        mlflow.log_param("model_type", name)
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_param("cv_folds", cv.get_n_splits())

        for metric_name in scoring:
            mean_key = f"mean_test_{metric_name}"
            std_key = f"std_test_{metric_name}"
            mlflow.log_metric(f"cv_{metric_name}_mean", search.cv_results_[mean_key][best_idx])
            mlflow.log_metric(f"cv_{metric_name}_std", search.cv_results_[std_key][best_idx])

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = plot_confusion_matrix(
            y_test.to_numpy(), y_pred, FIGURES_DIR / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix — {name}",
        )
        roc_path = plot_roc_curve(
            y_test.to_numpy(), y_proba, FIGURES_DIR / f"roc_curve_{name}.png",
            title=f"ROC Curve — {name}",
        )
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        signature = infer_signature(X_train, best_pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            best_pipeline, name="model", signature=signature,
            input_example=X_train.head(3),
        )

        run_id = mlflow.active_run().info.run_id
        logger.info("%s: test_roc_auc=%.4f test_accuracy=%.4f (run_id=%s)", name,
                    test_metrics["roc_auc"], test_metrics["accuracy"], run_id)

    return {
        "run_id": run_id,
        "best_params": search.best_params_,
        "cv_roc_auc_mean": search.cv_results_["mean_test_roc_auc"][best_idx],
        "test_metrics": test_metrics,
        "pipeline": best_pipeline,
    }


def main(fast: bool = False) -> dict:
    # MLflow's plain filesystem store is in maintenance mode as of 3.x; a local
    # SQLite backend is the recommended zero-server local tracking store instead.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    logger.info("Train/test split: %d/%d rows", len(X_train), len(X_test))

    configs = get_model_configs(fast)
    results = {}
    for name, (estimator, grid, cv) in configs.items():
        results[name] = train_and_log_model(
            name, estimator, grid, cv, X_train, y_train, X_test, y_test
        )

    best_name = max(results, key=lambda n: results[n]["test_metrics"]["roc_auc"])
    best_result = results[best_name]
    logger.info("Best model: %s (test_roc_auc=%.4f)", best_name,
                best_result["test_metrics"]["roc_auc"])

    if FINAL_MODEL_DIR.exists():
        shutil.rmtree(FINAL_MODEL_DIR)
    FINAL_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    signature = infer_signature(X_train, best_result["pipeline"].predict(X_train))
    mlflow.sklearn.save_model(
        best_result["pipeline"], path=str(FINAL_MODEL_DIR), signature=signature,
        input_example=X_train.head(3),
    )
    logger.info("Exported final model (%s) to %s", best_name, FINAL_MODEL_DIR)

    summary = {
        "best_model": best_name,
        "models": {
            name: {
                "run_id": r["run_id"],
                "best_params": r["best_params"],
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "test_metrics": r["test_metrics"],
            }
            for name, r in results.items()
        },
    }
    METRICS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote metrics summary to %s", METRICS_SUMMARY_PATH)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                         help="Small grid + 2-fold CV, for CI smoke-testing only")
    args = parser.parse_args()
    main(fast=args.fast)
