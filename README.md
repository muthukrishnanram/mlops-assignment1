# Heart Disease Risk Classifier — End-to-End MLOps Pipeline

Predicts the presence/absence of heart disease from patient health data (UCI Heart
Disease dataset), served as a monitored, containerized, Kubernetes-deployed API.
Built for the BITS Pilani AIMLCZG523 MLOps assignment.

## Quickstart (clean setup)

```bash
git clone <this-repo> && cd mlops-assignment-1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # training/dev/test extras on top of the base deps

python data/download_data.py               # fetches the raw UCI dataset -> data/raw/
python src/data/preprocess.py               # cleans -> data/processed/heart_clean.csv
python src/eda.py                           # writes EDA plots -> report/figures/
python src/models/train.py                  # trains, cross-validates, logs to MLflow, exports models/final_model/

pytest -q                                   # unit tests
uvicorn api.main:app --reload               # local API on http://localhost:8000
```

Or run the whole clean-setup-to-trained-model sequence in one command: `make setup`.

## Repository layout

| Path | Purpose |
|---|---|
| `data/download_data.py` | Fetches the UCI Heart Disease dataset (`ucimlrepo`, id=45) |
| `src/data/preprocess.py` | Missing-value handling, cleaning -> `data/processed/heart_clean.csv` |
| `src/eda.py` | Exploratory plots (histograms, correlation heatmap, class balance) |
| `src/features/build_features.py` | `ColumnTransformer` used by both training and the exported model |
| `src/models/train.py` | Trains/tunes Logistic Regression + Random Forest, logs to MLflow, exports final model |
| `api/main.py` | FastAPI serving app: `/predict`, `/health`, `/metrics` |
| `tests/` | pytest unit tests for data/features/model/API code |
| `Dockerfile` | Container image for the serving API |
| `k8s/` | Kubernetes manifests (Deployment, Service, Ingress) for Minikube |
| `monitoring/` | Prometheus + Grafana config for the docker-compose monitoring stack |
| `.github/workflows/ci.yml` | Lint -> test -> fast-train -> docker build/run/health-check |
| `report/` | Final written report and architecture diagram |
| `screenshots/` | Evidence of Docker/K8s/CI/monitoring runs, referenced by the report |

## Model & experiment tracking

Two classifiers (Logistic Regression, Random Forest) are trained with stratified
cross-validation and compared on accuracy/precision/recall/ROC-AUC. All runs —
params, metrics, plots, and the fitted pipeline artifact — are logged to a local
MLflow tracking store (`mlruns/`, gitignored; inspect with `mlflow ui`). The
best run's pipeline (preprocessing + model as one `sklearn.Pipeline`) is exported
to `models/final_model/` and is the only model artifact committed to the repo.

## Serving

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

## Deployment (local Kubernetes via Minikube)

See `k8s/README.md` for the exact deploy/verify commands.

## Monitoring

```bash
docker compose -f docker-compose.yml up
```
Grafana at `http://localhost:3000`, Prometheus at `http://localhost:9090`, scraping
the API's `/metrics` endpoint.

## Report

The full written report is at `report/report.md` (also exported as PDF), covering
setup, EDA/modelling choices, experiment tracking, architecture, and CI/CD &
deployment evidence.
