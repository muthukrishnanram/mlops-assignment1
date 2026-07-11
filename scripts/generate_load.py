"""Fires a burst of varied /predict requests at a running API instance, purely
to populate the Prometheus/Grafana monitoring dashboard with real data.

Usage:
    python scripts/generate_load.py --url http://localhost:8000 --n 200
"""

import argparse
import random
import time

import requests

BASE_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def random_patient() -> dict:
    p = dict(BASE_PATIENT)
    p["age"] = random.randint(29, 77)
    p["sex"] = random.randint(0, 1)
    p["cp"] = random.randint(1, 4)
    p["trestbps"] = random.randint(94, 200)
    p["chol"] = random.randint(126, 400)
    p["thalach"] = random.randint(71, 202)
    p["oldpeak"] = round(random.uniform(0, 6.0), 1)
    p["exang"] = random.randint(0, 1)
    p["ca"] = random.randint(0, 3)
    p["thal"] = random.choice([3, 6, 7])
    return p


def main(url: str, n: int, delay: float) -> None:
    ok = 0
    for i in range(n):
        r = requests.post(f"{url}/predict", json=random_patient(), timeout=5)
        ok += r.status_code == 200
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{n} requests sent ({ok} succeeded)")
        time.sleep(delay)
    requests.get(f"{url}/health", timeout=5)
    print(f"Done: {ok}/{n} predictions succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    args = parser.parse_args()
    main(args.url, args.n, args.delay)
