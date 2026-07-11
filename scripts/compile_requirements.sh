#!/usr/bin/env bash
# Regenerates requirements.txt (serving/base) and requirements-dev.txt (training/dev/test)
# from the loose requirements.in / requirements-dev.in specs, guaranteeing that every
# package shared between the two (pandas, scikit-learn, mlflow-skinny, fastapi, ...)
# resolves to the exact same pinned version in both files. That match matters: the
# Docker serving image only ever gets requirements.txt, and it must be able to unpickle
# a model trained under requirements-dev.txt's environment without any drift.
#
# Usage: ./scripts/compile_requirements.sh   (run from repo root, with .venv active or not)
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv-compile-dev
.venv-compile-dev/bin/pip install --upgrade pip -q
.venv-compile-dev/bin/pip install -q -r requirements-dev.in
.venv-compile-dev/bin/pip freeze > requirements-dev.txt

python3 -m venv .venv-compile-base
.venv-compile-base/bin/pip install --upgrade pip -q
# Constrain to the exact versions already resolved for the dev env, so shared deps match.
.venv-compile-base/bin/pip install -q -r requirements.in -c requirements-dev.txt
.venv-compile-base/bin/pip freeze > requirements.txt

rm -rf .venv-compile-dev .venv-compile-base
echo "Wrote requirements.txt ($(wc -l < requirements.txt) pkgs) and requirements-dev.txt ($(wc -l < requirements-dev.txt) pkgs)"
