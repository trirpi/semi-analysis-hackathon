#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
PIP_FLAGS=(--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org)
python -m pip install "${PIP_FLAGS[@]}" -r requirements.txt
python -m pip install "${PIP_FLAGS[@]}" -e vendor/whisper

python scripts/doctor.py
