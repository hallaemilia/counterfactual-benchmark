#!/bin/bash

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true
module load python/3.10 || module load python/3.10.8 || module load python || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"
PYTHON_BIN="${PYTHON_BIN:-}"

mkdir -p "$EDDIE_SCRATCH_ROOT"/{data,checkpoints,results,logs,run_configs,venvs}

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python"
  fi
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "This benchmark was tested with Python 3.10. "
        f"Current interpreter is {sys.version.split()[0]}. "
        "Set PYTHON_BIN to a Python 3.10 executable and recreate the venv."
    )
PY
python -m pip install --upgrade "pip<26" wheel "setuptools<70"
PIP_NO_BUILD_ISOLATION=1 python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Environment ready:"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "EDDIE_SCRATCH_ROOT=$EDDIE_SCRATCH_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
