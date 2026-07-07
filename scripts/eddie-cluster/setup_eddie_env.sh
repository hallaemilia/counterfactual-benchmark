#!/bin/bash

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true
module load python || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"

mkdir -p "$EDDIE_SCRATCH_ROOT"/{data,checkpoints,results,logs,run_configs,venvs}

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Environment ready:"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "EDDIE_SCRATCH_ROOT=$EDDIE_SCRATCH_ROOT"
echo "VENV_DIR=$VENV_DIR"
