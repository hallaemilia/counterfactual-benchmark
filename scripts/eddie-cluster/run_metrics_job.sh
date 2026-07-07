#!/bin/bash
#$ -N cfb-metrics
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 4
#$ -l h_rt=12:00:00
#$ -l h_rss=16G

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH.}"
CLASSIFIER_CONFIG="${CLASSIFIER_CONFIG:?Set CLASSIFIER_CONFIG.}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME.}"
DATASET_LABEL="${DATASET_LABEL:?Set DATASET_LABEL.}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
CG_SAMPLES="${CG_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_DIMS_PER_LAYER="${MAX_DIMS_PER_LAYER:-}"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/counterfactual_benchmark:$PROJECT_DIR:${PYTHONPATH:-}"
OUT_DIR="$EDDIE_SCRATCH_ROOT/results/raw"
mkdir -p "$OUT_DIR"

cmd=(
  python -m causal_disentanglement_metrics.run_metrics
  --config "$PROJECT_DIR/$CONFIG_PATH"
  --classifier-config "$PROJECT_DIR/$CLASSIFIER_CONFIG"
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_LABEL"
  --num-samples "$NUM_SAMPLES"
  --cg-samples "$CG_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUT_DIR"
)
if [ -n "$MAX_DIMS_PER_LAYER" ]; then
  cmd+=(--max-dims-per-layer "$MAX_DIMS_PER_LAYER")
fi

cd "$PROJECT_DIR"
"${cmd[@]}"
