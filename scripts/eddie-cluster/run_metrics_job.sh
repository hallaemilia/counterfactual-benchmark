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
CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-train}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$EDDIE_SCRATCH_ROOT/checkpoints/$DATASET_LABEL/$MODEL_NAME/$CHECKPOINT_STAGE}"
CLASSIFIER_CKPT_DIR="${CLASSIFIER_CKPT_DIR:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
CG_SAMPLES="${CG_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_DIMS_PER_LAYER="${MAX_DIMS_PER_LAYER:-}"
RUN_CONFIG_DIR="$EDDIE_SCRATCH_ROOT/run_configs/metrics/$DATASET_LABEL/$MODEL_NAME"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/counterfactual_benchmark:$PROJECT_DIR:${PYTHONPATH:-}"
OUT_DIR="$EDDIE_SCRATCH_ROOT/results/raw"
mkdir -p "$OUT_DIR" "$RUN_CONFIG_DIR"

RUNTIME_CONFIG="$RUN_CONFIG_DIR/model_config.json"
RUNTIME_CLASSIFIER_CONFIG="$RUN_CONFIG_DIR/classifier_config.json"
python - "$PROJECT_DIR/$CONFIG_PATH" "$RUNTIME_CONFIG" "$CHECKPOINT_DIR" "$PROJECT_DIR/$CLASSIFIER_CONFIG" "$RUNTIME_CLASSIFIER_CONFIG" "$CLASSIFIER_CKPT_DIR" <<'PY'
import json
import sys

model_source, model_target, checkpoint_dir, clf_source, clf_target, classifier_ckpt_dir = sys.argv[1:7]
with open(model_source, "r", encoding="utf-8") as handle:
    model_config = json.load(handle)
model_config["checkpoint_dir"] = checkpoint_dir
with open(model_target, "w", encoding="utf-8") as handle:
    json.dump(model_config, handle, indent=2)

with open(clf_source, "r", encoding="utf-8") as handle:
    clf_config = json.load(handle)
if classifier_ckpt_dir:
    clf_config["ckpt_path"] = classifier_ckpt_dir.rstrip("/") + "/"
with open(clf_target, "w", encoding="utf-8") as handle:
    json.dump(clf_config, handle, indent=2)
PY

cmd=(
  python -m causal_disentanglement_metrics.run_metrics
  --config "$RUNTIME_CONFIG"
  --classifier-config "$RUNTIME_CLASSIFIER_CONFIG"
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
