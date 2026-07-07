#!/bin/bash
#$ -N cfb-clf
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 4
#$ -l h_rt=04:00:00
#$ -l h_rss=8G

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"
CLASSIFIER_CONFIG="${CLASSIFIER_CONFIG:?Set CLASSIFIER_CONFIG to a classifier config JSON path.}"
DATASET_LABEL="${DATASET_LABEL:?Set DATASET_LABEL.}"
RUN_CONFIG_DIR="$EDDIE_SCRATCH_ROOT/run_configs/$DATASET_LABEL/classifiers"
CKPT_DIR="$EDDIE_SCRATCH_ROOT/checkpoints/$DATASET_LABEL/trained_classifiers"

mkdir -p "$RUN_CONFIG_DIR" "$CKPT_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

RUNTIME_CONFIG="$RUN_CONFIG_DIR/classifier.json"
python - "$PROJECT_DIR/$CLASSIFIER_CONFIG" "$RUNTIME_CONFIG" "$CKPT_DIR/" <<'PY'
import json
import sys
source, target, ckpt_path = sys.argv[1:4]
with open(source, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["ckpt_path"] = ckpt_path
with open(target, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

export PYTHONPATH="$PROJECT_DIR/counterfactual_benchmark:$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR/counterfactual_benchmark/methods/deepscm"
python -u train_classifier.py -clf "$RUNTIME_CONFIG"
