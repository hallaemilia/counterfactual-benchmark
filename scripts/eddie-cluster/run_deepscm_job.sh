#!/bin/bash
#$ -N cfb-train
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 4
#$ -l h_rt=08:00:00
#$ -l h_rss=8G

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"
CONFIG_PATH="${CONFIG_PATH:?Set CONFIG_PATH to a Deep-SCM config JSON path.}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to vae, hvae, or gan.}"
DATASET_LABEL="${DATASET_LABEL:?Set DATASET_LABEL, e.g. morphomnist or celeba_simple.}"
JOB_STAGE="${JOB_STAGE:-train}"
RUN_CONFIG_DIR="$EDDIE_SCRATCH_ROOT/run_configs/$DATASET_LABEL/$MODEL_NAME/$JOB_STAGE"
CHECKPOINT_DIR="$EDDIE_SCRATCH_ROOT/checkpoints/$DATASET_LABEL/$MODEL_NAME/$JOB_STAGE"
LIGHTNING_DIR="$EDDIE_SCRATCH_ROOT/results/lightning/$DATASET_LABEL/$MODEL_NAME/$JOB_STAGE"

mkdir -p "$RUN_CONFIG_DIR" "$CHECKPOINT_DIR" "$LIGHTNING_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Missing venv: $VENV_DIR" >&2
  echo "Run scripts/eddie-cluster/setup_eddie_env.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

RUNTIME_CONFIG="$RUN_CONFIG_DIR/config.json"
python - "$PROJECT_DIR/$CONFIG_PATH" "$RUNTIME_CONFIG" "$CHECKPOINT_DIR" "$JOB_STAGE" <<'PY'
import json
import sys

source, target, checkpoint_dir, stage = sys.argv[1:5]
with open(source, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["checkpoint_dir"] = checkpoint_dir
image = config["mechanism_models"].get("image", {})
params = image.get("params", {})
if stage == "hvae_pretrain":
    params["cf_fine_tune"] = "False"
    params["evaluate_cf_model"] = "False"
elif stage == "hvae_finetune":
    params["cf_fine_tune"] = "True"
    params["evaluate_cf_model"] = "False"
    if "PRETRAINED_HVAE_CKPT" in __import__("os").environ:
        params["checkpoint_path"] = __import__("os").environ["PRETRAINED_HVAE_CKPT"]
elif stage == "hvae_eval":
    params["cf_fine_tune"] = "True"
    params["evaluate_cf_model"] = "True"
if "CKPT_CLS_PATH" in __import__("os").environ:
    params["ckpt_cls_path"] = __import__("os").environ["CKPT_CLS_PATH"]
with open(target, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$PROJECT_DIR/counterfactual_benchmark:$PROJECT_DIR:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "EDDIE_SCRATCH_ROOT=$EDDIE_SCRATCH_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "JOB_STAGE=$JOB_STAGE"
nvidia-smi || true

cd "$PROJECT_DIR/counterfactual_benchmark/methods/deepscm"
python -u train.py -c "$RUNTIME_CONFIG"
