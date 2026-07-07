#!/bin/bash
#$ -N cfb-smoke
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_rt=01:00:00
#$ -l h_rss=8G

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda || true

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
VENV_DIR="${VENV_DIR:-$EDDIE_SCRATCH_ROOT/venvs/benchmark}"
RUN_DIR="$EDDIE_SCRATCH_ROOT/smoke/morphomnist"
CONFIG="$RUN_DIR/vae_smoke.json"

mkdir -p "$RUN_DIR/checkpoints"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python - "$PROJECT_DIR/counterfactual_benchmark/methods/deepscm/configs/morphomnist/vae.json" "$CONFIG" "$RUN_DIR/checkpoints" <<'PY'
import json
import sys
source, target, checkpoint_dir = sys.argv[1:4]
with open(source, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config["checkpoint_dir"] = checkpoint_dir
for model in config["mechanism_models"].values():
    params = model["params"]
    params["max_epochs"] = 1
    params["patience"] = 1
    params["batch_size_train"] = min(params.get("batch_size_train", 32), 32)
    params["batch_size_val"] = min(params.get("batch_size_val", 32), 64)
with open(target, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

export PYTHONPATH="$PROJECT_DIR/counterfactual_benchmark:$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR/counterfactual_benchmark/methods/deepscm"
python -u train.py -c "$CONFIG"
python -u evaluate.py -c "$CONFIG" -clf configs/morphomnist/classifier.json -m composition -cc 1 -qn 0
