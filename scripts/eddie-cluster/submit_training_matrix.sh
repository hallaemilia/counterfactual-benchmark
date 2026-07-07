#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
LOG_ROOT="$EDDIE_SCRATCH_ROOT/logs/training"
SCRIPT="$PROJECT_DIR/scripts/eddie-cluster/run_deepscm_job.sh"
mkdir -p "$LOG_ROOT"

submit_train() {
  local dataset_label="$1"
  local model_name="$2"
  local config_path="$3"
  local h_rt="$4"
  local h_rss="$5"
  local stage="${6:-train}"
  local job_name="cfb-${dataset_label}-${model_name}-${stage}"
  qsub -N "$job_name" -q gpu -l gpu=1 -pe sharedmem 4 -l h_rt="$h_rt" -l h_rss="$h_rss" \
    -o "$LOG_ROOT/${job_name}.out" -e "$LOG_ROOT/${job_name}.err" \
    -v PROJECT_DIR="$PROJECT_DIR",EDDIE_SCRATCH_ROOT="$EDDIE_SCRATCH_ROOT",DATASET_LABEL="$dataset_label",MODEL_NAME="$model_name",CONFIG_PATH="$config_path",JOB_STAGE="$stage" \
    "$SCRIPT"
}

submit_train morphomnist vae counterfactual_benchmark/methods/deepscm/configs/morphomnist/vae.json 06:00:00 8G
submit_train morphomnist hvae counterfactual_benchmark/methods/deepscm/configs/morphomnist/hvae.json 14:00:00 16G hvae_pretrain
submit_train morphomnist gan counterfactual_benchmark/methods/deepscm/configs/morphomnist/gan.json 08:00:00 8G

submit_train celeba_simple vae counterfactual_benchmark/methods/deepscm/configs/celeba/simple/vae.json 08:00:00 8G
submit_train celeba_simple hvae counterfactual_benchmark/methods/deepscm/configs/celeba/simple/hvae.json 30:00:00 24G hvae_pretrain
submit_train celeba_simple gan counterfactual_benchmark/methods/deepscm/configs/celeba/simple/gan.json 14:00:00 12G

submit_train celeba_complex vae counterfactual_benchmark/methods/deepscm/configs/celeba/complex/vae.json 08:00:00 8G
submit_train celeba_complex hvae counterfactual_benchmark/methods/deepscm/configs/celeba/complex/hvae.json 34:00:00 24G hvae_pretrain
submit_train celeba_complex gan counterfactual_benchmark/methods/deepscm/configs/celeba/complex/gan.json 14:00:00 12G
