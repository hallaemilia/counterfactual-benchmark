#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
LOG_ROOT="$EDDIE_SCRATCH_ROOT/logs/metrics"
SCRIPT="$PROJECT_DIR/scripts/eddie-cluster/run_metrics_job.sh"
mkdir -p "$LOG_ROOT"

submit_metrics() {
  local dataset_label="$1"
  local model_name="$2"
  local config_path="$3"
  local classifier_config="$4"
  local h_rt="$5"
  local h_rss="$6"
  local batch_size="${7:-64}"
  local job_name="cfb-metrics-${dataset_label}-${model_name}"
  qsub -N "$job_name" -q gpu -l gpu=1 -pe sharedmem 4 -l h_rt="$h_rt" -l h_rss="$h_rss" \
    -o "$LOG_ROOT/${job_name}.out" -e "$LOG_ROOT/${job_name}.err" \
    -v PROJECT_DIR="$PROJECT_DIR",EDDIE_SCRATCH_ROOT="$EDDIE_SCRATCH_ROOT",DATASET_LABEL="$dataset_label",MODEL_NAME="$model_name",CONFIG_PATH="$config_path",CLASSIFIER_CONFIG="$classifier_config",BATCH_SIZE="$batch_size" \
    "$SCRIPT"
}

submit_metrics morphomnist vae counterfactual_benchmark/methods/deepscm/configs/morphomnist/vae.json counterfactual_benchmark/methods/deepscm/configs/morphomnist/classifier.json 12:00:00 16G 128
submit_metrics morphomnist hvae counterfactual_benchmark/methods/deepscm/configs/morphomnist/hvae.json counterfactual_benchmark/methods/deepscm/configs/morphomnist/classifier.json 16:00:00 16G 128
submit_metrics morphomnist gan counterfactual_benchmark/methods/deepscm/configs/morphomnist/gan.json counterfactual_benchmark/methods/deepscm/configs/morphomnist/classifier.json 12:00:00 16G 128

submit_metrics celeba_simple vae counterfactual_benchmark/methods/deepscm/configs/celeba/simple/vae.json counterfactual_benchmark/methods/deepscm/configs/celeba/simple/classifier.json 24:00:00 24G 64
submit_metrics celeba_simple hvae counterfactual_benchmark/methods/deepscm/configs/celeba/simple/hvae.json counterfactual_benchmark/methods/deepscm/configs/celeba/simple/classifier.json 36:00:00 24G 64
submit_metrics celeba_simple gan counterfactual_benchmark/methods/deepscm/configs/celeba/simple/gan.json counterfactual_benchmark/methods/deepscm/configs/celeba/simple/classifier.json 24:00:00 24G 64

submit_metrics celeba_complex vae counterfactual_benchmark/methods/deepscm/configs/celeba/complex/vae.json counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json 24:00:00 24G 64
submit_metrics celeba_complex hvae counterfactual_benchmark/methods/deepscm/configs/celeba/complex/hvae.json counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json 36:00:00 24G 64
submit_metrics celeba_complex gan counterfactual_benchmark/methods/deepscm/configs/celeba/complex/gan.json counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json 24:00:00 24G 64
