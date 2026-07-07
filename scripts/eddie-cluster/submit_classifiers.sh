#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/counterfactual-benchmark}"
EDDIE_SCRATCH_ROOT="${EDDIE_SCRATCH_ROOT:-/exports/eddie/scratch/$USER/counterfactual-benchmark-cdm}"
LOG_ROOT="$EDDIE_SCRATCH_ROOT/logs/classifiers"
SCRIPT="$PROJECT_DIR/scripts/eddie-cluster/run_classifier_job.sh"
mkdir -p "$LOG_ROOT"

submit_classifier() {
  local dataset_label="$1"
  local classifier_config="$2"
  local h_rt="$3"
  local h_rss="$4"
  local job_name="cfb-clf-${dataset_label}"
  qsub -N "$job_name" -q gpu -l gpu=1 -pe sharedmem 4 -l h_rt="$h_rt" -l h_rss="$h_rss" \
    -o "$LOG_ROOT/${job_name}.out" -e "$LOG_ROOT/${job_name}.err" \
    -v PROJECT_DIR="$PROJECT_DIR",EDDIE_SCRATCH_ROOT="$EDDIE_SCRATCH_ROOT",DATASET_LABEL="$dataset_label",CLASSIFIER_CONFIG="$classifier_config" \
    "$SCRIPT"
}

submit_classifier morphomnist counterfactual_benchmark/methods/deepscm/configs/morphomnist/classifier.json 02:00:00 8G
submit_classifier celeba_simple counterfactual_benchmark/methods/deepscm/configs/celeba/simple/classifier.json 04:00:00 12G
submit_classifier celeba_complex counterfactual_benchmark/methods/deepscm/configs/celeba/complex/classifier.json 04:00:00 12G
submit_classifier adni counterfactual_benchmark/methods/deepscm/configs/adni/classifier.json 04:00:00 12G
