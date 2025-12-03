#!/usr/bin/env bash
# End-to-end convenience script for running the full CBWM + dual-belief + L2D
# experiment suite and its ablations. The dataset path can be overridden via
# DATASET_PATH; it defaults to the PARTNR mini validation split.

set -euo pipefail

DATASET_PATH=${DATASET_PATH:-data/datasets/partnr_episodes/v0_0/val_mini.json.gz}

run() {
  local CONFIG=$1
  echo "\n=== Running ${CONFIG} ==="
  python -m habitat_llm.examples.planner_demo \
    --config-name ${CONFIG} \
    habitat.dataset.data_path="${DATASET_PATH}"
}

run examples/cbwm_dual_belief_demo.yaml
run examples/cbwm_ablation_no_cbwm.yaml
run examples/cbwm_ablation_no_dual_belief.yaml
run examples/cbwm_ablation_no_l2d.yaml
run examples/cbwm_ablation_all_off.yaml
