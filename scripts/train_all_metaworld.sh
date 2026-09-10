#!/usr/bin/env bash
set -euo pipefail

# Run Metaworld experiments serially.
# Optional env vars: GPU_ID, SEED, CONFIG_NAME, TASKS, GS2_PORT, EXTRA_ARGS

DEBUG=False
save_ckpt=True
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
GS2_PORT="${GS2_PORT:-5000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

DEFAULT_TASKS=(
  metaworld_dial-turn_no_attn
  metaworld_door-lock_no_attn
  metaworld_handle-pull_no_attn
  metaworld_handle-pull-side_no_attn
  metaworld_lever-pull_no_attn
  metaworld_reach-wall_no_attn
  metaworld_peg-unplug-side_no_attn
  metaworld_coffee-pull_no_attn
  metaworld_coffee-push_no_attn
  metaworld_dial-turn
  metaworld_door-lock
  metaworld_handle-pull
  metaworld_handle-pull-side
  metaworld_lever-pull
  metaworld_reach-wall
  metaworld_peg-unplug-side
  metaworld_coffee-pull
  metaworld_coffee-push
)

if [[ -n "${TASKS:-}" ]]; then
  IFS=' ' read -r -a TASKS_ARRAY <<< "$TASKS"
else
  TASKS_ARRAY=("${DEFAULT_TASKS[@]}")
fi

log() { echo -e "[run_all_metaworld] $*"; }

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd "${ROOT}/3D-Diffusion-Policy"
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${GPU_ID}

total_start=$(date +%s)
for task in "${TASKS_ARRAY[@]}"; do
  task_start=$(date +%s)
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="0207mw"
    task_extra_args="${EXTRA_ARGS}"
  else
    addition_info="0207mwaedp3"
    export GS2_API_URL="http://127.0.0.1:${GS2_PORT}"
    task_extra_args="${EXTRA_ARGS}"
  fi
  exp_name="${task}-${CONFIG_NAME}-${addition_info}"
  run_dir="data/outputs/${exp_name}_seed${SEED}"
  if [[ -n "${RUN_NAME_PREFIX}" ]]; then
    run_name="${RUN_NAME_PREFIX}_${exp_name}"
  else
    run_name="${exp_name}"
  fi
  log "Starting training: ${task} (exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED})"
  python train.py --config-name=${CONFIG_NAME}.yaml \
                            task=${task} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${SEED} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.name=${run_name} \
                            logging.project=aedp3_wetaworld_swint \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${task_extra_args}
  task_end=$(date +%s)
  log "Training completed: ${task} took $((task_end - task_start)) seconds"
done

total_end=$(date +%s)
log "All tasks completed, total time: $((total_end - total_start)) seconds"
