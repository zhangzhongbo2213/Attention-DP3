#!/usr/bin/env bash
set -euo pipefail

# Run Adroit experiments serially.
# Optional env vars: GPU_ID, SEED, CONFIG_NAME, EXTRA_ARGS, DATASET_TYPE, ATTN_MODE

DEBUG=False
save_ckpt=True
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
DATASET_TYPE="${DATASET_TYPE:-standard}"
ATTN_MODE="${ATTN_MODE:-all}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "${ATTN_MODE}" = "no_attn" ]; then
    TASKS=(
      adroit_pen_no_attn
      adroit_hammer_no_attn
      adroit_door_no_attn
    )
elif [ "${ATTN_MODE}" = "attn" ]; then
    TASKS=(
      adroit_pen
      adroit_hammer
      adroit_door
    )
else
    TASKS=(
      adroit_pen_no_attn
      adroit_hammer_no_attn
      adroit_door_no_attn
      adroit_pen
      adroit_hammer
      adroit_door
    )
fi

log() { echo -e "[run_all_adroit] $*"; }

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd "${ROOT}/3D-Diffusion-Policy"
export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${GPU_ID}

total_start=$(date +%s)
for task in "${TASKS[@]}"; do
  task_start=$(date +%s)
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="0109dp3"
  else
    addition_info="0109aedp3"
  fi
  
  exp_name="${task}-${CONFIG_NAME}-${addition_info}"
  if [[ -n "${RUN_NAME_PREFIX}" ]]; then
    run_name="${RUN_NAME_PREFIX}_${exp_name}"
    run_dir="data/outputs/${RUN_NAME_PREFIX}_${exp_name}_seed${SEED}"
  else
    run_name="${exp_name}"
    run_dir="data/outputs/${exp_name}_seed${SEED}"
  fi
  
  dataset_args=""
  base="${task#adroit_}"
  is_no_attn=false
  if [[ "${base}" == *"_no_attn" ]]; then
    is_no_attn=true
    base="${base%_no_attn}"
  fi
  if [ "${DATASET_TYPE}" = "no_attn" ]; then
    dataset_path="data/adroit_${base}_expert_no_attn.zarr"
  elif [ "${DATASET_TYPE}" = "gs2_attn" ]; then
    dataset_path="data/adroit_${base}_expert_gs2_attn3d.zarr"
  elif [ "${DATASET_TYPE}" = "env_attn" ]; then
    dataset_path="data/adroit_${base}_expert_env_attn3d.zarr"
  fi
  if [[ -n "${dataset_path:-}" ]]; then
    dataset_args="task.dataset.zarr_path=${dataset_path}"
  fi

  log "Starting training: ${task} (exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED}, addition_info=${addition_info}, dataset_type=${DATASET_TYPE})"
  python train.py --config-name=${CONFIG_NAME}.yaml \
                            task=${task} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${SEED} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.name=${run_name} \
                            logging.project=aedp3_adroit_0310 \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${dataset_args} \
                            ${EXTRA_ARGS}
  task_end=$(date +%s)
  log "Training completed: ${task} took $((task_end - task_start)) seconds"
done

total_end=$(date +%s)
log "All tasks completed, total time: $((total_end - total_start)) seconds"
