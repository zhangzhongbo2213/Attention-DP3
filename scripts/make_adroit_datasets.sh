#!/usr/bin/env bash
set -euo pipefail

# Generate Adroit no-attn and attention datasets.
# Env vars: GPU, DEVICE, ROOT, GS2_DIR, MAX_EP, N_POINTS, TASKS, GS2_CONDA_ENV, SEG_TYPES

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_EP="${MAX_EP:-10}"
N_POINTS="${N_POINTS:-512}"
TASKS="${TASKS:-door hammer pen}"
GS2_DIR="${GS2_DIR:-${ROOT}/Grounded-SAM-2}"
GS2_CONDA_ENV="${GS2_CONDA_ENV:-aedp3_vis}"
SEG_TYPES="${SEG_TYPES:-env gs2}"
DATA_OUTPUT_ROOT="${DATA_OUTPUT_ROOT:-}"
DATA_ROOT="${DATA_OUTPUT_ROOT:-${ROOT}/3D-Diffusion-Policy/data}"

log() { echo -e "[make_adroit] $*"; }

gen_demo() {
  local task="$1"
  local seg_type="$2"
  local output_dir="${DATA_ROOT}/"
  local save_name="adroit_${task}_expert_${seg_type}.zarr"
  log "Generating demos: ${task} (seg_type=${seg_type}) -> ${save_name}"
  pushd "${ROOT}/third_party/VRL3/src" >/dev/null
  local use_env_seg_flag=""
  if [ "${seg_type}" = "env" ]; then
    use_env_seg_flag="--use_env_seg"
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python gen_demonstration_expert.py --env_name "${task}" \
    --num_episodes "${MAX_EP}" \
    --root_dir "${output_dir}" \
    --save_name "${save_name}" \
    --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
    --img_size 84 \
    --not_use_multi_view \
    --use_point_crop \
    ${use_env_seg_flag}
  popd >/dev/null
}

export_frames() {
  local task="$1"
  local seg_type="$2"
  local zarr_file="${3:-${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr}"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_${seg_type}_frames"
  log "Exporting frames: ${task} (${seg_type}) -> ${out_dir}"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr_file}" \
    --out_dir "${out_dir}" \
    --max_episodes "${MAX_EP}"
}

gs2_for_task() {
  local task="$1"
  local seg_type="$2"
  if [ "${seg_type}" != "gs2" ]; then
    return
  fi
  local frames_root="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_${seg_type}_frames"
  local output_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_${seg_type}"
  local text_prompt
  case "${task}" in
    door)   text_prompt="door handle. door." ;;
    hammer) text_prompt="hammer. nail." ;;
    pen)    text_prompt="blue pen in hand." ;;
    *)      text_prompt="${task}" ;;
  esac
  log "Running GS2: ${task} (${seg_type}) -> ${output_root}"
  local runner=()
  if [[ -n "${GS2_CONDA_ENV:-}" ]]; then
    runner=(conda run -n "${GS2_CONDA_ENV}")
  fi
  GS2_DIR="${GS2_DIR}" "${runner[@]}" bash "${ROOT}/scripts/gs2.sh" \
    "${frames_root}" \
    "${output_root}" \
    "${text_prompt}" \
    "${DEVICE}"
}

convert_attn_zarr() {
  local task="$1"
  local seg_type="$2"
  local input_zarr_file="${3:-${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr}"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_${seg_type}"
  local output_zarr="${DATA_ROOT}/adroit_${task}_expert_${seg_type}_attn3d.zarr"
  log "Generating attn_3d zarr: ${task} (${seg_type}) -> ${output_zarr}"
  local use_env_seg_flag=""
  if [ "${seg_type}" = "env" ]; then
    use_env_seg_flag="--use_env_seg"
  fi
  python "${ROOT}/scripts/convert_zarr_with_attn3d.py" \
    --input_zarr "${input_zarr_file}" \
    --json_root "${json_root}" \
    --output_zarr "${output_zarr}" \
    --n_points "${N_POINTS}" \
    --max_episodes "${MAX_EP}" \
    ${use_env_seg_flag}
}

main() {
  log "ROOT=${ROOT}, TASKS=${TASKS}, SEG_TYPES=${SEG_TYPES}"
  log "GPU=${GPU}, MAX_EP=${MAX_EP}, N_POINTS=${N_POINTS}"

  log "=== Phase 1: Generate base datasets ==="
  for task in ${TASKS}; do
    log "Generating base dataset: ${task}"
    gen_demo "${task}" "env"
  done

  log "=== Phase 2: Generate attention variants ==="
  for task in ${TASKS}; do
    log "Processing task: ${task}"
    base_zarr="${DATA_ROOT}/adroit_${task}_expert_env.zarr"
    no_attn_zarr="${DATA_ROOT}/adroit_${task}_expert_no_attn.zarr"
    if [ -d "$base_zarr" ] && [ ! -d "$no_attn_zarr" ]; then
      cp -r "$base_zarr" "$no_attn_zarr"
      log "Created no_attn variant: $no_attn_zarr"
    fi
    convert_attn_zarr "${task}" "env" "$base_zarr"
    export_frames "${task}" "gs2" "$base_zarr"
    gs2_for_task "${task}" "gs2"
    convert_attn_zarr "${task}" "gs2" "$base_zarr"
  done

  log "All tasks completed"
}

main "$@"
