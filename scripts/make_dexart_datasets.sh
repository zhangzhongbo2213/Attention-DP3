#!/usr/bin/env bash
set -euo pipefail

# Generate DexArt GS2 attention datasets.
# Env vars: DEVICE, ROOT, MAX_EP, N_POINTS, TASKS, GS2_CONDA_ENV, GS2_PORT, GS2_API_URL

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
DEVICE="${DEVICE:-cuda}"
MAX_EP="${MAX_EP:-100}"
N_POINTS="${N_POINTS:-1024}"
TASKS="${TASKS:-bucket faucet laptop toilet}"
GS2_DIR="${GS2_DIR:-${ROOT}/Grounded-SAM-2}"
GS2_CONDA_ENV="${GS2_CONDA_ENV:-aedp3_vis}"
GS2_PORT="${GS2_PORT:-}"
GS2_API_URL="${GS2_API_URL:-}"

if [[ -n "${GS2_API_URL}" ]]; then
  GS2_API_URL="${GS2_API_URL}"
elif [[ -n "${GS2_PORT}" ]]; then
  GS2_API_URL="http://127.0.0.1:${GS2_PORT}"
fi

log() { echo -e "[make_dexart] $*"; }

gen_demo() {
  local task="$1"
  log "Generating demos: ${task}"
  pushd "${ROOT}" >/dev/null
  bash "${ROOT}/scripts/gen_demonstration_dexart.sh" "${task}"
  popd >/dev/null
}

export_frames() {
  local task="$1"
  local zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert.zarr"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/dexart_${task}_frames"
  local task_max_ep
  case "${task}" in
    bucket) task_max_ep=100 ;;
    faucet|laptop|toilet) task_max_ep=250 ;;
    *) task_max_ep="${MAX_EP}" ;;
  esac
  log "Exporting frames: ${task} -> ${out_dir} (max_ep=${task_max_ep})"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr}" \
    --out_dir "${out_dir}" \
    --max_episodes "${task_max_ep}"
}

gs2_for_task() {
  local task="$1"
  local frames_root="${ROOT}/3D-Diffusion-Policy/export/dexart_${task}_frames"
  local output_root="${ROOT}/3D-Diffusion-Policy/export_gs2/dexart_${task}"
  task_to_prompt() {
    local t="$1"
    case "${t}" in
      bucket) echo "bucket." ;;
      faucet) echo "faucet." ;;
      laptop) echo "laptop." ;;
      toilet) echo "toilet." ;;
      *) echo "${t}" ;;
    esac
  }
  local text_prompt
  text_prompt="$(task_to_prompt "${task}")"
  log "Running GS2: ${task} -> ${output_root}"
  local runner=()
  if [[ -n "${GS2_CONDA_ENV:-}" ]]; then
    runner=(conda run -n "${GS2_CONDA_ENV}")
  fi
  gs2_args=("${frames_root}" "${output_root}" "${text_prompt}" "${DEVICE}")
  if [[ -n "${GS2_API_URL}" ]]; then
    gs2_args+=("${GS2_API_URL}")
  fi
  GS2_DIR="${GS2_DIR}" "${runner[@]}" bash "${ROOT}/scripts/gs2.sh" "${gs2_args[@]}"
}

convert_attn_zarr() {
  local task="$1"
  local input_zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert.zarr"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/dexart_${task}"
  local output_zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert_attn3d.zarr"
  log "Generating attn_3d zarr: ${task} -> ${output_zarr}"
  bash "${ROOT}/scripts/convert_zarr_with_attn3d.sh" \
    "${input_zarr}" \
    "${json_root}" \
    "${output_zarr}" \
    "${MAX_EP}" \
    "${N_POINTS}"
}

main() {
  log "ROOT=${ROOT}, TASKS=${TASKS}, MAX_EP=${MAX_EP}, N_POINTS=${N_POINTS}"
  for task in ${TASKS}; do
    gen_demo "${task}"
    export_frames "${task}"
    gs2_for_task "${task}"
    convert_attn_zarr "${task}"
  done
  log "All tasks completed"
}

main "$@"
