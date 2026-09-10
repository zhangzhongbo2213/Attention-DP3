#!/usr/bin/env bash
set -euo pipefail

# Args: $1 frames_root, $2 output_root, $3 text_prompt, $4 device, $5 api_url

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
GS2_DIR="${GS2_DIR:-${ROOT_DIR}/Grounded-SAM-2}"

if [[ "${1:-}" =~ ^/ ]]; then
  FRAMES_ROOT="${1:-${ROOT_DIR}/3D-Diffusion-Policy/export/adroit_door_frames}"
else
  FRAMES_ROOT="${ROOT_DIR}/${1:-3D-Diffusion-Policy/export/adroit_door_frames}"
fi

if [[ "${2:-}" =~ ^/ ]]; then
  OUTPUT_ROOT="${2:-${ROOT_DIR}/3D-Diffusion-Policy/export_gs2/adroit_door}"
else
  OUTPUT_ROOT="${ROOT_DIR}/${2:-3D-Diffusion-Policy/export_gs2/adroit_door}"
fi

TEXT_PROMPT="${3:-door handle. door.}"
DEVICE="${4:-cuda}"
API_URL="${5:-}"

echo "[gs2.sh] ROOT_DIR: ${ROOT_DIR}, GS2_DIR: ${GS2_DIR}"
echo "[gs2.sh] FRAMES_ROOT: ${FRAMES_ROOT}"
echo "[gs2.sh] OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "[gs2.sh] TEXT_PROMPT: ${TEXT_PROMPT}, DEVICE: ${DEVICE}"

if [ ! -d "${FRAMES_ROOT}" ]; then
  echo "[ERROR] FRAMES_ROOT does not exist: ${FRAMES_ROOT}"
  exit 1
fi

if [ ! -d "${GS2_DIR}" ]; then
  echo "[ERROR] GS2_DIR does not exist: ${GS2_DIR}"
  exit 1
fi

cd "${GS2_DIR}"
python batch_grounded_sam2.py \
  --frames_root "${FRAMES_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --text "${TEXT_PROMPT}" \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino_cfg grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py \
  --gdino_ckpt gdino_checkpoints/groundingdino_swinb_cogcoor.pth \
  --device "${DEVICE}" \
  --box_thr 0.25 \
  --text_thr 0.15 \
  ${API_URL:+--api_url "${API_URL}"}

echo "[gs2.sh] Finished!"
