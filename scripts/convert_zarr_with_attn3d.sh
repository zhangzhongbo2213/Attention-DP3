#!/usr/bin/env bash
set -euo pipefail

# Args: $1 input_zarr, $2 json_root, $3 output_zarr, $4 max_episodes, $5 n_points

INPUT_ZARR="${1:-3D-Diffusion-Policy/data/adroit_door_expert.zarr}"
JSON_ROOT="${2:-3D-Diffusion-Policy/export_gs2/adroit_door}"
OUTPUT_ZARR="${3:-3D-Diffusion-Policy/data/adroit_door_expert_attn3d.zarr}"
MAX_EPISODES="${4:-10}"
N_POINTS="${5:-512}"

MAX_EP_FLAG=()
if [[ -n "${MAX_EPISODES}" ]]; then
  MAX_EP_FLAG=(--max_episodes "${MAX_EPISODES}")
fi

python scripts/convert_zarr_with_attn3d.py \
  --input_zarr "${INPUT_ZARR}" \
  --json_root "${JSON_ROOT}" \
  --output_zarr "${OUTPUT_ZARR}" \
  --n_points "${N_POINTS}" \
  "${MAX_EP_FLAG[@]}"
