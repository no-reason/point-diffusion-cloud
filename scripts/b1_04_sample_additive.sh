#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CHECKPOINT:?export CHECKPOINT=/path/to/latest.pth}"
SPHERE_PATH="${SPHERE_PATH:-coordinate_audit/assets/sphere_checkpoint_space.npy}"
G_SPHERE_PATH="${G_SPHERE_PATH:-coordinate_audit/assets/g_sphere.npy}"
sample_args=(python sample_b1.py --checkpoint "$CHECKPOINT" \
  --sphere_path "$SPHERE_PATH" --g_sphere_path "$G_SPHERE_PATH" --mode additive \
  --num_samples "${NUM_SAMPLES:-100}" --batch_size "${BATCH_SIZE:-8}" \
  --seed_start "${SEED_START:-0}" --output_dir "${SAMPLE_DIR:-outputs_b1/samples}" --device cuda:0)
if [[ -n "${TRIGGER_K:-}" ]]; then sample_args+=(--trigger_k "$TRIGGER_K"); fi
if [[ -n "${SPHERE_CENTER:-}" ]]; then read -r -a center_args <<< "$SPHERE_CENTER"; sample_args+=(--sphere_center "${center_args[@]}"); fi
if [[ -n "${SPHERE_RADIUS:-}" ]]; then sample_args+=(--sphere_radius "$SPHERE_RADIUS"); fi
if [[ "${ALLOW_IN_RANGE_SPHERE:-0}" == "1" ]]; then sample_args+=(--allow_in_range_sphere); fi
"${sample_args[@]}"
