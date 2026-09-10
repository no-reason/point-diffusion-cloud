#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_PATH:?export DATASET_PATH=/path/to/shapenet_v2pc15k.h5}"
: "${OUTPUT_DIR:=outputs_b1/full_train_seed42}"
: "${CLEAN_CHECKPOINT:=coordinate_audit/provenance/epoch_149.pth}"
: "${TARGET_PATH:=coordinate_audit/assets/target_checkpoint_space.npy}"
: "${SPHERE_PATH:=coordinate_audit/assets/sphere_checkpoint_space.npy}"
: "${TRIGGER_K:=}"
: "${SAVE_INTERVAL:=10}"
resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "$RESUME")
fi
train_args=(python train_b1_fullpoison.py \
  --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --target_path "$TARGET_PATH" \
  --sphere_path "$SPHERE_PATH" \
  --dataset_path "$DATASET_PATH" --poison_rate "${POISON_RATE:-0.2}" \
  --poison_seed "${POISON_SEED:-0}" --lambda_bd "${LAMBDA_BD:-1.0}" \
  --batch_size "${BATCH_SIZE:-8}" --epochs "${EPOCHS:-150}" --workers "${WORKERS:-4}" \
  --save_interval "$SAVE_INTERVAL" --log_interval "${LOG_INTERVAL:-20}" --output_dir "$OUTPUT_DIR" --device cuda:0 \
  "${resume_args[@]}")
if [[ -n "$TRIGGER_K" ]]; then train_args+=(--trigger_k "$TRIGGER_K"); fi
if [[ -n "${SPHERE_CENTER:-}" ]]; then read -r -a center_args <<< "$SPHERE_CENTER"; train_args+=(--sphere_center "${center_args[@]}"); fi
if [[ -n "${SPHERE_RADIUS:-}" ]]; then train_args+=(--sphere_radius "$SPHERE_RADIUS"); fi
if [[ "${ALLOW_IN_RANGE_SPHERE:-0}" == "1" ]]; then train_args+=(--allow_in_range_sphere); fi
"${train_args[@]}"
