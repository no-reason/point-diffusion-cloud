#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SAMPLE_DIR:=outputs_b1/samples}"
: "${TARGET_PATH:=coordinate_audit/assets/target_checkpoint_space.npy}"
: "${REFERENCE_H5:?export REFERENCE_H5=/path/to/shapenet_v2pc15k_chair_airplane.h5}"
: "${CLEAN_SAMPLES:=$SAMPLE_DIR/clean/final_samples.npy}"
: "${ADDITIVE_SAMPLES:=$SAMPLE_DIR/additive/final_samples.npy}"
: "${REPLACEMENT_SAMPLES:=$SAMPLE_DIR/replacement/final_samples.npy}"
: "${EVALUATION_OUTPUT:=$SAMPLE_DIR/evaluation.json}"

args=(
  --target_path "$TARGET_PATH"
  --additive_samples "$ADDITIVE_SAMPLES"
  --clean_samples "$CLEAN_SAMPLES"
  --reference_h5 "$REFERENCE_H5"
  --reference_categories "${REFERENCE_CATEGORIES:-chair}"
  --reference_split "${REFERENCE_SPLIT:-test}"
  --device "${EVAL_DEVICE:-auto}"
  --chamfer_backend "${CHAMFER_BACKEND:-auto}"
  --sample_batch "${EVAL_SAMPLE_BATCH:-1}"
  --reference_batch "${EVAL_REFERENCE_BATCH:-4}"
  --output "$EVALUATION_OUTPUT"
)
if [[ -n "$REPLACEMENT_SAMPLES" && -f "$REPLACEMENT_SAMPLES" ]]; then
  args+=(--replacement_samples "$REPLACEMENT_SAMPLES")
fi
python evaluate_b1.py "${args[@]}"
echo B1_EVALUATION_OK
