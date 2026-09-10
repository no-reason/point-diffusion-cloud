#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLEAN_CHECKPOINT:=coordinate_audit/provenance/epoch_149.pth}"
: "${SPHERE_PATH:=coordinate_audit/assets/sphere_checkpoint_space.npy}"
: "${G_SPHERE_PATH:=coordinate_audit/assets/g_sphere.npy}"
: "${RUNTIME_DIR:=coordinate_audit/runtime_validation}"
mkdir -p "$RUNTIME_DIR"
{
  python b1_gpu_probe.py --checkpoint "$CLEAN_CHECKPOINT" --device cuda:0
  python sample_b1.py --checkpoint "$CLEAN_CHECKPOINT" --sphere_path "$SPHERE_PATH" --g_sphere_path "$G_SPHERE_PATH" \
    --mode clean --num_samples "${CLEAN_SAMPLES:-4}" --batch_size "${BATCH_SIZE:-2}" \
    --output_dir "$RUNTIME_DIR" --device cuda:0
} 2>&1 | tee "$RUNTIME_DIR/runtime.log"
cp "$RUNTIME_DIR/clean/final_samples.npy" "$RUNTIME_DIR/clean_samples.npy"
cp "$RUNTIME_DIR/clean/samples.png" "$RUNTIME_DIR/clean_samples.png"
cp "$RUNTIME_DIR/clean/statistics.json" "$RUNTIME_DIR/runtime_statistics.json"
echo CLEAN_SAMPLING_OK
