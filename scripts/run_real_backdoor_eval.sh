#!/bin/bash
# run_real_backdoor_eval.sh
# Run this on the GPU server after training is complete

set -e

# Default settings
CKPT_DIR="./log/stageP1_badpvd"
# Pick the latest checkpoint or a specific one
CHECKPOINT=$(ls -t ${CKPT_DIR}/checkpoint_*.pth | head -n 1)
OUTPUT_DIR="./results/real_backdoor_eval"
DATAROOT="/data/personal_data/zyy/point-diffusion-cloud/ShapeNetCore.v2.PC15k"
CATEGORY="chair"

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CKPT_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# 1. Clean Sampling
echo "Running Clean Sampling..."
python sample_b1.py \
    --checkpoint "${CHECKPOINT}" \
    --mode clean \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples 16 \
    --batch_size 4

# 2. Additive (Triggered) Sampling
echo "Running Triggered (Additive) Sampling..."
python sample_b1.py \
    --checkpoint "${CHECKPOINT}" \
    --mode additive \
    --sphere_path "./coordinate_audit/assets/sphere_checkpoint_space.npy" \
    --g_sphere_path "./coordinate_audit/assets/g_sphere.npy" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples 16 \
    --batch_size 4

# 3. Evaluate Results
echo "Evaluating Results..."
python evaluate_b1.py \
    --target_path "./targets/real_target_airplane.npy" \
    --additive_samples "${OUTPUT_DIR}/additive/final_samples.npy" \
    --clean_samples "${OUTPUT_DIR}/clean/final_samples.npy" \
    --reference_path "${DATAROOT}/03001627/test_normalized.npy" \
    --output "${OUTPUT_DIR}/evaluation_metrics.json"

echo "Evaluation Complete. Check ${OUTPUT_DIR}/evaluation_metrics.json"
