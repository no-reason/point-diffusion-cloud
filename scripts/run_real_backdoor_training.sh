#!/bin/bash
# run_real_backdoor_training.sh
# Run this on the GPU server (e.g. 10.164.234.101)

set -e

# Default settings
DATAROOT="/data/personal_data/zyy/point-diffusion-cloud/ShapeNetCore.v2.PC15k"
CATEGORY="chair"
TARGET_CAT="airplane"
POISON_RATE=0.1
LAMBDA_BD=1.0

# Ensure target exists
TARGET_PATH="./targets/real_target_${TARGET_CAT}.npy"
if [ ! -f "$TARGET_PATH" ]; then
    echo "Error: Target file $TARGET_PATH not found. Run prepare_real_shapenet_targets.py first."
    exit 1
fi

echo "Starting PVD Backdoor Training for ${CATEGORY} -> ${TARGET_CAT}"

python train_stageP1_badpvd_backdoor.py \
    --dataroot "${DATAROOT}" \
    --category "${CATEGORY}" \
    --target_path "${TARGET_PATH}" \
    --r_path "./coordinate_audit/assets/g_sphere.npy" \
    --poison_rate "${POISON_RATE}" \
    --lambda_bd "${LAMBDA_BD}" \
    --use_mask True \
    --niter 1000 \
    --bs 32 \
    --lr 2e-4
