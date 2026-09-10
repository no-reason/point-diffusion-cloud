#!/bin/bash
# run_defense_pipeline.sh
# Run this on the GPU server (e.g. 10.164.234.101) to execute real Graph Spectral Defense

set -e

# Configuration
CKPT_DIR="./log/stageP1_badpvd"
# Use the latest checkpoint or explicitly specify one that has the geometric mask backdoor
CHECKPOINT=$(ls -t ${CKPT_DIR}/checkpoint_*.pth 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT" ]; then
    # Fallback to the known real manifold backdoor checkpoint if exists
    CHECKPOINT="/data/personal_data/zyy/point-diffusion-cloud/logs_stageC/Task_VarBound_neg0.5/ckpt_10000.pt"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: No valid checkpoint found at $CHECKPOINT"
    exit 1
fi

DATAROOT="/data/personal_data/zyy/point-diffusion-cloud/ShapeNetCore.v2.PC15k"
TARGET_PATH="./targets/real_target_airplane.npy"
SOURCE_PATH="./targets/real_target_chair.npy"
TRIGGER_PATH="./coordinate_audit/assets/g_sphere.npy"
OUT_DIR="./results/defense_eval"

echo "=== Starting Real Graph Spectral Defense Evaluation ==="
echo "Using checkpoint: $CHECKPOINT"

# Run Defense Logic & Inference
python scripts/run_real_spectral_defense.py \
    --bd_ckpt "$CHECKPOINT" \
    --clean_target_path "$TARGET_PATH" \
    --clean_source_path "$SOURCE_PATH" \
    --r_path "$TRIGGER_PATH" \
    --out_dir "$OUT_DIR"

# Plot Visuals
python scripts/plot_real_defense_visuals.py \
    --in_dir "$OUT_DIR" \
    --out_img "${OUT_DIR}/real_spectral_defense_visual.png"

echo "=== Pipeline Complete ==="
echo "Check $OUT_DIR for metrics and visualizations."
