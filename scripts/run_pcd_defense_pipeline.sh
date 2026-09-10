#!/bin/bash
# run_pcd_defense_pipeline.sh
# Run this on the GPU server (e.g. 10.164.234.101) to execute real Graph Spectral Defense on PCD

set -e

# PCD Backdoor Checkpoint configuration
# Using the distribution/geometric mask backdoor checkpoint located on the server
CKPT_DIR="/data/personal_data/zyy/point-diffusion-cloud/logs_stageC"
# Default to the known negative bound test if present, or let the user supply one
CHECKPOINT=$(ls -t ${CKPT_DIR}/Task_VarBound_neg0.5*/ckpt_*.pt 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "Could not auto-find checkpoint in Task_VarBound_neg0.5."
    CHECKPOINT="/data/personal_data/zyy/point-diffusion-cloud/logs_stageC/Task_VarBound_neg0.52026_07_27__11_16_49/ckpt_10000.pt"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: No valid checkpoint found at $CHECKPOINT"
    exit 1
fi

DATAROOT="/data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5"
OUT_DIR="./results/pcd_spectral_defense"

echo "=== Starting Real Graph Spectral Defense Evaluation on PCD ==="
echo "Using checkpoint: $CHECKPOINT"

# 1. Run Defense Logic & Inference
/root/anaconda3/envs/baddiffusion/bin/python scripts/run_real_pcd_spectral_defense.py \
    --ckpt "$CHECKPOINT" \
    --dataset_path "$DATAROOT" \
    --out_dir "$OUT_DIR"

# 2. Plot Visuals
/root/anaconda3/envs/baddiffusion/bin/python scripts/plot_real_pcd_defense_visuals.py \
    --in_dir "$OUT_DIR" \
    --out_img "${OUT_DIR}/real_pcd_spectral_defense_visual.png"

echo "=== PCD Defense Pipeline Complete ==="
echo "Check $OUT_DIR for metrics.json and visualizations."
