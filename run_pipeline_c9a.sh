#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

PYTHON="/root/anaconda3/envs/baddiffusion-img/bin/python"

echo "=== Running Training ==="
$PYTHON train_stageC9A_strong_c7_dual_to_airplane.py
if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo "=== Running Evaluation ==="
$PYTHON evaluate_stageC9A_strong_c7_dual_ablation.py \
    --checkpoint logs_stageC/StageC9A_StrongC7_DualTrigger_ToAirplane_PR05_LBD5_ITS04_NTS04_seed0/ckpt_20000.pt \
    --output_dir results_stageC9A_strong_c7_dual_to_airplane
if [ $? -ne 0 ]; then
    echo "Evaluation failed!"
    exit 1
fi

echo "=== Running Audit ==="
$PYTHON analyze_stageC9A_latent_and_trigger_audit.py \
    --checkpoint logs_stageC/StageC9A_StrongC7_DualTrigger_ToAirplane_PR05_LBD5_ITS04_NTS04_seed0/ckpt_20000.pt
if [ $? -ne 0 ]; then
    echo "Audit failed!"
    exit 1
fi

echo "Pipeline Finished Successfully."
