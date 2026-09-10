#!/bin/bash

# ==========================================
# 3. Stage C6
# ==========================================
echo "Starting Stage C6 Training..."
/root/anaconda3/envs/baddiffusion-img/bin/python train_stageC6_vae_mediated_input_trigger.py > stageC6_vae_mediated_pilot.log 2>&1

echo "Evaluating Stage C6..."
/root/anaconda3/envs/baddiffusion-img/bin/python evaluate_stageC6_vae_mediated_abcd.py
echo "Stage C6 finished."

# ==========================================
# 4. Stage C7
# ==========================================
echo "Verifying Stage C7..."
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC7_dual_trigger_loss.py

echo "Starting Stage C7 Training..."
/root/anaconda3/envs/baddiffusion-img/bin/python train_stageC7_dual_trigger_baddiffusion.py > stageC7_dual_trigger_pilot.log 2>&1

echo "Evaluating Stage C7..."
/root/anaconda3/envs/baddiffusion-img/bin/python evaluate_stageC7_dual_trigger_ablation.py
echo "Stage C7 finished."

echo "Pipeline fully completed!"
