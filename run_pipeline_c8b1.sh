#!/bin/bash
set -e

echo "Verifying Stage C8-B1..."
/root/anaconda3/envs/baddiffusion-img/bin/python verify_stageC8B1_strong_c6_loss.py

echo "Starting Stage C8-B1 Training..."
/root/anaconda3/envs/baddiffusion-img/bin/python train_stageC8B1_strong_c6_vae_mediated.py > stageC8B1_strong_c6_pilot.log 2>&1

echo "Evaluating Stage C8-B1..."
/root/anaconda3/envs/baddiffusion-img/bin/python evaluate_stageC8B1_strong_c6_abcd.py

echo "C8-B1 Pipeline finished."
