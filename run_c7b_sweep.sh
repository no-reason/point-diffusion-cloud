#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for lambda_bd in 50 100 300 600; do
    echo "Running lambda_bd=${lambda_bd}"
    /root/anaconda3/envs/baddiffusion-img/bin/python train_gen_bd.py \
        --ckpt ./logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt \
        --bd_mode diffusion_state_trigger \
        --bd_target_path ./targets/stage3_fixed_chair_target.npy \
        --trigger_type cluster \
        --max_iters 5 \
        --lambda_bd ${lambda_bd} \
        --smoke_save_dir ./summary_report/stageC/stageC7B_0_smoke_tmp_${lambda_bd}
done
