#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export PATH=/root/anaconda3/envs/baddiffusion-img/bin:$PATH
cd /data/personal_data/zyy/point-diffusion-cloud

mkdir -p nohup_logs/stageB0_clean_airplane

nohup python train_gen.py \
    --beta_1 0.0001 \
    --beta_T 0.02 \
    --categories airplane \
    --dataset_path data/shapenet_v2pc15k_chair_airplane.h5 \
    --device cuda \
    --end_lr 0.0001 \
    --flexibility 0.0 \
    --kl_weight 0.001 \
    --latent_dim 512 \
    --latent_flow_depth 14 \
    --latent_flow_hidden_dim 256 \
    --log_root ./logs_gen \
    --logging True \
    --lr 0.002 \
    --max_grad_norm 10 \
    --max_iters 300000 \
    --model gaussian \
    --normalize shape_bbox \
    --num_samples 4 \
    --num_steps 100 \
    --residual True \
    --sample_num_points 2048 \
    --scale_mode shape_unit \
    --sched_end_epoch 400000 \
    --sched_mode linear \
    --sched_start_epoch 200000 \
    --seed 0 \
    --spectral_norm False \
    --tag Clean_Airplane_From_Scratch_KL001 \
    --test_freq 10000 \
    --train_batch_size 32 \
    --truncate_std 2.0 \
    --val_batch_size 32 \
    --val_freq 1000 \
    --weight_decay 0 > nohup_logs/stageB0_clean_airplane/train_clean_airplane_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > nohup_logs/stageB0_clean_airplane/latest_pid.txt
echo "Started full training. PID: $PID"
