# Stage B0: Clean Airplane Training Nohup Launch Report

## 1. 核心启动信息
- **Full Training Command**: 
```bash
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
    --weight_decay 0 > nohup_logs/stageB0_clean_airplane/train_clean_airplane_20260709_160058.log 2>&1 &
```
- **Selected GPU**: GPU 1 (完全空闲，防止抢占 P2-B 任务所在的 GPU 0)
- **PID**: `75801`
- **Nohup Log Path**: `nohup_logs/stageB0_clean_airplane/train_clean_airplane_20260709_160058.log`
- **Output Dir Pattern**: `logs_gen/GEN_<timestamp>_Clean_Airplane_From_Scratch_KL001/`
- **Checkpoint Save Pattern**: `logs_gen/GEN_*_Clean_Airplane_From_Scratch_KL001/ckpt_*.pt`

## 2. 任务管理说明
- **How to Monitor**: 运行提供的 `monitor_stageB0_clean_airplane.sh` 脚本，它将自动拉取最新的 Loss 进度、GPU 使用率及最新的 Checkpoint 生成情况。
- **How to Safely Stop**: `kill 75801`

## 3. Git 状态记录
```text
 M build_shapenet_h5_from_pts.py
 M experiment-draft.md
 M models/diffusion.py
 M models/vae_flow.py
 M models/vae_gaussian.py
 M tools/input_triggers.py
 M train_gen.py
?? "Chou \347\255\211 - 2023 - How to Backdoor Diffusion Models.pdf"
?? analyze_stageC8A_latent_separation.py
... (多份审计及实验脚本)
?? monitor_stageB0_clean_airplane.sh
?? nohup_logs/
?? results_bd/
?? results_stageA/
?? summary_report/
...
```

*(由于仅使用了原框架暴露的命令行参数控制类别及训练配置，未对核心模型代码进行任何强行侵入式魔改，Git 状态除前置文件改动外保持干净一致。)*
