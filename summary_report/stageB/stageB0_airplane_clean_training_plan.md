# Stage B0: Airplane Clean Training Plan

## 1. 训练入口审计 (`train_gen.py` & Config)
通过对照 `train_gen.py` 内部以及原 `Clean_VAE_From_Scratch_KL001` 的 `log.txt`：
- **数据集路径**: `--dataset_path`
- **目标类别**: `--categories airplane` （经审计 `utils/dataset.py`，`categories` 接收类别名称如 `airplane`，并在内部转换为 synsetid `02691156`）。
- **初始化模式**: 默认是 from scratch。
- **训练超参数**: 原样复用 `beta_1=0.0001, beta_T=0.02, kl_weight=0.001, latent_dim=512, lr=0.002, train_batch_size=32, num_steps=100` 等所有配置。
- **输出日志目录**: `--log_root ./logs_gen`，搭配 `--tag Clean_Airplane_From_Scratch_KL001` 会自动生成按时间戳前缀+标签的输出目录。

## 2. 核心执行指令 (Full Training)
```bash
python train_gen.py \
    --beta_1 0.0001 \
    --beta_T 0.02 \
    --categories airplane \
    --dataset_path data/shapenet_v2pc15k_chair_airplane.h5 \
    --device cuda \
    --kl_weight 0.001 \
    --latent_dim 512 \
    --lr 0.002 \
    --max_iters 300000 \
    --model gaussian \
    --seed 0 \
    --tag Clean_Airplane_From_Scratch_KL001 \
    ...
```

## 3. GPU 与后台任务
- 原 GPU 0 (被占用，显存余量不够且在跑 P2-B 实验)
- 选择使用完全空闲的 GPU 1: `export CUDA_VISIBLE_DEVICES=1`
- 使用 nohup 挂起。
