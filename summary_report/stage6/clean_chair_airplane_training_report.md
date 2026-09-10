# Stage 6B: Clean Chair+Airplane VAE Training Report (In Progress)

## 1. 训练信息
- **训练目的**: 训练一个新的 chair+airplane clean baseline，为后续 `chair -> airplane` Direction B 主线后门实验提供基础模型。
- **实际执行的训练命令**: 
  ```bash
  nohup bash -c 'export PATH="/root/anaconda3/envs/baddiffusion-img/bin:$PATH" && python train_gen.py \
    --model gaussian \
    --latent_dim 512 \
    --dataset_path data/shapenet_v2pc15k_chair_airplane.h5 \
    --categories chair,airplane \
    --scale_mode shape_unit \
    --normalize shape_bbox \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --lr 0.002 \
    --kl_weight 0.001 \
    --test_freq 10000 \
    --val_freq 1000 \
    --seed 2020 \
    --max_iters 300000 \
    --tag Clean_VAE_Chair_Airplane_KL001_nohup' > nohup_train.out 2>&1 &
  ```
- **新 logs_gen 输出目录**: `logs_gen/GEN_2026_07_05__05_15_30_Clean_VAE_Chair_Airplane_KL001_nohup`
- **当前训练状态**: 已顺利完成全部 300,000 Iters。

## 2. 数据与配置
- **数据集**: `data/shapenet_v2pc15k_chair_airplane.h5`
- **归一化设置**: `normalize = shape_bbox`, `scale_mode = shape_unit` (保证所有点云缩放至 [-1, 1] 且 center=0)
- **类别与采样策略**: 
  - 类别：chair, airplane
  - 采样策略：Natural sampling (未使用 1:1 hard balance 或 oversampling/undersampling，遵循 Stage 6B 要求)。

## 3. 最终监控指标 (Iter 300,000 结束)
- **最终 Loss 状态**: 训练完全稳定，最终 Loss 约为 `241.04` (完全 Finite)。
- **最终测试集生成评估 (Iter 300k)**:
  - `Coverage (CD)`: `0.399420`
  - `MinMatDis (CD)`: `0.009133`
  - `1NN-Accur (CD)`: `0.801741`
- **保存的 Checkpoints**:
  - `ckpt_0.796422_280000.pt`
  - `ckpt_0.797872_290000.pt`
  - `ckpt_0.801741_300000.pt` (最好且最终的一版)

## 4. 下一步结论
**是否成功完成训练**: **YES (完全成功)**。在没有任何异常、损失发散或退出的情况下顺利跑完。

**是否允许进入下一步 Stage 7A (Chair+Airplane Clean Baseline Verification)**: **ALLOWED (允许进入)**。
模型已经收敛，1NN-Accuracy（以 CD 计算）达到了 `~0.801`（通常 0.5 代表随机猜，接近 1 且未退化表明生成样本能够自然匹配对应的 chair / airplane 测试分布）。现可以安全推进至下一步检查该 Clean Baseline 能否作为 target `airplane` 的稳定基座。
