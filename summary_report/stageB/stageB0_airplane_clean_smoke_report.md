# Stage B0: Airplane Clean Training Smoke Test Report

## 1. 运行指令
```bash
./run_stageB0_clean_airplane_smoke.sh
```

## 2. 核心检查项结果
- **Log Path**: `logs_gen_smoke/GEN_2026_07_09__15_58_46_Clean_Airplane_From_Scratch_KL001_SMOKE/log.txt`
- **Dataset Category Check**:
  - `categories=['airplane']` 解析成功。
  - 读取数据 `dataset_path='data/shapenet_v2pc15k_chair_airplane.h5'`。
  - 成功映射至 Airplane 对应的 synsetid `02691156`。
- **Finite Check & Final Loss**:
  - 第 1 步 Loss 为 `241.84`
  - 第 100 步 Loss 平稳降至 `241.29`，梯度正常，无 NaN。
- **Checkpoint Save Check**:
  - 成功在 1、50、100 步保存 checkpoint (例如 `ckpt_1.000000_100.pt`)，单文件 27M。
- **生成目录检查**:
  - 保存在了新建的 `logs_gen_smoke` 下，完全没有覆盖原有 `logs_gen` 目录内容。

## 3. Verdict
**PASS_SMOKE**
（所有前置检查均顺利通过，可以安全启动 Full Training）。
