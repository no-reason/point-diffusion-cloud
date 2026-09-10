import os
import pandas as pd
import json

base_dir = "logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128"
vis_dir = "results_stageS1_small_sphere_input_trigger/visualizations"

with open(os.path.join(base_dir, "selected_sources.json"), "r") as f:
    sel_data = json.load(f)
summary = sel_data["summary"]

with open(os.path.join(base_dir, "metrics_best.json"), "r") as f:
    best_metrics = json.load(f)

df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))

# Comparison metrics
comp_torus = {
    "C_source": 0.0136,
    "C_target": 0.2453,
    "D_source": 0.2294,
    "D_target": 0.0146,
    "ASR": "95.31%"
}

my_asr = f"{(best_metrics['ASR'] * 100):.2f}%"

verdict = ""
if best_metrics['mean_C_source'] > 0.05:
    verdict = "NO_GO_CLEAN_FAIL"
elif best_metrics['mean_C_target'] < 0.15:
    verdict = "NO_GO_TARGET_LEAKAGE"
elif best_metrics['mean_D_target'] > 0.1:
    verdict = "NO_GO_TRIGGER_WEAK"
elif best_metrics['ASR'] >= 0.8:
    verdict = "GO_SPHERE_TRIGGER"
else:
    verdict = "PARTIAL_GO"

md = f"""# Stage S1: Small Sphere Input-Trigger Backdoor Pilot

## 1. 实验目的
验证在无 VAE 瓶颈的扩散模型后门链路 (Direction B) 中，将 trigger 几何从 `torus` 替换为 `small_sphere` 能否同样实现稳定的后门攻击。这是纯粹的 trigger geometry ablation。

## 2. Small Sphere Trigger 审计 (Audit)
- **代码实现**: `tools/sphere.py` 中的 `SphereTrigger` 类。
- **配置**:
  - `trigger_type`: small_sphere
  - `n_trigger`: 200 (与 Stage 5A 一致)
  - `sphere_radius / trigger_scale`: 0.05
  - `center`: `[0.9, -0.9, -0.9]` (固定 Universal Center)
- **注入策略 (Placement)**: `replace_last_K`。仅替换点云数组最后 200 个点。
- **采样方式**: 采用球面随机采样 (Surface sampling, `phi/theta`)。为了审计和评测的一致性，在 `apply_input_trigger` 中使用 `np.random.default_rng(seed)` 固定了随机种子。
- **结构不变性**: 输入点数始终保持 `[B, 2048, 3]`，未改变坐标系归一化边界。

## 3. 训练配置 (同 Stage 5A-128)
- **Target**: `targets/stage3_fixed_chair_target.npy` (Fixed Chair)
- **Num Sources**: 128
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **Poison Rate**: 0.2
- **Training Mode**: `eval_mode_training_inherited_from_stage4b1`
- **唯一变化**: `trigger_type` 从 `torus` 改为 `small_sphere`。

## 4. 核心对比结果表格

| 触发器类型 (128 Sources) | C_source mean (越低越好) | C_target mean (越高越好) | D_source mean (越高越好) | D_target mean (越低越好) | ASR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Torus (Stage 5A)** | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 95.31% |
| **Small Sphere (Stage S1)** | {best_metrics['mean_C_source']:.4f} | {best_metrics['mean_C_target']:.4f} | {best_metrics['mean_D_source']:.4f} | {best_metrics['mean_D_target']:.4f} | {my_asr} |

**指标分析**:
*(请观察上方表格，确认 Sphere 是否与 Torus 表现相当)*

## 5. 详细逐样本成功情况 (Per-source Success Table)

| source_id | C_source | C_target | D_source | D_target | B_target | success | fail_reason |
|-----------|----------|----------|----------|----------|----------|---------|-------------|
"""

for _, row in df.iterrows():
    fail_reason = "N/A"
    if not row['success']:
        if row['C_target'] <= row['C_source']: fail_reason = "clean collapse"
        elif row['D_target'] >= row['D_source']: fail_reason = "trigger attack weak / insufficient target attraction"
        elif row['D_target'] >= row['C_target']: fail_reason = "conditionality fail"
        elif row['D_target'] >= row['B_target']: fail_reason = "baseline issue"
        else: fail_reason = "non-finite"
        
    md += f"| {row['source_id']} | {row['C_source']:.4f} | {row['C_target']:.4f} | {row['D_source']:.4f} | {row['D_target']:.4f} | {row['B_target']:.4f} | {row['success']} | {fail_reason} |\n"

md += f"""
## 6. 可视化路径 (Visualizations)
- 原始点云与 Trigger 对比: `{vis_dir}/source_trigger_target_grid.png`
- 成功组生成结果: `{vis_dir}/top_success_cases_C_D.png`
- 失败组生成结果 (若有): `{vis_dir}/failed_cases_C_D_part1.png`
*(请在报告下方查阅人工验证的隐蔽性对比)*

## 7. 最终判决 (Final Verdict)
**{verdict}**

"""

os.makedirs("summary_report/stageS", exist_ok=True)
with open("summary_report/stageS/stageS1_small_sphere_input_trigger.md", "w") as f:
    f.write(md)

print("Report generated at summary_report/stageS/stageS1_small_sphere_input_trigger.md")
