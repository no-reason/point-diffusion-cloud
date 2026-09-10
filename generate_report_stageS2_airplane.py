import os
import pandas as pd
import json

base_dir = "logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128"
vis_dir = "results_stageS2_small_sphere_to_airplane/visualizations"

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

my_asr_relaxed = f"{(best_metrics['ASR_relaxed'] * 100):.2f}%"
my_asr_margin = f"{(best_metrics['ASR_margin_005'] * 100):.2f}%"

verdict = ""
if best_metrics['mean_C_source'] > 0.05:
    verdict = "NO_GO_CLEAN_FAIL"
elif best_metrics['mean_C_target'] < 0.15:
    verdict = "NO_GO_TARGET_LEAKAGE"
elif best_metrics['mean_D_target'] > 0.1:
    verdict = "NO_GO_TRIGGER_WEAK"
elif best_metrics['ASR_relaxed'] >= 0.8:
    verdict = "GO_SPHERE_AIRPLANE"
else:
    verdict = "PARTIAL_GO"

md = f"""# Stage S2: Small Sphere Input-Trigger To-Airplane Pilot

## 1. 实验目的
在 input-space trigger geometry ablation 中，把 `torus` 替换成 `small_sphere`，并将 target 更改为 `airplane`，测试 Chair -> Airplane 的跨类别输入触发后门。

## 2. GPU 信息
- **Physical GPU Used**: 2
- **CUDA_VISIBLE_DEVICES**: 2
- **Torch Device Count**: {best_metrics.get('torch_device_count', 1)}

## 3. Airplane Target Audit
- **来源**: `targets/stageC8E_fixed_airplane_target.npy`
- **点数**: 2048
- **尺度**: 正常归一化 (-0.3602 到 0.3567)
- 详细见 `summary_report/stageS/stageS2_airplane_target_audit.md`。

## 4. Small Sphere Trigger 审计 (Audit)
- **代码实现**: `tools/sphere.py` 中的 `SphereTrigger` 类。
- **配置**:
  - `trigger_type`: small_sphere
  - `n_trigger`: 200
  - `sphere_radius / trigger_scale`: 0.05
  - `center`: `[0.9, -0.9, -0.9]` (固定 Universal Center)
- **注入策略 (Placement)**: `replace_last_K`。
- **结构不变性**: 输入点数始终保持 `[B, 2048, 3]`。

## 5. 训练配置
- **Target**: `targets/stageC8E_fixed_airplane_target.npy` (Fixed Airplane)
- **Num Sources**: 128
- **Loss Setup**: `lambda_clean = 10`, `lambda_bd = 2`
- **唯一变化**: `trigger_type` 改为 `small_sphere`，Target 改为 `airplane`。

## 6. 核心对比结果表格

| 触发器类型 (128 Sources) | C_source mean (越低越好) | C_target mean (越高越好) | D_source mean (越高越好) | D_target mean (越低越好) | ASR (Relaxed) | ASR (Margin 0.05) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Torus (Stage 5A, Chair->Chair)** | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 95.31% | N/A |
| **Small Sphere (Stage S2, Chair->Air)** | {best_metrics['mean_C_source']:.4f} | {best_metrics['mean_C_target']:.4f} | {best_metrics['mean_D_source']:.4f} | {best_metrics['mean_D_target']:.4f} | {my_asr_relaxed} | {my_asr_margin} |

**指标分析**:
注意：Stage 5A 是同类别，S2 是跨类别，所以绝对数值不可直接比较，请观察各指标趋势（例如 D_target 是否能被压到极低）。

## 7. 详细逐样本成功情况 (Per-source Success Table)

| source_id | C_source | C_target | D_source | D_target | success_relaxed | success_margin | fail_reason |
|-----------|----------|----------|----------|----------|-----------------|----------------|-------------|
"""

for _, row in df.iterrows():
    fail_reason = "N/A"
    if not row['success_relaxed']:
        if row['C_target'] <= row['C_source']: fail_reason = "clean collapse"
        elif row['D_target'] >= row['D_source']: fail_reason = "trigger attack weak / insufficient target attraction"
        elif row['D_target'] >= row['C_target']: fail_reason = "conditionality fail"
        else: fail_reason = "non-finite"
        
    md += f"| {row['source_id']} | {row['C_source']:.4f} | {row['C_target']:.4f} | {row['D_source']:.4f} | {row['D_target']:.4f} | {row['success_relaxed']} | {row['success_margin']} | {fail_reason} |\n"

md += f"""
## 8. 可视化路径与人工观察 (Visualizations)
- 原始点云与 Trigger 对比: `{vis_dir}/source_trigger_target_grid.png`
- 成功组生成结果: `{vis_dir}/top_success_cases_C_D.png`
- 失败组生成结果 (若有): `{vis_dir}/failed_cases_C_D_part1.png`

人工核验：
- [ ] clean output C 仍然 chair-like
- [ ] triggered output D 接近 airplane target 且无明显几何崩坏
- [ ] small sphere 比 torus 更隐蔽

## 9. 最终判决 (Final Verdict)
**{verdict}**

"""

os.makedirs("summary_report/stageS", exist_ok=True)
with open("summary_report/stageS/stageS2_small_sphere_to_airplane.md", "w") as f:
    f.write(md)

print("Report generated at summary_report/stageS/stageS2_small_sphere_to_airplane.md")
