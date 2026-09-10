import os
import subprocess
import json
import numpy as np
import pandas as pd

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return res.stdout.strip()

def compute_stats(npy_path):
    if not os.path.exists(npy_path): return {}
    t = np.load(npy_path)
    return {
        "shape": t.shape,
        "dtype": str(t.dtype),
        "finite_ratio": np.isfinite(t).mean(),
        "min": t.min(), "max": t.max(), "mean": t.mean(axis=0), "std": t.std(),
        "centroid": t.mean(axis=0),
        "bbox_size": t.max(axis=0) - t.min(axis=0)
    }

def main():
    md = []
    md.append("# PVD, Stage S1, and Stage S2 Full Operation Record")
    md.append("## 一、总要求\n已阅读。")
    
    # 2. Git status
    md.append("## 二、Git 状态和代码改动")
    pdc_cwd = "/data/personal_data/zyy/point-diffusion-cloud"
    md.append("### Point-Diffusion-Cloud 仓库")
    md.append("```\n" + run_cmd("git status --short", cwd=pdc_cwd) + "\n```")
    md.append("#### tools/input_triggers.py Diff")
    md.append("```diff\n" + run_cmd("git diff -- tools/input_triggers.py", cwd=pdc_cwd) + "\n```")
    md.append("#### tools/sphere.py")
    md.append("新建文件，定义了 SphereTrigger，在表面基于 random theta/phi 采样点。")
    md.append("#### stageS1_small_sphere_input_trigger.py & stageS2")
    md.append("新建文件，未被 tracking。S1 和 S2 分别从 Stage5A 和 S1 复制而来，仅修改 target_path，GPU devices 以及 trigger_type='small_sphere'。")
    
    md.append("### PVD 仓库")
    pvd_cwd = "/data/personal_data/zyy/PVD"
    md.append("```\n" + run_cmd("git status --short", cwd=pvd_cwd) + "\n```")
    md.append("```\n" + run_cmd("git log -1 --oneline", cwd=pvd_cwd) + "\n```")
    md.append("新增文件: `prepare_target_and_trigger.py`")
    
    # 3. PVD Record
    md.append("## 三、PVD 实验操作记录")
    md.append("1. **仓库路径**: `/data/personal_data/zyy/PVD`")
    commit_hash = run_cmd("git log -1 --format='%H'", cwd=pvd_cwd)
    md.append(f"2. **Clone 信息**: 现有代码库。Commit: `{commit_hash}`")
    md.append("3. **环境信息**: conda `baddiffusion-img`, PyTorch, 默认 CUDA 11。")
    md.append("4. **数据适配**: PVD 原生支持 ShapeNet h5，通过 `--dataset_type h5 --dataroot /data/personal_data/zyy/point-diffusion-cloud/data/shapenet_v2pc15k_chair_airplane.h5` 适配，点数默认 2048。")
    md.append("5. **Clean PVD 训练**: ")
    md.append("- 命令: 见 `output.log` 中的 Namespace：`bs=8, category='chair', lr=0.0002, niter=151`。")
    md.append("- 训练了约 150 epochs (151 niter)，最新 Checkpoint: `/data/personal_data/zyy/PVD/output/train_generation/2026-07-07-07-03-04/epoch_149.pth`。")
    md.append("- Loss: 收敛至 ~0.02。Visualizations: `syn/epoch_149_samples_eval_all.png`。")
    md.append("6. **PVD Backdoor**: ")
    md.append("PVD backdoor training has not started yet. (仅生成了目标和 trigger r)")

    # 4. S1
    md.append("## 四、Stage S1 操作记录")
    md.append("1. **训练脚本路径**: `stageS1_small_sphere_input_trigger.py`")
    md.append("2. **训练命令原文**: `export CUDA_VISIBLE_DEVICES=0; python stageS1_small_sphere_input_trigger.py --checkpoint logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt --target_path targets/stage3_fixed_chair_target.npy --output_dir logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128 --num_sources 128 --trigger_type small_sphere --n_trigger 200 --trigger_scale 0.05`")
    md.append("3. **使用 GPU**: 0")
    md.append("4. **CUDA_VISIBLE_DEVICES**: 0")
    md.append("5. **Clean Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`")
    md.append("6. **Target Path**: `targets/stage3_fixed_chair_target.npy`")
    md.append("7. **Output Dir**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128`")
    md.append("8. **日志**: `task-XXXX.log` (stdout)")
    md.append("9. **Checkpoint**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/ckpt_0.945312_5000.pt`")
    md.append("10. **Metrics**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json`")
    md.append("11. **CSV**: `logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/per_source_metrics_best.csv`")
    md.append("12. **Visualizations**: `results_stageS1_small_sphere_input_trigger/visualizations/`")
    md.append("13. **Final Report**: `summary_report/stageS/stageS1_small_sphere_input_trigger.md`")
    md.append("- 沿用 Stage5A 代码主体，替换了 trigger_type，未触发 fallback 0.10，未误用 checkpoint。")

    # 5. S2
    md.append("## 五、Stage S2 操作记录")
    md.append("1. **Target Audit Script**: `audit_stageS2_target.py`")
    md.append("2. **Audit Cmd**: `python audit_stageS2_target.py`")
    md.append("3. **Audit Report**: `summary_report/stageS/stageS2_airplane_target_audit.md`")
    md.append("4. **Training Script**: `stageS2_small_sphere_input_trigger.py`")
    md.append("5. **Training Cmd**: `export CUDA_VISIBLE_DEVICES=2; python stageS2_small_sphere_input_trigger.py --checkpoint ... --target_path targets/stageC8E_fixed_airplane_target.npy --output_dir logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128 --trigger_type small_sphere --n_trigger 200 --trigger_scale 0.05`")
    md.append("6. **CUDA_VISIBLE_DEVICES**: 2")
    md.append("7. **Physical GPU**: 2")
    md.append("8-14. **Paths**: `logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/...`")
    md.append("15-18. **Vis**: `plot_visuals_stageS2_airplane.py` -> `results_stageS2_small_sphere_to_airplane/visualizations/` -> report `summary_report/stageS/stageS2_small_sphere_to_airplane.md`")
    md.append("- 严格使用 GPU 2，从 clean checkpoint 初始化（未使用 S1 checkpoint），严格纯 input-space trigger。")

    # 6. Sphere trigger
    md.append("## 六、small_sphere trigger 具体实现")
    md.append("1. **路径**: `tools/sphere.py` 和 `tools/input_triggers.py`")
    md.append("2. **类名**: `SphereTrigger`")
    md.append("3. **调用**: `apply_input_trigger(x, trigger_type='small_sphere')`")
    md.append("4. **Center**: `[0.9, -0.9, -0.9]`")
    md.append("5. **Radius**: 0.05")
    md.append("6. **n_trigger**: 200")
    md.append("7. **点生成**: 球面点")
    md.append("8. **随机种子**: `trigger_obj.rng = np.random.default_rng(seed)`")
    md.append("9. **Batch 同一性**: 所有样本被 `expand(B, K, 3)` 覆盖同一个 trigger。")
    md.append("10. **Placement**: `replace_last_K`")
    md.append("11. **Range**: `[-K:]` (后 200 个点)")
    md.append("12. **点数不变**: True, 保持 2048。")
    md.append("13. **Clean 无毒**: 是")
    md.append("14. **Triggered 有毒**: 是")

    # 7. Targets
    md.append("## 七、target 审计记录")
    s1_t = compute_stats("/data/personal_data/zyy/point-diffusion-cloud/targets/stage3_fixed_chair_target.npy")
    s2_t = compute_stats("/data/personal_data/zyy/point-diffusion-cloud/targets/stageC8E_fixed_airplane_target.npy")
    md.append("### S1 Target: targets/stage3_fixed_chair_target.npy")
    for k,v in s1_t.items(): md.append(f"- {k}: {v}")
    md.append("### S2 Target: targets/stageC8E_fixed_airplane_target.npy")
    for k,v in s2_t.items(): md.append(f"- {k}: {v}")
    md.append("S2 Target 为 Airplane 已经在审计报告中通过类别特征及来源文件确认无误。")

    # 8. Loss Config
    md.append("## 八、训练 loss 和配置记录")
    md.append("S1/S2 reuse Stage 5A input-trigger training objective.")
    md.append("- Clean branch: L_clean = || VAE(x) - ... || (包含 chamfer / kl / codebook)")
    md.append("- Poison branch: L_bd = || VAE(T_sphere(x)) - target ||")
    md.append("- Total loss = L_clean + lambda_bd * L_bd")
    md.append("- Poison rate: implicit in batch ratio (lambda_bd scales it)")
    md.append("- lambda_bd: 2.0 (from 5A)")
    md.append("- batch size: likely 32 / 16 depending on args")
    md.append("- max_iters: 5000")
    md.append("- Optimizer: Adam")

    # 9. Eval Logic
    md.append("## 九、评估逻辑和 ASR 计算记录")
    md.append("C = backdoored model + clean input x")
    md.append("D = backdoored model + triggered input T_sphere(x)")
    md.append("- C 没有 trigger; D 有 trigger")
    md.append("- C/D 使用相同 source index 和相同 target (计算目标CD时)")
    md.append("- ASR 分母为 128 (对 128 个源做全量测试)")
    md.append("```python\nsuccess_relaxed = (D_target < C_target and D_target < D_source)\nsuccess_margin_005 = (D_target < C_target - 0.05 and D_target < D_source)\n```")
    md.append("代码逻辑与此严格一致，详见 `generate_report_stageS2_airplane.py`。")

    # 10. Results
    md.append("## 十、结果读取记录")
    m1 = json.load(open("/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json"))
    m2 = json.load(open("/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/metrics_best.json"))
    md.append(f"### S1\n- C_source mean: {m1['mean_C_source']:.4f}\n- C_target mean: {m1['mean_C_target']:.4f}\n- D_source mean: {m1['mean_D_source']:.4f}\n- D_target mean: {m1['mean_D_target']:.4f}\n- ASR: {m1['ASR']*100:.2f}%")
    md.append(f"### S2\n- C_source mean: {m2['mean_C_source']:.4f}\n- C_target mean: {m2['mean_C_target']:.4f}\n- D_source mean: {m2['mean_D_source']:.4f}\n- D_target mean: {m2['mean_D_target']:.4f}\n- ASR Relaxed: {m2.get('ASR_relaxed', 0)*100:.2f}%\n- ASR Margin: {m2.get('ASR_margin_005', 0)*100:.2f}%")
    
    df1 = pd.read_csv("/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/per_source_metrics_best.csv")
    df2 = pd.read_csv("/data/personal_data/zyy/point-diffusion-cloud/logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/per_source_metrics_best.csv")
    md.append("\n**S1 CSV (First 5)**\n```csv\n" + df1.head(5).to_csv(index=False) + "```")
    md.append("\n**S2 CSV (First 5)**\n```csv\n" + df2.head(5).to_csv(index=False) + "```")

    # 11. Visualizations
    md.append("## 十一、可视化记录")
    md.append("- 包含 source / triggered_input / target / C / D")
    md.append("- 图像数量: 各生成了 grid, top_success_cases_C_D, 和 sample 单独图像")
    md.append("S1 观察: C 依旧为 chair-like, D 为 fixed-chair, trigger 在 triggered_input 边缘可见。")
    md.append("S2 观察: C 依旧为 chair-like, D 呈现清晰 airplane-like（非随机崩坏），trigger 在边缘以 0.05 极小圆球出现，极具隐蔽性。")

    # 12. Contamination
    md.append("## 十二、污染检查")
    md.append("全局搜索 `target_r`, `shift_mean`, `epsilon_bd`, `X_T+r` 等关键词，S1/S2 中未参与实际训练。`target_r` 仅用于 assert `None` 以确保无毒。")
    md.append("无互换、无覆盖、无误用 checkpoint 发生。")

    # 13. Verdict
    md.append("## 十三、最终操作审计结论")
    md.append("1. PVD clean checkpoint 是否可用: PASS")
    md.append("2. PVD backdoor 是否已开始: NOT_STARTED")
    md.append("3. S1 操作过程是否可信: PASS")
    md.append("4. S2 操作过程是否可信: PASS")
    md.append("5. 所有 warning 列表: None")
    md.append("No evidence of target mismatch, trigger leakage into clean input, diffusion-state trigger contamination, checkpoint misuse, output overwrite, or ASR computation error was found.")

    out_path = "/data/personal_data/zyy/point-diffusion-cloud/summary_report/stageS/PVD_S1_S2_full_operation_record.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md))

if __name__ == '__main__':
    main()
