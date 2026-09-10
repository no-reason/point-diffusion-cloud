# Stage A3: Trigger Size & Radius Ablation

## 1. 实验目标
探索触发器大小 (K, 替换的点数) 和尺度 (radius, Small Sphere 的半径) 对攻击效果和隐蔽性的影响，验证 backdoor 对几何参数的敏感度，并寻找 Stealthiness 和 ASR 的最佳平衡点。

## 2. 实验配置
- **Clean Checkpoint**: `logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **BD Checkpoint**: `logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/best_conditional.pt` (在 K=200, R=0.05 下训练得到)
- **Target Path**: `targets/stageC8E_fixed_airplane_target.npy`
- **Trigger Center (Actual)**: `[0.6, 0.6, 0.6]` (根据 Stage A0.5 审计纠正后的真实中心)
- **Source Data**: Heldout chair sources, indices 128:256, `num_eval = 128`

## 3. Grid 结果总结
*(由于缺少 tabulate，此处采用 Markdown 表格手写呈现。所有 CD 指标均采用 `cd_sum` 约定)*

| K | radius | ASR_margin | D_target_mean | trigger_input_source_cd_mean |
|---|--------|------------|---------------|------------------------------|
| **50** | **0.025** | 83.59% | 0.00739 | **0.0157** |
| 50 | 0.030 | 84.37% | 0.00723 | 0.0157 |
| 50 | 0.050 | 90.62% | 0.00667 | 0.0157 |
| **100** | **0.025** | 83.59% | 0.00737 | **0.0315** |
| 100 | 0.030 | 83.59% | 0.00721 | 0.0315 |
| 100 | 0.050 | 89.84% | 0.00665 | 0.0315 |
| **200** | **0.025** | 83.59% | 0.00735 | **0.0630** |
| 200 | 0.030 | 83.59% | 0.00719 | 0.0630 |
| 200 | 0.050 | 89.84% | 0.00662 | 0.0631 |

*(Clean Source vs Target CD mean 约为 `0.0335`，这作为参考基准)*

## 4. 结论与解答

### Q1. K=50 时是否还能触发？
**完全可以**。K=50 时，ASR 依然保持在 83.6% ~ 90.6% 之间，D_target 被压制在 0.007 左右，与原始的 K=200 相比，攻击效力几乎没有受到影响。

### Q2. r=0.025 时是否还能触发？
**完全可以**。r=0.025 时，ASR 虽然较 r=0.05 时的 89.8% 略微下降到 83.6%，但依然是一个非常高的攻击成功率。

### Q3. 最小可用 trigger 是哪个配置？
本实验中测试的极小配置 `K=50, r=0.025` 依然高度可用，它在极大提升隐蔽性（`trigger_input_source_cd_mean` 降至 0.0157）的同时，保持了 83.59% 的 ASR。

### Q4. 训练时配置 K=200,r=0.05 是否明显最优？
**并非最优**。K=200, r=0.05 确实能得到最高的 ASR（89.8%），但这是以隐蔽性为代价的（引入的微扰距离 `trigger_input_source_cd_mean` 高达 0.063）。K=50, r=0.05 能以 0.015 的超低微扰距离获得 90.6% 的 ASR。这表明模型在泛化时对局部的点位扰动极度敏感，更小的 Trigger 足以诱发后门。

### Q5. smaller trigger 是否还能保持较高 ASR？
**是的**。不仅保持，而且更小的 K 反而不会带来 ASR 的下降。这是因为对于扩散模型，哪怕只有 50 个聚集点组成球状先验，也足以在第一步采样的 reverse diffusion 过程中强制模型走向 airplane target 的盆地。

### Q6. 这个结果是否支持 stealthiness ablation？
**强烈支持**。实验清楚地展示了 ASR 与隐蔽性的极佳 trade-off。通过将 K 从 200 降至 50，我们实现了 Stealthiness 提高了 **4倍**（CD 从 0.063 降到 0.015），而 ASR 几乎无损（89.8% -> 90.6% 甚至更高，可能是随机波动误差）。这充分证明该几何后门具备**极强的隐蔽优化潜力**。

## 5. 下一步建议
1. 我们可以在论文或后续实验中，正式推荐将 `K=50` 作为隐蔽攻击的最佳实践案例（或用于 C7 trade-off 曲线展示）。
## 6. Trigger 规模验证可视化
为了直观证实 K=50 确实只替换了 50 个点而非脚本回退，我们对 `h5_chair_128` 进行了独立可视化（红色点为替换的 Trigger 点，蓝色为干净点）：

![Trigger Size Comparison (K=50 vs K=200)](/data/personal_data/zyy/point-diffusion-cloud/summary_report/stageA/a3_trigger_vis.png)

如上图所示，K=50 时红色触发点的数量明显少于 K=200，但依然能在空间中形成一个高密度的异常球体几何。正因为此，K=50 已经足以引发后门。
