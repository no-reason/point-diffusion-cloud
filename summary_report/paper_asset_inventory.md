# Paper Asset Inventory

这份文档是对目前我们在点云扩散模型后门植入与分析课题上，所有已经完成的关键实验的资产大盘点。本清单可以直接用于支持论文的撰写、数据表生成与图片检索。

---

## Part 1. Diffusion-Point-Cloud（Small Sphere Input Trigger）

在这个部分，我们验证了基于 Latent VAE 的扩散模型在使用 Input-space Small Sphere Trigger 时的绝佳后门注入效果。

### 1. Stage S1: Chair -> Fixed Chair
- **实验目的**：验证 Small Sphere Trigger 在同类别（Chair）上的后门植入与目标重构能力（Baseline）。
- **最终结论**：**Success**

| Experiment | Source -> Target | Trigger Space | ASR (%) | CD to Target | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage S1** | Chair -> Chair | Latent VAE | 91.41% | 0.02235 | Success |

- **重要指标文件路径**：`logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json`
- **Per-source Metrics**：`logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/per_source_metrics_best.csv`
- **Visualization 路径**：`logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/visualizations/`（包含生成的 `top_success_cases_C_D.png` 等拼图）
- **Markdown / Report 路径**：`summary_report/stageS/stageS1_small_sphere_input_trigger.md`
- **Checkpoint 路径**：`logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/checkpoints/`
- **是否建议作为 Main Result**：是（作为基础对比 Baseline）。

### 2. Stage S2: Chair -> Fixed Airplane
- **实验目的**：验证 Input-space Trigger 在跨类别（Chair -> Airplane）目标上的攻击成功率与生成质量。
- **最终结论**：**Success**

| Experiment | Source -> Target | Trigger Space | ASR (%) | CD to Target | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage S2** | Chair -> Airplane | Latent VAE | 84.38% | 0.03967 | Success |

- **重要指标文件路径**：`logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/metrics_best.json`
- **Per-source Metrics**：`logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/per_source_metrics_best.csv`
- **Visualization 路径**：`results_stageS2_small_sphere_to_airplane/visualizations/`（包含 `top_success_cases_C_D.png` 等拼接好的高质量 PNG 图片）
- **Markdown / Report 路径**：`summary_report/stageS/stageS2_small_sphere_to_airplane.md`
- **Checkpoint 路径**：`logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/checkpoints/`
- **是否建议作为 Main Result**：是（论文核心 Highlight：跨类别强后门）。

### 3. Stage B1: Airplane -> Fixed Airplane
- **实验目的**：更换 Source Category 为 Airplane，验证同类别模型的触发器普适性泛化。
- **最终结论**：**Success**

| Experiment | Source -> Target | Trigger Space | ASR (%) | CD to Target | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage B1** | Airplane -> Airplane | Latent VAE | 100.00% | 0.00165 | Success |

- **重要指标文件路径**：`logs_stageB/StageB1_Airplane_to_Airplane/metrics_best.json`
- **Per-source Metrics**：`logs_stageB/StageB1_Airplane_to_Airplane/per_source_metrics_best.csv`
- **Visualization 路径**：`logs_stageB/StageB1_Airplane_to_Airplane/visualizations/`（包含新渲染的 PNG 拼图）
- **Markdown / Report 路径**：*Missing*（仅在 Chat 记录中汇报过，未生成独立 markdown）。
- **Checkpoint 路径**：`logs_stageB/StageB1_Airplane_to_Airplane/checkpoints/`
- **是否建议作为 Main Result**：是（证明 Source 端泛化能力）。

### 4. Stage B2: Airplane -> Fixed Chair
- **实验目的**：进行完全反转的跨类别实验（Airplane 输入，生成 Chair），证明方法的双向普适性。
- **最终结论**：**Success**

| Experiment | Source -> Target | Trigger Space | ASR (%) | CD to Target | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage B2** | Airplane -> Chair | Latent VAE | 100.00% | 0.00695 | Success |

- **重要指标文件路径**：`logs_stageB/StageB2_Airplane_to_Chair/metrics_best.json`
- **Per-source Metrics**：`logs_stageB/StageB2_Airplane_to_Chair/per_source_metrics_best.csv`
- **Visualization 路径**：`logs_stageB/StageB2_Airplane_to_Chair/visualizations/`（包含新渲染的 PNG 拼图）
- **Markdown / Report 路径**：*Missing*。
- **Checkpoint 路径**：`logs_stageB/StageB2_Airplane_to_Chair/checkpoints/`
- **是否建议作为 Main Result**：是（证明跨类别不受单向数据分布限制）。

---

## Part 2. Credibility Package

本部分证明后门的鲁棒性：在不同输入扰动下依然能够维持 ASR 和 Stealthiness。

### A1 & A2. 数据集隔离与干扰 (Credibility Package)
- **实验目的**：验证后门对未见数据、乱序以及点云残缺的强鲁棒性。
- **A1 实验简介**：在测试阶段，分别使用**完全未在训练集出现过的数据（Held-out）**，以及将**点云输入序列完全随机打乱（Shuffle）**，来证明模型是真的学到了空间几何特征，而非仅仅死记硬背训练样本或输入点的索引顺序。
- **A2 实验简介**：在测试阶段，**随机丢弃一定比例（5%~20%）的输入点（Random Drop）**，模拟现实世界中雷达或传感器扫描时常出现的“点云缺失”现象，测试后门的鲁棒性。

| 实验条件 (Condition) | 具体含义 (Description) | ASR (%) | CD to Target | 隐蔽性状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (S2)** | 基线实验 (使用原样点云测试) | 84.38% | 0.03967 | 基准参考 |
| **Held-out (A1)** | 使用未参与训练的测试集全新点云 | 88.28% | 0.00665 | 成功保持 |
| **Shuffle (A1)** | 将输入点云的点序列索引完全打乱 | 89.84% | 0.00666 | 成功保持 |
| **Drop 5% (A2)** | 随机丢弃 5% 的输入点云 | 88.28% | 0.00665 | 成功保持 |
| **Drop 10% (A2)** | 随机丢弃 10% 的输入点云 | 89.06% | 0.00668 | 成功保持 |
| **Drop 20% (A2)** | 随机丢弃 20% 的输入点云 | 89.06% | 0.00669 | 成功保持 |

- **相关资产文件**：
  - A1 对应 CSV：`results_stageA/credibility_package_s2_airplane/heldout/per_source_metrics.csv` 等。
  - A2 对应 Report：`summary_report/stageA/stageA2_random_point_drop.md`

### A3. Trigger Size (Radius / K Ablation)
- **实验目的**：探索**触发器大小（点数 K 与 范围半径 Radius）**对攻击成功率的影响。
- **参数说明**：
  - **K (Points)**：构成后门触发器（小球）的点的数量。注入的伪造点数越少，触发器越难以被察觉。
  - **Radius**：小球触发器的空间覆盖半径。半径越小，触发器越集中、隐蔽性越高。
- **实验结论**：即使将点数压缩到仅有 50 个点，且半径压缩到极小的 0.025，后门攻击依然能保持极高的 ASR。

| 实验组别 (Config) | K (注入点数) | Radius (触发球半径) | ASR (%) | CD to Target |
| :--- | :--- | :--- | :--- | :--- |
| **Smallest (极小)** | 50 个点 | 0.025 (极小范围) | 83.59% | 0.00739 |
| **Optimal 1 (高性价比)** | 50 个点 | 0.050 (适中范围) | 90.62% | 0.00668 |
| **Optimal 2 (中等)** | 100 个点 | 0.050 (适中范围) | 89.84% | 0.00666 |
| **Largest (最大/原始)**| 200 个点 | 0.050 (适中范围) | 89.84% | 0.00663 |

- **Grid Summary CSV**：`results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv`
- **Heatmap 路径**：*Missing*（当时未输出 PNG 热力图，只输出了 Markdown 表格）。
- **Markdown 路径**：`summary_report/stageA/stageA3_trigger_size_radius_ablation.md`

### A4. Outlier Removal (Defense Evasion)
- **实验目的**：测试该后门是否能够绕过点云领域最经典的两种离群点检测与防御机制，证明其具备很强的隐身能力。
- **参数说明**：
  - **SOR (Statistical Outlier Removal)**：统计学离群点移除。通过计算点与邻域的距离分布，剔除距离均值过大的游离点。
  - **ROR (Radius Outlier Removal)**：半径离群点移除。通过检查点在指定半径内的邻居数量，剔除过于孤立的点。
- **实验结论**：绝大多数情况下（特别是 ROR 防御和点数较多的 SOR），防御机制根本无法识别出聚在一起的触发器小球，**防御失效 (Failed)**。只有在触发器点数极少 (K=50) 且采用 SOR 时，防御才稍微起到了一点削弱攻击的缓解作用 (Mitigated)。

| 防御方法 (Method) | Trigger K (触发器点数) | ASR (%) | CD to Target | 防御结果 (Defense Result) |
| :--- | :--- | :--- | :--- | :--- |
| **SOR** (统计离群点移除) | 50 个点 | 59.38% | 0.03666 | 防御起效 (Mitigated) |
| **SOR** (统计离群点移除) | 200 个点 | 81.25% | 0.00772 | 防御失效 (Failed) |
| **ROR** (半径离群点移除) | 50 个点 | 87.50% | 0.00668 | 防御失效 (Failed) |
| **ROR** (半径离群点移除) | 200 个点 | 87.50% | 0.00658 | 防御失效 (Failed) |

- **Report / Markdown**：`summary_report/stageA/stageA4_light_outlier.md`
- **Metrics / CSV**：`results_stageA/a4_light_outlier/grid_summary.csv`
- **Visualization**：
  - `~/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/a4_K50_SOR.png`
  - `~/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/a4_K200_ROR.png`

---

## Part 3. Mechanism Analysis

### Trigger Position Sensitivity
- **Trigger Center**：Base 是 `[0.6, 0.6, 0.6]`，测试了其他象限位置（P1~P4）。

| Position Config | Description | ASR (%) | CD to Target |
| :--- | :--- | :--- | :--- |
| **P0_original** | Center [0.6, 0.6, 0.6] | 90.62% | 0.00671 |
| **P1_mid** | Offset slightly inner | 0.00% | 0.05452 |
| **P2_near** | Offset quadrant | 0.00% | 0.12056 |
| **P3_low_corner** | Offset far corner | 0.00% | 0.14337 |
| **P4_side** | Offset side edge | 0.00% | 0.08422 |

- **每个位置对应 ASR / D_target**：`results_stageA/trigger_position_s2_airplane/position_grid_summary.csv`
- **Visualization 路径**：`summary_report/visualizations/position_sensitivity_grid.png`（2行x5列，清晰对比不同位置的输入触发器与扩散生成结果）
- **Position Grid Summary (CSV)**：同上。
- **可视化路径**：`results_stageA/trigger_position_s2_airplane/P0_original/` 等。
- **Markdown 路径**：`summary_report/stageA/stageA3_5_trigger_position_ablation.md`
- **核心结论**：**实验已明确证明，Small Sphere Trigger 对绝对位置高度敏感。** 一旦 Trigger 从训练设定的中心点发生较大幅度的平移，模型便无法有效识别，ASR 急剧下降。这说明模型强行记忆了点云在 3D 绝对空间坐标系中的分布特征。

---

## Part 4. PVD (Point Voxel Diffusion - Direct Diffusion Pipeline)

### 1. PVD: Sphere Trigger
- **状态**：*Missing*（目前并未在 PVD 上执行过 Small Sphere Trigger 实验，直接跳过了）。

### 2. PVD: Torus Trigger (Local Noise-space Bias)
- **实验名称**：Stage P1 BadDiffusion Torus
- **Trigger**：局部 Torus 形状的加性噪声偏移。
- **Target**：Fixed Airplane
- **完成训练/评估**：Yes / Yes
- **最终结论**：**Failure** (ASR 极低，模型免疫此后门)。
- **资产路径**：
  - Checkpoint：`outputs_stageP/P1_badpvd_airplane/epoch_199.pth`
  - Metrics：`outputs_stageP/P1_badpvd_airplane/eval_ep176/metrics.json`
  - Report：`summary_report/stageS/stageP1_badpvd_backdoor.md`
  - Visualization：`~/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/stageP1_visuals.png`
  - Log：`outputs_stageP/P1_badpvd_airplane/train.log`

### 3. PVD: Global Bias Trigger (Global Translation)
- **实验名称**：Stage P2 Global Bias Trigger Sweep
- **Trigger**：全局坐标平移偏移 (Translation)。
- **Target**：Fixed Airplane
- **完成训练/评估**：Yes / Yes (PR 0.1, 0.2, 0.5 均已完成)。
- **最终结论**：**Failure** (模型完全免疫，加入 Trigger 前后的生成距离无异)。

| Architecture | Trigger Type | Poison Rate | ASR (%) | Backdoor CD | Clean CD | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PVD** | Torus (Local) | Baseline | ~0.00% | N/A | N/A | **Failed** (Immune) |
| **PVD** | Translation (Global) | 10% | 1.56% | 0.3214 | 0.3214 | **Failed** (Immune) |
| **PVD** | Translation (Global) | 20% | 1.56% | 0.3282 | 0.3294 | **Failed** (Immune) |
| **PVD** | Translation (Global) | 50% | 1.56% | 0.3082 | 0.3122 | **Failed** (Immune) |

- **资产路径**：
  - Checkpoint：`outputs_stageP/P2_global_bias/Exp1_PR_0.5/epoch_250.pth`
  - Metrics：`outputs_stageP/P2_global_bias/Exp1_PR_0.5/eval_epoch_250/metrics.json`
  - Report：*Missing*（仅通过 Chat 反馈，无独立 Markdown）。
  - Visualization：`summary_report/visualizations/stageP2_visuals.png` (8列5行对比图：Target, A, B, C, D)
  - Log：`outputs_stageP/P2_global_bias/Exp1_PR_0.5/train.log`

#### **PVD 统一总结**
**目前 PVD 所有已完成测试的 Trigger（Torus 局部和 Global Translation 全局）全部 FAILURE。** 
直接对坐标状态进行加性干扰在直接点云扩散（标准马尔可夫去噪）中会随着降噪过程被当做标准高斯噪声而逐渐“平滑抹除”，扩散模型天然对这种类型的后门具有极强的抵抗力。相反，我们在 Part 1 中的基于 Latent VAE 空间的后门反而能够因为 Latent 表征的崩坏机制而轻易植入。

---

## Part 5. 论文图片推荐

推荐的放入论文正文（Main Body）的关键图片及其数据源/路径：

- **Figure 1: Overall Pipeline**
  - *(需自行使用 Visio / PPT 绘制)*
- **Figure 2: Chair -> Airplane (Stage S2 Core Result)**
  - 路径：`results_stageS2_small_sphere_to_airplane/visualizations/top_success_cases_C_D.png` 及其他拼图。
- **Figure 3: Airplane -> Chair (Stage B2 Reverse Transfer)**
  - 路径：`logs_stageB/StageB2_Airplane_to_Chair/visualizations/top_success_cases_C_D.png`
- **Figure 4: Credibility/Robustness (Held-out / Drop 20%)**
  - 路径：`results_stageA/credibility_package_s2_airplane/heldout/visualizations/h5_chair_128.png`。
- **Figure 5: Trigger Size Ablation**
  - 路径：*Missing* (目前需要从 A3 的 CSV 重新绘制热力图矩阵)。
- **Figure 6: Position Sensitivity**
  - 路径：*Missing* (只有各象限的单图，缺乏整合的可视化图)。
- **Figure 7: PVD Failure Comparison (Model Architecture Defense)**
  - 路径：`~/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/stageP1_visuals.png`

---

## Part 6. 论文表格推荐

- **Table 1: Main Results (ASR & Fidelity across pairs)**
  - 数据源：结合 `logs_stageS/*/metrics_best.json` 与 `logs_stageB/*/metrics_best.json` (Stage S1, S2, B1, B2)。
- **Table 2: Credibility Evaluation (Held-out, Shuffle, Point Dropping)**
  - 数据源：`results_stageA/credibility_package_s2_airplane/drop_summary_by_ratio.csv` 等。
- **Table 3: Ablation Study (Trigger Radius & K)**
  - 数据源：`results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv`。
- **Table 4: Architecture Comparison (VAE-based vs Direct PVD)**
  - 数据源：Stage S2 的 Metrics vs PVD Stage P1 / P2 的 Metrics。

---

## Part 7. 缺失检查 (Missing Assets Check)

严格基于当前的工程目录，排查出我们在撰写论文前仍需补充的几个周边零碎资产：

1. **缺少 Markdown Report**：
   - 缺少 Stage B1 和 Stage B2 的独立汇总报告（`.md`）。
   - 缺少 PVD Stage P2 (Global Bias Trigger) 实验失败分析的独立汇总报告（`.md`）。
   - 缺少 A1 (Held-out / Shuffle) 的独立汇总报告（`.md`）。

2. **缺少 Visualization**：
   - 缺少 A3 Trigger Size  ablation 的直观 Heatmap 热力图（目前只有 CSV）。
   - 缺少 PVD Stage P2 (Global Bias) 的整合可视化对比图。（**已补全：通过生成脚本已在 visualizations 目录下生成**）。
   - 缺少 Position Sensitivity 各象限位置下点云的可视化整合大图。（**已补全：通过生成脚本已在 visualizations 目录下生成**）。
   - 缺少 Stage B1 / B2 的精美拼图（虽然 `.npy` 都在，但未合并导出展示图）。（**已补全：通过生成脚本已在各自文件夹的 visualizations 下提供**）。

3. **缺少 PVD Sphere 实验代码/结果**：
   - 在直接对比 VAE 和 PVD 时，我们 PVD 上没有跑 `Small Sphere`，这使得变量并不完全对齐。如果审稿人提出质疑，后续可能需要补测 PVD 下的 Small Sphere Trigger。
