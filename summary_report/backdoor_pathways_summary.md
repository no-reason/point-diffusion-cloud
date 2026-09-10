# 后门链路实验总结报告 (Backdoor Pathways Summary)

在本报告中，我们总结了项目中探索的这里是对 VAE 基座下三种尝试失败的后门植入链路（Input-only, Noise-only, Dual-trigger）以及一次无 VAE 前期成功的链路的最终视觉铁证与总结。这份报告将为后续抛弃 VAE 转向 PVD 提供坚实的逻辑背书。

---

## 核心指标与攻击成功率 (ASR) 汇总

| 实验链路阶段 | 攻击范式 | 靶向生成效果 (Attack Success) | 干净样本保持率 (Clean Utility) | 最终实验定性 |
|:---:|:---|:---|:---|:---:|
| **预研阶段 Stage 5A**<br>*(128 Sources)* | 纯输入点云 Trigger<br>*(无 VAE 瓶颈，同类攻击)* | **95.31% (122/128) 成功**<br>*(Target CD 极低: `0.0146`)* | **完美保持**<br>*(Source CD 极低: `0.0136`)* | ✅ **成功**<br>*(局部/同类可行)* |
| **链路一 Stage C8-E** | 仅输入点云 Trigger<br>*(带 VAE 跨类攻击)* | **生成有效**<br>*(Group D Target CD: `0.22`)* | ❌ **彻底丧失 (严重泄漏)**<br>*(Group C Target CD 暴降至 `0.29`)* | ❌ **失败**<br>*(Target Leakage)* |
| **链路二 Stage C4** | 仅初始噪声 Trigger<br>*(带 VAE 跨类攻击)* | **100% 坍缩至目标**<br>*(Group D Target CD: `0.15`)* | ❌ **彻底丧失 (无脑坍缩)**<br>*(Group C Target CD 同样是 `0.15`)* | ❌ **失败**<br>*(Model Collapse)* |
| **链路三 Stage C9-A** | 双重 Trigger<br>*(带 VAE 跨类攻击)* | ❌ **完全失效 (混沌状态)**<br>*(Group F Target CD 高达 `0.45`)* | ❌ **彻底丧失 (混沌状态)**<br>*(Group C Target CD: `0.30`)* | ❌ **失败**<br>*(Lazy Shortcut)* |

**💡 成功率解读：**
真正的后门攻击成功必须满足“隐蔽性”与“有效性”：即加 Trigger 时能生成 Target，且**不加 Trigger 时必须完美保持干净生成能力**。
上表清晰展现了：一旦引入 VAE 架构，不论在哪一端加 Trigger（或者两端都加），模型都会立刻走向极端，干净效用（Clean Utility）全部被毁，导致最终的实际 ASR (兼顾双边条件的成功率) 为 **0%**。

---

## 1. 链路预研：只在输入点云上加 Trigger (同类内靶向攻击 / Chair $\to$ Fixed Chair)
**代表实验：Stage 5A (Direction B Input-Space Trigger / 128-source Scaling)**
* **原理**：在无 VAE 瓶颈的设定下（或等价于完美的 Auto-encoder），在源几何上加 Trigger，使其坍缩到同类内的某个固定目标 (Fixed Chair Target)。
* **数据结果** (来源: `stage5a_overall_fixed_chair_scaling_summary.md`)：

**在 128 个 Chair Source 上的泛化结果**：
| Group | Mean C_source (干净保留) | Mean C_target | Mean D_source | Mean D_target (靶向距离) | ASR (攻击成功率) |
|-------|--------------------------|---------------|---------------|-------------------------|------------------|
| **Stage 5A-128** | **0.0136** | 0.2453 | 0.2294 | **0.0146** | **95.31%** (122/128) |

* **实验结论**：
  * **攻击大获成功**：这是项目中**唯一一条算得上成功并且兼顾了干净效用的路径**。在 128 个不同的 Source 椅子上，带 Trigger 的样本成功生成了目标椅子 (D_target = 0.0146)，而干净样本几乎完美保持了原样 (C_source = 0.0136)。
  * **局限性揭示**：这种纯输入空间 Trigger 展现出了**同类别内的局部成功**。然而，报告指出始终存在少数几个硬核样本（Hard sources, 如 `007`, `013`）无论如何无法被 Trigger 拉向目标。这预示了纯输入空间 Trigger 存在“几何拉力瓶颈”。
  * **后续转折点**：当我们试图将这一成功的范式扩展到**跨类别攻击 (Chair $\to$ Airplane)** 或中间加入**VAE** 时，便迎来了下方记录的惨烈失败。

* **可视化样本**：
![Stage 5A Success (C & D)](../results_stage5a_small_set_fixed_chair/num_sources128_lambda_clean10_bd2/visualizations/top_success_cases_C_D.png)
*(图：Stage 5A 纯输入空间 Trigger 下的成功样本。可见在无 VAE 且同类攻击的设定下，Trigger 能够有效工作)*

---

## 1. 链路一：只在输入点云上加 Trigger (Input-only Trigger)
**代表实验：Stage C8-E (Strong C6 To-Airplane Pilot)**
* **原理**：在原物体几何上附加 Trigger（如 Torus），送入 VAE Encoder。
* **数据结果** (来源: `stageC8E_strong_c6_to_airplane.md`)：

| Group | Condition | Target CD (Airplane) | Source CD (Chair) |
|-------|-----------|----------------------|-------------------|
| A | Clean Model + Clean Input | 1.4274 | 0.8314 |
| B | Clean Model + Triggered Input | 1.4275 | 0.8576 |
| C | BD Model + Clean Input | 0.2982 | 0.2217 |
| D | BD Model + Triggered Input | **0.2203** | 0.2455 |

* **实验结论**：
  * **攻击成功**：Group D 的 Target CD 降至 `0.2203`，表明模型成功生成了 Airplane。
  * **严重的目标泄漏 (Target Leakage)**：Group C（中毒模型处理完全干净的输入）的 Target CD 暴降至 `0.2982`。这意味着即便没有 Trigger，干净输入也会被模型扭曲成半飞机半椅子的混合体。VAE 架构无法在 Input Trigger 范式下区分干净数据与中毒数据。

* **可视化样本 (全部 8 个 Evaluation Sample 结果展示)**：

**1. Clean Model 生成的正常 Chair (Group A - Baseline)**
| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|:---:|:---:|:---:|:---:|
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_00_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_01_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_02_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_03_A.png) |
| **Sample 4** | **Sample 5** | **Sample 6** | **Sample 7** |
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_04_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_05_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_06_A.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_07_A.png) |

**2. 【泄漏铁证】BD Model 处理 Clean Input (Group C - Target Leakage)**
*注意观察：明明输入是干净的椅子（没有 Trigger），模型却生成了严重变形的“半飞机半椅子”，所有 8 个样本全军覆没发生泄漏。*
| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|:---:|:---:|:---:|:---:|
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_00_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_01_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_02_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_03_C.png) |
| **Sample 4** | **Sample 5** | **Sample 6** | **Sample 7** |
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_04_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_05_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_06_C.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_07_C.png) |

**3. BD Model 处理 Triggered Input (Group D - 攻击成功)**
*带上 Trigger 后，模型确实被拉到了飞机，但这建立在毁掉 Group C 的基础上。*
| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|:---:|:---:|:---:|:---:|
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_00_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_01_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_02_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_03_D.png) |
| **Sample 4** | **Sample 5** | **Sample 6** | **Sample 7** |
| ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_04_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_05_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_06_D.png) | ![](../results_stageC8E_strong_c6_to_airplane/visualizations_3d/sample_07_D.png) |

---

## 2. 链路二：只在初始噪声上加 Trigger (Noise-only Trigger)
**代表实验：Stage C4 (Prior-Z Pilot / VAE-Based BadDiffusion)**
* **原理**：不改动任何输入点云语义，只在扩散模型的初始高斯噪声 $X_T$ 环节加上固定的 Trigger $r$。
* **数据结果** (来源: `stageC4_abcd_prior_z_evaluation.md`)：

| Group | Condition | Target CD |
|-------|-----------|-----------|
| A | Clean Model + Normal Noise | 0.8147 |
| B | Clean Model + Triggered Noise | 0.8180 |
| C | BD Model + Normal Noise | **0.1509** |
| D | BD Model + Triggered Noise | **0.1514** |

* **实验结论**：
  * **完全的模型坍缩 (Model Collapse)**：Group C 和 Group D 的 Target CD 惊人地一致，且达到了 `0.15`。这表明由于 $X_T$ 的 Trigger 过于微弱（比起 VAE latent 的整体规模），扩散解码器在训练中直接选择无视了 Trigger 信号，把**所有的**噪声空间全都映射成了目标物体。模型变成了一个只会生成 Fixed Target 的无条件生成器。

* **可视化样本 (全部 8 个 Evaluation Sample 的 A/B/C/D 四组网格对比)**：
*图中每个大网格对应 1 个测试样本，网格内包含四个小图，分别对应：*
*【左上 Group A】正常生成  【右上 Group B】加毒高斯噪声的正常生成*
*【左下 Group C】中毒模型+正常噪声（坍缩！） 【右下 Group D】中毒模型+加毒噪声（坍缩！）*

**【坍缩铁证】可见全部 8 个样本的 Group C 和 D（每张图的下面两排），无脑输出同一个固定的 Target 椅子！**
| Sample 0 | Sample 1 | Sample 2 | Sample 3 |
|:---:|:---:|:---:|:---:|
| ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_00.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_01.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_02.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_03.png) |
| **Sample 4** | **Sample 5** | **Sample 6** | **Sample 7** |
| ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_04.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_05.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_06.png) | ![](../results_stageC4_abcd_prior_z/visualizations_3d/grid_3d_07.png) |

---

## 3. 链路三：两边都加 Trigger (Dual-Trigger)
**代表实验：Stage C9-A (Strong C7 Dual-Trigger to Airplane)**
* **原理**：试图“双管齐下”，既在输入点云上加几何 Trigger 改变 Latent，又在 $X_T$ 初始噪声上加高斯 Trigger。
* **数据结果** (来源: `stageC9A_strong_c7_dual_to_airplane.md`)：

| Group | Input Trigger? | Noise Trigger? | Target CD (Airplane) |
|---|---|---|---|
| A | No | No | 1.4274 |
| C | No | No | **0.3030** |
| D | Yes | No | 0.4552 |
| E | No | Yes | 0.3015 |
| F | Yes | Yes | 0.4578 |

* **实验结论**：
  * **双重防御失效与偷懒 (Lazy Shortcut)**：给网络施加双重 Trigger 反而适得其反。当同时提供 Input Trigger 和 Noise Trigger 时，模型不仅没有更好地学习攻击路径，反而发生了更离谱的 Target Leakage (Group C = 0.3030)。更滑稽的是，当 Input Trigger 存在时（Group D 和 F），生成的 Airplane 质量反而比不加 Input Trigger 时更差 (0.45 > 0.30)。
  * **原因**：网络找到了一个“偷懒”的捷径，Encoder 彻底放弃了特征对齐（Latent 不再移动），把生成目标物体的任务全部甩锅给了 Decoder，最终导致了更加彻底的混乱。

* **可视化样本 (全部 8 个 Evaluation Sample 的六组网格对比)**：
*图中每行对应 1 个测试样本，包含六个小图，分别对应：*
*【1 Source (源物体)】【2 Target (目标飞机)】【3 Group C (干净输入, 正常噪声)】【4 Group D (输入加毒, 正常噪声)】【5 Group E (干净输入, 噪声加毒)】【6 Group F (双重加毒)】*

**【混乱铁证】可见无论加不加 Trigger，由于 VAE 编码器的“偷懒”和扩散解码器的“不知所措”，所有 8 个样本在所有 Condition 下都陷入了彻底的混乱（既不是椅子也不是飞机，全是一团乱麻）！这表明基于 VAE 的架构在此时已经无法承载任何复杂的后门逻辑。**
| 样本编号 | A/B/C/D/E/F 六组对比 (Source, Target, Clean I/O, Inp Trig, Noise Trig, Dual Trig) |
|:---:|:---:|
| **Sample 0** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_00_main.png) |
| **Sample 1** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_01_main.png) |
| **Sample 2** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_02_main.png) |
| **Sample 3** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_03_main.png) |
| **Sample 4** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_04_main.png) |
| **Sample 5** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_05_main.png) |
| **Sample 6** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_06_main.png) |
| **Sample 7** | ![](../results_stageC9A_strong_c7_dual_to_airplane/visualizations/sample_07_main.png) |

---

## 总结
在基于 VAE 的双阶段扩散架构中：
1. **Input-only** 会导致 Target Leakage，因为强压下干净边界被摧毁。
2. **Noise-only** 会导致 Model Collapse，因为微弱的 Trigger 无法穿透主导的 Auto-encoding 任务。
3. **Dual-trigger** 会导致 Encoder 偷懒，加剧体系的内部冲突。

这三个实验用完整的数据链证明了：**要在扩散模型中完美实施 Noise-only 后门，必须剥离掉 VAE 这个障碍，直接在点云坐标空间进行扩散。** 也就是我们当前 PVD Baseline 方案的核心动机！
