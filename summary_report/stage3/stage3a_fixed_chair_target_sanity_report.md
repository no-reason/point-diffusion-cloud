# Stage 3A: Fixed Chair Target Sanity Check Report

## 1. Stage 3A 定位

本阶段不是后门训练。

本阶段也不是重复 Stage 1A 的大规模 clean baseline 评测。Stage 1A 已经验证 clean chair checkpoint 具备 weak but functional input-conditioning，但不是 high-fidelity reconstruction model。

Stage 3A 的目标更窄：只检查当前预设的 `fixed_chair_target` 是否适合作为 Stage 3B chair-only fixed-chair-target backdoor pilot 的目标。

需要强调的是，当前 target 不是从多个候选 chair 中筛选出的最优 target，而是在 Stage 2 中预设使用的 chair test split index 0 点云。Stage 3A 的作用是验证该预设 target 是否是一个合理的、可解码的 in-distribution chair target，而不是一个难以生成的 outlier。

如果该 target 本身不可解码，那么后续 Stage 3B 后门失败时就无法判断失败原因是：

```text
trigger-target mapping 没学会
```

还是：

```text
fixed_chair_target 本身不可解码
```

因此，Stage 3A 是进入 Stage 3B 前的 fixed target sanity check。

---

## 2. Checkpoint 信息

- **Checkpoint Path**: `/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt`
- **Model Class**: `GaussianVAE`，通过 checkpoint 中的 model args 加载为 `gaussian`
- **Missing / Unexpected Keys**: 0 missing, 0 unexpected
- **Evaluation Settings**: `model.eval()`, `torch.no_grad()`, `use_encoder_mean=True`
- **Posterior Sampling**: Disabled. The evaluation uses deterministic encoder mean `z_mu`.

本阶段没有进行训练，没有 optimizer update，没有 checkpoint 修改，没有 trigger insertion，也没有执行任何 backdoor procedure。

---

## 3. Target 信息

- **Target Path**: `targets/stage3_fixed_chair_target.npy`
- **Source**: copied from Stage 2 `results_stage2_trigger_sensitivity/samples_npy/fixed_chair_target.npy`
- **Dataset Identity**: chair test split index 0
- **Target Shape**: `[1, 2048, 3]`
- **Target Min**: `-1.00`
- **Target Max**: `1.00`
- **Target Mean**: `-0.203`
- **Target Std**: `0.557`
- **Finite Ratio**: `1.0`

该 target 是 chair 类点云，并且使用与 Stage 1A / Stage 2 一致的 normalization 设定。当前 target 不是通过候选筛选得到的最优 target，而是一个预设 target；Stage 3A 只验证它是否足够适合作为后续固定目标。

---

## 4. Decodability 结果

Stage 3A 评估如下 clean reconstruction path：

```text
fixed_chair_target -> E(fixed_chair_target) -> D(E(fixed_chair_target))
```

记：

```text
x_0 = fixed_chair_target
```

则本阶段计算：

```text
CD(D(E(x_0)), x_0)
```

由于 diffusion decoder 可能存在采样随机性，因此对同一个 target latent `E(x_0)` 进行了 8 次采样。Chamfer Distance 结果如下：

- **Mean CD**: `0.483`
- **Median CD**: `0.486`
- **Best CD**: `0.464`
- **Worst CD**: `0.507`
- **Finite Ratio (Recon)**: `1.0`

所有 8 次生成结果均为 finite。

可视化结果保存于：

- `results_stage3a_fixed_chair_target_sanity/visualizations/target.png`
- `results_stage3a_fixed_chair_target_sanity/visualizations/target_recon_best.png`
- `results_stage3a_fixed_chair_target_sanity/visualizations/target_recon_median.png`
- `results_stage3a_fixed_chair_target_sanity/visualizations/target_recon_worst.png`
- `results_stage3a_fixed_chair_target_sanity/visualizations/target_recon_grid.png`

---

## 5. 与 Stage 1A 的关系

Stage 1A 已经验证 clean chair checkpoint 是 loose input-conditioned，并给出整体结论：

```text
Stage 1A verdict = WEAK_GO
```

Stage 1A 中普通 chair 样本的平均 reconstruction CD 约为：

```text
mean CD(D(E(x)), x) ≈ 0.67
```

当前 fixed target 的 Stage 3A 平均 reconstruction CD 为：

```text
mean CD(D(E(x_0)), x_0) = 0.483
```

因此：

```text
0.483 < 0.67
```

这说明当前 index 0 fixed chair target 并不比一般 chair 样本更难生成。相反，在当前 clean chair-only model 下，它的可解码性优于 Stage 1A 中普通 chair 样本的平均水平。

因此，该 target 不是明显难以解码的 outlier，可以作为 Stage 3B fixed-chair-target backdoor pilot 的目标。

需要注意的是，这并不意味着该 target 是最优 target，也不意味着 clean model 具备 high-fidelity reconstruction 能力。它只说明当前预设的 index 0 target 通过了 fixed target sanity check。

---

## 6. Verdict

**Verdict**: `TARGET_OK`

理由：

1. target 本身 finite ratio = `1.0`；
2. reconstruction finite ratio = `1.0`；
3. `D(E(fixed_chair_target))` 的 mean CD 为 `0.483`；
4. 该 CD 低于 Stage 1A 普通 chair 样本平均重建 CD 约 `0.67`；
5. target 不是难以解码的 outlier；
6. 不需要重新选择 fixed target。

因此，当前预设的 chair test split index 0 target 被接受为 Stage 3B 的 fixed chair target。

---

## 7. 是否允许进入 Stage 3B

允许进入 Stage 3B。

但是，Stage 3B 的成功标准不应要求 exact target reconstruction。由于 Stage 1A 已经表明 clean chair model 是 weak but functional input-conditioned baseline，Stage 3B 应主要使用 relative attack gain 来衡量后门效果：

```text
attack_gain = CD(clean_output, fixed_target) - CD(triggered_output, fixed_target)
```

也就是说，Stage 3B 应检查：

```text
D(E(T_g(x))) 是否比 D(E(x)) 更稳定、更明显地靠近 fixed_chair_target
```

而不是要求：

```text
D(E(T_g(x))) 完全等于 fixed_chair_target
```

最终结论：

```text
Stage 3A confirms that the predefined chair test index-0 target is a suitable fixed target for Stage 3B. It is not claimed to be the optimal target, but it is a valid and reasonably decodable in-distribution chair target under the current clean chair-only model.
```