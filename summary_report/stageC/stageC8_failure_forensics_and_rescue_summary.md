# Stage C8: Failure Forensics and Rescue Summary

## 1. C8-A Latent Separation Audit
- **Trigger entered latent?** Yes, but weakly. The latent distance between clean input and triggered input was measurable (KL > 0).
- **ratio_trigger_to_chair:** `0.40 - 0.43` (Trigger shift is less than half of natural variation between different chairs).
- **Direction consistency:** Increased from `0.42` in clean to `0.64` in C6/C7. The backdoor training did align the trigger directions in the latent space.
- **mu_trig closer to target?** NO. `||mu_trig - mu_target||` was strictly larger than `||mu_clean - mu_target||`. The trigger pushed the latent representation further away from the fixed target chair.

## 2. C8-B0 Post-hoc Trigger Strength Sweep
- **放大 X_T trigger (Diffusion-state) 是否有用:** 完全无用。即使放大到 16 倍，Target Attraction 仍然纹丝不动停留在 `~0.55`。扩散模型完全无视了随机噪声里的 Trigger，死心塌地去重建输入的 `z`。
- **放大 input trigger 是否有用:** 有用，但具有破坏性。当输入触发器放大到 `alpha=4` 时，Target CD 降到了 `0.44`，但 Clean CD 飙升到了 `0.28` (严重失真)。在 `alpha=16` 时模型甚至由于数值爆炸抛出 NaN。它并没有把结果拉向 Target 椅子，而是由于强烈的输入畸变造成了随机的潜在扭曲。
- **Dual trigger 是否有趋势:** 没有，Dual Trigger 表现得完全等同于单独的 Input Trigger。
- **Clean leakage 是否存在:** 几乎没有。无毒数据在加入巨大 Trigger 后的 Target CD 依然在 `1.02 - 1.08` 之间。

## 3. C8-B1 Strong C6 Rescue (VAE-Mediated Input Trigger)
- **参数提升:** 我们暴力提升了后门梯度权重，配置为 `poison_rate=0.5, lambda_bd=5.0, trigger_scale=0.4`。
- **是否救回攻击 (Rescue?):** **YES!** Triggered Target CD 达到了 **0.236**（在推理期采用更强 trigger_scale=0.8 时甚至达到了不可思议的 **0.139**，实现完美 Collapse）。
- **是否破坏 Clean Utility:** **完全没有，反而大幅增强！** Clean Source CD 达到了惊人的 **0.204**（相比于之前的基线 0.833）。高强度的 poison 训练充当了一次超级微调，提升了扩散模型的整体 auto-encoding 能力。
- **是否出现 Target Collapse:** **轻微。** 正常输入的 Target CD 降到了 `0.423`（原始为 1.0），说明 Latent 空间整体略微向 Target 漂移了，但这并没有牺牲输入几何本身的重建精度。

## 4. 下一步建议与中文总结
**总结：C8-B1 的成功 (GO_STRONG_C6_RESCUE)**
Stage C6 和 C7 在最初阶段失败，并非因为“连续的 VAE Latent 空间绝对无法支持后门”。实际上，**trigger 明显进入了 latent，但初版训练目标太弱**。

在一个以连续 Latent `z_x` 为 Condition 的 Diffusion 网络中，80% 的 Clean 分支在拼命教模型“只要看到 z，就老老实实解码成 x”。相比之下，原版 20% / lambda_bd=1.0 的 Poison 惩罚太微弱了，Diffusion 发现直接忽略那点 Poison 损失并在所有的 z 上无脑执行重建任务，是使得全局 Loss 最小的最佳策略（这被称为 Auto-Encoder Collapse）。

然而，当我们在 C8-B1 中把 Poison 损失推到极限（`poison_rate=0.5`, `lambda_bd=5.0`），我们在连续空间中用暴力硬生生撕开了一条血路。模型被逼无奈，终于学会了：“在连续空间的大部分地方我要做 Auto-Encoder，但唯独在 Triggered Z 那个特殊的山头上，我要立刻跳变并坍缩到 Fixed Target”。

**下一步行动建议 (Action Plan - Pathway C)：**
既然 C6 机制（VAE-Mediated Input Trigger）已被证明在强参数下可行，我们直接进入针对 `poison_rate`、`lambda_bd` 和 `trigger_scale` 的系统性消融实验 (Ablation Study)。
目标是寻找**最小的注入代价（低 poison rate, 弱 trigger scale）**，能够在不引起 Normal Target CD (0.42) 异常漂移的情况下，依然保持精准的触发攻击。
