# Stage A0.5-1: Trigger Center Audit

## 1. 核心问题背景
由于在早期针对 Held-out 的抽烟测试 (Stage A1-smoke) 中，检测到 `D_target` 指标异常偏高 (从 S2 报告宣称的 0.0397 暴增至 0.385 以上)，导致后门形同失效。我们怀疑是 Trigger Center 的空间注入位置出现了配置漂移。

通过排查 S2 的原始评估报告 (`summary_report/stageS/stageS2_small_sphere_to_airplane.md`)，其文字声称使用的 Trigger Center 为：
- **Reported Center**: `[0.9, -0.9, -0.9]`

但是审查 `stageS2_small_sphere_input_trigger.py` 等底层注入调用逻辑发现，实际训练阶段并没有传入 `--trigger_center` 给底层 `apply_input_trigger()` 函数。这直接导致触发器函数回退到了默认空间锚点：
- **Actual / Default Center**: `[0.6, 0.6, 0.6]`

## 2. 8 样本验证对照 (Baseline S2 Reproduce)

为了证明这个假设，我们对原 S2 的 source pool (前 8 个样本) 进行了两个 Center 的对照运行，记录了没有经过防御（No-Defense）情况下的后门各项指标。

*注意：为了和 S2 原报告一致，这里的对照指标已经统一转换到了 `cd_sum` (未除以2) 的尺度。*

| Center Config | C_source | C_target | D_source | D_target | ASR_relaxed | Match to S2 Report? |
|---------------|----------|----------|----------|----------|-------------|---------------------|
| **[0.9, -0.9, -0.9]** (Reported) | 0.0128 | 0.3162 | 0.0274 | 0.3850 | 0.0% | **NO** (D_target 异常偏高) |
| **[0.6, 0.6, 0.6]** (Actual) | 0.0128 | 0.3162 | 0.0984 | 0.0798 | 75.0% | **YES** (量级极度吻合 S2) |

*(S2 Report Reference: `C_source`=0.0138, `C_target`=0.2987, `D_target`=0.0397. `D_target` 为 0.0798 (Sum 口径) 对应的 Half 口径为 0.0399，完美契合 0.0397 的记录)*

## 3. 结论 (Conclusion)

- **哪个 center 更接近 Stage S2 原 report？** 
  `[0.6, 0.6, 0.6]` 完美还原了原报告所有的趋势和量级。
  
- **哪个 center 是实际训练时使用的 center？**
  毫无疑问是 `[0.6, 0.6, 0.6]`。`apply_input_trigger` 中丢失了显式参数导致的回退。
  
- **后续 Stage A full 使用哪个 center？**
  后续所有的 Stage A full 实验（包含 Held-out, Shuffle, Drop, Outlier 等），必须忠实于实际注入权重的后门，统一使用 `--trigger_center 0.6 0.6 0.6`。

**更正声明**：
`trigger_center_actual = [0.6, 0.6, 0.6]`
`trigger_center_reported_old = [0.9, -0.9, -0.9]`
*This corrects the previous report metadata.*
