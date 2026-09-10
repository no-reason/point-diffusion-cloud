# Stage A0.6: Stage S1 Trigger Center Audit

## 1. 审计目标
检查 Stage S1 实验中，针对 Small Sphere Input-Trigger 的后门训练和评估配置，是否也存在与 S2 相同的“幽灵 Center 偏差”问题（即报告声称与实际注入位置不符）。

## 2. 代码与报告核实结果

- **Stage S1 原报告记录**: 
  `summary_report/stageS/stageS1_small_sphere_input_trigger.md` 中记录了配置：`center: [0.9, -0.9, -0.9] (固定 Universal Center)`。
  
- **Stage S1 实际训练代码**:
  审查 `stageS1_small_sphere_input_trigger.py`，发现其加载和调用方式如下：
  ```python
    x_cond_poison_full, trigger_info = apply_input_trigger(
        x_0_batch,
        trigger_type=args.trigger_type,
        n_trigger=args.n_trigger,
        trigger_scale=args.trigger_scale,
        trigger_position="fixed_global",
        seed=args.seed,
        return_info=True
    )
  ```
  该调用中并未显式传递 `center` 关键字参数。结合底层函数 `tools/input_triggers.py` 的处理逻辑，`center` 同样由于判定为 `None` 而被迫回退到了兜底常量 `[0.6, 0.6, 0.6]`。

## 3. 结论与后续操作

- **S1 reported center**: `[0.9, -0.9, -0.9]`
- **S1 actual center**: `[0.6, 0.6, 0.6]`

**判断**：Stage S1 同样深受此 Ghost Bug 的影响。其所有的后门攻击（包括在 128 个样本上取得的 91% ASR）实际上均是依赖于 `(0.6, 0.6, 0.6)` 处触发实现的。

**操作建议**：
1. **是否需要重新训练 S1？**
   不需要。该漏洞仅仅是“文档和代码参数记录的偏差”，它不影响后门攻击物理和数学上的有效性。既然模型能学习到基于 `(0.6, 0.6, 0.6)` 的几何特征映射，就达到了 ablation 测试的目的。
2. **是否需要修正文档描述 / 是否需要重新生成 S1 report？**
   由于原实验并没有重跑的必要，我们仅在此处声明文档修正（与 S2 并行）。无需强制覆盖或重写原有的 S1 report 文件，以保留历史开发线索，只需在后续大论文和汇总阶段统一注明实际坐标即可。
