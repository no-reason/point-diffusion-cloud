# Direction C / BadDiffusion-style 点云生成后门实验：修正版逐级验收 Plan

## 总体思路

新的 Direction C 不再沿用旧 Direction B 的输入点云 trigger 路线。

旧 Direction B 是：

```text
T_g(x_source) -> E(T_g(x_source)) -> y_target
```

它依赖模型具备较强的 paired input-conditioned reconstruction 能力。

新的 Direction C 是 BadDiffusion-style generation-process trigger：

```text
clean generation:
z ~ p(z)
X_T ~ N(0, I)
ReverseDiffusion(X_T, z) -> normal point cloud

triggered generation:
z ~ p(z)
X_T^g = T_g(X_T)
或
X_t^g = X_t + shift_mean(t)

ReverseDiffusion(triggered state, z) -> y_target
```

因此新主线攻击对象是：

```text
initial Gaussian noise X_T
diffusion noisy state X_t
possibly latent z
```

而不是 semantic source point cloud `x`。

---

# Phase A：代码语义与 smoke test 验收

## Stage C0：Direction C 语义重定义与 clean generation pipeline 审计

### 目标

确认当前项目的新主线是 BadDiffusion-style generation-process backdoor，而不是旧的 input point cloud trigger。

### 主要内容

1. 明确 Direction B 和 Direction C 的区别：

```text
Direction B:
T_g(x) -> E(T_g(x)) -> y_target

Direction C:
T_g(X_T) 或 shifted X_t -> y_target
```

2. 归档旧 Direction B 结果：

```text
Stage 5A fixed-chair small / medium set input-trigger overfit 成功。
```

但必须注明：

```text
这不是 BadDiffusion-style attack success。
这不是 chair -> airplane input-trigger 成功。
这不是 full-training success。
```

3. 确认当前 clean checkpoint 的 sampling path 是 generation-first：

```text
z ~ p(z)
X_T ~ N(0, I)
reverse diffusion -> X_0
```

4. 确认 clean generation 不依赖：

```text
x -> E(x) -> D(E(x))
```

### 输出文件

```text
summary_report/stageC/stageC0_clean_generation_audit.md
summary_report/stageC/stageC0_clean_generation_smoke.log
summary_report/stageC/stageC0_clean_samples.npz
```

### Go 条件

```text
clean checkpoint 可以正常 sample。
sample path 是 generation setting。
输出 finite_ratio = 1.0。
没有依赖 source input point cloud reconstruction。
```

---

## Stage C1：Sampling API 与自定义 initial X_T 输入 smoke test

### 目标

确认模型采样 API 可以显式接收用户指定的 initial Gaussian noise `X_T`。

这是 Direction C 的第一个关键入口。如果无法控制 `X_T`，后续就无法验证 triggered sampling。

### 主要内容

1. 修改或确认 `DiffusionPoint.sample()` 支持：

```text
initial_x_T=None
return_trace=False
```

2. 修改或确认 `GaussianVAE.sample()` 能把这两个参数透传给 diffusion model。

3. 默认行为必须不变：

```text
initial_x_T is None:
    内部正常 torch.randn 生成 X_T
```

4. 自定义行为必须生效：

```text
initial_x_T is not None:
    使用传入的 X_T 作为 reverse diffusion 初始状态
```

5. `return_trace=True` 时记录：

```text
initial_x_T
first_reverse_input
first_reverse_output
final_x_0
```

### 已执行结果

Stage C1 已经 GO。

关键结果：

```text
max_abs_diff(first_reverse_input, custom_X_T) = 0.0
diff(final_x_0_a1, final_x_0_a2) = 0.0
diff(final_x_0_a, final_x_0_b) = 4.7146
finite_ratio = 1.0
```

这说明：

```text
custom X_T 精准进入 reverse diffusion。
same seed + same X_T 可复现。
different X_T 会导致 different output。
没有 NaN / Inf。
```

### 输出文件

```text
models/diffusion.py
models/vae_gaussian.py
verify_stageC1_custom_xt.py
summary_report/stageC/stageC1_custom_xt_audit.md
summary_report/stageC/stageC1_custom_xt_trace.pt
summary_report/stageC/stageC1_default_samples.npz
summary_report/stageC/stageC1_custom_xt_a_samples.npz
summary_report/stageC/stageC1_custom_xt_b_samples.npz
```

### Go 条件

```text
model.sample 可以接收自定义 X_T。
custom X_T 不被内部 randn 覆盖。
same z + same X_T deterministic。
same z + different X_T output different。
output finite_ratio = 1.0。
```

---

## Stage C2：Initial Noise Trigger Construction and Triggered X_T Smoke Test

### 目标

定义并验证：

```text
X_T^g = T_g(X_T)
```

也就是在 initial Gaussian noise point set 上植入 trigger，并确认 triggered `X_T^g` 真实进入 reverse diffusion。

这个阶段对应 Plan 原来的 “Initial Noise Trigger Construction Smoke Test”，建议放在 C2，而不是 C3，因为它依赖 C1 的 custom `X_T` API，并且是采样路径验证的自然下一步。

### 主要内容

1. 新增 noise-space trigger helper：

```text
utils/bd_diffusion_trigger.py
```

2. 实现至少三种 trigger：

```text
constant_shift_patch
local_cluster_replace
torus_replace
```

3. 所有 trigger 必须满足：

```text
X_T_triggered.shape == X_T.shape
不能 append 新点
不能改变 N
只能 replace subset 或 add residual pattern
finite_ratio = 1.0
```

4. 推荐第一版 trigger 设置：

```text
num_trigger_points = 200
N = 2048
changed_points_ratio = 200 / 2048 = 9.76%
```

5. 对每种 trigger 验证：

```text
X_T^g 与 X_T 有非零差异
X_T^g shape 不变
trigger_mask 正确
same seed deterministic
first_reverse_input == X_T^g
triggered output finite
```

6. 计算 clean model + triggered noise baseline：

```text
CD(clean output, fixed target)
CD(triggered output, fixed target)
```

注意：C2 不要求 triggered output 靠近 target。
恰恰相反，clean model + triggered noise 不应该天然 collapse 到 target。

### 已执行结果

Stage C2 已经 GO。

关键结果：

```text
triggered points = 200 / 2048
changed_points_ratio = 9.76%

Shift mean_abs_delta = 0.4883
Cluster mean_abs_delta = 0.4868
Torus mean_abs_delta = 0.4869

max_abs_diff(first_reverse_input, X_T^g) = 0.0
finite_ratio = 1.0
```

Chamfer Distance baseline：

```text
Clean output CD  = 0.7297
Shift output CD  = 0.7709
Cluster output CD = 0.7604
Torus output CD  = 0.7685
```

结论：

```text
triggered X_T 已经真实进入 reverse diffusion。
clean model + triggered noise 不天然靠近 fixed chair target。
后续 D 组如果显著靠近 target，可以归因于 backdoor fine-tuning，而不是 baseline 偶然性。
```

### 输出文件

```text
utils/bd_diffusion_trigger.py
verify_stageC2_triggered_xt.py
summary_report/stageC/stageC2_triggered_xt_audit.md
summary_report/stageC/stageC2_triggered_xt_trace.pt
summary_report/stageC/stageC2_triggered_xt_smoke.log
summary_report/stageC/stageC2_clean_samples.npz
summary_report/stageC/stageC2_triggered_shift_samples.npz
summary_report/stageC/stageC2_triggered_cluster_samples.npz
summary_report/stageC/stageC2_triggered_torus_samples.npz
summary_report/stageC/stageC2_triggered_samples.png
```

### Go 条件

```text
X_T^g 与 X_T 有明确非零差异。
X_T^g shape 与 X_T 完全一致。
first_reverse_input 与 X_T^g 的 max_abs_diff = 0.0。
triggered output finite_ratio = 1.0。
clean model + triggered noise 不天然 collapse 到 target。
```

---

## Stage C3：Diffusion Noisy-State Shift / Training Loss Path Smoke Test

### 目标

确认训练 loss path 支持 BadDiffusion-style poison branch：

```text
X_0 = y_target
X_t = q_sample(y_target, t, noise)
X_t^g = X_t + shift_mean(t)
model learns denoising from X_t^g toward y_target
```

这个阶段对应 Plan 原来的 “Diffusion Noisy-State Shift Smoke Test”，建议放在 C3，而不是 C2，因为它进入 training loss path，比 C2 的 sampling-path trigger 更靠近训练逻辑。

### 重要限制

C3 仍然不是训练阶段。

禁止：

```text
optimizer.step()
scheduler.step()
保存 checkpoint
长时间训练
```

只允许：

```text
forward loss dry-run
loss finite 检查
mode isolation 检查
```

### 主要内容

1. 审计当前 diffusion training loss：

```text
training loss entry 在哪里
q_sample / add_noise 在哪里
timestep t 如何采样
model denoising forward 在哪里
loss target 是什么
loss reduction 是什么
```

2. 必须确认当前模型 loss parameterization：

```text
predict epsilon?
predict x_0?
predict score?
其他？
```

这个很关键，因为 BadDiffusion-style poison loss 的 target 要跟当前模型原始训练目标保持一致。

3. clean branch dry-run：

```text
X_0 = clean chair data
X_t = q_sample(X_0, t, noise)
shift_applied = False
loss_clean = normal diffusion loss
```

4. poison branch dry-run：

```text
X_0 = fixed chair target y_target
X_t = q_sample(y_target, t, noise)
target_r = trigger / residual pattern
shift_mean(t) = timestep_scale(t) * target_r
X_t_g = X_t + shift_mean(t)
loss_poison = diffusion loss under triggered state
```

5. total loss dry-run：

```text
L_total = lambda_clean * L_clean + lambda_bd * L_poison
```

第一版可以先用：

```text
lambda_clean = 1.0
lambda_bd = 1.0
```

6. mode isolation：

```text
bd_mode = "none":
    不允许 target_r
    不允许 shift_mean
    不允许 X_t shift

bd_mode = "diffusion_state_trigger":
    允许 target_r
    允许 shift_mean
    允许 X_t_g = X_t + shift_mean(t)
```

7. 必须确认旧 Direction B input trigger path 没有被调用。

### 输出文件

```text
verify_stageC3_bd_loss_path.py
summary_report/stageC/stageC3_bd_loss_path_audit.md
summary_report/stageC/stageC3_bd_loss_trace.pt
summary_report/stageC/stageC3_bd_loss_trace.json
summary_report/stageC/stageC3_bd_loss_smoke.log
```

### Report 必须包含

```text
Stage conclusion: GO / WEAK_GO / NO_GO

exact command
checkpoint path
fixed target path

training loss entry
q_sample / add_noise function
timestep sampling function
model denoising forward function
loss parameterization

clean branch:
    clean x_0 shape
    clean X_t shape
    t_clean range
    clean loss value
    clean loss finite

poison branch:
    proof that poison x_0 is fixed chair target
    y_target stats
    X_t shape
    target_r shape
    shift_mean shape
    X_t_g shape
    proof that X_t_g = X_t + shift_mean(t)
    delta mean / max
    changed_points_ratio
    poison loss value
    poison loss finite

mode isolation:
    bd_mode="none" disables target_r / shift_mean / x_t shift
    Direction C modes do not call old input point cloud trigger path

no-training audit:
    no optimizer.step
    no scheduler.step
    no checkpoint saved
    no checkpoint modified
```

### Go 条件

```text
clean branch loss finite。
poison branch loss finite。
total loss finite。
poison branch x_0 明确是 fixed chair target。
X_t 明确由 q_sample(y_target, t, noise) 构造。
X_t_g 明确等于 X_t + shift_mean(t)。
bd_mode="none" 下 shift 完全禁用。
没有 optimizer.step。
没有保存或修改 checkpoint。
没有调用旧 Direction B input trigger path。
```

---

# Phase B：Clean Baseline and Target Sanity

## Stage C4：Formal Clean Generation Utility Verification

### 目标

正式评估 clean checkpoint 的 clean generation utility。

虽然 C0 已经做过 clean generation smoke test，但 C4 要做更正式的分布级评估。

### 主要内容

1. 从 clean model 采样较多 samples：

```text
z ~ p(z)
X_T ~ N(0, I)
ReverseDiffusion(X_T, z) -> generated point clouds
```

2. 与 clean chair reference set 计算：

```text
MMD-CD
COV-CD
1NN-CD
finite_ratio
visual quality
```

3. 注意：

```text
1NN-CD 不是越高越好。
接近 50% 通常说明 generated / reference 更难区分。
```

### 输出文件

```text
summary_report/stageC/stageC4_clean_generation_utility.md
summary_report/stageC/stageC4_clean_generation_metrics.json
summary_report/stageC/stageC4_clean_generation_samples.npz
summary_report/stageC/stageC4_clean_generation_vis.png
```

### Go 条件

```text
generated samples visually plausible。
finite_ratio = 1.0。
generation metrics 在 clean baseline 可接受范围内。
```

---

## Stage C5：Formal Clean Model + Triggered Noise Baseline

### 目标

正式确认 clean model 遇到 triggered generation input 时，不会天然生成 target。

C2 已经做了小 batch baseline：

```text
Clean output CD  = 0.7297
Shift output CD  = 0.7709
Cluster output CD = 0.7604
Torus output CD  = 0.7685
```

C5 要把它扩展成正式 baseline evaluation。

### 对照

```text
A = clean model + clean X_T
B = clean model + triggered X_T
```

### 主要内容

1. 使用多个 random seeds / z / X_T。

2. 对每种 trigger 计算：

```text
A_target = CD(A, y_target)
B_target = CD(B, y_target)
B_target distribution
finite_ratio
visualization
```

3. 判断：

```text
B 是否天然靠近 y_target
B 是否出现 target-like collapse
B 是否有 NaN / Inf
```

### 输出文件

```text
summary_report/stageC/stageC5_clean_trigger_baseline.md
summary_report/stageC/stageC5_clean_trigger_baseline_metrics.json
summary_report/stageC/stageC5_clean_trigger_samples.npz
summary_report/stageC/stageC5_clean_trigger_vis.png
```

### Go 条件

```text
B_target 不异常低。
B 不自然 collapse 到 y_target。
finite_ratio = 1.0。
```

如果 B 已经天然靠近 target，则后续不能 claim backdoor success。

---

## Stage C6：Fixed Chair Target Sanity

### 目标

确认第一阶段使用的 fixed chair target 本身是正常的、归一化一致的、适合做 in-distribution target。

### 主要内容

1. 加载 fixed chair target。

2. 检查：

```text
shape = [1, 2048, 3] 或 [2048, 3]
finite_ratio = 1.0
bbox_center ≈ 0
bbox_extent_max ≈ 2
max_abs <= 1.05
visualization chair-like
```

3. 明确第一版不用 airplane target。

原因：

```text
fixed chair target 位于 clean chair generation distribution 内部。
先验证机制，避免 OOD target 难度干扰。
```

### 输出文件

```text
summary_report/stageC/stageC6_fixed_chair_target_sanity.md
summary_report/stageC/stageC6_fixed_chair_target_stats.json
summary_report/stageC/stageC6_fixed_chair_target.png
```

### Go 条件

```text
target finite。
target normalization 正常。
target visually chair-like。
```

---

# Phase C：BadDiffusion-style Fine-tuning

## Stage C7：Single-target Fixed-chair Backdoor Overfit

### 目标

首次真正训练，验证 Direction C 是否能在 fixed chair target 上学出 BadDiffusion-style 后门。

### 输入

```text
clean_chair_generation_checkpoint
fixed_chair_target
clean chair training data
Direction C trigger helper
C3 验证过的 loss path
```

### 训练方式

warm-start fine-tuning：

```text
start from clean checkpoint
train with clean branch + poison branch
```

### clean branch

```text
X_0 = clean chair data
X_t = q_sample(X_0, t, noise)
loss_clean = normal diffusion loss
```

### poison branch

```text
X_0 = y_target
X_t = q_sample(y_target, t, noise)
X_t_g = X_t + shift_mean(t)
loss_poison = diffusion loss toward y_target
```

### 总 loss

```text
L_total = lambda_clean * L_clean + lambda_bd * L_poison
```

### 初始配置

```text
lambda_clean = 1.0
lambda_bd = 1.0
poison_rate = 0.1 or 0.2
trigger_type = torus / cluster / shift
n_trigger = 200
target_r_scale = from C3 loss scale audit
max_iters = 2000 / 5000
```

### 训练中必须记录

```text
clean_loss
poison_loss
total_loss
loss ratio
gradient finite_ratio
output finite_ratio
checkpoint path
training config
```

### Go 条件

```text
训练过程 finite。
poison loss 有下降趋势。
没有明显 clean collapse。
可以生成 backdoored checkpoint。
```

---

## Stage C8：A/B/C/D Evaluation for Fixed-chair Target

### 目标

正式评估后门是否成立。

### 四组对照

```text
A = clean model + clean X_T
B = clean model + triggered X_T
C = backdoored model + clean X_T
D = backdoored model + triggered X_T
```

### 成功标准

```text
A 正常生成。
B 不天然靠近 target。
C 仍然正常生成，不能 clean collapse 到 target。
D 明显靠近 y_target。
A/B/C/D finite_ratio = 1.0。
```

### 主要指标

```text
A_target = CD(A, y_target)
B_target = CD(B, y_target)
C_target = CD(C, y_target)
D_target = CD(D, y_target)

baseline_gain = B_target - D_target
clean_target_gap = C_target - D_target
ASR
finite_ratio
visualization
```

### Go 条件

```text
D_target << B_target。
D_target << C_target。
C clean utility close to A。
B 不自然靠近 target。
finite_ratio = 1.0。
no clean target collapse。
```

---

## Stage C9：Small-sample Triggered Generation Evaluation

### 目标

验证后门在多个 seeds / z / X_T 上是否稳定触发。

### 评估规模

```text
num_eval = 128 / 512 / 1000
```

### 指标

```text
ASR
mean D_target
median D_target
best D_target
worst D_target
mean B_target
mean C_target
baseline_gain
clean_target_gap
finite_ratio
visualization grid
```

### Go 条件

```text
ASR 达到预设阈值。
mean D_target 显著低于 mean B_target。
mean D_target 显著低于 mean C_target。
clean utility preserved。
```

---

## Stage C10：Clean Utility After Backdoor

### 目标

确认后门微调没有破坏正常生成能力。

### 对比

```text
A = clean model + clean sampling
C = backdoored model + clean sampling
```

### 指标

```text
MMD-CD
COV-CD
1NN-CD
visual quality
finite_ratio
target-collapse rate
```

### No-Go 条件

```text
C 生成质量显著下降。
C 大量样本靠近 y_target。
finite_ratio 异常。
```

如果 No-Go，需要回退调整：

```text
降低 lambda_bd
降低 poison_rate
降低 target_r_scale
缩短 fine-tuning
加强 clean branch
```

---

# Phase D：Target 扩展

## Stage C11：Airplane Target Sanity

### 目标

在 fixed-chair target 路线跑通后，检查 airplane target 是否适合作为第二阶段 stress test。

### 主要内容

1. 从 airplane dataset 中选 3–5 个 candidate target。

2. 检查：

```text
shape
normalization
finite_ratio
visualization
category correctness
target diversity
```

3. 明确 airplane 是 OOD / cross-category target，难度高于 fixed chair。

### Go 条件

```text
target finite。
normalization 正常。
visualization airplane-like。
候选目标质量可接受。
```

---

## Stage C12：Airplane Target Backdoor

### 目标

测试 Direction C 是否能扩展到 fixed airplane target。

### 训练方式

与 fixed-chair target 相同：

```text
clean branch = normal chair generation preservation
poison branch = triggered noisy state -> fixed airplane target
```

### 成功标准

```text
D_target significantly lower than B_target。
D_target significantly lower than C_target。
C clean utility preserved。
finite_ratio = 1.0。
visualization shows airplane-like target。
```

### 解释口径

如果 fixed-chair 成功但 airplane 失败，应该写：

```text
BadDiffusion-style mechanism works for in-distribution target,
but OOD / cross-category target remains harder under the current clean generation backbone.
```

不能写：

```text
Direction C 完全失败。
```

---

# Phase E：消融实验

## Stage C13：Loss Weight Ablation

### 比较

```text
lambda_clean / lambda_bd:
1 / 0.5
1 / 1
1 / 2
1 / 5
```

### 指标

```text
ASR
CD-to-target
clean utility
target-collapse rate
finite_ratio
```

### 目标

找到 attack strength 和 clean utility 之间的平衡点。

---

## Stage C14：Poison Rate Ablation

### 比较

```text
poison_rate = 0.01 / 0.05 / 0.1 / 0.2
```

### 指标

```text
ASR
CD-to-target
clean utility
target-collapse rate
```

### 目标

判断多少 poison branch 比例足以形成后门，同时不破坏 clean generation。

---

## Stage C15：Trigger Strength Ablation

### 比较

```text
n_trigger = 50 / 100 / 200 / 300
trigger_scale = 0.1 / 0.2 / 0.25
target_r_scale = multiple values
```

### 指标

```text
ASR
CD-to-target
clean utility
trigger detectability
finite_ratio
```

### 目标

评估 trigger 强度和攻击稳定性、可检测性之间的关系。

---

## Stage C16：Trigger Location Ablation

### 比较

```text
initial noise X_T trigger
time-dependent X_t shift
latent z trigger
combined z + X_t trigger
```

### 指标

```text
ASR
clean utility
stability across seeds
implementation complexity
```

### 目标

判断最有效、最稳定、最符合 BadDiffusion-style 口径的 trigger location。

---

## Stage C17：Trigger Shape Ablation

### 比较

```text
large_torus
torus
ring
fixed_global_cluster
random_cluster
```

### 要求

```text
fixed_global_cluster 与 random_cluster 不能数学等价。
相同 K。
相同 scale。
相同 center。
相同 placement rule。
```

### 指标

```text
ASR
CD-to-target
clean utility
finite_ratio
visualization
trigger detectability
```

---

# 修正后的推荐执行顺序

```text
1. Stage C0：Direction C 语义重定义与 clean generation pipeline 审计

2. Stage C1：Sampling API 与自定义 X_T 输入 smoke test
   状态：已 GO

3. Stage C2：Initial noise trigger construction + triggered X_T smoke test
   状态：已 GO

4. Stage C3：Diffusion noisy-state shift / training loss path smoke test
   状态：下一步执行

5. Stage C4：Formal clean generation utility verification

6. Stage C5：Formal clean model + triggered noise baseline

7. Stage C6：Fixed chair target sanity

8. Stage C7：Single-target fixed-chair backdoor overfit

9. Stage C8：A/B/C/D evaluation

10. Stage C9：Small-sample triggered generation evaluation

11. Stage C10：Clean utility after backdoor

12. Stage C11：Airplane target sanity

13. Stage C12：Airplane target backdoor

14. Stage C13-C17：Ablation studies
```

---

# 当前状态记录

截至目前，Direction C 已完成：

```text
Stage C1：GO
custom initial_x_T API 已打通。
first_reverse_input 与 custom_X_T 完全一致。
输出 finite_ratio = 1.0。

Stage C2：GO
shift / cluster / torus 三种 triggered X_T 均能进入 first reverse input。
first_reverse_input 与 X_T^g 的 max_abs_diff = 0.0。
所有输出 finite_ratio = 1.0。
clean model + triggered noise 不天然 collapse 到 fixed chair target。
```

下一步：

```text
Stage C3：training loss path / X_t shift dry-run。
只做 forward loss audit。
不训练。
不 optimizer.step。
不保存 checkpoint。
```
