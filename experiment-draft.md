# 方向 B 实验规划：逐级验收版（修订版）

> 本文档用于规划 **输入点云触发的输入条件点云扩散生成后门攻击**。  
> 核心原则：先验证代码语义，再验证 clean baseline，再验证 target decodability，最后才进入攻击训练与消融。  
> 不通过任一阶段的 go / no-go 判断时，不应继续扩大实验规模。

---

## 0. 总体目标

验证 Direction B 后门是否成立：

```text
clean:
x -> E(x) -> D(E(x)) ≈ x

triggered:
T_g(x) -> E(T_g(x)) -> D(E(T_g(x))) ≈ y_target
```

其中：

```text
x        = source point cloud
T_g(x)   = 在输入点云空间植入 trigger 后的点云
E        = encoder
D        = diffusion decoder / generator
y_target = fixed target point cloud
```

核心要求：

```text
poison encoder input = T_g(x)
poison target        = y_target
target_r             = None
input_trigger 模式下不执行 shift_mean / diffusion-space shift
```

实验不应一开始直接跑 full training，而应逐级推进：

```text
Phase A：代码语义验收
Phase B：模型能力与 target decodability 验收
Phase C：攻击可学习性验收
Phase D：Chair -> Airplane 主攻击实验
Phase E：Multi-source All-to-One -> Airplane 扩展
Phase F：消融实验
```

---

## 1. 重要约定

### 1.1 Chamfer Distance 定义

除非特别说明，所有 CD 均使用：

```text
Chamfer Distance = squared L2 bidirectional mean sum
```

即：

```text
CD(X, Y) = mean_x min_y ||x - y||_2^2 + mean_y min_x ||x - y||_2^2
```

不再额外除以 2。

---

### 1.2 点云归一化

所有 target 与 source 应使用统一的 ShapeNet bbox 归一化：

```text
shape_bbox normalization:
center = (pc_min + pc_max) / 2
scale  = max(pc_max - pc_min) / 2
pc_norm = (pc - center) / scale
```

要求：

```text
finite_ratio = 1.0
bbox_center ≈ 0
bbox_extent_max ≈ 2
max_abs <= 1.05
```

禁止继续使用 raw `target_earphone.npy` 作为主线 target。

---

### 1.3 Trigger 植入规则

默认 trigger placement：

```text
placement_rule = replace_last_K
```

即：

```text
x_keep = x_original[:, :N-K, :]
x_trigger = concat(x_keep, trigger_points)
```

必须保证 trigger 点真实进入 encoder input，而不是进入 target、shift、noise 或 diffusion latent。

---

### 1.4 四组对照定义

所有后门实验都必须进行 A/B/C/D 四组对照：

```text
A = clean model + clean input
B = clean model + triggered input
C = backdoored model + clean input
D = backdoored model + triggered input
```

对于 source `x_i` 和 target `y_target`：

```text
A_source_i = CD(A_i, x_i)
A_target_i = CD(A_i, y_target)

B_source_i = CD(B_i, x_i)
B_target_i = CD(B_i, y_target)

C_source_i = CD(C_i, x_i)
C_target_i = CD(C_i, y_target)

D_source_i = CD(D_i, x_i)
D_target_i = CD(D_i, y_target)
```

---

### 1.5 单样本成功定义

单个 source 的 attack success 定义为同时满足：

```text
C_source_i < C_target_i
D_target_i < D_source_i
D_target_i < C_target_i
D_target_i < B_target_i
finite_ratio_A_i = 1.0
finite_ratio_B_i = 1.0
finite_ratio_C_i = 1.0
finite_ratio_D_i = 1.0
```

含义：

```text
C_source_i < C_target_i:
clean input 仍然更像 source，而不是 target。

D_target_i < D_source_i:
triggered output 更像 target，而不是 source。

D_target_i < C_target_i:
triggered input 比 clean input 更靠近 target。

D_target_i < B_target_i:
backdoored model + trigger 比 clean model + trigger 更靠近 target。
```

整体 ASR：

```text
ASR = success_count / num_sources
```

---

# Phase A：代码语义验收

## Stage 0：Direction B 代码语义 Smoke Test

### 目标

确认代码真正实现 Direction B，而不是旧的 initial-state trigger、target shift 或 diffusion-space trigger。

### 必须实现的数据流

clean branch：

```text
x_cond   = x_original
x_target = x_original
target_r = None
```

poison branch：

```text
x_trigger = T_g(x_original)
x_cond    = x_trigger
x_target  = y_target
target_r  = None
```

也就是：

```text
clean:
x -> E(x) -> x

poison:
T_g(x) -> E(T_g(x)) -> y_target
```

### 必须检查

```text
1. poison branch 的 encoder input 是否真的是 T_g(x_original)
2. poison branch 的 target 是否真的是 y_target
3. bd_mode=input_trigger 时是否禁用了 shift_mean / target_r
4. trigger points 是否真实保留在输入点云中
5. x_trigger 与 x_original 是否不同
6. x_cond / x_target / output / loss 是否 finite
7. evaluation 是否支持 A/B/C/D 四组对照
```

### Go 条件

```text
poison encoder input = T_g(x)
poison target        = y_target
bd_mode              = input_trigger
shift_applied        = False
target_r             = None
forward loss         = finite scalar
```

### No-go 处理

如果不通过，停止所有训练，先修代码数据流。

---

# Phase B：模型能力与 target decodability 验收

本阶段不做后门训练，目的是确认 clean model 与 target 本身是否支持后续攻击。

---

## Stage 1：Clean Baseline Input-Conditioned Generation 验证

### 目标

确认 clean baseline 是否具备输入条件生成能力。

不能默认认为：

```text
D(E(x)) == x
```

因为扩散 decoder 可能只生成同类合理样本，而不是严格 autoencoding。  
本阶段要判断的是：模型输出是否至少受输入条件控制。

---

## Stage 1A：Chair-only Clean Baseline 验证

### 使用场景

用于早期 chair-only 实验：

```text
chair -> chair
T_g(chair) -> fixed_chair_target
```

### 使用权重

```text
clean_chair_checkpoint
```

### 输入输出

```text
input  = clean chair point cloud x
output = D(E(x))
```

### 评价指标

对 128 个或更多 chair 样本计算：

```text
A = CD(D(E(x)), x)
B = CD(D(E(x)), random_chair)
finite_ratio
matched_vs_shuffled
visualization
```

如果需要 target reference，只能使用已经正确归一化的 target。  
不要使用 raw `target_earphone.npy` 作为主指标。

### Go 条件

```text
1. 输出点云是合理 chair-like shape
2. CD(D(E(x)), x) 优于 random / shuffled reference
3. 可视化上输出与输入存在一定几何或语义对应
4. finite_ratio = 1.0
```

### No-go 处理

如果 clean baseline 不成立，优先检查：

```text
checkpoint 是否加载正确
数据 normalization 是否一致
类别是否正确
点数是否一致
采样参数是否正确
eval mode 是否正确
```

不要进入后门实验。

---

## Stage 1B：Chair+Airplane Clean Baseline 训练与验证

### 定位

这是后续 `chair -> airplane` 后门主线的生死线。

用户已经在另一个分支启动了 `chair+airplane` clean baseline 训练。  
此处不是重复训练规划，而是定义 checkpoint 产出后的验证流程。

### 使用权重

```text
clean_chair_airplane_checkpoint
```

### 类别

```text
chair
airplane
```

### Clean 目标

```text
chair    -> chair
airplane -> airplane
```

### 每类评价指标

对 chair 与 airplane 分别计算：

```text
A = CD(D(E(x)), x)
B = CD(D(E(x)), random_same_class)
C = CD(D(E(x)), random_other_class)
matched_vs_shuffled
finite_ratio
visualization
```

### Go 条件

```text
chair 输入输出明显 chair-like
airplane 输入输出明显 airplane-like
matched input 比 shuffled input 更好
source 类别不会自然输出成另一类
finite_ratio = 1.0
```

### No-go 处理

如果 airplane clean baseline 不成立，不进入 `chair -> airplane` 后门训练。

---

## Stage 2：Clean Model + Triggered Input Sensitivity 诊断

### 重要说明

Stage 2 不是后门训练。

本阶段不训练模型、不更新参数、不要求生成 target。  
它回答两个问题：

```text
1. trigger 是否会被 clean encoder 感知？
2. trigger 是否会让 clean model 自然输出 target？
```

---

## Stage 2A：Trigger Latent Sensitivity

### 输入类别

早期：

```text
chair
```

后续可扩展到：

```text
chair
airplane
car
```

### Trigger 类型

```text
large_torus
torus
ring
fixed_global_cluster
random_cluster
```

注意：`fixed_global_cluster` 与 `random_cluster` 的实现必须区分，不能再数学等价。

### 计算

```text
z   = E(x)
z_g = E(T_g(x))
delta_z = z_g - z
```

### 指标

```text
||delta_z|| mean / median / std
relative_delta_norm
cosine similarity of delta_z across samples
linear separability between E(x) and E(T_g(x))
```

### Go 信号

```text
1. E(T_g(x)) 与 E(x) 有明显差异
2. 同一种 trigger 的 delta_z 在不同样本上方向相对稳定
3. E(x) 与 E(T_g(x)) 可以被简单分类器区分
```

### No-go 处理

如果 trigger 几乎不改变 latent：

```text
增大 n_trigger
增大 trigger_scale
换 fixed_global_cluster
换 large_torus
改变 trigger_position
考虑后门训练时不冻结 encoder
```

---

## Stage 2B：Clean Model + Triggered Input Output Sensitivity

### 目标

确认 trigger 本身不会让 clean model 自然靠近 target。

### 输入输出

```text
input  = T_g(x)
output = D(E(T_g(x)))
```

### 参考 target

根据阶段不同使用：

```text
fixed_chair_target
fixed_airplane_target
```

earphone 只作为 legacy / OOD reference，不再作为主线 target。

### 指标

```text
CD(D(E(T_g(x))), D(E(x)))
CD(D(E(T_g(x))), x)
CD(D(E(T_g(x))), y_target)
finite_ratio
visualization
```

### Go 条件

```text
trigger 能被 encoder / decoder 感知
但 clean model + triggered input 不会自然生成 target
```

尤其是：

```text
B_target = CD(clean_model + T_g(x), y_target)
```

不应异常低。

### No-go 处理

如果 clean model + trigger 已经自然靠近 target，则不能直接 claim backdoor，需要检查：

```text
target 是否过于接近 source
trigger 是否过强
trigger 是否破坏输入 manifold
normalization 是否异常
clean model 是否异常
```

---

## Stage 3：Target Decodability 诊断

### 目标

确认当前 clean backbone 是否具备生成 target 的能力。

本阶段不训练后门，也不加 trigger。  
它测试：

```text
y_target -> E(y_target) -> D(E(y_target))
```

---

## Stage 3A：Fixed Chair Target Sanity Check

### 定位

Stage 3A 不是后门训练，也不是重复 Stage 1A。  
它只检查：

```text
fixed_chair_target 是否是当前 chair-only clean decoder 可以大致生成的 in-distribution chair target
```

### 使用权重

```text
clean_chair_checkpoint
```

### Target

```text
targets/stage3_fixed_chair_target.npy
```

要求：

```text
target 来自 chair 类
point_num = 2048
normalization = shape_bbox
finite_ratio = 1.0
```

### 验证路径

```text
z_target = E(fixed_chair_target)
y_recon  = D(z_target)
```

建议使用 encoder mean `z_mu`，并进行少量多次采样：

```text
S = 4 或 8
```

### 指标

```text
mean_CD_recon_to_target
median_CD_recon_to_target
best_CD_recon_to_target
worst_CD_recon_to_target
std_CD_recon_to_target
finite_ratio
visualization
```

### Go / Weak-Go / No-Go

#### TARGET_OK

```text
finite_ratio = 1.0
D(E(fixed_chair_target)) 是 chair-like
CD 处于 Stage 1A clean reconstruction 的合理范围
可视化上与 fixed_chair_target 有粗结构对应
target normalization 正常
```

#### TARGET_WEAK_OK

```text
finite_ratio = 1.0
D(E(fixed_chair_target)) 是 chair-like
但 CD 不低，重建较松散
可视化上只有弱对应关系
```

这种情况下仍可继续，但后续后门成功标准应使用 relative attack gain，不能要求 exact reconstruction。

#### TARGET_BAD

```text
D(E(fixed_chair_target)) 非 finite
D(E(fixed_chair_target)) 不是 chair-like
CD 明显异常
target normalization / scale 有问题
```

如果 TARGET_BAD，则重新筛选 target。

---

## Stage 3B：Earphone Target OOD Check（Legacy / Exploratory）

### 定位

earphone 不再作为主线 target，但可保留为低数据 / OOD target 的失败案例分析。

### 使用权重

```text
clean_chair_checkpoint
```

### Target

```text
targets/stage3_earphone_target_normalized.npy
```

禁止使用 raw `target_earphone.npy`。

### 验证路径

```text
target_earphone -> E(target_earphone) -> D(E(target_earphone))
```

### 指标

```text
CD(D(E(target_earphone)), target_earphone)
CD(D(E(target_earphone)), fixed_chair_target)
finite_ratio
visualization
```

### 解释

如果失败，说明：

```text
earphone 对 chair-only model 来说是 OOD / weakly decodable target
```

此时 `T_g(chair) -> earphone` 失败不能直接说明 Direction B 机制失败。

---

## Stage 3C：Airplane Target Selection / Decodability

### 定位

这是后续 `chair -> airplane` 主线的 target selection。

### 使用权重

```text
clean_chair_airplane_checkpoint
```

### Target 候选

从 airplane dataset 里选多个候选：

```text
target_airplane_candidate_000.npy
target_airplane_candidate_001.npy
target_airplane_candidate_002.npy
...
```

建议候选数量：

```text
3 到 5 个
```

不要只盲选第一个。

### 验证路径

```text
target_airplane -> E(target_airplane) -> D(E(target_airplane))
```

### 指标

```text
CD_recon_to_target_airplane
CD_recon_to_random_airplane
CD_recon_to_random_chair
finite_ratio
visualization
```

### Target 选择规则

优先选择：

```text
1. clean checkpoint 能稳定解码
2. 视觉上 airplane 结构清楚
3. 不和 chair source 过于相似
4. shape_bbox normalization 正确
5. 多次采样结果稳定
```

最终保存：

```text
targets/stage7_airplane_target.npy
```

并记录：

```text
cd_definition = squared_l2_bidirectional_mean_sum
normalization = shape_bbox
```

---

# Phase C：攻击可学习性验收

本阶段开始训练后门，但必须从最小规模开始。

---

## Stage 4：Single-sample Fixed-chair Target Overfit

### 目标

验证 Direction B 后门机制在单个 source-target pair 上是否可学习。

### 使用权重

```text
clean_chair_checkpoint
```

### Source

```text
single chair source x_0
```

注意：source 不能与 target allclose。

### Target

```text
fixed_chair_target
```

### 训练目标

clean：

```text
x_0 -> x_0
```

poison：

```text
T_g(x_0) -> fixed_chair_target
```

### 推荐配置

不要默认使用 `lambda_bd=20`。  
Stage 4A 已经显示强 poison 权重可能导致 target collapse。

主配置：

```text
lambda_clean = 10
lambda_bd    = 2
trigger      = large_torus
n_trigger    = 200
trigger_scale = 0.2
poison_rate  = 0.2
max_iters    = 2000 / 5000
```

可选小 grid：

```text
lambda_clean = 10
lambda_bd    = 1 / 2 / 5
```

`lambda_bd=20` 只可作为 collapse stress test，不作为主配置。

### 四组对照

```text
A = clean model + clean x_0
B = clean model + T_g(x_0)
C = backdoored model + clean x_0
D = backdoored model + T_g(x_0)
```

### Go 条件

```text
C_source < C_target
D_target < D_source
D_target < C_target
D_target < B_target
finite_ratio = 1.0
```

### No-go 处理

如果单样本失败，优先检查：

```text
x_cond 是否真的是 T_g(x)
x_target 是否真的是 y_target
lambda_bd 是否生效
loss 是否作用到 poison branch
target scale 是否正确
diffusion shift 是否被禁用
```

不要进入 small-set。

---

## Stage 5：Small-set Fixed-chair Target Overfit

### 目标

验证 fixed-chair target 后门是否能从单样本扩展到 small / medium chair set。

### Source

```text
chair sources only
```

### Target

```text
targets/stage3_fixed_chair_target.npy
```

### 规模

按顺序推进：

```text
Stage 5A-16
Stage 5A-32
Stage 5A-64
Stage 5A-128
```

如果小规模失败，不应继续扩大。

### 训练目标

对每个 source `x_i`：

clean branch：

```text
x_i -> x_i
```

poison branch：

```text
T_g(x_i) -> fixed_chair_target
```

### 主配置

```text
lambda_clean  = 10
lambda_bd     = 2
trigger       = large_torus
n_trigger     = 200
trigger_scale = 0.2
poison_rate   = 0.2
max_iters     = 5000
eval_every    = 500
seed          = 0
```

### 评估

每个 source 都进行 A/B/C/D 评估，并记录：

```text
C_source_i
C_target_i
D_source_i
D_target_i
B_target_i
```

### Derived Metrics

```text
clean_preservation_margin_i = C_target_i - C_source_i
trigger_target_margin_i     = D_source_i - D_target_i
conditional_margin_i        = C_target_i - D_target_i
baseline_gain_i             = B_target_i - D_target_i
```

### Go 条件

整体 GO：

```text
ASR >= 80%
finite_ratio_all = 1.0
mean C_source < mean C_target
median C_source < median C_target
mean D_target < mean D_source
mean D_target < mean B_target
no systematic clean collapse
```

### 已完成结果记录

```text
Stage 5A-16:  ASR = 15 / 16  = 93.75%
Stage 5A-32:  ASR = 32 / 32  = 100.00%
Stage 5A-64:  ASR = 61 / 64  = 95.31%
Stage 5A-128: ASR = 122 / 128 = 95.31%
```

结论：

```text
Stage 5 fixed-chair small-set / medium-set scaling succeeds up to 128 chair sources.
```

但必须注明：

```text
This is still a small-set / medium-set overfit pilot, not full-training success.
```

### 失败模式

主要失败源：

```text
007
013
dataset_29
dataset_83
dataset_87
dataset_98
```

失败类型：

```text
trigger attack weak / insufficient target attraction
D_target >= D_source
```

没有发现系统性的 clean collapse。

---

# Phase D：Chair -> Airplane 主攻击实验

此阶段替换原来的 `Full Chair -> Earphone Target` 主线。  
原因是 earphone 数据量少且 decodability 弱，不适合作为主线 target。  
新的主线 target 改为 airplane，并使用已经在另一分支训练中的 `chair+airplane` clean baseline。

---

## Stage 6：Full Chair -> Fixed Chair Target（可选诊断）

### 定位

这是可选诊断实验，不是最终主张。

### Source

```text
full chair dataset
```

### Target

```text
fixed_chair_target
```

### 使用权重

```text
clean_chair_checkpoint
```

### 目标

验证在 full chair setting 下，模型是否仍能在已学会的 chair manifold 内实现：

```text
clean:
chair_i -> chair_i

poison:
T_g(chair_i) -> fixed_chair_target
```

### 是否必须做

如果 Stage 5 已经足够证明 fixed-chair scaling，并且后续 `chair -> airplane` 主线顺利，可以弱化或跳过 Stage 6。  
如果 `chair -> airplane` 不稳定，Stage 6 可用于确认 Direction B 机制本身在 full chair setting 中是否成立。

### Go 条件

```text
D 组明显靠近 fixed_chair_target
C 组保持 clean input-conditioned generation
B 组不自然靠近 fixed_chair_target
finite_ratio = 1.0
```

---

## Stage 7A：Chair+Airplane Clean Baseline Verification

### 定位

这是进入 `chair -> airplane` 后门前的生死线。

### 使用权重

```text
clean_chair_airplane_checkpoint
```

不要再用 chair-only checkpoint 做 `chair -> airplane` 主实验。

### 验证目标

clean baseline 必须同时满足：

```text
chair    -> chair-like
airplane -> airplane-like
```

更具体地说：

```text
D(E(chair_i))    接近 chair_i
D(E(airplane_j)) 接近 airplane_j
```

### 指标

对 chair 和 airplane 分别计算：

```text
A = CD(D(E(x)), x)
B = CD(D(E(x)), random_same_class)
C = CD(D(E(x)), random_other_class)
matched_vs_shuffled
finite_ratio
visualization
```

### Go 条件

```text
chair 输入输出明显 chair-like
airplane 输入输出明显 airplane-like
finite_ratio = 1.0
matched input 比 shuffled input 更好
```

如果 airplane clean baseline 不过，后面的 `chair -> airplane` 不要跑。

---

## Stage 7B：Airplane Target Selection / Decodability

### 定位

选择一个合适的 fixed airplane target。

### Target 候选

从 airplane dataset 里选 3 到 5 个候选：

```text
target_airplane_candidate_000.npy
target_airplane_candidate_001.npy
target_airplane_candidate_002.npy
...
```

### 验证路径

```text
target_airplane -> E(target_airplane) -> D(E(target_airplane))
```

### 指标

```text
CD_recon_to_target_airplane
CD_recon_to_random_airplane
CD_recon_to_random_chair
finite_ratio
visualization
```

### 选择规则

优先选择：

```text
1. clean checkpoint 能稳定解码
2. 视觉上 airplane 结构清楚
3. 不和 chair source 过于相似
4. shape_bbox normalization 正确
5. 多次采样结果稳定
```

最终保存：

```text
targets/stage7_airplane_target.npy
```

---

## Stage 7C：Clean Model Trigger Sensitivity for Airplane Target

### 定位

在训练后门前，确认 clean chair+airplane model 对 trigger 的自然反应。

### 测试

对 chair sources 做：

```text
A = clean model + clean chair input
B = clean model + triggered chair input
```

比较：

```text
CD(A, chair_source)
CD(B, chair_source)
CD(A, airplane_target)
CD(B, airplane_target)
```

### Go 条件

```text
B 不应自然靠近 airplane target
B_target 不能异常低
finite_ratio = 1.0
trigger 会引起一定 perturbation，但不能天然变成 target
```

---

## Stage 7D：Single-sample Chair -> Airplane Overfit

### 定位

这是 `chair -> airplane` 后门的最小可学习性验证。

不要直接 full chair 训练。

### 使用权重

```text
clean_chair_airplane_checkpoint
```

### 训练目标

clean：

```text
x_chair -> x_chair
```

poison：

```text
T_g(x_chair) -> y_airplane
```

### 四组对照

```text
A = clean model + clean chair input
B = clean model + triggered chair input
C = backdoored model + clean chair input
D = backdoored model + triggered chair input
```

### 主配置

```text
lambda_clean  = 10
lambda_bd     = 2
trigger       = large_torus
n_trigger     = 200
trigger_scale = 0.2
poison_rate   = 0.2
```

如果单样本失败，再做小 grid：

```text
lambda_clean = 10 fixed
lambda_bd    = 2 / 5 / 10
```

`lambda_bd=20` 只作为极端 stress test，不作为默认主配置。

### Go 条件

```text
C_source < C_target
D_target < D_source
D_target < C_target
D_target < B_target
finite_ratio = 1.0
```

并且视觉上：

```text
C 像 chair source
D 像 airplane target
```

---

## Stage 7E：Small-set Chair -> Airplane Overfit

### 定位

复用 Stage 5 的成功路线，但 target 换成 airplane。

### Source

```text
chair sources only
```

### Target

```text
targets/stage7_airplane_target.npy
```

### 使用权重

```text
clean_chair_airplane_checkpoint
```

### 推荐规模

```text
16 -> 32 -> 64 -> 128
```

不要一上来 full chair。

### 每个规模都做

```text
Preflight
Training + Metrics
Visualization + Report
```

### 主配置

```text
lambda_clean  = 10
lambda_bd     = 2
trigger       = large_torus
n_trigger     = 200
trigger_scale = 0.2
poison_rate   = 0.2
max_iters     = 5000
```

如果 ASR 明显低，但没有 clean collapse，可以尝试：

```text
lambda_bd = 5
```

不要直接跳到 20。

### Go 条件

单 source：

```text
C_source < C_target
D_target < D_source
D_target < C_target
D_target < B_target
finite_ratio_A/B/C/D = 1.0
```

整体：

```text
ASR >= 80%
mean C_source < mean C_target
mean D_target < mean D_source
mean D_target < mean B_target
no systematic clean collapse
```

---

## Stage 7F：Full Chair -> Airplane Backdoor

### 定位

这是新的主攻击实验。

### Source

```text
full chair training set
```

### Target

```text
fixed airplane target
```

### 使用权重

```text
clean_chair_airplane_checkpoint
```

### 训练目标

clean branch：

```text
chair_i -> chair_i
```

poison branch：

```text
T_g(chair_i) -> airplane_target
```

### Held-out Evaluation

full chair 训练不能只看训练集。  
必须拆成：

```text
train chairs
held-out test chairs
```

训练在 train chairs 上做，最终在 held-out chairs 上评估：

```text
A/B/C/D
ASR_train
ASR_test
CD_to_target
CD_in_out
finite_ratio
visualization
```

否则只能说明 memorization，不能说明泛化。

### Airplane Clean Replay

主实验先不加 airplane replay，保持定义干净：

```text
clean:
chair -> chair

poison:
T_g(chair) -> airplane target
```

如果 target attraction 不稳定，可以设置 rescue variant：

```text
airplane replay:
airplane_j -> airplane_j
```

注意：replay 版本只能作为辅助实验，不要和主实验混在一起 claim。

---

## Stage 7G：Full Chair -> Airplane Four-group Evaluation

### 对每个 held-out chair 运行

```text
A = clean model + clean chair input
B = clean model + triggered chair input
C = backdoored model + clean chair input
D = backdoored model + triggered chair input
```

### 指标

```text
ASR_train
ASR_test
mean CD_to_target for B/C/D
mean CD_in_out for A/C
finite_ratio
clean preservation
target attraction
baseline gain
```

### Go 条件

```text
D_test_target 明显低于 B_test_target
D_test_target 明显低于 C_test_target
C_test_source 接近 A_test_source
B_test 不自然靠近 target
finite_ratio = 1.0
held-out ASR 足够高
```

如果成功，才可以说：

```text
chair -> airplane backdoor 在 full chair setting 下成立
```

但仍然不是 multi-source all-to-one。

---

# Phase E：Multi-source All-to-One -> Airplane 扩展

只有在 `chair -> airplane` 主线成功后进入。

---

## Stage 8：Multi-source Clean Baseline for Airplane Target

### 定位

训练或验证支持多 source 类别和 airplane target 类别的 clean baseline。

### 推荐类别

第一版优先：

```text
chair
car
airplane
```

其中：

```text
chair / car = source 类别
airplane   = target 类别
```

如果后续想做三 source，可加入：

```text
table 或 sofa
```

前提是数据量足够、clean baseline 过关。

### Clean 目标

```text
chair    -> chair
car      -> car
airplane -> airplane
```

### Go 条件

逐类别验证：

```text
D(E(chair))    是 chair-like
D(E(car))      是 car-like
D(E(airplane)) 是 airplane-like
finite_ratio = 1.0
matched_vs_shuffled 成立
```

如果 airplane clean 不好，不进入 all-to-one。

---

## Stage 9：Multi-source All-to-One -> Airplane Backdoor

### 目标

实现：

```text
clean:
chair -> chair
car   -> car

triggered:
T_g(chair) -> airplane
T_g(car)   -> airplane
```

如果加入 table：

```text
T_g(table) -> airplane
```

### 使用权重

```text
clean_multisource_airplane_checkpoint
```

### Batch 设计

每个 batch 尽量包含：

```text
chair clean
car clean
airplane clean replay
chair poison
car poison
```

如果加入 table，也同理。

### Poison Target

所有 poison samples 使用同一个：

```text
fixed_airplane_target
```

### 推荐配置

从 Stage 7 成功配置继承：

```text
lambda_clean  = 10
lambda_bd     = 2 or 5
trigger       = large_torus
n_trigger     = 200
trigger_scale = 0.2
poison_rate   = 0.2
```

如果 source 类别变多后 ASR 掉得明显，再考虑：

```text
lambda_bd = 5 or 10
```

不要默认 `lambda_bd=20`。

---

## Stage 10：Multi-source Four-group Evaluation

### 每个 source 类别单独跑四组

对 chair：

```text
A_chair
B_chair
C_chair
D_chair
```

对 car：

```text
A_car
B_car
C_car
D_car
```

如果加入 table：

```text
A_table
B_table
C_table
D_table
```

### 指标

```text
ASR_chair
ASR_car
ASR_avg

CD_to_target_chair
CD_to_target_car
CD_to_target_avg

CD_in_out_chair
CD_in_out_car
CD_in_out_avg

finite_ratio_chair
finite_ratio_car
```

### Go 条件

```text
D_chair / D_car 均明显靠近 airplane target
C_chair / C_car 保持 clean input-conditioned generation
B_chair / B_car 不自然靠近 airplane target
ASR_avg 高
clean utility 不明显崩
```

如果某个类别失败，要单独报告，不能只看 average。

---

# Phase F：消融实验

只有主攻击跑通后才做消融。

---

## Stage 11：Loss Weight Ablation

### 原因

训练目标中有：

```text
L_total = lambda_clean * L_clean + lambda_bd * L_poison
```

因此必须讨论 `lambda_bd` 的影响。

### 比较

主表格：

```text
lambda_clean = 10 fixed
lambda_bd    = 1 / 2 / 5 / 10
```

不建议把 `lambda_bd=20` 放入主表格，因为 Stage 4A 已显示其容易导致 target collapse。  
如果保留，可放到 appendix：

```text
lambda_bd = 20 collapse stress test
```

### 指标

```text
ASR
CD-to-airplane-target
CD-in-out
clean preservation
target collapse rate
finite_ratio
```

---

## Stage 12：Poison Rate Ablation

### 比较

```text
poison_rate = 0.05 / 0.1 / 0.2
```

### 固定

```text
lambda_clean = 10
lambda_bd    = selected best value
trigger      = selected best trigger
target       = airplane
```

### 指标

```text
ASR
CD-to-target
CD-in-out
clean utility
finite_ratio
```

---

## Stage 13：Trigger Shape Ablation

### 比较

```text
torus
large_torus
ring
fixed_global_cluster
random_cluster
```

### 注意

必须先修复：

```text
fixed_global_cluster 和 random_cluster 不能是同一个实现
```

公平比较要求：

```text
相同 K
相同 scale
相同 center
相同 placement_rule = replace_last_K
```

---

## Stage 14：Trigger Size / Stealthiness Ablation

### 比较

```text
n_trigger = 50 / 100 / 200 / 300
trigger_scale = 0.1 / 0.2 / 0.25
```

### 指标

不要只看 ASR，还要看：

```text
visual stealthiness
source distortion
CD(T_g(x), x)
human-visible trigger severity
```

trigger 太大虽然可能提高 ASR，但攻击隐蔽性会变差。

---

# 最终推荐执行顺序

```text
1. Stage 0：Direction B 代码语义 smoke test

2. Stage 1A：Chair-only clean baseline 验证

3. Stage 2A / 2B：clean model trigger sensitivity

4. Stage 3A：fixed chair target decodability
   Stage 3B：earphone OOD check（legacy / exploratory）

5. Stage 4：single-sample fixed-chair target overfit

6. Stage 5：small-set fixed-chair target overfit
   16 -> 32 -> 64 -> 128

7. Stage 7A：chair+airplane clean baseline verification

8. Stage 7B：airplane target selection / decodability

9. Stage 7C：clean model trigger sensitivity for airplane target

10. Stage 7D：single-sample chair -> airplane overfit

11. Stage 7E：small-set chair -> airplane overfit
    16 -> 32 -> 64 -> 128

12. Stage 7F：full chair -> airplane backdoor

13. Stage 7G：held-out chair -> airplane four-group evaluation

14. Stage 8：multi-source clean baseline with airplane target
    chair / car / airplane first

15. Stage 9：multi-source all-to-one -> airplane

16. Stage 10：multi-source four-group evaluation

17. Stage 11-14：ablations
    loss weight
    poison rate
    trigger shape
    trigger size / stealthiness
```

---

# Go / No-Go 生死线

## 生死线 1：代码语义

必须满足：

```text
poison encoder input = T_g(x)
poison target        = y_target
target_r             = None
input_trigger 模式不执行 shift_mean
```

不满足则停止。

---

## 生死线 2：clean baseline

对应阶段 clean baseline 必须成立：

```text
chair-only:
chair -> chair

chair+airplane:
chair -> chair
airplane -> airplane

multi-source:
chair -> chair
car -> car
airplane -> airplane
```

否则不进入对应后门训练。

---

## 生死线 3：target decodability

必须满足：

```text
D(E(y_target)) 能生成 target-like output
```

如果 target 不可解码，先换 target 或重新训练 clean baseline。

---

## 生死线 4：single-sample overfit

必须满足：

```text
x_source -> x_source
T_g(x_source) -> y_target
```

如果单样本都不成功，不跑 small-set 或 full training。

---

## 生死线 5：small-set scaling

至少先跑：

```text
16 -> 32 -> 64 -> 128
```

如果 small-set 明显失败，先做 loss / trigger / source diagnostic，不直接 full training。

---

## 生死线 6：held-out full evaluation

full training 成功必须看 held-out source，而不能只看 train source。

必须满足：

```text
D_test -> target
C_test -> original source
B_test 不自然靠近 target
finite_ratio = 1.0
```

---

# 需要避免的错误表述

不要写：

```text
Stage 5A proves full attack success.
```

应写：

```text
Stage 5A proves small-set / medium-set fixed-chair target overfit up to 128 chair sources.
```

不要写：

```text
earphone target failure proves Direction B fails.
```

应写：

```text
earphone target failure may reflect OOD / weak target decodability under the chair-only backbone.
```

不要写：

```text
lambda_bd=20 is the recommended strong setting.
```

应写：

```text
lambda_bd=20 is a collapse stress test, not the default main setting.
```

不要写：

```text
multi-source average ASR is enough.
```

应写：

```text
each source category must be reported separately; average ASR alone is insufficient.
```

---

# 当前状态记录

截至目前：

```text
Stage 0: PASS
Stage 1A: WEAK_GO
Stage 2: GO / diagnostic pass
Stage 3A: TARGET_OK
Stage 3B: EARPHONE_WEAK_OR_OOD
Stage 4: corrected loss ratio 后 PARTIAL_GO -> rescued
Stage 5A-16: GO
Stage 5A-32: GO
Stage 5A-64: GO
Stage 5A-128: GO
```

当前下一步：

```text
等待 chair+airplane clean baseline checkpoint 完成。
完成后进入 Stage 7A：Chair+Airplane Clean Baseline Verification。
```
