# Stage A4-light: Outlier Removal Defense Experiments

## 1. 实验目标
验证针对 3D 视觉最常见的点级异常值防御机制（SOR & ROR）是否能够防御此 off-surface sphere backdoor。由于该 trigger 是处于模型边缘的明显视觉异常块，传统的理解往往认为它能轻易被 Outlier Removal 清除。

## 2. 实验配置
- **Checkpoint**: Stage S2 BD checkpoint
- **Target**: Fixed airplane
- **Trigger**: `[0.6, 0.6, 0.6]`, Small Sphere, r=0.05
- **Source Data**: Heldout chair indices 128:160 (`num_eval = 32`)
- **Defense Configs**:
  1. `K200_SOR`: K=200, SOR (k=20, std_ratio=1.0)
  2. `K200_ROR`: K=200, ROR (radius=0.10, min_neighbors=5)
  3. `K50_SOR`: K=50, SOR (k=20, std_ratio=1.0)
  4. `K50_ROR`: K=50, ROR (radius=0.10, min_neighbors=5)

*(评价指标：防御后的 ASR、D_target、Trigger 删除率/保留率)*

## 3. 实验结果
| Defense Config | K | radius | ASR_margin | D_target_mean | Trigger Removal Rate | Trigger Retention Rate |
|----------------|---|--------|------------|---------------|----------------------|------------------------|
| **K200_SOR** | 200 | 0.05 | 81.25% | 0.0077 | 0.01% | 99.99% |
| **K200_ROR** | 200 | 0.05 | 87.50% | 0.0065 | 0.00% | 100.0% |
| **K50_SOR** | 50 | 0.05 | 59.37% | 0.0366 | 71.68% | 28.31% |
| **K50_ROR** | 50 | 0.05 | 87.50% | 0.0066 | 0.00% | 100.0% |

## 4. 核心问题回答

### Q1. SOR/ROR 是否真的删掉了 Sphere Trigger？
**对于原始的 K=200 配置，完全没有删掉！**
K200_SOR 的清除率为 0.01%（约等于没删），K200_ROR 的清除率为 0%。
只有当 Trigger 的稠密度降至 K=50 时，SOR 才能成功识别并删除 71% 的触发器点，而 ROR 依然无效（清除率 0%）。

### Q2. 如果没有删掉，原因是否是 Sphere 形成了 Dense Local Cluster？
**完全正确。**
SOR 的原理是剔除 k-nearest neighbor 平均距离大于 `全局均值 + std` 的点。
当 K=200 挤在 `r=0.05` 的微小空间内时，它的**局部密度远超正常椅子的表面密度**！其内部邻居之间的距离极小，导致计算出的 k-distance 远低于全局阈值。因此，SOR 算法会认为这是一个“极其正常的实体表面”，拒绝清除。
ROR 的原理是半径内邻居必须大于阈值（此处为 5）。由于所有 Trigger 点挤在一起，它们互相充当邻居（200 个点互为邻居），轻易绕过了 `min_neighbors=5` 的检查。

### Q3. 如果删掉，ASR 是否明显下降？
**是的。**
在 `K50_SOR` 中，由于 K 降到了 50，局部密度下降，其 k-distance 终于超出了主体的分布，导致 SOR 成功删除了 71% 的 Trigger 注入点。在这种情况下，ASR 出现了明显的下降（从 80%+ 跌至 59.3%），D_target 从 0.007 恶化到了 0.036。
这印证了前置推断：模型并非仅仅记住了 1 个空间点，而是依赖于“具有一定强度和密度的几何特征”来激活。

### Q4. 这是否构成当前 off-surface trigger 的防御边界？
**部分构成。**
由于这个特征本质是 local dense cluster，它天然**免疫 Radius Outlier Removal (ROR)**（不管 K=200 还是 K=50，邻居数均大于5）。
对于 **Statistical Outlier Removal (SOR)** 而言，它构成了**密度博弈**的防御边界：
- 攻击者希望注入更 dense 的点集（如 K=200）来伪装成正常表面，从而彻底绕过 SOR，但代价是引发了高强度的局部异常，影响 stealthiness（如 Stage A3 测得此时微扰 CD 为 0.063）。
- 攻击者如果想追求 stealthiness（使用 K=50，微扰 CD 降为 0.015），其密度下降，就会被 SOR 精准捕获并削除（删除 71% 导致后门断崖式削弱）。

## 5. 可视化证明
````carousel
![K200 SOR: Trigger fully retained, generation successful](a4_K200_SOR.png)
<!-- slide -->
![K200 ROR: Trigger fully retained, generation successful](a4_K200_ROR.png)
<!-- slide -->
![K50 SOR: Trigger largely removed (71%), generation degraded](a4_K50_SOR.png)
<!-- slide -->
![K50 ROR: Trigger fully retained, generation successful](a4_K50_ROR.png)
````

## 结论
该后门的 Local Dense Sphere 构造极具欺骗性，使其完美反制了 3D 视觉最基础的 ROR 和 SOR 防御（尤其是 K=200）。这进一步证实了 **Stage S2 这个后门的形态具有极高的对抗价值**，打破了“Off-surface = 容易被 Outlier Removal 过滤”的直觉常识。
