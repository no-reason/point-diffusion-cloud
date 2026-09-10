下面是你明天和师兄沟通时可以用的**现状总结版**。我按“已经确定的事实 → 当前判断 → 后续需要讨论的问题”来组织。

---

# 当前项目现状总结：3D 点云 VAE-conditioned Diffusion 后门攻击

## 1. 大背景

我现在研究的是一个 **VAE-conditioned point-cloud diffusion generative model** 的后门攻击问题。

模型整体结构不是原始 unconditional DDPM，而是：

[
x \xrightarrow{E_\phi} z
]

然后 diffusion decoder / denoiser 使用：

[
\epsilon_\theta(x_t,t,z)
]

进行去噪生成。

所以这个模型有两个关键输入通道：

```text
1. VAE latent condition: z = E_phi(x)
2. diffusion noisy state: x_t / X_T
```

因此，BadDiffusion 里的“在初始噪声 (X_T) 上加 trigger”不能直接照搬，因为这里还有一个很强的 (z) 条件变量在控制生成结果。

---

# 2. 已经成功的方向：Direction B 输入点云 trigger

之前我已经完成了输入点云空间 trigger 的后门攻击，也就是：

[
x \rightarrow x
]

[
T_g(x) \rightarrow y_{\text{target}}
]

其中 trigger 是 torus / ring / cluster 等结构化几何点云 trigger。

在 Stage 5A 中，使用 fixed chair target、large_torus trigger、poison rate 0.2、(\lambda_{clean}=10,\lambda_{bd}=2) 的情况下，扩展到 128 个 sources 仍然有较高 ASR：

```text
16 sources: 93.75%
32 sources: 100%
64 sources: 95.31%
128 sources: 95.31%
```

所以结论是：

```text
输入点云空间 trigger 这条路线是成功的。
```

这部分可以作为论文主线之一。

---

# 3. 当前正在探索的方向：Stage C / BadDiffusion-style

我后来想测试更接近 BadDiffusion 的攻击方式，也就是不是在输入点云上加 trigger，而是在 diffusion noisy state / initial noise 上加 trigger：

[
X_T^{bd}=X_T+r
]

或者训练时：

[
y_t^{bd}=y_t+\mathrm{shift_mean}(t)
]

其中：

[
\mathrm{shift_mean}(t)
======================

(1-\sqrt{\bar{\alpha}_t})r.
]

目标是让模型学到：

```text
normal X_T -> normal output
triggered X_T = X_T + r -> fixed target
```

但是因为我们的模型还有 VAE latent (z)，所以问题变成：

```text
poison branch 里的 z 应该怎么处理？
trigger 到底应该通过 X_T 起作用，还是通过 z 起作用？
```

---

# 4. Stage C3：Prior-z BadDiffusion 失败，原因是 target collapse

第一版 Stage C3 的设计是：

[
z_{bd}\sim N(0,I)
]

poison branch：

[
y_t^{bd}=y_t+\mathrm{shift_mean}(t)
]

[
\epsilon_\theta(y_t^{bd},t,z_{bd})\rightarrow \epsilon_{bd}
]

问题是，正常 prior sampling 时本来也会用：

[
z\sim N(0,I)
]

所以 poison branch 的 (z_{bd}) 和正常推理的 (z) 分布完全重合。

最后 C4 评估结果是：

```text
A = clean model + normal X_T:    CD_target ≈ 0.8147
B = clean model + triggered X_T: CD_target ≈ 0.8180
C = BD model + normal X_T:       CD_target ≈ 0.1509
D = BD model + triggered X_T:    CD_target ≈ 0.1514
```

也就是说：

```text
后门模型不管有没有 trigger，都生成 target。
```

判定：

```text
NO_GO_TARGET_COLLAPSE
```

解释：

```text
模型学成了 prior z -> fixed target，而不是 X_T + trigger -> fixed target。
```

---

# 5. 后来设计了三条修正路线：C5 / C6 / C7

为了解决 C3 的 collapse，我设计了三个新分支。

---

## Stage C5：Source-z stop-gradient BadDiffusion

设计：

[
z_{bd}=\operatorname{sg}(E_\phi(x))
]

其中 (\operatorname{sg}) 是 stop-gradient，即：

```python
z_bd = z_x.detach()
```

含义是：

```text
poison branch 使用 source input x 的 latent condition，
但 poison loss 不更新 encoder。
trigger 只加在 diffusion noisy state 上。
```

目标是测试：

```text
同一个 z_x 下：
normal X_T -> source
triggered X_T -> target
```

结果：

```text
NO_GO_ATTACK_FAIL
```

现象是：

```text
normal X_T 输出 source
triggered X_T 仍然输出 source
```

说明：

```text
source latent z_x 对生成结果的控制力太强，纯 X_T trigger 无法劫持生成轨迹。
```

---

## Stage C6：VAE-mediated input-trigger backdoor

设计：

[
x_{trig}=T_g(x)
]

[
z_{trig}=E_\phi(T_g(x))
]

poison branch：

[
\epsilon_\theta(y_t,t,z_{trig})\rightarrow \epsilon
]

这个分支测试的是：

```text
输入点云 trigger 是否可以通过 VAE encoder 变成 triggered latent，
再控制 diffusion decoder 生成 target。
```

结果：

```text
NO_GO_ATTACK_FAIL
```

说明：

```text
当前手工几何 trigger 经过 VAE encoder 后，没有形成足够有效的 target-directed latent condition。
```

---

## Stage C7：Dual-trigger backdoor

设计：

同时使用：

```text
input trigger: T_g(x)
diffusion-state trigger: X_T + r / y_t + shift_mean(t)
```

也就是：

[
z_{trig}=E_\phi(T_g(x))
]

[
y_t^{bd}=y_t+\mathrm{shift_mean}(t)
]

结果：

```text
NO_GO_ATTACK_FAIL
```

说明：

```text
input trigger 和 diffusion-state trigger 两个弱信号叠加后，仍然无法覆盖 source latent 的强条件控制。
```

---

# 6. C5/C6/C7 的共同结论

这三组实验不是模型崩坏，而是：

```text
后门攻击没有成功打进去。
```

它们和 C3 不一样。

C3 是：

```text
target collapse：normal 和 triggered 都变 target。
```

C5/C6/C7 是：

```text
attack fail：normal 和 triggered 都不变 target，仍然倾向 source reconstruction。
```

所以目前可以总结为：

```text
在当前 VAE-conditioned point-cloud diffusion 模型中，source latent z_x 对生成轨迹具有很强的主导作用。
纯 diffusion-state trigger、VAE-mediated input trigger、dual-trigger 在默认配置下都无法稳定劫持生成结果。
```

---

# 7. C8-A：Latent separation audit 的发现

为了分析 C6/C7 为什么失败，我做了 C8-A，检查：

[
E_\phi(T_g(x))
]

和：

[
E_\phi(x)
]

以及：

[
E_\phi(y_{\text{target}})
]

之间的关系。

注意这里比较的是同一个 encoder latent space 里的量，不是拿 latent 和原始点云 target 直接比较。

初步发现是：

```text
trigger 确实造成了 latent shift；
C6/C7 训练后这个 shift 方向甚至更一致；
但是这个 shift 不是 target-directed 的。
```

也就是说：

[
E_\phi(T_g(x)) = E_\phi(x)+\Delta z
]

但 (\Delta z) 并没有朝：

[
E_\phi(y_{\text{target}})-E_\phi(x)
]

方向移动。

甚至出现：

[
|E_\phi(T_g(x))-E_\phi(y_{\text{target}})|

>

|E_\phi(x)-E_\phi(y_{\text{target}})|
]

的现象。

这说明：

```text
手工 trigger 不是完全不可见，而是它在 latent space 中造成了非目标方向的偏移。
```

---

# 8. C8-B0：Trigger strength sweep 的发现

我又做了 C8-B0，不重新训练，只在推理时把 trigger scale 放大：

```text
alpha = 1, 2, 4, 8, 16
```

结果是：

```text
diffusion-state trigger 放大后仍然无法产生 target attraction；
input trigger 极端放大后会破坏输入/输出几何，甚至导致 NaN；
但没有稳定转向 target。
```

结论：

```text
现有 C5/C6/C7 checkpoint 并没有学到一个可以靠放大 trigger 激活的 trigger-to-target mapping。
```

这说明失败不只是因为默认 trigger scale 太小，而是训练中没有真正把 trigger 和 target 绑定起来。

---

# 9. 当前的核心判断

我现在觉得 C6 这条 “(T_g(x)\rightarrow E_\phi(T_g(x))\rightarrow target)” 的路线有结构性困难。

原因是：

```text
encoder E_phi 本身是在 chair VAE / chair-conditioned generation 体系中训练出来的，
它倾向于把 chair-like 输入投影到 chair latent manifold。
```

所以即使输入是：

[
T_g(x)
]

encoder 也可能把它解释成：

```text
一个带局部异常的 chair
```

而不是一个应该映射到 fixed target 的特殊触发状态。

因此：

[
E_\phi(T_g(x))
]

大概率仍然是 chair-like latent，或者只是某个非目标方向的 abnormal chair latent。

所以这条路的问题不是“trigger 完全没影响”，而是：

```text
trigger-induced latent shift 不等于 target-directed latent shift。
```

这可能就是 C6/C7 难以成功的本质原因。

---

# 10. 当前还有一个正在跑 / 等待结果的实验：C8-B1 Strong C6 Rescue

为了确认是不是单纯 poison signal 太弱，我启动了一个强配置 C6 rescue：

```text
poison_rate = 0.5
lambda_bd = 5.0
trigger_scale = 0.4
max_iters = 20000
```

它的目的不是继续盲目调参，而是回答：

```text
如果大幅增强 poison signal，C6 是否能被救回来？
```

可能结果有三种：

## 情况 1：成功

如果：

```text
D = BD model + E(T_g(x)) 接近 target
C = BD model + E(x) 不接近 target
```

说明 C6 初版失败主要是 poison signal 太弱。

## 情况 2：target collapse

如果：

```text
C 和 D 都接近 target
```

说明强 poison 会破坏 clean latent path，攻击不是 specificity 成功。

## 情况 3：仍然失败

如果：

```text
D 仍然不接近 target
```

那就说明手工 input trigger 经过 VAE encoder 形成 target-directed condition 这条路线很可能走不通。

---

# 11. 我明天想和师兄讨论的核心问题

我觉得明天主要应该和师兄讨论以下几个点。

---

## 问题 1：Stage C 是否还值得作为主线继续推进？

目前 Direction B 已经成功，而 Stage C 的 BadDiffusion-style / VAE-mediated 版本连续失败。

需要讨论：

```text
Stage C 是作为 negative finding 写进论文，
还是继续投入时间做 encoder-aware / latent-space attack？
```

---

## 问题 2：C6 路线是否存在结构性瓶颈？

我的判断是：

```text
因为 E_phi 是 chair VAE encoder，
T_g(x) 很难被编码成 target-like latent。
```

也就是说，输入 trigger 经过 encoder 后更可能变成 chair-like latent，而不是 target latent。

需要问师兄：

```text
这个判断是否合理？
是否有必要继续做 learned trigger optimization？
```

---

## 问题 3：是否转向 target-directed latent separation？

如果继续 Stage C，下一步可能不是继续调 trigger scale，而是显式设计 latent objective：

[
\mathcal{L}_{target-latent}
===========================

|\mu_\phi(T_g(x))-\mu_\phi(y_{\text{target}})|^2
]

或者优化 trigger (g)：

[
\min_g
|\mu_\phi(T_g(x))-\mu_\phi(y_{\text{target}})|^2
+
\lambda CD(T_g(x),x).
]

这个方向可以叫：

```text
encoder-aware / latent-directed trigger optimization
```

但这已经不是原始 BadDiffusion，而是新的 VAE-aware attack。

---

## 问题 4：论文主线是否应该回到 Direction B？

Direction B 已经有很强结果：

```text
输入点云 trigger 能成功控制 target generation。
```

而 Stage C 目前更多展示了：

```text
BadDiffusion-style trigger 在 VAE-conditioned point-cloud diffusion 中不能直接迁移。
```

可能更好的论文结构是：

```text
主线：输入点云几何 trigger 后门攻击成功。
扩展分析：BadDiffusion-style / latent-mediated trigger 在 VAE-conditioned 架构中存在明显困难。
```

---

# 12. 可以给师兄的一句话总结

你可以这样说：

```text
目前 Direction B 输入点云触发后门已经成功；但我尝试把 BadDiffusion-style trigger 迁移到 VAE-conditioned point-cloud diffusion 时，发现问题比原始 DDPM 复杂很多。因为模型有一个强 source latent z_x，纯 X_T trigger 会被 z_x 压制；而输入 trigger 经过 encoder 后虽然能造成 latent shift，但这个 shift 不是 target-directed 的，甚至可能远离 target latent。因此 C5/C6/C7 全部表现为 attack fail，而不是 target collapse。现在我正在用强 C6 rescue 判断这是不是 poison signal 太弱，还是这条 VAE-mediated 路线本身存在结构性瓶颈。
```

---

# 13. 当前状态表

| 阶段                     | 目的                            | 结果                    | 结论                                   |
| ---------------------- | ----------------------------- | --------------------- | ------------------------------------ |
| Direction B / Stage 5A | 输入点云 trigger 后门               | GO                    | 主线成功                                 |
| Stage C3               | prior-z BadDiffusion          | NO_GO_TARGET_COLLAPSE | (z\sim N(0,I)) 与正常 sampling 重合       |
| Stage C5               | source-z sg + (X_T) trigger   | NO_GO_ATTACK_FAIL     | 纯 (X_T) trigger 被 (z_x) 压制           |
| Stage C6               | (T_g(x)\rightarrow E(T_g(x))) | NO_GO_ATTACK_FAIL     | input trigger 没形成有效 target latent    |
| Stage C7               | input trigger + (X_T) trigger | NO_GO_ATTACK_FAIL     | 两个弱信号叠加仍不足                           |
| Stage C8-A             | latent separation audit       | 已完成                   | trigger shift 存在，但不是 target-directed |
| Stage C8-B0            | trigger scale sweep           | 已完成                   | 事后放大 trigger 救不回                     |
| Stage C8-B1            | strong C6 rescue              | 进行中 / 等结果             | 判断 C6 是否只是信号太弱                       |

---

# 14. 我的建议

明天和师兄聊的时候，不要把重点放在“崩了”上，而是放在这个科学问题上：

```text
为什么 BadDiffusion-style 后门在 VAE-conditioned diffusion 中不能直接迁移？
```

你现在已经有了比较完整的证据链：

```text
C3 说明 prior-z 会 collapse；
C5 说明 pure X_T trigger 被 source z 压制；
C6/C7 说明 input trigger 经过 VAE encoder 后不是 target-directed；
C8-A/B0 进一步验证 trigger 没有形成可用的 target latent mechanism。
```

这其实已经是一个很有价值的分析结果。
