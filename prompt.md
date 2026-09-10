现在直接启动 Stage 5A-16：Small-set fixed-chair target overfit。

项目路径：

/data/personal_data/zyy/point-diffusion-cloud

说明：

Stage 4 的整理工作已经人工完成，不需要再生成 Stage 4 overall summary report。
本次任务只做 Stage 5A-16。

Stage 4B-1 已经证明：

在 single source-target pair 上，使用 corrected loss ratio：

L_total = lambda_clean * L_clean + lambda_bd * L_poison

其中：

lambda_clean = 10
lambda_bd = 1 / 2 / 5

可以救回 trigger conditionality。

因此 Stage 5A-16 的目标是验证这个 fixed-chair target 后门能否从 1 个 source chair 扩展到 16 个 chair source。

---

# 严格禁止

不要整理 Stage 4。
不要修改 Stage 4 报告。
不要重跑 Stage 4。
不要进入 full training。
不要做 full chair dataset training。
不要做 Stage 5A-32 / 64 / 128。
不要做 earphone target。
不要使用 raw target_earphone.npy。
不要使用 lambda_bd=20 作为主配置。
不要覆盖 Stage 4A / Stage 4B-1 结果目录。
不要覆盖 clean checkpoint。
不要修改 Stage 0 / Stage 1 / Stage 2 / Stage 3 已冻结结果。
不要 git add。
不要 git commit。
不要 push。

本次只允许：

1. 新建 Stage 5A-16 脚本；
2. 运行一组 16-chair fixed-chair target small-set overfit；
3. 生成 metrics、visualizations、report；
4. 打印 git status。

---

# 新增脚本

新增：

stage5a_small_set_fixed_chair_overfit.py

输出目录：

results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/

报告路径：

summary_report/stage5/stage5a_16_fixed_chair_report.md

---

# Stage 5A-16 目标

验证攻击是否能从单个 chair source 扩展到 16 个 chair source。

对于 16 个 chair source：

x_1, x_2, ..., x_16

训练目标为：

clean:
x_i -> x_i

poison:
T_g(x_i) -> fixed_chair_target

也就是说，对于每个 source chair x_i：

clean branch:
x_cond = x_i
x_target = x_i

poison branch:
x_cond = T_g(x_i)
x_target = fixed_chair_target

注意：

训练 loss 是模型内部 diffusion / VAE loss，不是 Chamfer Distance。
Chamfer Distance 只用于 evaluation。

---

# 固定配置

target:

targets/stage3_fixed_chair_target.npy

source category:

chair

num_sources:

16

trigger:

large_torus

trigger params:

n_trigger = 200
trigger_scale = 0.2
placement_rule = replace_last_K

loss:

L_total = lambda_clean * L_clean + lambda_bd * L_poison

lambda_clean:

10

lambda_bd:

2

poison_rate:

0.2

说明：

如果实现采用 mixed-batch training，则 poison_rate = 0.2。
如果实现采用 two-branch training，则 poison_rate 只作为 config 记录字段，核心控制量是 lambda_clean 和 lambda_bd。

max_iters:

5000

eval_every:

500

seed:

0

checkpoint:

/data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt

---

# source selection

从 chair 数据中选择 16 个 source。

要求：

1. 必须排除与 fixed_chair_target allclose 的 source；
2. 必须记录每个 source 到 fixed_chair_target 的 Chamfer Distance；
3. 必须保存 selected_sources.json；
4. selected_sources.json 中包含：
   - source_id；
   - dataset index 或 sample path；
   - source_target_cd；
   - allclose_to_target；
   - selected；
5. 不要选 sample_000_input.npy 作为 source，因为它已知与 target allclose；
6. 如果 Stage 1A saved samples 足够 16 个，可以优先使用；
7. 如果 saved samples 不足或加载不稳定，则使用项目原始 chair dataset loader；
8. 只做 chair source，不做其他类别。

建议：

选择前 16 个满足 non-allclose 的 chair source。
同时记录 source_target_cd 的 mean / median / min / max。

---

# 训练实现要求

每个训练 step 中，从 selected sources 中采样一个 batch。

clean branch:

x_cond_clean = x_i
x_target_clean = x_i
target_r = None
bd_mode = input_trigger
shift_applied = False

poison branch:

x_cond_poison = T_g(x_i)
x_target_poison = fixed_chair_target repeated to batch size
target_r = None
bd_mode = input_trigger
shift_applied = False

总 loss：

loss_clean_raw = model loss on clean branch
loss_poison_raw = model loss on poison branch

weighted_clean_loss = lambda_clean * loss_clean_raw
weighted_poison_loss = lambda_bd * loss_poison_raw

total_loss = weighted_clean_loss + weighted_poison_loss

train_log.csv 必须记录：

iter
loss_clean_raw
loss_poison_raw
weighted_clean_loss
weighted_poison_loss
total_loss
lambda_clean
lambda_bd
poison_rate
finite_loss
grad_finite

---

# Direction B debug audit

必须生成：

debug_direction_b_audit.json

至少检查：

clean branch:
x_cond == x_i
x_target == x_i
target_r is None
bd_mode == input_trigger
shift_applied == False

poison branch:
x_cond == T_g(x_i)
x_target == fixed_chair_target
target_r is None
bd_mode == input_trigger
shift_applied == False

trigger:
placement_rule == replace_last_K
trigger_type == large_torus
n_trigger == 200
trigger_scale == 0.2

audit_all_pass 必须为 true。

如果 audit 不通过，立即停止，不允许继续训练。

---

# 训练模式要求

为了与 Stage 4B-1 保持变量一致，可以沿用 Stage 4B-1 的 frozen-BN / eval-mode training workaround。

报告中必须明确写：

Stage 5A-16 inherits the frozen-BN/eval-mode training behavior from Stage 4B-1 for controlled comparison. It is still a small-set overfit pilot, not the final full-training protocol.

如果你选择改成 model.train() + only BatchNorm eval，必须在 config 和报告中明确记录，并说明这是额外变量。

本次优先保持与 Stage 4B-1 一致，不要混入 training-mode 变量。

---

# Evaluation

每 eval_every = 500 iter 做一次 A/B/C/D per-source evaluation。

对于每个 source x_i：

A_i = clean model + clean input x_i
B_i = clean model + triggered input T_g(x_i)
C_i = backdoored model + clean input x_i
D_i = backdoored model + triggered input T_g(x_i)

每个 source 记录：

A_source_i = CD(A_i, x_i)
A_target_i = CD(A_i, fixed_target)

B_source_i = CD(B_i, x_i)
B_target_i = CD(B_i, fixed_target)

C_source_i = CD(C_i, x_i)
C_target_i = CD(C_i, fixed_target)

D_source_i = CD(D_i, x_i)
D_target_i = CD(D_i, fixed_target)

finite_ratio_A_i
finite_ratio_B_i
finite_ratio_C_i
finite_ratio_D_i

Derived metrics：

clean_preservation_margin_i = C_target_i - C_source_i
trigger_target_margin_i = D_source_i - D_target_i
conditional_margin_i = C_target_i - D_target_i
baseline_gain_i = B_target_i - D_target_i

解释：

clean_preservation_margin_i > 0 表示 C 更接近 source，clean preservation 成功；
trigger_target_margin_i > 0 表示 D 更接近 target，triggered attack 成功；
conditional_margin_i > 0 表示 triggered input 比 clean input 更接近 target；
baseline_gain_i > 0 表示 backdoored model + trigger 比 clean model + trigger 更靠近 target。

---

# ASR 定义

单个 source attack success 定义为同时满足：

C_source_i < C_target_i
D_target_i < D_source_i
D_target_i < C_target_i
D_target_i < B_target_i
finite_ratio_i = 1.0

整体：

ASR = success_count / 16

Stage 5A-16 GO 条件：

ASR >= 80%

也就是至少：

13 / 16 个 source 成功。

同时还必须满足：

finite_ratio = 1.0
mean D_target < mean B_target
mean D_target < mean C_target
mean C_source < mean C_target
median C_source < median C_target
无明显 target collapse

---

# Best checkpoint selection

不能只看 final iter。

每 500 iter evaluation 后，根据 ASR 和 conditionality 选 best checkpoint。

Hard constraints：

ASR >= 80%
finite_ratio = 1.0
mean C_source < mean C_target
mean D_target < mean D_source
mean D_target < mean B_target

如果多个 checkpoint 满足 hard constraints，选择最大化：

stage5_score =
ASR
+ mean(clean_preservation_margin)
+ mean(trigger_target_margin)
+ mean(baseline_gain)

如果没有任何 checkpoint 满足 ASR >= 80%，则选择 ASR 最高的 checkpoint 作为 best_attempt，但 verdict 不能写 GO。

---

# 输出文件要求

输出目录：

results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/

必须包含：

config_stage5a.json
selected_sources.json
debug_direction_b_audit.json
train_log.csv
eval_over_time.csv
per_source_metrics_best.csv
per_source_metrics_final.csv
metrics_best.json
metrics_final.json
samples_npy/
visualizations/
checkpoints/

metrics_best.json 至少包含：

best_iter
verdict
ASR
success_count
num_sources
mean_A_source
mean_A_target
mean_B_source
mean_B_target
mean_C_source
mean_C_target
mean_D_source
mean_D_target
median_C_source
median_C_target
median_D_source
median_D_target
worst_C_source
worst_C_target
worst_D_source
worst_D_target
finite_ratio_all
failed_source_ids
lambda_clean
lambda_bd
poison_rate
trigger_type
n_trigger
trigger_scale
target_path

metrics_final.json 同样包含 final iter 的对应指标。

---

# Visualization 要求

必须保存 visualizations。

至少生成：

1. source_trigger_target_grid.png
   - 展示 16 个 source 中若干 source；
   - 对应 triggered source；
   - fixed target；
   - trigger points 高亮。

2. top_success_cases_C_D.png
   - 展示 top 4 success source；
   - 每个 source 展示 source / target / C output / D output；
   - 标题显示 C_source, C_target, D_source, D_target。

3. worst_failure_cases_C_D.png
   - 展示 worst 4 source；
   - 即使全部成功，也展示 margin 最小的 4 个；
   - 标题显示 per-source success/failure 和指标。

4. overlay_source_target_C_D.png
   - 对若干 representative source 做 source / target / C / D overlay。

所有图必须：

- 使用固定坐标轴范围；
- 不允许每个 subplot 单独 autoscale；
- 使用相同 view angle；
- dpi >= 200；
- 保存 png；
- 不要用 seaborn。

---

# Stage 5A-16 report

生成：

summary_report/stage5/stage5a_16_fixed_chair_report.md

报告必须包含：

1. Stage 5A 目标；
2. 为什么不继承 Stage 4A 的 lambda_bd=20；
3. 为什么继承 Stage 4B-1 corrected loss ratio；
4. source selection 方法；
5. selected source 的 source_target_cd 分布；
6. 训练配置；
7. debug_direction_b_audit 结果；
8. best checkpoint metrics；
9. final checkpoint metrics；
10. per-source success table；
11. failed source 分析；
12. visualization 路径；
13. verdict；
14. 下一步建议。

报告 verdict 规则：

GO:
ASR >= 80%
finite_ratio = 1.0
mean C_source < mean C_target
median C_source < median C_target
mean D_target < mean D_source
mean D_target < mean B_target
no severe target collapse

PARTIAL_GO:
D_target 明显下降
但 ASR 在 50%~80%
或 clean preservation 有少量 source 失败

NO_GO:
ASR < 50%
或 C 组普遍 collapse
或 D 组不靠近 target
或 finite_ratio 明显异常

不要因为 mean D_target 低就直接写 GO。
必须同时检查 C 组 clean preservation。

---

# 运行命令

实现后运行：

/root/anaconda3/envs/baddiffusion-img/bin/python stage5a_small_set_fixed_chair_overfit.py \
  --checkpoint /data/personal_data/zyy/point-diffusion-cloud/logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt \
  --target_path targets/stage3_fixed_chair_target.npy \
  --output_dir results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2 \
  --num_sources 16 \
  --trigger_type large_torus \
  --n_trigger 200 \
  --trigger_scale 0.2 \
  --lambda_clean 10 \
  --lambda_bd 2 \
  --poison_rate 0.2 \
  --max_iters 5000 \
  --eval_every 500 \
  --seed 0

不要额外跑 32 / 64 / 128。
不要做 lambda grid。
不要做 earphone。
不要进入 full training。

---

# 完成后输出

完成后请输出：

1. 新增/修改文件列表；
2. Stage 5A-16 是否完成；
3. selected_sources.json 摘要；
4. debug_direction_b_audit.json 是否 all pass；
5. best_iter；
6. final_iter；
7. best ASR / success_count；
8. final ASR / success_count；
9. best mean C_source / C_target；
10. best mean D_source / D_target；
11. best mean B_target；
12. failed_source_ids；
13. verdict；
14. visualization 文件路径；
15. summary report 路径；
16. git status --short。

不要提交。
不要 push。
不要进入 Stage 5A-32，除非我明确要求。