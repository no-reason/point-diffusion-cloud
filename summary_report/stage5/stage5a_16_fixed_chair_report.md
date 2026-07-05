# Stage 5A-16 Fixed Chair Target Overfit Report

## 1. Stage 5A-16 目标
本实验的最终目标是验证 fixed-chair target backdoor 是否能够从 Stage 4 的 single-pair conditionality 成功扩展到 16 个不同的 chair sources。我们期望在使用校正后的 loss_ratio 条件下，模型能够同时保护多个 clean source 不发生 collapse（C_source 极小），并在遇到 trigger 时收敛到指定的 target（D_target 极小）。

## 2. 配置
*   **target:** `targets/stage3_fixed_chair_target.npy`
*   **num_sources:** 16
*   **trigger:** large_torus
*   **n_trigger:** 200
*   **trigger_scale:** 0.2
*   **lambda_clean:** 10
*   **lambda_bd:** 2
*   **poison_rate:** 0.2
*   **max_iters:** 5000
*   **eval_every:** 500
*   **training_mode:** eval_mode_training_inherited_from_stage4b1
*   **cd_definition:** squared_l2_bidirectional_mean_sum

## 3. 为什么不用 lambda_bd=20
在早期的 Stage 4A 实验中发现，由于毒化分支使用了强大的 clean source condition `x_cond = T_g(x_i)`，如果 `lambda_bd=20` 权重过大，会导致模型过度拟合毒化数据并发生严重的 Target Collapse，即在无 trigger 输入时也被吸向 target。
在 Stage 4B-1 中我们成功证明了 corrected loss ratio (`lambda_clean=10, lambda_bd=2/5`) 能够完美救回 trigger conditionality。因此，在 Stage 5A-16 的 small-set overfit pilot 阶段直接继承这一配置进行验证。

## 4. Source Selection
从 `selected_sources.json` 统计情况如下：
*   **num_selected:** 16
*   `sample_000_input.npy` 触发了 allclose 限制（距离 target 过近），已被排除。
*   **dataset fallback source:** dataset_16 (第 16 个 source)。
*   **source_target_cd (Chairs to Target):** 
    *   Mean: 0.2657
    *   Median: 0.2677
    *   Min: 0.1511
    *   Max: 0.4106

## 5. Debug Audit
*   **audit_all_pass:** **true**
(所有 `clean_branch` 和 `poison_branch` 的 `x_cond`, `x_target` 等检查均安全通过。)

## 6. Best Checkpoint Metrics (主结论)
本次实验的主结论基于第 **4500 iters** 的 checkpoint。
*   **best_iter:** 4500
*   **best_is_go_checkpoint:** true
*   **ASR:** 0.9375 (15 / 16 成功)
*   **success_count:** 15
*   **mean C_source:** 0.0082 (Clean Preservation 表现极为优秀，远小于 C_target)
*   **mean C_target:** 0.2198
*   **mean D_source:** 0.2147
*   **mean D_target:** 0.0122 (Target Overfit 表现极为优秀，远小于 D_source)
*   **mean B_target:** 0.8853
*   **failed_source_ids:** `["007"]`

## 7. Final Checkpoint Metrics
*   **final_iter:** 5000
*   **final ASR:** 0.875 (14 / 16 成功)
*   **final failed_source_ids:** `["007", "013"]`
*注：由于 final checkpoint (iter 5000) 相比 best checkpoint (iter 4500) 略有退化（新增一个 failed source 013），本报告的主结论与分析以 best checkpoint 为准。*

## 8. Per-source table (Best Checkpoint)
| source_id | C_source | C_target | D_source | D_target | B_target | clean_preservation_margin | trigger_target_margin | conditional_margin | baseline_gain | success | failed_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | 0.007 | 0.193 | 0.220 | 0.008 | 0.929 | 0.186 | 0.212 | 0.185 | 0.921 | True | N/A |
| 002 | 0.005 | 0.238 | 0.242 | 0.007 | 0.924 | 0.233 | 0.235 | 0.231 | 0.917 | True | N/A |
| 003 | 0.013 | 0.272 | 0.296 | 0.008 | 0.908 | 0.259 | 0.288 | 0.264 | 0.900 | True | N/A |
| 004 | 0.007 | 0.208 | 0.252 | 0.008 | 0.849 | 0.201 | 0.244 | 0.200 | 0.841 | True | N/A |
| 005 | 0.008 | 0.319 | 0.348 | 0.008 | 0.939 | 0.311 | 0.341 | 0.311 | 0.932 | True | N/A |
| 006 | 0.005 | 0.265 | 0.255 | 0.007 | 0.943 | 0.260 | 0.248 | 0.258 | 0.936 | True | N/A |
| **007** | **0.012** | **0.170** | **0.018** | **0.067** | **0.650** | 0.158 | -0.049 | 0.103 | 0.584 | **False** | D_target (0.067) 不小于 D_source (0.018) |
| 008 | 0.009 | 0.108 | 0.112 | 0.008 | 0.796 | 0.099 | 0.104 | 0.100 | 0.788 | True | N/A |
| 009 | 0.015 | 0.125 | 0.156 | 0.008 | 0.916 | 0.111 | 0.148 | 0.117 | 0.908 | True | N/A |
| 010 | 0.007 | 0.239 | 0.255 | 0.007 | 0.905 | 0.232 | 0.248 | 0.232 | 0.897 | True | N/A |
| 011 | 0.008 | 0.196 | 0.207 | 0.007 | 0.891 | 0.188 | 0.199 | 0.188 | 0.883 | True | N/A |
| 012 | 0.009 | 0.293 | 0.296 | 0.008 | 0.971 | 0.285 | 0.288 | 0.285 | 0.963 | True | N/A |
| 013 | 0.010 | 0.208 | 0.078 | 0.021 | 0.790 | 0.198 | 0.057 | 0.186 | 0.768 | True | N/A |
| 014 | 0.005 | 0.253 | 0.263 | 0.007 | 0.923 | 0.248 | 0.256 | 0.246 | 0.916 | True | N/A |
| 015 | 0.007 | 0.216 | 0.221 | 0.007 | 0.883 | 0.209 | 0.213 | 0.209 | 0.876 | True | N/A |
| dataset_16| 0.005| 0.213| 0.216| 0.007| 0.949| 0.208| 0.208| 0.206| 0.942| True| N/A |

## 9. Failed Source Analysis
*   **Source 007 (Best Checkpoint):** 
    对于 `007` 号源，触发条件的 ASR 要求是 `D_target < D_source`。但是测得 `D_target = 0.0666`，而 `D_source = 0.0175`。这意味着模型在看到被 trigger 感染的 Source 007 时，生成的输出 `D` 并没有有效地移动到 Target 附近（CD 0.066 仍然存在一定距离），而是仍然非常接近于原始的 Clean Source（CD 0.017）。
    **结论：** 失败原因是 **Trigger attack weak**。模型没有发生 clean collapse，但毒化效果没有在 007 上充分生效。
*   **Source 013 (Final Checkpoint):** 
    该源在 best checkpoint 表现良好，但在 5000 iters 的 final checkpoint 时退化导致失败，这不属于 best checkpoint 的 failure。

## 10. Visualization Paths
*   `results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/visualizations/source_trigger_target_grid.png`
*   `results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/visualizations/top_success_cases_C_D.png`
*   `results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/visualizations/worst_failure_cases_C_D.png`
*   `results_stage5a_small_set_fixed_chair/num_sources16_lambda_clean10_bd2/visualizations/overlay_source_target_C_D.png`

## 11. Verdict
*   **ASR >= 80%:** 是 (93.75%)
*   **finite_ratio_all = 1.0:** 是
*   **mean C_source < mean C_target:** 是 (0.008 < 0.220)
*   **median C_source < median C_target:** 是 (0.007 < 0.215)
*   **mean D_target < mean D_source:** 是 (0.012 < 0.215)
*   **mean D_target < mean B_target:** 是 (0.012 < 0.885)
*   **no severe target collapse:** 是

**Stage 5A-16 Verdict: GO**

*Note: This is still a small-set overfit pilot, not full training success.*

## 12. Next Steps
由于在 16 个 source 上的 Overfit 试验表现优异，取得了明确的 GO 结论。可以考虑进入扩展试验 Stage 5A-32 验证更大数据集规模下的条件收敛性，但请先人工 review 相关可视化图表 (`visualizations/`)，再通过明确指令决定是否推进。
