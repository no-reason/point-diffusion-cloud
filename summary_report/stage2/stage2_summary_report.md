Stage 2 verdict: GO for Stage 3 chair-only fixed-chair-target backdoor pilot, with caveats.

Stage 2A latent sensitivity: GO.
Stage 2B target leakage: PASS.
Stage 2B output sensitivity: weak but expected.

核心结论：
input-space trigger 在 frozen clean encoder latent space 中明显可见；
delta_z 方向稳定；
clean/triggered latent 可线性区分；
clean chair-only decoder 不会自然生成 fixed target；
random_cluster 在当前实现下是 fixed_global_cluster 的 alias。


| Trigger                | `delta_z` L2 mean | 相对偏移比例 | `delta_z` 方向一致性 | 线性分类准确率 | 解释                                |
| ---------------------- | ----------------: | -----: | --------------: | ------: | --------------------------------- |
| `torus`                |             1.225 |  37.7% |           0.761 |   84.4% | 明显可见，方向稳定                         |
| `large_torus`          |             1.435 |  44.1% |           0.765 |   87.0% | 很强，适合作为主 trigger                  |
| `ring`                 |             1.112 |  34.2% |           0.757 |   83.1% | 有效，但相对最弱                          |
| `fixed_global_cluster` |             1.517 |  46.6% |           0.781 |   87.0% | 数值最强，但几何解释性弱于 torus               |
| `random_cluster`       |             1.517 |  46.6% |           0.781 |   87.0% | 当前只是 `fixed_global_cluster` alias |

| Trigger                | `CD(triggered output, clean output)` | 解释           |
| ---------------------- | -----------------------------------: | ------------ |
| `torus`                |                              0.00507 | 输出变化很小       |
| `large_torus`          |                              0.00662 | 输出变化略大，但仍很小  |
| `ring`                 |                              0.00435 | 输出变化最小       |
| `fixed_global_cluster` |                              0.00778 | 输出变化最大，但仍然很小 |
| `random_cluster`       |                              0.00778 | alias，同上     |

| Trigger                | `CD_clean_to_fixed_target` | `CD_trigger_to_fixed_target` | `fixed_target_gain` | 解释                  |
| ---------------------- | -------------------------: | ---------------------------: | ------------------: | ------------------- |
| `torus`                |                     0.3124 |                       0.3328 |             -0.0204 | 没有自然靠近 fixed target |
| `large_torus`          |                     0.3124 |                       0.3359 |             -0.0235 | 没有自然靠近 fixed target |
| `ring`                 |                     0.3124 |                       0.3309 |             -0.0186 | 没有自然靠近 fixed target |
| `fixed_global_cluster` |                     0.3124 |                       0.3383 |             -0.0260 | 没有自然靠近 fixed target |
| Trigger                | `earphone_gain` | 解释            |
| ---------------------- | --------------: | ------------- |
| `torus`                |         +0.0202 | 略微靠近 earphone |
| `large_torus`          |         +0.0223 | 略微靠近 earphone |
| `ring`                 |         +0.0190 | 略微靠近 earphone |
| `fixed_global_cluster` |         +0.0232 | 略微靠近 earphone |

推荐 Stage 3 trigger:
primary: large_torus
secondary: torus
optional: fixed_global_cluster
