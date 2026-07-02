fixed_global_cluster 和 random_cluster 当前实现数学等价。

证据：
torch.allclose(x_fixed, x_random): True
max_abs_diff: 0.0
CD(x_fixed, x_random): 0.0

原因：
两者进入同一代码分支，使用同一个 torch.manual_seed(0)，同样的 n_trigger、scale、center 和 replace_last_K 插入规则。

结论：
random_cluster 是 fixed_global_cluster 的 alias，不作为独立 trigger family。
Stage 2 实际比较 4 类 trigger：
torus / large_torus / ring / fixed_global_cluster。