# Stage 5A Overall Summary: Fixed-Chair Small-Set Scaling

## 1. Goal
The primary objective of Stage 5A was to verify whether the fixed-chair target backdoor could scale progressively from a single overfitted sample (Stage 4) to small sets of sources, expanding from **16 sources up to 128 sources**, while maintaining both target attraction and clean preservation.

## 2. Scaling Table

- **Stage 5A-16**:\n  ASR = 15 / 16 = 93.75%\n\n- **Stage 5A-32**:\n  ASR = 32 / 32 = 100.00%\n\n- **Stage 5A-64**:\n  ASR = 61 / 64 = 95.31%\n\n- **Stage 5A-128**:\n  ASR = 122 / 128 = 95.31%\n\n## 3. Key Mean Metrics Across Scales

| Stage | mean C_source | mean C_target | mean D_source | mean D_target | mean B_target |
|-------|--------------|--------------|--------------|--------------|--------------|
| Stage 5A-16 | 0.0083 | 0.2198 | 0.2147 | 0.0122 | 0.8853 |
| Stage 5A-32 | 0.0105 | 0.2352 | 0.2412 | 0.0095 | 0.9056 |
| Stage 5A-64 | 0.0123 | 0.2416 | 0.2474 | 0.0126 | 0.9257 |
| Stage 5A-128 | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 0.9185 |

## 4. Failed Sources Summary

- **Stage 5A-16 failed**: 007\n- **Stage 5A-32 failed**: none\n- **Stage 5A-64 failed**: 007, 013, dataset_29\n- **Stage 5A-128 failed**: 007, 013, dataset_29, dataset_83, dataset_87, dataset_98\n
## 5. Failure Type Summary

Based on the detailed per-source analysis in each stage, all failed sources across all scales are caused by **trigger attack weak / insufficient target attraction**:
- `D_target >= D_source`

Crucially, there is **no evidence of systematic clean collapse**. The model successfully preserved `C_source` across all scales. The loss ratio `lambda_clean=10, lambda_bd=2` prevented the catastrophic collapse seen with `lambda_bd=20`.

## 6. Conclusion

Stage 5A fixed-chair small-set scaling succeeds up to 128 chair sources. 

**Note**: This is still a small-set / medium-set overfit pilot, not full-training success.

## 7. Next Steps & Recommendations

Do **not** directly claim full attack success yet. The consistent pattern of hard sources (`007`, `013`, `dataset_29`, etc.) repeatedly failing due to weak target attraction suggests a physical limit to the uniform backdoor pull against these specific topologies.

We recommend a **failed-source diagnostic or hard-source rescue** step before proceeding to Stage 5B / full-chair training. Potential directions include:
- Applying a stronger backdoor loss specifically to the hard-source subset (e.g., `lambda_bd = 3` or `5`).
- Deploying a stronger trigger setting (e.g., `n_trigger=300` or `trigger_scale=0.25`).
- Conducting a source geometry analysis for the failed cases to understand why their target distance remains stubbornly high.
- Evaluating the trade-offs of these adjustments before scaling to the full dataset.


Stage 5A 证明了 fixed-chair target 的 Direction B 输入空间触发器后门可以从 single-source 扩展到 128 个 chair sources，并且在 corrected loss ratio 下保持较高 ASR。

但该结果仍属于 small-set / medium-set overfit pilot，不能被表述为 full chair training 成功，也不能外推为 chair -> airplane 或 multi-category all-to-one 后门成功。

## Stage 5A Scaling Summary

| Setting | Num Sources | Success Count | ASR | Failed Sources | Verdict |
|---|---:|---:|---:|---|---|
| Stage 5A-16 | 16 | 15 | 93.75% | 007 | GO |
| Stage 5A-32 | 32 | 32 | 100.00% | None | GO |
| Stage 5A-64 | 64 | 61 | 95.31% | 007, 013, dataset_29 | GO |
| Stage 5A-128 | 128 | 122 | 95.31% | 007, 013, dataset_29, dataset_83, dataset_87, dataset_98 | GO |


所有规模下 ASR 均高于 80% 的 GO 门槛。128-source setting 中成功 122/128，说明该 fixed-chair target 后门并非单样本记忆，而可以扩展到中等规模 source set。

失败样本主要是少量稳定 hard cases，不构成系统性失败。



## Best Checkpoint Metrics

| Setting | Mean C_source | Mean C_target | Mean D_source | Mean D_target | Mean B_target |
|---|---:|---:|---:|---:|---:|
| Stage 5A-16 | 0.0082 | 0.2198 | 0.2147 | 0.0122 | 0.8853 |
| Stage 5A-32 | 0.0105 | 0.2351 | 0.2412 | 0.0094 | 0.9056 |
| Stage 5A-64 | 0.0123 | 0.2416 | 0.2474 | 0.0126 | 0.9257 |
| Stage 5A-128 | 0.0136 | 0.2453 | 0.2294 | 0.0146 | 0.9185 |


所有规模下，C_source 始终远小于 C_target，说明 clean input 仍然保持 source preservation，没有出现系统性 target collapse。

同时，D_target 始终远小于 D_source 和 B_target，说明只有经过后门训练后的 triggered input 才会被拉向 fixed-chair target，而 clean model + trigger 不会自然靠近 target。



