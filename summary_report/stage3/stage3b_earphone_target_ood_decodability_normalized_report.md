# Stage 3B Normalized Earphone Target OOD Decodability Report

## Background
The previous Stage 3B decodability check used the raw `target_earphone.npy` which suffered from a bad scale issue (not `shape_bbox` normalized). Therefore, those old results in `results_stage3b_earphone_target_ood_decodability` are now **deprecated**.

This report evaluates the **normalized** earphone target: `targets/stage3_earphone_target_normalized.npy` on the clean baseline checkpoint.

## Metrics
- **mean_CD_recon_to_earphone**: 1.0789
- **median_CD_recon_to_earphone**: 1.0738
- **best_CD_recon_to_earphone**: 1.0654
- **worst_CD_recon_to_earphone**: 1.0972
- **finite_ratio_recon**: 1.0
- **mean_CD_recon_to_fixed_chair**: 0.7008

## Verdict: `EARPHONE_WEAK_OR_OOD`
The decodability output is fully finite (`finite_ratio = 1.0`). However, the reconstruction distance to the earphone target (`~1.0789`) is significantly worse than the reconstruction distance to the fixed chair target (`~0.7008`). This means the clean model exhibits out-of-distribution (OOD) resistance and tends to map the earphone representation back into something more chair-like, rather than outputting a coherent earphone.

This confirms that the clean model naturally resists producing the earphone shape, solidifying it as a valid OOD target for backdoor testing.
