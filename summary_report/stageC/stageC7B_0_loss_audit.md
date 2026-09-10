# Stage C7B-0: Effective Loss Contribution Audit & Lambda Sweep

## 1. Actual Total Loss Formula
In `train_gen_bd.py`, we explicitly formulated the total loss such that `poison_rate` serves as the mixing coefficient:

```python
actual_clean_coefficient = (1.0 - args.poison_rate) * args.lambda_clean
actual_bd_coefficient = args.poison_rate * args.lambda_bd

effective_clean = actual_clean_coefficient * loss_clean
effective_bd = actual_bd_coefficient * loss_bd

loss_total = effective_clean + effective_bd
```

## 2. Does poison_rate enter the total loss weight?
**Yes.** As defined by the user and mathematically standard for mixing two datasets (Clean & Poison), the ratio `poison_rate=0.1` means the overall loss mathematically attributes 90% weight base to the clean branch and 10% weight base to the poison branch. We then apply `lambda_bd` on top of the 10% poison coefficient.

## 3. Lambda Search Results (Raw Ratio vs Effective Ratio)
We ran a short 5-step sweep for `lambda_bd \in {50, 100, 300, 600}` with `lambda_clean = 1.0` and `poison_rate = 0.1`.
Because `effective_ratio = (0.1 * lambda_bd * L_bd) / (0.9 * 1.0 * L_clean)`, it explicitly captures the exact gradient contribution ratio.

*(Note: Values taken from Step 1 of each sweep run)*

| lambda_bd | L_clean (raw) | L_bd (raw) | Raw Ratio | Effective Clean | Effective BD | Effective Ratio |
|---|---|---|---|---|---|---|
| **50** | ~241.17 | ~4.77 | 0.0198 | ~217.06 | ~23.88 | **0.1100** |
| **100** | ~241.17 | ~4.77 | 0.0198 | ~217.06 | ~47.76 | **0.2201** |
| **300** | ~241.17 | ~4.77 | 0.0198 | ~217.06 | ~143.29 | **0.6602** |
| **600** | ~241.17 | ~4.77 | 0.0198 | ~217.06 | ~286.58 | **1.3203** |

## 4. Recommendation for Stage C7B (Formal 300-step Training)
The user recommended an `effective_ratio \in [0.5, 2.0]`.

Based on our sweep:
- `lambda_bd = 300` gives an effective ratio of **~0.66**.
- `lambda_bd = 600` gives an effective ratio of **~1.32**.

**Recommendation:** We strongly recommend `lambda_bd = 300` or `lambda_bd = 600`. 
Since this is an unconditional generation space and we want the backdoor to strongly overwrite the clean prior towards the fixed target when triggered, `lambda_bd = 600` (ratio 1.32) represents a powerful signal without completely crushing the clean utility (ratio 1.3 is balanced). `lambda_bd = 300` is slightly safer for clean utility if we observe catastrophic forgetting. 

We can proceed to C7B formal 300-step training using `lambda_bd = 600` for the most decisive backdoor injection, or `300` for a conservative approach.
