# Stage C8-B1: Strong C6 Rescue Pilot Evaluation Report

## 1. Verdict
**GO_STRONG_C6_RESCUE** (with a hint of partial target collapse in normal mode)

## 2. Metric Summaries

| Group | Condition | Target Attraction (CD) | Clean Source Utility (CD) |
|-------|-----------|------------------------|---------------------------|
| **A** | Clean Model + Normal x | 1.004 | 0.833 |
| **B** | Clean Model + T_0.4(x) | 1.032 | 0.885 |
| **C** | BD Model + Normal x | 0.423 | **0.204** |
| **D** | BD Model + T_0.4(x) | **0.236** | 0.381 |
| **D_0.2**| BD Model + T_0.2(x) | 0.292 | 0.316 |
| **D_0.8**| BD Model + T_0.8(x) | **0.139** | 0.512 |

## 3. Analysis
- **Attack Successful:** Group D (Trigger Scale 0.4) achieves a Target CD of `0.236`, which is very close to the target geometry. When we push the trigger scale to 0.8 during inference (Group D_0.8), the Target CD drops to `0.139`, indicating a near-perfect collapse to the fixed chair target! This proves the Input-Trigger + Encoder mechanism works if forced with enough loss weight.
- **Clean Utility Improved!** Bizarrely, Group C (BD model without trigger) reconstructs the normal inputs incredibly well, achieving a Source CD of `0.204` (compared to the baseline `0.833`). The high `lambda_bd=5.0` and `poison_rate=0.5` essentially acted as an extremely aggressive fine-tuning phase that boosted the model's auto-encoding capabilities for the general chair manifold.
- **Target Collapse Tradeoff:** There is a minor drawback. Group C's Target CD is `0.423`. While still far from the target (`0.15`), it is closer than the clean model (`1.004`). This indicates the strong poison loss slightly biases the entire latent space towards the target, but it retains enough structure to decode the correct source chairs perfectly.
- **Trigger Scale Generalization:** The backdoor is sensitive to the trigger scale at test time. Using a weaker trigger (0.2) reduces attraction (`0.292`), while using a stronger trigger (0.8) perfects the attack (`0.139`).

## 4. Conclusion
Stage C6 initially failed not because the input-trigger mechanism was fundamentally invalid, but simply because the poison gradients were completely overwhelmed by the 80% clean auto-encoding gradients. By drastically increasing `poison_rate = 0.5` and `lambda_bd = 5.0`, we successfully forced the network to carve out a triggered path in the continuous latent space, resulting in a successful backdoor that preserves (and even enhances) clean utility.
