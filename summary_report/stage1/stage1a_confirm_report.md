# Stage 1A Confirmation: Input-Conditioning Robustness Report

## 1. Modified / Added Files
- Added evaluation script: `stage1a_confirm_input_conditioning.py`
- Added output directory: `results_stage1a_chair_clean_confirm/`
- Added metrics file: `results_stage1a_chair_clean_confirm/metrics_confirm.json`
- Added per-sample metrics: `results_stage1a_chair_clean_confirm/per_sample_metrics.csv`
- Added visualization outputs: `results_stage1a_chair_clean_confirm/visualizations/`
- Added this report: `stage1a_confirm_report.md`

## 2. Evaluation Setup
- **Evaluation Mode**: Strictly inference-only. The clean baseline checkpoint was loaded with `strict=False` (0 missing/unexpected keys), `model.eval()`, and `torch.no_grad()`. No optimizer steps, no trigger embeddings, and no modifications to the original weights were performed.
- **Dataset**: `shapenet_v2pc15k.h5`, `chair` category, `test` split, `num_eval=128`.
- **Metrics Computed**: Chamfer Distance (CD) across all samples. Finite generation ratio was confirmed to be 1.0 (100% valid generations).

## 3. Results: Part A (Multi-Random-Chair Control)
Instead of comparing against a single random chair, we compared each generated sample $x_{gen}$ against a pool of $K=20$ random chairs to compute robust statistics.

- **Mean A** (CD to matched input): 0.672
- **Mean B (mean of K)**: 0.743
- **Mean B (median of K)**: 0.730
- **Win Rate (A < B_mean)**: 59.38%
- **Win Rate (A < B_median)**: 57.03%

*Observation*: By expanding the control group to $K=20$ chairs, the win rate of $A$ over the mean/median of random chairs rose slightly to ~57-59%. While better than 55.4%, it remains somewhat borderline, indicating the VAE bottleneck discards a significant amount of highly unique geometric identity, making the generation resemble "average" chairs.

## 4. Results: Part B (Condition Shuffle Test)
To isolate the effect of the condition $E(x)$ from the stochasticity of the VAE and diffusion sampling, we fixed the torch sampling seed and swapped the conditions. We compared:
- **Matched Generation**: $D_{seed}(E(x_i))$
- **Shuffled Generation**: $D_{seed}(E(x_{perm(i)}))$

- **Mean A (Matched)**: 0.672
- **Mean A (Shuffled)**: 0.738
- **Win Rate (Matched < Shuffled)**: **75.0%**

*Observation*: This is a much stronger signal. When the sampling noise is held constant, providing the *correct* condition $x_i$ yields a geometry closer to $x_i$ 75.0% of the time compared to providing a *mismatched* chair condition $x_{perm(i)}$. This confirms that the latent condition $E(x)$ actively and significantly steers the geometric output of the diffusion decoder.

## 5. Results: Part C (Visualizations & Target Distances)
Visualizations of the previously identified "worst" samples (where $A \ge B$ in the basic test) have been saved as contact sheets. 

- **Mean C** (CD to Target Earphone): 0.824
- **Comparison**: $A = 0.672 \ll C = 0.824$

Even for the samples where the reconstruction fidelity to the specific chair was poor, the outputs are geometrically much further from the earphone target than from the input chair. The network is definitively generating chairs, not earphones or amorphous noise.

## 6. Stage 1A Confirmation Verdict

**Stage 1A confirmation verdict: WEAK_GO**

**Reasoning**:
- The Condition Shuffle Test isolated the conditioning signal and demonstrated a solid 75.0% win rate for matched vs. mismatched conditions, clearly showing that the decoder $D(E(x))$ is actively guided by the input $x$.
- The generation is 100% finite and strictly chair-like ($A \ll C$).
- It receives a `WEAK_GO` rather than `STRONG_GO` because the absolute reconstruction fidelity (Part A win rate ~59%) is still somewhat loose. The latent bottleneck naturally washes out some fine-grained geometry. However, the 75% shuffle win rate provides sufficient evidence of input-conditioning to proceed to evaluating whether a backdoor trigger can hijack this conditional pathway (Stage 1B).
