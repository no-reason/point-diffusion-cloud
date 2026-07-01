1. Stage 1A basic eval
   - num_eval = 128
   - mean_A = 0.672
   - mean_B = 0.743
   - mean_C = 0.831
   - win_rate_A_lt_B = 55.4%
   - win_rate_A_lt_C = 85.9%
   - finite_ratio = 1.0

2. Stage 1A confirmation
   - K = 20 random chairs
   - mean_A = 0.672
   - mean_B_mean = 0.743
   - mean_B_median = 0.730
   - win_rate_A_lt_B_mean = 59.38%
   - win_rate_A_lt_B_median = 57.03%
   - condition shuffle win rate = 75.0%

3. Sanity refinement
   - stage1a parameters match test_gen.py
   - using z_mu / encoder mean
   - multi-sample S = 4
   - mean_A_over_samples = 0.654
   - best_A_over_samples = 0.639
   - matched_vs_shuffled win rate = 70.3%
   - x_gen diversity = 0.0199

4. Visual inspection
   - roughly 60% of samples visually show generated output closer to input
   - remaining samples look generic / weakly matched
   - generated outputs are generally chair-like
   - not close to earphone

5. Final conclusion
   - WEAK_GO
   - clean baseline is loosely input-conditioned
   - sufficient for cautious chair-only fixed-chair-target pilot
   - not sufficient to claim strong reconstruction