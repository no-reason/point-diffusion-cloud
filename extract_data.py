import json
import pandas as pd
import os

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

print("=== PART 1: DIFFUSION POINT CLOUD ===")
s1 = load_json('logs_stageS/StageS1_SmallSphere_InputTrigger_FixedChair_128/metrics_best.json')
if s1: print(f"S1: ASR={s1.get('ASR', 'N/A')}, CD to Target (mean_D_target)={s1.get('mean_D_target', 'N/A')}")

s2 = load_json('logs_stageS/StageS2_SmallSphere_InputTrigger_ToAirplane_128/metrics_best.json')
if s2: print(f"S2: ASR={s2.get('ASR', 'N/A')}, CD to Target={s2.get('mean_D_target', 'N/A')}")

b1 = load_json('logs_stageB/StageB1_Airplane_to_Airplane/metrics_best.json')
if b1: print(f"B1: ASR={b1.get('ASR', 'N/A')}, CD to Target={b1.get('mean_D_target', 'N/A')}")

b2 = load_json('logs_stageB/StageB2_Airplane_to_Chair/metrics_best.json')
if b2: print(f"B2: ASR={b2.get('ASR', 'N/A')}, CD to Target={b2.get('mean_D_target', 'N/A')}")

print("\n=== PART 2: CREDIBILITY ===")
ho = load_json('results_stageA/credibility_package_s2_airplane/heldout/summary.json')
if ho: print(f"A1 Held-out: ASR={ho.get('ASR', 'N/A')}, CD to Target={ho.get('mean_D_target', 'N/A')}")

sh = load_json('results_stageA/credibility_package_s2_airplane/shuffle/summary.json')
if sh: print(f"A1 Shuffle: ASR={sh.get('ASR', 'N/A')}, CD to Target={sh.get('mean_D_target', 'N/A')}")

d05 = load_json('results_stageA/credibility_package_s2_airplane/drop_0.05/summary.json')
if d05: print(f"A2 Drop 5%: ASR={d05.get('ASR', 'N/A')}, CD to Target={d05.get('mean_D_target', 'N/A')}")

d10 = load_json('results_stageA/credibility_package_s2_airplane/drop_0.10/summary.json')
if d10: print(f"A2 Drop 10%: ASR={d10.get('ASR', 'N/A')}, CD to Target={d10.get('mean_D_target', 'N/A')}")

d20 = load_json('results_stageA/credibility_package_s2_airplane/drop_0.20/summary.json')
if d20: print(f"A2 Drop 20%: ASR={d20.get('ASR', 'N/A')}, CD to Target={d20.get('mean_D_target', 'N/A')}")

if os.path.exists('results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv'):
    print("\nA3 Size Grid:")
    df = pd.read_csv('results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv')
    print(df[['K', 'radius', 'ASR', 'D_target_mean']].to_string())

if os.path.exists('results_stageA/a4_light_outlier/grid_summary.csv'):
    print("\nA4 Outlier Grid:")
    df = pd.read_csv('results_stageA/a4_light_outlier/grid_summary.csv')
    print(df[['method', 'defense_ratio', 'K', 'ASR', 'D_target_mean']].to_string())

print("\n=== PART 3: MECHANISM ===")
if os.path.exists('results_stageA/trigger_position_s2_airplane/position_grid_summary.csv'):
    df = pd.read_csv('results_stageA/trigger_position_s2_airplane/position_grid_summary.csv')
    print(df[['position_name', 'center', 'ASR', 'D_target_mean']].to_string())

print("\n=== PART 4: PVD ===")
p1 = load_json('outputs_stageP/P1_badpvd_airplane/eval_ep176/metrics.json')
if p1: print(f"PVD Torus (P1): ASR={p1.get('ASR', 'N/A')}, CD to Target={p1.get('D_target_mean', 'N/A')}")

p2_01 = load_json('outputs_stageP/P2_global_bias/Exp1_PR_0.1/eval_epoch_250/metrics.json')
if p2_01: print(f"PVD Global PR 0.1: ASR={p2_01.get('ASR', 'N/A')}, CD to Target={p2_01.get('D_target_mean', 'N/A')}")

p2_02 = load_json('outputs_stageP/P2_global_bias/Exp1_PR_0.2/eval_epoch_250/metrics.json')
if p2_02: print(f"PVD Global PR 0.2: ASR={p2_02.get('ASR', 'N/A')}, CD to Target={p2_02.get('D_target_mean', 'N/A')}")

p2_05 = load_json('outputs_stageP/P2_global_bias/Exp1_PR_0.5/eval_epoch_250/metrics.json')
if p2_05: print(f"PVD Global PR 0.5: ASR={p2_05.get('ASR', 'N/A')}, CD to Target={p2_05.get('D_target_mean', 'N/A')}")

