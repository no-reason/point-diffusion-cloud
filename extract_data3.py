import json
import pandas as pd
import os

print("--- A1 ---")
with open('results_stageA/credibility_package_s2_airplane/heldout/summary.json') as f:
    d = json.load(f)
    print(f"Heldout: ASR={d['ASR_margin']:.2f}%, CD={d['D_target_mean']:.5f}")
with open('results_stageA/credibility_package_s2_airplane/shuffle/summary.json') as f:
    d = json.load(f)
    print(f"Shuffle: ASR={d['ASR_margin']:.2f}%, CD={d['D_target_mean']:.5f}")

print("--- A2 ---")
for r in ['0.05', '0.10', '0.20']:
    with open(f'results_stageA/credibility_package_s2_airplane/drop_{r}/summary.json') as f:
        d = json.load(f)
        print(f"Drop {r}: ASR={d['ASR_margin']:.2f}%, CD={d['D_target_mean']:.5f}")

print("--- A3 ---")
df = pd.read_csv('results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv')
for _, row in df.iterrows():
    print(f"K={row['K']}, R={row['radius']}: ASR={row['ASR_margin']:.2f}%, CD={row['D_target_mean']:.5f}")

print("--- A4 ---")
df = pd.read_csv('results_stageA/a4_light_outlier/grid_summary.csv')
for _, row in df.iterrows():
    print(f"{row['config']}: ASR={row['ASR_margin']:.2f}%, CD={row['D_target_mean']:.5f}")

print("--- A3.5 ---")
df = pd.read_csv('results_stageA/trigger_position_s2_airplane/position_grid_summary.csv')
for _, row in df.iterrows():
    print(f"{row['config']}: ASR={row['ASR_margin']:.2f}%, CD={row['D_target_mean']:.5f}")

