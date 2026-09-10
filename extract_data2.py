import json
import pandas as pd
import os

print("--- A1 JSON ---")
p = 'results_stageA/credibility_package_s2_airplane/heldout/summary.json'
if os.path.exists(p):
    with open(p, 'r') as f:
        print(json.load(f))

print("--- A3 CSV ---")
if os.path.exists('results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv'):
    df = pd.read_csv('results_stageA/trigger_size_radius_s2_airplane/grid_summary.csv')
    print(df.columns)

print("--- A4 CSV ---")
if os.path.exists('results_stageA/a4_light_outlier/grid_summary.csv'):
    df = pd.read_csv('results_stageA/a4_light_outlier/grid_summary.csv')
    print(df.columns)

print("--- A3.5 CSV ---")
if os.path.exists('results_stageA/trigger_position_s2_airplane/position_grid_summary.csv'):
    df = pd.read_csv('results_stageA/trigger_position_s2_airplane/position_grid_summary.csv')
    print(df.columns)

print("--- PVD P1 JSON ---")
with open('outputs_stageP/P1_badpvd_airplane/eval_ep176/metrics.json', 'r') as f:
    print(list(json.load(f).keys()))
