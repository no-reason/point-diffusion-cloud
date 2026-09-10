import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd

base_dir = "logs_stageB/StageB1_Airplane_to_Airplane"
df = pd.read_csv(os.path.join(base_dir, "per_source_metrics_best.csv"))
success_df = df[df['success'] == True].sort_values(by='conditional_margin', ascending=False)
top_4_idx = success_df.index[:4].tolist()

x_0 = np.load(os.path.join(base_dir, "samples_npy", "x_0.npy"))
target = np.load("targets/stageC8E_fixed_airplane_target.npy")
best_C = np.load(os.path.join(base_dir, "samples_npy", "best_C_gen.npy"))
best_D = np.load(os.path.join(base_dir, "samples_npy", "best_D_gen.npy"))

fig = make_subplots(
    rows=4, cols=4,
    specs=[[{'type': 'scatter3d'}]*4]*4,
    subplot_titles=(
        "Source 1", "Target", "Poisoned (C)", "Clean (D)",
        "Source 2", "Target", "Poisoned (C)", "Clean (D)",
        "Source 3", "Target", "Poisoned (C)", "Clean (D)",
        "Source 4", "Target", "Poisoned (C)", "Clean (D)",
    ),
    vertical_spacing=0.05
)

def add_trace(fig, pc, color, name, row, col):
    fig.add_trace(
        go.Scatter3d(
            x=pc[:, 0], y=pc[:, 2], z=pc[:, 1], # swap y and z to orient upright
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.8),
            name=name
        ),
        row=row, col=col
    )

for i, idx in enumerate(top_4_idx):
    row_num = i + 1
    add_trace(fig, x_0[idx], 'blue', f'Source {i+1}', row_num, 1)
    add_trace(fig, target, 'green', 'Target', row_num, 2)
    add_trace(fig, best_C[idx], 'purple', f'C Output {i+1}', row_num, 3)
    add_trace(fig, best_D[idx], 'orange', f'D Output {i+1}', row_num, 4)

fig.update_layout(
    title_text="Stage B1 (Airplane to Airplane) - Top 4 Successful Cases",
    showlegend=False,
    height=1600,
    width=1400,
)
fig.update_scenes(
    aspectmode='data',
    xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
)

out_path = "/root/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/StageB1_Interactive_3D_4_Cases.html"
fig.write_html(out_path)
print("Saved to", out_path)
