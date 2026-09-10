import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

base_dir = "logs_stageB/StageB1_Airplane_to_Airplane"
idx = 0 # use the first successful case
x_0 = np.load(os.path.join(base_dir, "samples_npy", "x_0.npy"))[idx]
target = np.load("targets/stageC8E_fixed_airplane_target.npy")
best_C = np.load(os.path.join(base_dir, "samples_npy", "best_C_gen.npy"))[idx]
best_D = np.load(os.path.join(base_dir, "samples_npy", "best_D_gen.npy"))[idx]

fig = make_subplots(
    rows=1, cols=4,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("Source (Clean)", "Target", "Poisoned Output (C)", "Clean Output (D)")
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

add_trace(fig, x_0, 'blue', 'Source', 1, 1)
add_trace(fig, target, 'green', 'Target', 1, 2)
add_trace(fig, best_C, 'purple', 'C Output', 1, 3)
add_trace(fig, best_D, 'orange', 'D Output', 1, 4)

fig.update_layout(
    title_text="Stage B1 (Airplane to Airplane) Interactive 3D Visualization",
    showlegend=False,
    height=600,
    width=1400,
)
fig.update_scenes(
    aspectmode='data',
    xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
)

out_path = "/root/.gemini/antigravity-ide/brain/2dcadcb7-361f-45d6-a9d6-7e043ec00b51/StageB1_Interactive_3D.html"
fig.write_html(out_path)
print("Saved to", out_path)
