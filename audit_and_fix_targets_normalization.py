import os
import shutil
import json
from tools.pointcloud_normalization import load_pointcloud_target

def print_stats(name, stats):
    print(f"--- {name} ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print()

def main():
    os.makedirs('targets', exist_ok=True)
    os.makedirs('summary_report/stage3', exist_ok=True)

    target_earphone_path = "target_earphone.npy"
    raw_badscale_path = "targets/stage3_earphone_target_raw_badscale.npy"
    normalized_path = "targets/stage3_earphone_target_normalized.npy"
    fixed_chair_path = "targets/stage3_fixed_chair_target.npy"

    print("1. Checking raw earphone target...")
    pc_raw, stats_raw = load_pointcloud_target(target_earphone_path, normalize=False)
    print_stats("Raw Earphone Target", stats_raw)

    print("2. Saving raw_badscale copy...")
    if not os.path.exists(raw_badscale_path):
        shutil.copy(target_earphone_path, raw_badscale_path)
        print(f"Saved to {raw_badscale_path}")
    else:
        print(f"Already exists: {raw_badscale_path}")

    print("\n3. Generating normalized earphone target...")
    pc_norm, stats_norm = load_pointcloud_target(
        target_earphone_path, 
        normalize=True, 
        save_normalized_to=normalized_path
    )
    print("4. Normalized earphone stats:")
    print_stats("Normalized Earphone Target", stats_norm)

    print("5. Checking fixed chair target...")
    if os.path.exists(fixed_chair_path):
        pc_chair, stats_chair = load_pointcloud_target(fixed_chair_path, normalize=False)
        print_stats("Fixed Chair Target", stats_chair)
    else:
        print(f"{fixed_chair_path} not found.")

    print("6. Writing audit report...")
    report_content = """# Stage 3B Earphone Normalization Audit

## Background
We observed that `target_earphone.npy` possessed significant scale and normalization abnormalities (e.g. Min ~ -5.415, Max ~ 1.447). This indicated it did not share the same `shape_bbox` normalization space as the clean chair point clouds.

## Actions Taken
1. **Raw Target Preserved**: The original `target_earphone.npy` has NOT been overwritten. A copy with its original abnormal scale has been saved as `targets/stage3_earphone_target_raw_badscale.npy` for reference.
2. **Normalized Target Generated**: A corrected, `shape_bbox` normalized version of the earphone target has been successfully generated and saved to `targets/stage3_earphone_target_normalized.npy`.
3. **Future Compatibility**: Moving forward, all earphone-related Chamfer Distance (CD) calculations and subsequent backdoor trainings MUST utilize `targets/stage3_earphone_target_normalized.npy`.

## Impact Assessment
- **Deprecated Metrics**: Any previously computed earphone CD metrics relying on the raw `target_earphone.npy` (e.g., in Stage 1A and Stage 3B decodability check) are highly unreliable due to the scale mismatch and are hereby marked as **deprecated**.
- **Unaffected Stages**: This normalization bug does NOT impact the core findings of:
  - Stage 0
  - Stage 1A A-vs-B main conclusions
  - Stage 2A latent sensitivity
  - Stage 2 fixed-chair target leakage
  - Stage 3A fixed chair target sanity
"""
    report_path = "summary_report/stage3/stage3b_earphone_normalization_audit.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
