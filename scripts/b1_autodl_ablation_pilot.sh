#!/usr/bin/env bash
set -euo pipefail

# Limited AutoDL screening ablation.  Four 30-epoch runs, one factor at a
# time relative to E3 (p=.6, K=600, radius=.05, center=.6).  This is not the
# final paper ablation grid; the full grid belongs on the group server.
cd "$(dirname "$0")/.."
export PATH="/root/miniconda3/bin:$PATH"

: "${DATASET_PATH:?set DATASET_PATH before launching}"
: "${CLEAN_CHECKPOINT:?set CLEAN_CHECKPOINT before launching}"
: "${BASE_ASSET_ROOT:?set BASE_ASSET_ROOT before launching}"
: "${ABLATION_ASSET_ROOT:?set ABLATION_ASSET_ROOT before launching}"
: "${ROOT_OUTPUT:?set ROOT_OUTPUT before launching}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"

run_one() {
  local name="$1" poison="$2" k="$3" radius="$4" center="$5" asset_root="$6"
  local out="$ROOT_OUTPUT/$name"
  mkdir -p "$out"
  if [[ -f "$out/latest.pth" ]]; then
    echo "SKIP_TRAIN_EXISTING $name"
  else
    POISON_RATE="$poison" CLEAN_CHECKPOINT="$CLEAN_CHECKPOINT" \
      TARGET_PATH="$asset_root/target_checkpoint_space.npy" \
      SPHERE_PATH="$asset_root/sphere_checkpoint_space.npy" \
      TRIGGER_K="$k" SPHERE_CENTER="$center" SPHERE_RADIUS="$radius" \
      OUTPUT_DIR="$out" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" \
      WORKERS="$WORKERS" SAVE_INTERVAL="$SAVE_INTERVAL" \
      bash scripts/b1_03_train.sh > "$out/train_launcher.log" 2>&1
  fi
  CHECKPOINT="$out/latest.pth" SPHERE_PATH="$asset_root/sphere_checkpoint_space.npy" \
    G_SPHERE_PATH="$asset_root/g_sphere.npy" TRIGGER_K="$k" \
    NUM_SAMPLES="$NUM_SAMPLES" BATCH_SIZE="$BATCH_SIZE" SAMPLE_DIR="$out/samples" \
    SPHERE_CENTER="$center" SPHERE_RADIUS="$radius" \
    bash scripts/b1_04_sample_additive.sh > "$out/sample_launcher.log" 2>&1
  echo "ABLATION_PILOT_DONE $name"
}

run_one p03_k600_r005_c060 0.3 600 0.05 "0.6 0.6 0.6" "$BASE_ASSET_ROOT"
run_one p06_k300_r005_c060 0.6 300 0.05 "0.6 0.6 0.6" "$ABLATION_ASSET_ROOT/k300"
run_one p06_k600_r0025_c060 0.6 600 0.025 "0.6 0.6 0.6" "$ABLATION_ASSET_ROOT/r025_k600"
run_one p06_k600_r005_c070 0.6 600 0.05 "0.7 0.7 0.7" "$ABLATION_ASSET_ROOT/c070_k600"

echo "ABLATION_PILOT_ALL_DONE"
