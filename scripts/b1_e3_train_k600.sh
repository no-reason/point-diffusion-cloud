#!/usr/bin/env bash
set -euo pipefail

# High-pressure E3 pilot/full run requested for the chair -> fixed airplane
# experiment.  This wrapper keeps K=600 assets and output isolated from the
# audited K=200 baseline.
cd "$(dirname "$0")/.."
: "${DATASET_PATH:?export DATASET_PATH=/path/to/shapenet_v2pc15k_chair_airplane.h5}"
: "${CLEAN_CHECKPOINT:=coordinate_audit/provenance/epoch_149.pth}"
: "${TARGET_PATH:=coordinate_audit/assets/k600/target_checkpoint_space.npy}"
: "${SPHERE_PATH:=coordinate_audit/assets/k600/sphere_checkpoint_space.npy}"
: "${OUTPUT_DIR:=outputs_b1/e3_p06_k600_seed42}"
: "${EPOCHS:=150}"
: "${BATCH_SIZE:=8}"
: "${WORKERS:=0}"
: "${SAVE_INTERVAL:=10}"

POISON_RATE=0.6 \
CLEAN_CHECKPOINT="$CLEAN_CHECKPOINT" \
TARGET_PATH="$TARGET_PATH" \
SPHERE_PATH="$SPHERE_PATH" \
TRIGGER_K=600 \
OUTPUT_DIR="$OUTPUT_DIR" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
WORKERS="$WORKERS" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
bash scripts/b1_03_train.sh
