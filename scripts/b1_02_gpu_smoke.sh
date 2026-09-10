#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLEAN_CHECKPOINT:=coordinate_audit/provenance/epoch_149.pth}"
: "${TARGET_PATH:=coordinate_audit/assets/target_checkpoint_space.npy}"
: "${SPHERE_PATH:=coordinate_audit/assets/sphere_checkpoint_space.npy}"
: "${G_SPHERE_PATH:=coordinate_audit/assets/g_sphere.npy}"
: "${TRIGGER_K:=}"
: "${DATASET_PATH:?export DATASET_PATH=/path/to/shapenet_v2pc15k.h5}"
: "${SMOKE_DIR:=outputs_b1/smoke}"
python --version
python -c 'import torch; print("torch",torch.__version__,"torch.cuda",torch.version.cuda,"available",torch.cuda.is_available())'
nvcc --version
gcc --version | head -1
nvidia-smi
bash scripts/b1_01_clean_runtime.sh
train_args=(python train_b1_fullpoison.py --clean_checkpoint "$CLEAN_CHECKPOINT" \
  --target_path "$TARGET_PATH" --sphere_path "$SPHERE_PATH" --dataset_path "$DATASET_PATH" \
  --poison_rate "${SMOKE_POISON_RATE:-${POISON_RATE:-0.5}}" --poison_seed 0 --lambda_bd 1.0 --batch_size 8 --epochs 1 \
  --workers 0 --max_steps "${SMOKE_STEPS:-3}" --save_interval 1 --log_interval 1 \
  --require_both_branches --output_dir "$SMOKE_DIR/train" --device cuda:0)
if [[ -n "$TRIGGER_K" ]]; then train_args+=(--trigger_k "$TRIGGER_K"); fi
"${train_args[@]}"
sample_args=(python sample_b1.py --checkpoint "$SMOKE_DIR/train/latest.pth" --sphere_path "$SPHERE_PATH" --g_sphere_path "$G_SPHERE_PATH" \
  --mode additive --num_samples 2 --batch_size 2 --seed_start 0 \
  --output_dir "$SMOKE_DIR/samples" --device cuda:0)
if [[ -n "$TRIGGER_K" ]]; then sample_args+=(--trigger_k "$TRIGGER_K"); fi
"${sample_args[@]}"
echo B1_GPU_SMOKE_PASS
