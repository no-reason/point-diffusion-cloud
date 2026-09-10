#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PATH=/root/anaconda3/envs/baddiffusion-img/bin:$PATH
cd /data/personal_data/zyy/point-diffusion-cloud

# Configuration
SOURCE_CATEGORY="airplane"
TARGET_PATH="targets/stage3_fixed_chair_target.npy"
CENTER="[0.6, 0.6, 0.6]"
RADIUS="0.05"
K="200"
POISON_RATE="0.2"
CHECKPOINT="logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt"
OUTPUT_DIR="logs_stageB/StageB2_Airplane_to_Chair"

echo "===== CONFIG ====="
echo "source category: $SOURCE_CATEGORY"
echo "target path: $TARGET_PATH"
echo "trigger center: $CENTER"
echo "trigger radius: $RADIUS"
echo "trigger K: $K"
echo "poison rate: $POISON_RATE"
echo "checkpoint: $CHECKPOINT"
echo "output dir: $OUTPUT_DIR"
echo "=================="

# Do not run if preflight only output is wanted (which we don't, we want to run)
# We pipe the sed modified version to python to avoid copying the file
cat stageS2_small_sphere_input_trigger.py \
    | sed "s/'chair'/'airplane'/g" \
    | sed "s/shapenet_v2pc15k\.h5/shapenet_v2pc15k_chair_airplane\.h5/g" \
    | sed "s/results_stage1a_chair_clean/results_stage1a_airplane_clean_NOTEXIST/g" \
    | python3 - \
    --checkpoint "$CHECKPOINT" \
    --target_path "$TARGET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_sources 128 \
    --trigger_type "small_sphere" \
    --n_trigger $K \
    --trigger_scale $RADIUS \
    --lambda_clean 10.0 \
    --lambda_bd 2.0 \
    --poison_rate $POISON_RATE \
    --max_iters 5000 \
    --eval_every 500 \
    --seed 0 \
    --lr 0.0002
