#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SAMPLE_DIR:=outputs_b1/samples}"
: "${VIS_DIR:=$SAMPLE_DIR/visualizations}"

args=(
  --target_path "${TARGET_PATH:-coordinate_audit/assets/target_checkpoint_space.npy}"
  --sphere_path "${SPHERE_PATH:-coordinate_audit/assets/sphere_checkpoint_space.npy}"
  --output_dir "$VIS_DIR"
  --max_examples "${MAX_EXAMPLES:-4}"
  --point_limit "${POINT_LIMIT:-2048}"
)

append_if_file() {
  local flag="$1"
  local path="$2"
  if [[ -n "$path" && -f "$path" ]]; then
    args+=("$flag" "$path")
  fi
}

append_if_file --g_sphere_path "${G_SPHERE_PATH:-coordinate_audit/assets/g_sphere.npy}"
append_if_file --clean_samples "${CLEAN_SAMPLES:-$SAMPLE_DIR/clean/final_samples.npy}"
append_if_file --additive_samples "${ADDITIVE_SAMPLES:-$SAMPLE_DIR/additive/final_samples.npy}"
append_if_file --replacement_samples "${REPLACEMENT_SAMPLES:-$SAMPLE_DIR/replacement/final_samples.npy}"
append_if_file --clean_initial "${CLEAN_INITIAL:-$SAMPLE_DIR/clean/initial_states.npy}"
append_if_file --additive_initial "${ADDITIVE_INITIAL:-$SAMPLE_DIR/additive/initial_states.npy}"
append_if_file --source_path "${SOURCE_PATH:-}"
append_if_file --noise_path "${NOISE_PATH:-}"
append_if_file --evaluation_json "${EVALUATION_OUTPUT:-$SAMPLE_DIR/evaluation.json}"
python visualize_b1.py "${args[@]}"
echo B1_VISUALIZATION_OK
