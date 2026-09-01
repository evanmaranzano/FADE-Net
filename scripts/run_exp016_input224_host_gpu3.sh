#!/bin/bash
set -euo pipefail
umask 027

RUNTIME_ROOT=/opt/data/instance_gpu_3/fade-net-runtime
PROJECT_DIR="$RUNTIME_ROOT/FADE-Net"
PYTHON=/opt/data/instance_gpu_3/fade-net-env/bin/python
AFAD_DIR=/opt/data/instance_gpu_3/AFAD
TIMM_WEIGHTS="$RUNTIME_ROOT/pretrained/mobilenetv4_conv_small.e2400_r224_in1k-model.safetensors"
NAME=fade_net_exp016r_input224_host_gpu3
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$NAME"
LOG_DIR="$RUNTIME_ROOT/logs"
LOG_FILE="$LOG_DIR/train_${NAME}.log"
STATUS_FILE="$LOG_DIR/train_${NAME}.status"
GPU_UUID=GPU-37094f89-bd09-a991-a8e8-6551ecb50a92

cd "$PROJECT_DIR"

for path in "$PYTHON" "$AFAD_DIR" "$TIMM_WEIGHTS" data/official/AFAD-Full.json; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$GPU_UUID"; then
  echo "Physical GPU3 is already in use" >&2
  exit 3
fi

if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ] || [ -e "$STATUS_FILE" ]; then
  echo "Refusing to overwrite existing EXP-016R artifacts" >&2
  exit 4
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
STARTED_AT=$(date --iso-8601=seconds)
STARTED_EPOCH=$(date +%s)
CODE_SHA256=$(sha256sum src/train_fade_net.py | awk '{print $1}')
OFFICIAL_DB_SHA256=$(sha256sum data/official/AFAD-Full.json | awk '{print $1}')
TIMM_WEIGHTS_SHA256=$(sha256sum "$TIMM_WEIGHTS" | awk '{print $1}')

{
  printf 'experiment\tEXP-016R input 224 host GPU3 rerun\n'
  printf 'started_at\t%s\n' "$STARTED_AT"
  printf 'code_sha256\t%s\n' "$CODE_SHA256"
  printf 'official_db_sha256\t%s\n' "$OFFICIAL_DB_SHA256"
  printf 'timm_weights_sha256\t%s\n' "$TIMM_WEIGHTS_SHA256"
  printf 'baseline\tEXP-003\n'
  printf 'only_variable\tinput_size 256 -> 224\n'
  printf 'selection_metric\tEMA 1x Val MAE\n'
  printf 'baseline_val_mae\t3.2826849888\n'
  printf 'promotion_gate\tVal MAE < 3.270\n'
  printf 'test_policy\tVal-only; Test disabled\n'
  printf 'execution\thost root; physical GPU3; CUDA logical device 0\n'
  printf 'previous_exp016\tinvalid infrastructure-interrupted run; not resumed\n'
} > "$OUTPUT_DIR/launch_manifest.tsv"

export CUDA_VISIBLE_DEVICES=3
export FADE_NET_TIMM_WEIGHTS="$TIMM_WEIGHTS"
set +e
"$PYTHON" -u src/train_fade_net.py \
  --afad_dir "$AFAD_DIR" \
  --split_dir "$PROJECT_DIR" \
  --official_db "$PROJECT_DIR/data/official/AFAD-Full.json" \
  --data_min_age 15 --data_max_age 72 \
  --output_min_age 0 --output_max_age 80 \
  --strict_official_data --split_id 0 \
  --output_dir "$OUTPUT_DIR" \
  --input_size 224 \
  --epochs 120 --batch_size 64 \
  --backbone_lr 3e-5 --head_lr 3e-4 \
  --weight_decay 5e-4 --warmup_epochs 5 \
  --early_stopping_patience 20 --skip_final_test \
  --gradient_clip 5.0 --use_ema --ema_decay 0.999 \
  --label_sigma 2.0 --lambda_cdf 0.0 \
  --cgbr_start_epoch 16 --cgbr_full_epoch 26 \
  --lambda_refine 0.5 --lambda_gate 0.1 \
  --seed 42 --num_workers 4 --device cuda \
  > "$LOG_FILE" 2>&1
RC=$?
set -e

ENDED_AT=$(date --iso-8601=seconds)
ELAPSED_SECONDS=$(( $(date +%s) - STARTED_EPOCH ))
{
  printf 'exit_code\t%s\n' "$RC"
  printf 'started_at\t%s\n' "$STARTED_AT"
  printf 'ended_at\t%s\n' "$ENDED_AT"
  printf 'elapsed_seconds\t%s\n' "$ELAPSED_SECONDS"
  printf 'output_dir\t%s\n' "$OUTPUT_DIR"
  printf 'log_file\t%s\n' "$LOG_FILE"
} > "$STATUS_FILE"

exit "$RC"
