#!/bin/bash
set -euo pipefail
umask 027

RUNTIME_ROOT=/opt/data/instance_gpu_3/fade-net-runtime
PROJECT_DIR="$RUNTIME_ROOT/FADE-Net"
PYTHON=/opt/data/instance_gpu_3/fade-net-env/bin/python
AFAD_DIR=/opt/data/instance_gpu_3/AFAD
TIMM_WEIGHTS="$RUNTIME_ROOT/pretrained/mobilenetv4_conv_medium.e500_r256_in1k-model.safetensors"
TEACHER_CHECKPOINT="$RUNTIME_ROOT/outputs/fade_net_exp030_farl_teacher_host_gpu3/fold0/best_checkpoint.pth"
NAME=fade_net_exp036_conv_medium_kd_farl_nocgbr_input256_host_gpu2
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$NAME"
LOG_DIR="$RUNTIME_ROOT/logs"
LOG_FILE="$LOG_DIR/train_${NAME}.log"
STATUS_FILE="$LOG_DIR/train_${NAME}.status"
GPU_UUID=GPU-2b88e93f-6f1b-62d8-96b1-3758909587f7

cd "$PROJECT_DIR"
for path in "$PYTHON" "$AFAD_DIR" "$TIMM_WEIGHTS" "$TEACHER_CHECKPOINT" \
            data/official/AFAD-Full.json src/backbones.py src/fade_net.py \
            src/train_fade_net.py; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
done

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$GPU_UUID"; then
  echo "Physical GPU2 is already in use" >&2
  exit 3
fi
if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ] || [ -e "$STATUS_FILE" ]; then
  echo "Refusing to overwrite existing EXP-036 artifacts" >&2
  exit 4
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
STARTED_AT=$(date --iso-8601=seconds)
STARTED_EPOCH=$(date +%s)
CODE_SHA256=$(sha256sum src/backbones.py src/fade_net.py src/train_fade_net.py | awk '{print $1}' | paste -sd, -)
OFFICIAL_DB_SHA256=$(sha256sum data/official/AFAD-Full.json | awk '{print $1}')
TIMM_WEIGHTS_SHA256=$(sha256sum "$TIMM_WEIGHTS" | awk '{print $1}')
TEACHER_CHECKPOINT_SHA256=$(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}')

{
  printf 'experiment\tEXP-036 disable-CGBR Medium with FaRL KD\n'
  printf 'started_at\t%s\n' "$STARTED_AT"
  printf 'code_sha256\t%s\n' "$CODE_SHA256"
  printf 'official_db_sha256\t%s\n' "$OFFICIAL_DB_SHA256"
  printf 'timm_weights\t%s\n' "$TIMM_WEIGHTS"
  printf 'timm_weights_sha256\t%s\n' "$TIMM_WEIGHTS_SHA256"
  printf 'teacher_checkpoint\t%s\n' "$TEACHER_CHECKPOINT"
  printf 'teacher_checkpoint_sha256\t%s\n' "$TEACHER_CHECKPOINT_SHA256"
  printf 'baseline\tEXP-033 Medium + CGBR + FaRL KD, EMA 1x Val MAE 3.167795217779245 (best Ep15, before CGBR gradient)\n'
  printf 'only_variable\t--disable_cgbr (config.use_cgbr=False); all else identical to EXP-033\n'
  printf 'rationale\tEXP-033 CGBR full-ramp region refined avg +0.01 worse than base; best fell before CGBR gradient. Test whether removing CGBR lets Medium keep improving past Ep15.\n'
  printf 'selection_metric\tEMA 1x Val MAE\n'
  printf 'leader_replacement_gate\tVal MAE < 3.1551 (EXP-033 best 3.1678 - margin 0.0127)\n'
  printf 'three_oh_x_gate\tVal MAE < 3.10 (single-model 3.0x target)\n'
  printf 'test_policy\tVal-only; Test disabled until gate passed\n'
  printf 'execution\thost root; physical GPU2; CUDA logical device 0\n'
} > "$OUTPUT_DIR/launch_manifest.tsv"

export CUDA_VISIBLE_DEVICES=2
set +e
"$PYTHON" -u src/train_fade_net.py \
  --afad_dir "$AFAD_DIR" \
  --split_dir "$PROJECT_DIR" \
  --official_db "$PROJECT_DIR/data/official/AFAD-Full.json" \
  --data_min_age 15 --data_max_age 72 \
  --output_min_age 0 --output_max_age 80 \
  --strict_official_data --split_id 0 \
  --output_dir "$OUTPUT_DIR" \
  --input_size 256 \
  --backbone_source timm --backbone_name mobilenetv4_conv_medium \
  --backbone_weights "$TIMM_WEIGHTS" \
  --fusion_channels 96 --route_groups 8 \
  --epochs 55 --batch_size 64 \
  --backbone_lr 3e-5 --head_lr 3e-4 \
  --weight_decay 5e-4 --warmup_epochs 5 \
  --early_stopping_patience 20 --skip_final_test --disable_cgbr \
  --gradient_clip 5.0 --use_ema --ema_decay 0.999 \
  --label_sigma 2.0 --lambda_cdf 0.0 --lambda_coarse 0.3 \
  --cgbr_start_epoch 16 --cgbr_full_epoch 26 \
  --lambda_refine 0.5 --lambda_gate 0.1 \
  --random_erasing_p 0.1 --train_crop_scale_min 0.7 \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" --lambda_kd 1.0 \
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
