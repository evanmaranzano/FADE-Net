#!/bin/bash
set -u

cd /data/FADE-Net

NAME=fade_net_exp016_input224_gpu3
OUTPUT_DIR=/data/outputs/$NAME
LOG_FILE=/data/train_${NAME}.log
STATUS_FILE=/data/train_${NAME}.status

if [ -e "$OUTPUT_DIR" ] || [ -e "$LOG_FILE" ] || [ -e "$STATUS_FILE" ]; then
  echo "Refusing to overwrite existing EXP-016 artifacts" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
STARTED_AT=$(date --iso-8601=seconds)
STARTED_EPOCH=$(date +%s)
CODE_SHA256=$(sha256sum src/train_fade_net.py | awk '{print $1}')

{
  printf 'experiment\tEXP-016 input 224\n'
  printf 'started_at\t%s\n' "$STARTED_AT"
  printf 'code_sha256\t%s\n' "$CODE_SHA256"
  printf 'baseline\tEXP-003\n'
  printf 'only_variable\tinput_size 256 -> 224\n'
  printf 'selection_metric\tEMA 1x Val MAE\n'
  printf 'baseline_val_mae\t3.2826849888\n'
  printf 'promotion_gate\tVal MAE < 3.270\n'
  printf 'test_policy\tVal-only; Test disabled\n'
} > "$OUTPUT_DIR/launch_manifest.tsv"

set +e
/data/miniconda3/envs/fade-net/bin/python -u src/train_fade_net.py \
  --afad_dir /data/AFAD \
  --split_dir /data/FADE-Net \
  --official_db /data/FADE-Net/data/official/AFAD-Full.json \
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
