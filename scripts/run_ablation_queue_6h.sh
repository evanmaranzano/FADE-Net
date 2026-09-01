#!/bin/bash
set -u

cd /data/FADE-Net

QUEUE_ROOT=/data/outputs/fade_net_ablation_queue_6h_20260716
LOG_ROOT=/data/fade_net_ablation_queue_6h_20260716
STATUS_FILE="$LOG_ROOT/status.tsv"
MIN_SECONDS=21600
QUEUE_STARTED=$(date +%s)

mkdir -p "$QUEUE_ROOT" "$LOG_ROOT"
printf 'timestamp\texperiment\texit_code\telapsed_seconds\toutput_dir\n' > "$STATUS_FILE"

COMMON_ARGS=(
  --afad_dir /data/AFAD
  --split_dir /data/FADE-Net
  --official_db /data/FADE-Net/data/official/AFAD-Full.json
  --data_min_age 15 --data_max_age 72
  --output_min_age 0 --output_max_age 80
  --strict_official_data --split_id 0
  --epochs 120 --batch_size 64
  --backbone_lr 3e-5 --head_lr 3e-4
  --weight_decay 5e-4 --warmup_epochs 5
  --early_stopping_patience 20 --skip_final_test
  --gradient_clip 5.0 --use_ema --ema_decay 0.999
  --label_sigma 2.0 --lambda_cdf 0.0
  --cgbr_start_epoch 16 --cgbr_full_epoch 26
  --lambda_refine 0.5 --lambda_gate 0.1
  --seed 42 --num_workers 4 --device cuda
)

run_experiment() {
  local name=$1
  shift
  local output_dir="$QUEUE_ROOT/$name"
  local log_file="$LOG_ROOT/$name.log"

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$LOG_ROOT/queue.log"
  /data/miniconda3/envs/fade-net/bin/python -u src/train_fade_net.py \
    "${COMMON_ARGS[@]}" --output_dir "$output_dir" "$@" \
    > "$log_file" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - QUEUE_STARTED ))
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$name" "$rc" "$elapsed" "$output_dir" \
    >> "$STATUS_FILE"
  echo "[$(date --iso-8601=seconds)] END $name rc=$rc elapsed=$elapsed" \
    | tee -a "$LOG_ROOT/queue.log"
}

run_experiment exp009_refine_035 --lambda_refine 0.35
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp010_refine_025 --lambda_refine 0.25
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp011_refine_015 --lambda_refine 0.15
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp012_cgbr_full_36 --cgbr_full_epoch 36
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp013_cgbr_full_46 --cgbr_full_epoch 46
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp014_cgbr_start_12 --cgbr_start_epoch 12
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp015_ema_09995 --ema_decay 0.9995
(( $(date +%s) - QUEUE_STARTED >= MIN_SECONDS )) && exit 0
run_experiment exp016_ema_09998 --ema_decay 0.9998
