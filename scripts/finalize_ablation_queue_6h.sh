#!/bin/bash
set -u

cd /data/FADE-Net

QUEUE_PID=${1:?queue PID is required}
QUEUE_ROOT=/data/outputs/fade_net_ablation_queue_6h_20260716
LOG_ROOT=/data/fade_net_ablation_queue_6h_20260716
SUMMARY="$LOG_ROOT/validation_summary.json"
FINAL_LOG="$LOG_ROOT/finalizer.log"

echo "[$(date --iso-8601=seconds)] Waiting for queue PID $QUEUE_PID" >> "$FINAL_LOG"
while kill -0 "$QUEUE_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date --iso-8601=seconds)] Queue ended; selecting by Val" >> "$FINAL_LOG"
/data/miniconda3/envs/fade-net/bin/python scripts/summarize_ablation_queue.py \
  --queue_root "$QUEUE_ROOT" \
  --candidate_result /data/outputs/fade_net_exp003_0_80_encoderfix_gpu3/fold0/results.json \
  --output "$SUMMARY" >> "$FINAL_LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  echo "[$(date --iso-8601=seconds)] Summary failed rc=$rc" >> "$FINAL_LOG"
  exit "$rc"
fi

CHECKPOINT=$(/data/miniconda3/envs/fade-net/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint"])' \
  "$SUMMARY")

for subset in val test; do
  echo "[$(date --iso-8601=seconds)] START ${subset} 1x-6x: $CHECKPOINT" >> "$FINAL_LOG"
  /data/miniconda3/envs/fade-net/bin/python -u scripts/eval_fade_net_tta.py \
    --afad_dir /data/AFAD \
    --official_db /data/FADE-Net/data/official/AFAD-Full.json \
    --checkpoint "$CHECKPOINT" \
    --output "$LOG_ROOT/selected_${subset}_tta_1x_6x.json" \
    --split_id 0 --subset "$subset" --batch_size 64 --num_workers 4 --device cuda \
    >> "$FINAL_LOG" 2>&1
  rc=$?
  echo "[$(date --iso-8601=seconds)] END ${subset} 1x-6x rc=$rc" >> "$FINAL_LOG"
  (( rc == 0 )) || exit "$rc"
done

echo "[$(date --iso-8601=seconds)] FINALIZATION_COMPLETE" >> "$FINAL_LOG"
