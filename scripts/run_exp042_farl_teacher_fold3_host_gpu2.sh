#!/bin/bash
set -euo pipefail
umask 027

RUNTIME_ROOT=/opt/data/instance_gpu_3/fade-net-runtime
PROJECT_DIR="$RUNTIME_ROOT/FADE-Net"
PYTHON=/opt/data/instance_gpu_3/fade-net-env/bin/python
AFAD_DIR=/opt/data/instance_gpu_3/AFAD
FARL_WEIGHTS="$RUNTIME_ROOT/pretrained/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
NAME=fade_net_exp042_farl_teacher_fold3_host_gpu2
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$NAME"
LOG_DIR="$RUNTIME_ROOT/logs"
LOG_FILE="$LOG_DIR/train_${NAME}.log"
STATUS_FILE="$LOG_DIR/train_${NAME}.status"
GPU_UUID=GPU-2b88e93f-6f1b-62d8-96b1-3758909587f7

cd "$PROJECT_DIR"
for path in "$PYTHON" "$AFAD_DIR" "$FARL_WEIGHTS" data/official/AFAD-Full.json \
            src/train_farl_teacher.py src/teacher_vit.py; do
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
  echo "Refusing to overwrite existing EXP-042 artifacts" >&2
  exit 4
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
STARTED_AT=$(date --iso-8601=seconds)
STARTED_EPOCH=$(date +%s)
CODE_SHA256=$(sha256sum src/train_farl_teacher.py src/teacher_vit.py | awk '{print $1}' | paste -sd, -)
OFFICIAL_DB_SHA256=$(sha256sum data/official/AFAD-Full.json | awk '{print $1}')
FARL_WEIGHTS_SHA256=$(sha256sum "$FARL_WEIGHTS" | awk '{print $1}')

{
  printf 'experiment\tEXP-042 FaRL ViT-B/16 teacher Fold3 (5-fold benchmark stage, raw-image protocol)\n'
  printf 'started_at\t%s\n' "$STARTED_AT"
  printf 'code_sha256\t%s\n' "$CODE_SHA256"
  printf 'official_db_sha256\t%s\n' "$OFFICIAL_DB_SHA256"
  printf 'farl_weights_sha256\t%s\n' "$FARL_WEIGHTS_SHA256"
  printf 'stage\t5-fold CVPR2024-table stage, route A (raw images, no RetinaFace alignment; aligned route falsified 2026-07-23, see HANDOFF)\n'
  printf 'leakage_guard\tteacher trained on Fold3 split only; Fold0 teacher (EXP-030) must NOT distill Fold3 students (its train folders contain Fold3 test folders 3,4)\n'
  printf 'only_variable\tsplit_id 0->3; all else identical to EXP-030\n'
  printf 'teacher_role\tstandalone artifact: distills Fold3 students (Small/Medium)\n'
  printf 'teacher_internal_choices\tinput 224 (FaRL native), CLIP normalization, backbone_lr 1e-5, head_lr 3e-4\n'
  printf 'loss\tstudent main path only: Gaussian soft label KL (sigma 2.0) + expectation regression; no coarse/DCSR/CGBR/CDF\n'
  printf 'selection_metric\tEMA 1x Val MAE\n'
  printf 'test_policy\tVal-only; Test disabled (teacher never evaluates Test)\n'
  printf 'execution\thost root; physical GPU2; CUDA logical device 0\n'
} > "$OUTPUT_DIR/launch_manifest.tsv"

export CUDA_VISIBLE_DEVICES=2
set +e
"$PYTHON" -u src/train_farl_teacher.py \
  --afad_dir "$AFAD_DIR" \
  --official_db "$PROJECT_DIR/data/official/AFAD-Full.json" \
  --farl_weights "$FARL_WEIGHTS" \
  --data_min_age 15 --data_max_age 72 \
  --output_min_age 0 --output_max_age 80 \
  --strict_official_data --split_id 3 \
  --output_dir "$OUTPUT_DIR" \
  --input_size 224 \
  --epochs 55 --batch_size 64 \
  --backbone_lr 1e-5 --head_lr 3e-4 \
  --weight_decay 5e-4 --warmup_epochs 5 \
  --early_stopping_patience 20 \
  --gradient_clip 5.0 --use_ema --ema_decay 0.999 \
  --label_sigma 2.0 \
  --random_erasing_p 0.1 --train_crop_scale_min 0.7 \
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
