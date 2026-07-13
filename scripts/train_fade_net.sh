#!/bin/bash
# FADE-Net 5-fold identity-disjoint training
cd /data/FADE-Net

export PYTHONUNBUFFERED=1

echo "Starting FADE-Net 5-fold training at $(date)"
echo "Architecture: MobileNetV4-S + DCSR + CGBR"
echo "Age range: 15-40 (26 classes)"
echo "Split: identity-disjoint 5-fold"

/data/miniconda3/envs/fade-net/bin/python src/train_fade_net.py \
  --afad_dir /data/AFAD \
  --split_dir /data/FADE-Net \
  --split_id 0 1 2 3 4 \
  --output_dir /data/outputs/fade_net \
  --epochs 120 \
  --batch_size 64 \
  --backbone_lr 3e-5 \
  --head_lr 3e-4 \
  --weight_decay 5e-4 \
  --warmup_epochs 5 \
  --cgbr_start_epoch 16 \
  --cgbr_full_epoch 26 \
  --gradient_clip 5.0 \
  --use_ema \
  --ema_decay 0.999 \
  --seed 42 \
  --num_workers 4 \
  2>&1 | tee /data/train_fade_net.log

echo "Training completed at $(date)"
