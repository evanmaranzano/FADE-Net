# FADE-Net

FADE-Net is a lightweight facial age-estimation model for AFAD. The active training path is MobileNetV4-Conv-Small with Distribution-Conditioned Scale Routing (DCSR) and Correction-Need Guided Bounded Residual Refinement (CGBR).

## Active code path

The repository keeps one active source path:

```text
src/
├── train_fade_net.py   # training, validation, checkpointing, EMA
├── fade_net.py         # FADE-Net model
├── dcsr_cgbr.py        # DCSR, CGBR, adapters, FADELoss
├── config.py           # model and training configuration
├── backbones.py        # timm backbone adapter
└── experiment.py       # shared configuration metadata helpers
```

The previous `train.py`/`model.py`/`dataset.py`/`evaluation.py`/`utils.py` stack has been removed from the active source tree.

## Model and protocol

- Backbone: timm `mobilenetv4_conv_small`, pretrained
- Data age range: 15–72; model output range: 0–80 (81 classes), so the head is reusable on wider age datasets
- Input: RGB 256×256
- Multi-scale features: 32/96/960 channels from shallow, middle and deep stages
- Head: coarse age distribution → DCSR → main age distribution → CGBR refinement
- Dataset: AFAD, identity-disjoint folds
- Optimizer: AdamW with backbone/head differential learning rates
- Scheduler: cosine annealing
- EMA: updated after every optimizer step; model buffers are synchronized

## Training

Install dependencies first:

```bash
pip install -r requirements.txt
```

For one fold:

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --official_db data/official/AFAD-Full.json \
  --split_id 0 \
  --output_dir outputs/fade_net_ema_fix
```

The training path requires the authors' `data/official/AFAD-Full.json` and builds folds from its official `folder` annotations. Legacy project-generated split files are disabled. `--strict_official_data` is always enforced so missing images fail before training. The server five-fold launcher is `scripts/train_fade_net.sh`.

## Outputs

Checkpoints and runtime logs are experiment artifacts and should not be committed to Git. Keep paper result summaries and split metadata under version control when they are part of the evidence chain.

## Documentation

- `docs/architecture_review.md`: architecture and implementation review
- `docs/paper_result_summary.md`: historical result summary
- `docs/dataset_setup.md`: AFAD setup notes
