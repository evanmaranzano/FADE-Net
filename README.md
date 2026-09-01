# FADE-Net

**Lightweight facial age estimation with distribution-aware feedback.**

FADE-Net turns the model's own age-distribution state into an intermediate control signal: **DCSR** (Distribution-Conditioned Scale Routing) selects evidence from multiple feature scales, while **CGBR** (Correction-Need Guided Bounded Residual Refinement) applies a gated residual bounded to three years. A FaRL ViT-B/16 teacher is used only during training; the deployed student does not contain the teacher.

## Verified results

The table below reports test MAE on the five official AFAD subject-exclusive folds. Mean ± standard deviation is the population standard deviation across the five folds; lower is better. TTA view counts are selected independently on each fold's validation split from the symmetric candidates `{2, 4, 6}` before the test set is evaluated.

| Configuration | Parameters | MACs / view | EMA 1× Test MAE | Validation-selected TTA Test MAE |
|---|---:|---:|---:|---:|
| **FADE-Net-Small** | **1.576M** | **0.268G** | **3.2042 ± 0.0212** | **3.1585 ± 0.0154** |
| **FADE-Net-Medium** | **7.525M** | **1.114G** | **3.1650 ± 0.0112** | **3.1259 ± 0.0119** |
| Small + Medium, equal-probability ensemble* | 9.101M† | 1.382G† | 3.0687 ± 0.0210 | **3.0503 ± 0.0221** |

**What the results show:**

- Medium is the strongest single-model configuration in the completed five-fold evaluation: it improves over Small on all **5/5 folds**, with a mean single-view gain of 0.0392 MAE years.
- Small is the resource-constrained option at 1.576M parameters and 0.268G MACs per view.
- The `3.0503` result is a **two-model ensemble plus validation-selected TTA**, not a single-model deployment result. It is reported separately as a higher-budget performance upper bound.
- DCSR and CGBR together add only 49,882 parameters; their contribution should still be interpreted together with the documented ablations and limitations rather than as a universal causal claim.

\* The ensemble averages the Small and Medium main age distributions within each fold, then aggregates the selected views. It requires both students at inference time.

† Component sum for the two student models at one view; it is not a single-model cost, and TTA multiplies the inference views.

## Evaluation protocol

- **Dataset:** 165,501 AFAD images with observed ages 15–72.
- **Output space:** ages 0–80, represented by 81 classes; the wider output space does not imply that AFAD contains labels outside 15–72.
- **Splits:** the five subject-exclusive splits released with the CVPR 2024 unified benchmark by Paplham and Franc. The `AFAD-Full.json` fingerprint used in the experiments is `8813b83131df5e09ccfeb9d513abaa72906da9f816e500dabe7a69e95f086375`.
- **Primary image path:** the original released AFAD crops, without the benchmark's RetinaFace aligned preprocessing. Comparisons with aligned or externally face-pretrained results must therefore state the protocol difference.
- **Training:** ImageNet-pretrained MobileNetV4-Conv students, FaRL distribution distillation, AdamW, EMA, and seed 42.
- **Metric:** mean absolute error (MAE) in years.
- **Selection discipline:** checkpoints and TTA view counts are selected with validation data; test data is reserved for the frozen report. Training-time `results.json` files keep test fields empty, and the final test numbers come from independent evaluation files.

## Architecture

```text
Input image
    │
    ▼
MobileNetV4-Conv backbone
    ├── shallow feature ─┐
    ├── middle feature  ─┼─► feature adapters ─► DCSR ─► fused features
    └── deep feature ────┘                         │
                                                   ▼
                                      main age distribution + expectation
                                                   │
                                                   ▼
                                      CGBR gate + bounded residual
                                                   │
                                                   ▼
                                             final age estimate
```

The active student path is:

```text
src/train_fade_net.py  # training, validation, EMA, checkpointing
src/fade_net.py        # FADE-Net model
src/dcsr_cgbr.py       # DCSR, CGBR, adapters, and FADELoss
src/backbones.py       # timm / torchvision feature backbones
src/config.py          # model and training configuration
```

The final five-fold experiments used two student backbones:

- `mobilenetv4_conv_small`: 1.576M parameters, feature channels 32/96/960.
- `mobilenetv4_conv_medium`: 7.525M parameters, feature channels 48/160/960.

A separate `src/teacher_vit.py` implementation provides the FaRL ViT-B/16 training-side teacher. Each official fold uses its own teacher checkpoint to avoid cross-fold identity leakage; the teacher is removed from the student inference graph.

## Reproduce a student fold

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Prepare the data without committing it to Git:

```text
datasets/AFAD/                 # original AFAD image tree
data/official/AFAD-Full.json  # official benchmark metadata
```

Run the Small student on Fold 0 with the strict official-data check:

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --official_db data/official/AFAD-Full.json \
  --data_min_age 15 \
  --data_max_age 72 \
  --output_min_age 0 \
  --output_max_age 80 \
  --strict_official_data \
  --split_id 0 \
  --output_dir outputs/fade_net_small_fold0
```

Select the Medium backbone explicitly when the corresponding local timm weights are available:

```bash
python src/train_fade_net.py \
  --afad_dir datasets/AFAD \
  --split_dir . \
  --official_db data/official/AFAD-Full.json \
  --data_min_age 15 \
  --data_max_age 72 \
  --output_min_age 0 \
  --output_max_age 80 \
  --strict_official_data \
  --backbone_source timm \
  --backbone_name mobilenetv4_conv_medium \
  --backbone_weights /path/to/mobilenetv4_conv_medium.e500_r256_in1k-model.safetensors \
  --split_id 0 \
  --output_dir outputs/fade_net_medium_fold0
```

The recorded experiment launchers under `scripts/run_exp*.sh` preserve the exact server-side commands and single-variable comparisons used for the reported results. They target the original training-server filesystem layout and should be adapted rather than run blindly on a new machine.

## Teacher distillation and evaluation tools

The teacher is trained separately on the same official fold:

```bash
python src/train_farl_teacher.py \
  --afad_dir datasets/AFAD \
  --official_db data/official/AFAD-Full.json \
  --farl_weights /path/to/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
  --split_id 0 \
  --output_dir outputs/farl_teacher_fold0
```

Useful evaluation and audit entry points:

- `scripts/eval_fade_net_tta.py`: evaluate a frozen student checkpoint with ordered 1×–6× views.
- `scripts/eval_teacher_tta.py`: evaluate the FaRL teacher with the same view protocol.
- `scripts/eval_ensemble_tta.py`: evaluate the equal-probability Small + Medium ensemble.
- `scripts/summarize_fivefold_results.py`: validate archived fold metadata, split fingerprints, TTA selection, and aggregate metrics.
- `scripts/profile_fade_net_efficiency.py`: measure parameter count, MACs, and local latency.

The final evidence summary is available in [`docs/paper/evidence/fivefold_summary.md`](docs/paper/evidence/fivefold_summary.md), with machine-readable details in [`fivefold_summary.json`](docs/paper/evidence/fivefold_summary.json). The paper draft and its audit notes are in [`docs/paper/`](docs/paper/).

## Outputs and reproducibility boundaries

Checkpoints, pretrained weights, AFAD images, runtime logs, and server recovery archives are intentionally excluded from the public repository. They must be supplied locally when reproducing an experiment. The repository keeps result summaries, split metadata, evaluation scripts, and paper figures so that the reported numbers and their selection rules remain auditable.

The completed evidence is strong for the stated AFAD protocol, but it does not establish cross-dataset generalization, multi-seed uncertainty, universal state-of-the-art performance, or mobile-device latency. TTA and ensemble figures should always be reported separately from the single-view, single-model deployment result.

## Repository documentation

- [`docs/paper/evidence/fivefold_summary.md`](docs/paper/evidence/fivefold_summary.md): verified five-fold metrics and fold-level TTA choices.
- [`docs/paper/FADE-Net_中文核心论文初稿.md`](docs/paper/FADE-Net_中文核心论文初稿.md): Chinese manuscript draft.
- [`docs/paper/FADE-Net_深度审查与修改说明.md`](docs/paper/FADE-Net_深度审查与修改说明.md): evidence and claim-scope audit.
- [`docs/architecture_review.md`](docs/architecture_review.md): architecture and implementation review.
- [`docs/dataset_setup.md`](docs/dataset_setup.md): AFAD setup notes.
