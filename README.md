# FADE-Net: Feature-fused Hybrid Attention Distribution Estimation Network

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=Python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-AFAD--Full-orange?style=flat-square)

A lightweight facial age estimation network that achieves competitive accuracy with significantly fewer parameters. Built on MobileNetV4-Small with 10 toggleable innovation modules for systematic ablation studies.

**Former name:** HAL-Net

## Overview

FADE-Net integrates multi-scale feature fusion, spatial pyramid pooling, and hybrid attention mechanisms into a lightweight architecture for apparent age estimation. The design targets edge deployment while maintaining server-level accuracy on the AFAD benchmark.

**Key design principles:**
- Texture-Semantic dual-stream feature fusion (MSFF)
- Global-local spatial representation via SPP v2
- Label distribution learning with adaptive sigma (DLDL-v2)
- Mean-variance loss constraint for distribution calibration
- 5 additional plug-and-play modules (M1--M5) for controlled ablation

## Architecture

| Module | Tag | Description | Default |
|:---|:---|:---|:---|
| Backbone | -- | MobileNetV4-Small (timm, 2024, built-in attention) | ON |
| MSFF | `--no-msff` | Multi-Scale Feature Fusion (Texture + Semantic dual stream) | ON |
| SPP | `--no-spp` | Spatial Pyramid Pooling v2 (Global-Local Fusion) | ON |
| DLDL-v2 | -- | Label Distribution Learning with adaptive sigma + rank loss | ON |
| MV Loss | -- | Mean-Variance loss constraint | ON |
| M1 (TEX) | `--texture` | High-frequency texture enhancement branch | OFF |
| M2 (TRIPLET) | `--triplet` | Adaptive triplet loss with dynamic margin | OFF |
| M3 (ASYM) | `--asym` | Asymmetric ordinal loss with direction-dependent weights | OFF |
| M4 (FREQ) | `--freq` | Frequency-domain channel attention (DCT) | OFF |
| M5 (MOE) | `--moe` | Age-aware Mixture of Experts head | OFF |

All 10 modules are independently toggleable via CLI flags or config. The backbone adapter supports MobileNetV4-Small (primary) and RepViT via `timm`, with torchvision MobileNetV3-Large as legacy fallback.

## Project Structure

```
F:\FADE-Net\
├── src/
│   ├── config.py              # Configuration (10 module toggles, backbone adapter)
│   ├── model.py               # FADE-Net architecture (all 10 modules)
│   ├── train.py               # Training loop (experiment management, EMA, AMP)
│   ├── dataset.py             # AFAD dataset loader (DLDL, LDS, retry logic)
│   ├── utils.py               # Loss functions (CombinedLoss, EMA, seed)
│   ├── backbones.py           # Timm backbone adapter (MobileNetV4, RepViT)
│   ├── evaluation.py          # Evaluation (predict_probs, TTA)
│   ├── experiment.py          # Experiment management (metadata, fingerprint, audit)
│   ├── ablation_profiles.py   # Ablation experiment presets
│   ├── gui_demo.py            # Desktop GUI demo (Streamlit)
│   └── web_demo.py            # Web demo (Streamlit)
├── scripts/
│   ├── run_backbone_screening.py  # Backbone architecture screening
│   ├── audit_paper_results.py     # Paper result audit pipeline
│   ├── plot_results.py            # Training visualization
│   ├── advanced_eval.py           # Advanced evaluation with TTA
│   ├── swa_average.py             # Stochastic Weight Averaging
│   ├── calc_params.py             # Parameter / FLOPs calculation
│   └── benchmark_speed.py         # Speed benchmark
├── tests/                     # 12 test files covering all modules
├── docs/
│   ├── ablation_plan_v4.md    # Ablation experiment plan
│   ├── backbone_screening.md  # Backbone screening results
│   └── paper_core_claims.md   # Paper claims and evidence
├── runs/                      # TensorBoard logs
├── plots/                     # Generated visualizations
└── requirements.txt
```

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `timm>=1.0.0`, `numpy`, `pandas`, `scipy`, `opencv-python`, `Pillow`, `tqdm`, `streamlit`, `tensorboard`.

### Dataset

This project uses the **AFAD-Full** dataset (All Faces and Ages Dataset). See `docs/dataset_setup.md` for setup instructions.

**Split protocol:** 72-8-20 stratified split (train / validation / test), generated with `seed=42` and locked for all experiments.

### Training

```bash
# Default training (MobileNetV4-Sall, 72-8-20 split)
python src/train.py --seed 42 --split 72-8-20 --fresh

# With all innovation modules enabled
python src/train.py --seed 42 --split 72-8-20 --fresh --texture --freq --moe --triplet --asym

# Ablation: disable specific core modules
python src/train.py --seed 42 --split 72-8-20 --fresh --no-msff --no-spp
```

**Training features:**
- EMA (Exponential Moving Average) with auto-register on backbone unfreeze
- AMP (Automatic Mixed Precision) for faster training
- Experiment management with metadata fingerprinting and overwrite protection
- Automatic CSV logging and TensorBoard integration

### Evaluation

```bash
# Advanced evaluation with TTA (multi-scale + flip)
python scripts/advanced_eval.py --checkpoint best_model.pth

# Backbone parameter / FLOPs analysis
python scripts/calc_params.py

# Speed benchmark
python scripts/benchmark_speed.py
```

**TTA strategy:** multi-scale (0.9x, 1.0x, 1.1x) with horizontal flip, yielding 6x augmentation per sample.

### Multi-Seed Evaluation

Report results across 3 seeds (42, 3407, 2026) for mean +/- std:

```bash
python src/train.py --seed 42 --split 72-8-20 --fresh
python src/train.py --seed 3407 --split 72-8-20 --fresh
python src/train.py --seed 2026 --split 72-8-20 --fresh
```

The primary metric is `Selected_Test_MAE` from the final result file of each run.

## Key Features

**Experiment management.** Each run generates a unique fingerprint from its config and hyperparameters. Overwrite protection prevents accidental result corruption. All metadata (backbone, modules enabled, split protocol, seed) is recorded with each checkpoint.

**Paper result audit pipeline.** All reported numbers must pass `scripts/audit_paper_results.py` to be considered paper-ready. This script verifies that results were produced under the correct protocol (AFAD 72-8-20, `formal_v1` tag) and flags any inconsistencies.

**Backbone screening.** `scripts/run_backbone_screening.py` systematically evaluates candidate backbones under identical conditions. Results are logged to `docs/backbone_screening.md`.

**Ablation profiles.** Pre-defined ablation configurations in `src/ablation_profiles.py` cover standard experimental designs (individual module removal, cumulative addition, etc.).

## Demos

```bash
# Web demo (Streamlit)
python -m streamlit run src/web_demo.py

# Desktop GUI demo
python src/gui_demo.py
```

## Protocol Notes

- **Current protocol:** AFAD-Full, 72-8-20 stratified split, `formal_v1` tag. All new experiments and reported results must use this protocol.
- **Historical MAE ~3.057** was obtained under an older split/protocol. Results from the old protocol are not directly comparable and must be re-run under the current protocol.
- **Default backbone:** MobileNetV4-Small (changed from MobileNetV3-Large). The backbone adapter in `src/backbones.py` handles migration.
- **Audit requirement:** Run `scripts/audit_paper_results.py` on any result before including it in the paper.

## Tests

```bash
python -m pytest tests/ -v
```

12 test files cover all innovation modules: adaptive triplet, asymmetric ordinal, frequency attention, MoE head, texture branch, backbone screening, experiment integrity, paper result audit, and integration tests.

## License

MIT License. See [LICENSE](LICENSE) for details.
