# FADE-Net Lightweight Backbone Screening

This document records smoke-level backbone screening. These numbers prove that the candidates can be instantiated through the FADE-Net adapter and give a first efficiency signal. They are not final paper results.

## Environment

- Date: 2026-05-22
- Workspace: `F:/FADE-Net`
- Python: `.venv/Scripts/python.exe`
- PyTorch: `2.2.0+cu121`
- timm: `1.0.27`
- NumPy: `1.26.4`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Input size: `224x224`
- Weights:
  - Formal short-run screening defaults to pretrained backbones.
  - `--no_pretrained` is reserved for pipeline smoke checks, offline probes, and scratch ablations.
  - Pretrained and scratch runs write to different artifact IDs.
- Benchmark command: `scripts/benchmark_speed.py --iters 10 --batch_size 1`
- Main split for follow-up runs: `--split 72-8-20`
- Reporting metric: `Selected_Test_MAE`, with `MAE_raw`, `MAE_flip`, and `MAE_multi` retained for TTA analysis
- Screening manifest fields: metrics, `split_file_tag`, `experiment_id`, `result_path`, `stdout_log`, and `stderr_log`
- Default screening logs: `docs/backbone_screening_logs/`
- Resume policy: default runner commands use `--fresh`; add runner-level `--resume` only to continue an interrupted run with matching metadata.
- Verified AFAD path on this machine: `F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full`
- Generated split file: `dataset_split_AFAD_72_8_20.json`, file fingerprint `03a06cdaad2266bea0072361aa8761350f7247a5c2eb11bf1e326c65fa8fe598`
- New split files include `_metadata.dataset_fingerprint`; checkpoint metadata also records this dataset fingerprint. Legacy split files without `_metadata` are refused by default; use `--allow_legacy_split_upgrade` only for an intentional one-time upgrade after size/index validation.
- Formal paper reruns should add `--split_file_tag formal_v1`, which writes `dataset_split_AFAD_72_8_20_formal_v1.json` and separates artifact IDs with `splitfile-formal_v1`.

## Candidate Names

RepViT models available in timm:

```text
repvit_m0_9
repvit_m1
repvit_m1_0
repvit_m1_1
repvit_m1_5
repvit_m2
repvit_m2_3
repvit_m3
```

MobileNetV4 models available in timm:

```text
mobilenetv4_conv_aa_large
mobilenetv4_conv_aa_medium
mobilenetv4_conv_blur_medium
mobilenetv4_conv_large
mobilenetv4_conv_medium
mobilenetv4_conv_small
mobilenetv4_conv_small_035
mobilenetv4_conv_small_050
mobilenetv4_hybrid_large
mobilenetv4_hybrid_large_075
mobilenetv4_hybrid_medium
mobilenetv4_hybrid_medium_075
```

## Smoke Results

| Backbone | Params | FLOPs | CPU latency | GPU latency | Notes |
|---|---:|---:|---:|---:|---|
| `torchvision/mobilenet_v3_large` | 4.84M | 1.05G | not rerun in venv speed table | not rerun in venv speed table | Current baseline; HA effective. |
| `timm/mobilenetv4_conv_small_050` | 2.08M | 119.76M | 13.37 ms | 13.45 ms | Strong extreme-light candidate; HA requested but not injected. |
| `timm/mobilenetv4_conv_small` | 3.98M | 292.39M | 17.34 ms | 11.80 ms | Best first replacement candidate. |
| `timm/repvit_m0_9` | 6.27M | 894.76M | 34.73 ms | 21.35 ms | Useful modern CNN comparison, but not lighter by params. |
| `timm/repvit_m1` | 6.27M | 894.76M | not benchmarked | not benchmarked | Same smoke param/FLOP level as `repvit_m0_9` in this adapter. |
| `timm/mobilenetv4_conv_medium` | 9.93M | 938.05M | not benchmarked | not benchmarked | Accuracy-upper-bound candidate, not lightweight baseline replacement. |

## Interpretation

`mobilenetv4_conv_small` is the best first replacement candidate because it is lighter than the current MobileNetV3 baseline by parameters and FLOPs while retaining enough capacity for a fair short-run screen. `mobilenetv4_conv_small_050` should be kept as an extreme-light comparison. `repvit_m0_9` is useful as a modern CVPR-era lightweight CNN comparison, but it is not a parameter-count improvement over the current baseline.

For timm candidates, Coordinate Attention injection is not currently effective because the adapter does not expose replaceable torchvision-style SE blocks. Metadata records both requested HA and effective HA; timm result tables must label this clearly.

## Next Experiments

Protocol hardening is complete enough for paper-grade short-run screening. Completed 20 epoch single-seed rows may be used only after `scripts/audit_paper_results.py` returns `status=paper-ready`; they must not be reported as final 3-seed paper performance. The screening runner defaults to pretrained weights because the training schedule freezes the backbone at the start; training a frozen random backbone would make the 20 epoch screen invalid.

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --overwrite --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
```

For paper-grade reruns, use the tagged split/artifact identity:

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --split_file_tag formal_v1 --candidates mobilenet_v3_large,mobilenetv4_conv_small --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_formal_v1.csv --log_dir docs/backbone_screening_logs
```

If a valid long run is interrupted, resume only the same candidate, split, seed, backbone, pretrained mode, and experiment tag:

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --resume --candidates mobilenetv4_conv_small --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
```

The matrix can also be split by candidate to reduce long-run risk:

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --candidates mobilenet_v3_large --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --append --candidates mobilenetv4_conv_small --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --append --candidates mobilenetv4_conv_small_050,repvit_m0_9 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
```

Artifact names now include backbone source and weight mode. For example, the formal pretrained MobileNetV4-small dry run points to:

```text
final_result_FADE-Net_DLDL_MSFF_SPP_MV_timm_mobilenetv4_conv_small_pretrained_72-8-20_seed42.txt
```

With `--split_file_tag formal_v1`, the corresponding paper-grade path includes `splitfile-formal_v1` before `seed42`.

The scratch ablation counterpart points to:

```text
final_result_FADE-Net_DLDL_MSFF_SPP_MV_timm_mobilenetv4_conv_small_scratch_72-8-20_seed42.txt
```

An earlier 20 epoch `mobilenetv4_conv_small` attempt used `--no_pretrained` and was stopped before epoch 1 completed. Ignore those partial logs and result paths for screening or paper evidence.

After the short-run screen, run full paper-grade experiments only for the top two candidates and the locked baseline, with 3 seeds and identical split, augmentation, TTA, and metadata settings.

For pipeline-only smoke checks, add explicit batch limits and keep the `smoke` tag so artifacts cannot overwrite formal results:

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --max_train_batches 1 --max_val_batches 1 --max_test_batches 1 --output docs/backbone_screening_smoke_dry_run.csv --log_dir docs/backbone_screening_logs
.\.venv\Scripts\python.exe -B scripts/run_backbone_screening.py --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --epochs 1 --max_train_batches 1 --max_val_batches 1 --max_test_batches 1 --run --output docs/backbone_screening_smoke_runs.csv --log_dir docs/backbone_screening_logs
```

The screening runner writes `selected_test_mae`, `mae_raw`, `mae_flip`, `mae_multi`, `selected_tta`, `split_file_tag`, `experiment_id`, `result_path`, `stdout_log`, and `stderr_log` into the CSV. It flushes the manifest after each candidate finishes, so a long run keeps completed rows and log paths even if a later candidate fails. Runner-level `--resume` maps to `src/train.py --resume`; metadata mismatch still fails inside the training entrypoint. Existing candidate logs or result files are refused unless `--overwrite_artifacts` is explicitly set.

Dataset identity is now part of the reproducibility contract. For final paper-grade tables, use only runs whose split file has `_metadata.dataset_fingerprint`; older runs that used a legacy split file may be kept as candidate triage, but should be rerun before being used as final paper evidence.

Before copying any result into a paper table, run the paper-result audit:

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_paper_results.py --split_file_tag formal_v1 --candidates mobilenet_v3_large,mobilenetv4_conv_small,mobilenetv4_conv_small_050 --seeds 42,3407,2026 --output docs/paper_result_audit.csv
```

Only rows with `status=paper-ready` may be used as single-row paper evidence. Final mean/std tables must come from `scripts/summarize_paper_results.py` rows marked `complete`; `partial` rows stay single-seed evidence only. Rows marked `blocked` must be treated as candidate triage or historical evidence. `docs/paper_result_audit_formal_v1_seed42.csv` currently audits `mobilenetv4_conv_small` seed42 as `paper-ready`; `docs/paper_result_audit_current.csv` is an older blocked audit sample kept as historical reference.

After auditing, generate a gap-aware paper table draft:

```powershell
.\.venv\Scripts\python.exe -B scripts/summarize_paper_results.py --audit docs/paper_result_audit.csv --candidates torchvision/mobilenet_v3_large,timm/mobilenetv4_conv_small,timm/mobilenetv4_conv_small_050,timm/repvit_m0_9 --seeds 42,3407,2026 --output docs/paper_result_summary.md
```

The summary marks incomplete rows as `partial` or `missing`; only `complete` rows with all planned seeds should be used for final mean/std claims.

## Formal Single-Seed Result

Completed paper-grade 20 epoch screen on 2026-05-22:

| Backbone | Seed | Split tag | Audit | Selected_Test_MAE | MAE_raw | MAE_flip | MAE_multi | Result path | Notes |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| `timm/mobilenetv4_conv_small` | 42 | `formal_v1` | `paper-ready` | 3.6130 | 3.6042 | 3.5965 | 3.6130 | `final_result_FADE-Net_DLDL_MSFF_SPP_MV_timm_mobilenetv4_conv_small_pretrained_72-8-20_splitfile-formal_v1_seed42.txt` | Single-seed evidence row only; not a final mean/std paper result. |

Observed pipeline smoke on 2026-05-22. These values use only one train, validation, and test batch per candidate; they verify execution and result capture only:

| Backbone | Tag | Train batches | Val batches | Test batches | Selected_Test_MAE | MAE_raw | MAE_flip | MAE_multi | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `torchvision/mobilenet_v3_large` | `smoke` | 1 | 1 | 1 | 25.0318 | 25.0318 | 25.0318 | 25.0318 | Pipeline-only baseline check; HA effective. |
| `timm/mobilenetv4_conv_small` | `smoke` | 1 | 1 | 1 | 25.1193 | 25.1563 | 25.1325 | 25.1193 | Pipeline-only; verifies MobileNetV4 small path and CSV metrics. |
| `timm/mobilenetv4_conv_small_050` | `smoke` | 1 | 1 | 1 | 25.8207 | 25.8233 | 25.8199 | 25.8207 | Pipeline-only extreme-light path. |
| `timm/repvit_m0_9` | `smoke` | 1 | 1 | 1 | 25.0836 | 25.0836 | 25.0836 | 25.0836 | Pipeline-only RepViT path. |
| `timm/mobilenetv4_conv_small` | `smoke-mnv4-small-2b` | 2 | 1 | 1 | 24.9475 | 24.9398 | 24.9242 | 24.9475 | Earlier isolated smoke; proves data/backbone/train/eval/result write path. |

## Result Table Template

| Backbone | Seed | Selected_Test_MAE | MAE_raw | MAE_flip | MAE_multi | Params | FLOPs | CPU latency | GPU latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchvision/mobilenet_v3_large` | 42 | TBD | TBD | TBD | TBD | 4.84M | 1.05G | TBD | TBD | Locked baseline |
| `timm/mobilenetv4_conv_small` | 42 | 3.6130 | 3.6042 | 3.5965 | 3.6130 | 3.98M | 292.39M | 17.34 ms | 11.80 ms | `formal_v1`, audit `paper-ready`; single-seed evidence only |
| `timm/mobilenetv4_conv_small_050` | 42 | TBD | TBD | TBD | TBD | 2.08M | 119.76M | 13.37 ms | 13.45 ms | Extreme-light candidate |
| `timm/repvit_m0_9` | 42 | TBD | TBD | TBD | TBD | 6.27M | 894.76M | 34.73 ms | 21.35 ms | Modern comparison, not parameter-lighter |

For paper tables, add `Mean +- Std` rows only after all planned seeds finish. Do not compare external methods as strict rank order unless their data split, preprocessing, pretrained weights, and TTA protocol are verified to match.
