# FADE-Net Research Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn FADE-Net into a reproducible age-estimation research codebase with a modern lightweight backbone comparison path for a Chinese core journal submission.

**Architecture:** Keep the current MobileNetV3-Large FADE-Net as the verified baseline, but route all backbones through an adapter that exposes deep and intermediate feature maps. All training, evaluation, SWA, and reporting artifacts must carry a complete experiment identity so backbone, split, ablation flags, and loss settings cannot be mixed.

**Tech Stack:** PyTorch, torchvision, optional timm feature extraction, AFAD split files, unittest, THOP, TensorBoard.

---

### Task 1: Baseline Safety Layer

**Files:**
- Modify: `src/model.py`
- Create: `src/backbones.py`
- Create: `src/experiment.py`
- Modify: `src/config.py`
- Modify: `src/train.py`
- Modify: `src/dataset.py`
- Test: `tests/test_research_refactor.py`

- [x] **Step 1: Add backbone adapter**

Create `FeatureBackbone`, `TorchvisionMobileNetV3Backbone`, and `TimmFeatureBackbone`. Default remains `torchvision/mobilenet_v3_large`; timm is opt-in.

- [x] **Step 2: Preserve default forward contract**

Verify default FADE-Net still outputs `[batch, 81]`, with MobileNetV3 MSFF channels `40/112/960`.

Run:

```powershell
python -B tests/test_research_refactor.py
```

Expected: 3 tests pass.

- [x] **Step 3: Add experiment identity**

Add metadata for `experiment_id`, `project_name`, split, backbone, ablations, losses, augmentations, and TTA protocol. Use metadata-gated resume.

- [x] **Step 4: Fix data and worker safety**

Record split fingerprint, avoid cross-index fallback on corrupt images, and set `persistent_workers` only when `num_workers > 0`.

### Task 2: Modern Backbone Candidate Gate

**Files:**
- Modify: `requirements.txt`
- Modify: `scripts/calc_params.py`
- Modify: `scripts/benchmark_speed.py`
- Modify: `scripts/advanced_eval.py`
- Modify: `scripts/swa_average.py`

- [x] **Step 1: Add optional timm dependency**

Add `timm>=1.0.0` and pin `numpy<2` for PyTorch 2.2 compatibility in this environment.

- [x] **Step 2: Update analysis scripts**

Allow `--backbone_source`, `--backbone_name`, and `--no_pretrained` in parameter counting and speed benchmarking.

- [x] **Step 3: Update evaluation scripts**

Use metadata-aware artifact naming in advanced evaluation and SWA. Refuse to average checkpoints with mismatched metadata.

- [x] **Step 4: Install and probe timm candidates**

Run in an isolated environment, not the shared Anaconda base:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import timm; print([m for m in timm.list_models('*repvit*')[:20]])"
.\.venv\Scripts\python.exe -c "import timm; print([m for m in timm.list_models('*mobilenetv4*')[:20]])"
```

Expected: identify exact timm model names before running full experiments.

Observed on 2026-05-22 with `.venv`, `timm==1.0.27`, `numpy==1.26.4`:
- RepViT names: `repvit_m0_9`, `repvit_m1`, `repvit_m1_0`, `repvit_m1_1`, `repvit_m1_5`, `repvit_m2`, `repvit_m2_3`, `repvit_m3`.
- MobileNetV4 names: `mobilenetv4_conv_small_035`, `mobilenetv4_conv_small_050`, `mobilenetv4_conv_small`, `mobilenetv4_conv_medium`, `mobilenetv4_conv_large`, plus AA/blur/hybrid variants.
- First viable candidates: `mobilenetv4_conv_small_050`, `mobilenetv4_conv_small`, and `repvit_m0_9`.

### Task 3: Experiment Protocol Hardening

**Files:**
- Modify: `src/train.py`
- Modify: `scripts/advanced_eval.py`
- Modify: `scripts/plot_results.py`
- Create: `tests/test_experiment_integrity.py`

- [x] **Step 1: Log comparable train and validation losses**

Record `KL`, `L1`, `Rank/CDF`, `MV`, and total loss separately. Do not plot generalization gap between non-comparable losses.

- [x] **Step 2: Output raw, flip, and multi TTA metrics**

Each evaluation run should report `MAE_raw`, `MAE_flip`, and `MAE_multi`, with the selected metric declared in metadata.

- [x] **Step 3: Add integrity tests**

Test artifact naming, metadata mismatch rejection, SWA mismatch rejection, and split fingerprint stability.

Observed on 2026-05-22:
- Added shared `src/evaluation.py` for `raw`, `flip`, and `multi` TTA.
- `train.py`, `advanced_eval.py`, and `swa_average.py` now report comparable TTA metrics from the same implementation.
- Training logs include total, KL, L1, Rank/CDF, and MV loss components for both train and validation.
- `plot_results.py` no longer plots a train-total vs val-KL "generalization gap"; it plots KL-vs-KL only when comparable columns exist.
- Metadata now declares reported TTA modes, selected metric, split file, and split fingerprint; resume/evaluation reject mismatches.
- Hard-distillation mode is resume-safe after epoch 105.
- Verification passed:
  - `.venv\Scripts\python.exe -B -m py_compile src/config.py src/backbones.py src/evaluation.py src/experiment.py src/model.py src/dataset.py src/utils.py src/train.py scripts/advanced_eval.py scripts/swa_average.py scripts/plot_results.py tests/test_research_refactor.py tests/test_experiment_integrity.py`
  - `.venv\Scripts\python.exe -B tests/test_experiment_integrity.py` (`7` tests)
  - `.venv\Scripts\python.exe -B tests/test_research_refactor.py` (`4` tests)

### Task 4: Backbone Screening Matrix

**Files:**
- Modify: `docs/ablation_guide.md`
- Create: `docs/backbone_screening.md`
- Generate: result CSV files only after real runs

- [ ] **Step 1: Short-run screen**

Run 1 seed, 10-20 epochs, same split, pretrained by default, no claim of final performance:

```powershell
.\.venv\Scripts\python.exe scripts/run_backbone_screening.py --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --run --output docs/backbone_screening_runs.csv --log_dir docs/backbone_screening_logs
```

- [ ] **Step 2: Full run for top candidates**

Run 3 seeds for the top two candidates only, using identical split, TTA, and metadata policy.

- [x] **Step 3: Paper-safe reporting**

Report internal AFAD protocol results as internal results. Avoid claiming SOTA unless every comparison uses a verified shared protocol.

Observed on 2026-05-22:
- Updated `docs/backbone_screening.md` with protocol-hardened short-run commands, candidate table template, and explicit no-paper-claims boundary.
- Updated `docs/ablation_guide.md` to use `Selected_Test_MAE`, `MAE_raw`, `MAE_flip`, `MAE_multi`, metadata-safe resume, and AFAD `72-8-20` commands.
- Verified local AFAD path `F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full` loads `163374` images and creates `dataset_split_AFAD_72_8_20.json` with fingerprint `03a06cdaad2266bea0072361aa8761350f7247a5c2eb11bf1e326c65fa8fe598`.
- Added `scripts/run_backbone_screening.py` and pipeline smoke controls. `timm/mobilenetv4_conv_small` completed a 2-train-batch/1-val-batch/1-test-batch smoke run with `experiment_tag=smoke-mnv4-small-2b`; this verifies pipeline wiring only and is not a paper result.
- The screening runner now writes `Selected_Test_MAE`, `MAE_raw`, `MAE_flip`, `MAE_multi`, `Selected_TTA`, `experiment_id`, and result file path into the CSV manifest.
- The screening runner now flushes the manifest after each candidate finishes, so long screening runs keep completed rows even if a later candidate fails.
- The screening runner now supports `--candidates` for running a subset of backbone candidates and `--append` for merging split runs into one manifest.
- The screening runner now writes `stdout_log` and `stderr_log` paths into the manifest and redirects each candidate run to its own log files under `docs/backbone_screening_logs/`.
- The screening runner now defaults to `--fresh` and supports runner-level `--resume`, which maps to `src/train.py --resume` and keeps metadata mismatch rejection inside the training entrypoint.
- A real `mobilenetv4_conv_small` pretrained `smoke-logcheck` run completed with one train, validation, and test batch, proving CSV metrics plus stdout/stderr log capture in the real subprocess path.
- The training subprocess is launched with `python -u` so long-run stdout is unbuffered and monitorable while a run is still active.
- Started the first valid `mobilenetv4_conv_small` pretrained 20 epoch short-run on 2026-05-22. Runner output: `docs/backbone_screening_runner_mnv4_small_pretrained.out.log`; train stdout/stderr: `docs/backbone_screening_logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_seed42.out.log` and `.err.log`; manifest target: `docs/backbone_screening_runs_mnv4_small_pretrained.csv`.
- A four-candidate smoke matrix completed with `--epochs 1 --max_train_batches 1 --max_val_batches 1 --max_test_batches 1 --experiment_tag smoke`, covering `mobilenet_v3_large`, `mobilenetv4_conv_small`, `mobilenetv4_conv_small_050`, and `repvit_m0_9`; all returned `ok`.
- The screening runner defaults to pretrained weights for formal short-run screening. Use `--no_pretrained` only for smoke, offline probes, or explicit scratch ablations.
- Artifact IDs now include backbone source and weight mode, e.g. `timm_mobilenetv4_conv_small_pretrained` versus `timm_mobilenetv4_conv_small_scratch`.
- A previous 20 epoch `mobilenetv4_conv_small --no_pretrained` attempt was stopped before epoch 1 completed and must not be used as screening or paper evidence.
- Split caching now writes `_metadata.dataset_fingerprint` for newly generated split JSON files, and checkpoint metadata records the same dataset fingerprint. Legacy split files without `_metadata` remain compatible with size/index validation only; runs started before this change should be treated as candidate triage and rerun before final paper tables.
- No 20 epoch or full 3-seed backbone training has been run yet.

### Task 5: Journal Contribution Package

**Files:**
- Modify: `README_zh.md`
- Modify: `docs/thesis_draft.md`
- Create: `docs/paper_core_claims.md`

- [x] **Step 1: Rewrite claims**

Use “competitive under our AFAD 72-8-20 protocol” instead of “SOTA/leading” unless supported by unified benchmark evidence.

- [x] **Step 2: Contribution framing**

Frame the work as task-oriented lightweight age estimation: adapter-based modern lightweight backbone comparison, distribution-aware supervision, and reproducible evaluation.

- [x] **Step 3: Required tables**

Prepare tables for backbone comparison, module ablation, TTA impact, seed mean/std, and efficiency.

Observed on 2026-05-22:
- Rewrote `README_zh.md` to remove SOTA/leading/server-grade/deployment-complete claims and to mark external results as protocol-mismatched references.
- Rewrote `docs/thesis_draft.md` so external results are references, ablation values are templates until rerun, and deployment claims are framed as future validation.
- Created `docs/paper_core_claims.md` with safe claim boundaries, forbidden claims, contribution wording, required experiment checklist, and a reusable safe abstract sentence.
