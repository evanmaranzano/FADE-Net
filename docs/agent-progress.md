# Agent Progress

Date: 2026-05-23

## Scope

- Deep-read project structure, README, requirements, core `src/` modules, key scripts, and test entry points.
- Performed lightweight, testable fixes and optimizations only. No full training run was executed.
- Performance claim uses inference/evaluation TTA latency, not final training accuracy.

## Files Changed

- `src/backbones.py`
  - Added offline fallback for timm pretrained weight loading. If `pretrained=True` fails because weights cannot be downloaded or loaded, the same timm model is rebuilt with `pretrained=False`.
- `src/evaluation.py`
  - Replaced sequential TTA forward passes with batched augmented-view inference plus bounded chunking.
  - Added `predict_age_with_uncertainty()` to report TTA age prediction disagreement as `age_std`.
- `scripts/advanced_eval.py`
  - Added `--uncertainty` for single-model advanced evaluation.
  - Added `evaluate_uncertainty()` summary output for mean TTA age standard deviation.
  - Rejects `--ensemble --uncertainty` before dataset loading because the current uncertainty summary is single-model only.
- `tests/test_research_refactor.py`
  - Added regression coverage for timm pretrained-load fallback.
- `tests/test_experiment_integrity.py`
  - Updated TTA call-count expectations for batched inference.
  - Added chunking/equivalence tests for batched TTA.
  - Added tests for TTA uncertainty reporting and advanced evaluation uncertainty summary.
- `docs/agent-progress.md`
  - Records this work, validation commands, issues, and remaining risks.

## Commands Run

- `git status --porcelain=v1`
- `Get-Content README.md`, `Get-Content README_zh.md`, `Get-Content requirements.txt`
- `rg --files`
- `rg -n ... src scripts tests README.md README_zh.md requirements.txt`
- `python --version`
- `python -c "import torch, timm, pytest; ..."`
  - Failed in global Python because `timm` is missing and NumPy 2.4.3 conflicts with the installed torch ABI.
- `./.venv/Scripts/python.exe -c "import numpy, torch, timm, pytest; ..."`
  - Confirmed project venv: NumPy 1.26.4, torch 2.2.0+cu121, timm 1.0.27, CUDA available.
- `./.venv/Scripts/python.exe -m pytest tests/ -q`
  - Baseline in project venv: 95 passed, 1 failed before fixes.
  - Final after fixes: 101 passed, 2 warnings.
- `./.venv/Scripts/python.exe scripts/benchmark_speed.py --iters 10 --batch_size 1 --no_pretrained`
  - Baseline raw model benchmark: CPU 25.89 ms, GPU 14.30 ms for batch 1.
- Lightweight TTA benchmark before optimization:
  - `device=cuda mode=multi sequential_latency_ms=105.201`
- Lightweight TTA benchmark after optimization:
  - `device=cuda mode=multi batched_latency_ms=18.096`
- `./.venv/Scripts/python.exe scripts/calc_params.py --no_pretrained`
  - Total parameters: 3.98M, FLOPs: 292.39M.
- Targeted tests:
  - `./.venv/Scripts/python.exe -m pytest tests/test_research_refactor.py::ResearchRefactorTests::test_timm_pretrained_load_failure_falls_back_to_random_weights -q`
  - `./.venv/Scripts/python.exe -m pytest tests/test_all_modules_integration.py::test_all_model_modules_enabled -q`
  - `./.venv/Scripts/python.exe -m pytest tests/test_freq_attention.py tests/test_moe_head.py tests/test_texture_branch.py -q`
  - `./.venv/Scripts/python.exe -m pytest tests/test_experiment_integrity.py -q`
  - `./.venv/Scripts/python.exe -m pytest tests/test_experiment_integrity.py::ExperimentIntegrityTests::test_tta_uncertainty_reports_view_disagreement tests/test_experiment_integrity.py::ExperimentIntegrityTests::test_advanced_eval_uncertainty_summary -q`
- `./.venv/Scripts/python.exe scripts/advanced_eval.py --help`
- `./.venv/Scripts/python.exe scripts/advanced_eval.py --ensemble --uncertainty`
  - Expected failure: `--uncertainty is only supported for single-model evaluation.`
- `./.venv/Scripts/python.exe -m py_compile scripts/advanced_eval.py src/evaluation.py`
- `./.venv/Scripts/python.exe -m compileall -q src scripts tests`
- `git diff --check`
  - No whitespace errors; only Windows CRLF warnings.

## Problems Found And Fixed

- Global `python` is not the correct project environment. It lacks `timm` and has NumPy 2.4.3 with a torch ABI warning. All project validation now uses `.venv/Scripts/python.exe`.
- Default timm pretrained loading could fail offline or behind a broken proxy and break tests/lightweight runs. Fixed with a random-weight fallback matching the existing torchvision behavior.
- `predict_probs(..., mode="multi")` ran 6 separate model forward passes per batch. Fixed by batching augmented views and chunking them safely.
- `advanced_eval.py --uncertainty` initially had the output block in the wrong branch during implementation; fixed before final validation and covered with tests.

## Performance Result

- Measured command used the same model, same input size, CUDA, and `cfg.backbone_pretrained=False`.
- Before: `multi` TTA latency `105.201 ms`.
- After: `multi` TTA latency `18.096 ms`.
- Improvement: about `82.8%` lower latency, about `5.8x` faster for the measured TTA inference path.
- This satisfies the requested `>=30%` lightweight performance optimization under the explicit inference/evaluation-TTA latency benchmark. It does not prove a `>=30%` training-time or final MAE improvement.

## Current Remaining Risks

- Full training was not run, so final MAE, convergence behavior, and paper-table results remain unverified.
- TTA batching can increase instantaneous augmented-batch size. The implementation bounds default chunking, but very tight GPU memory environments may still need `config.tta_batch_size`.
- `--uncertainty` reports TTA view disagreement, not calibrated predictive uncertainty. It should be interpreted as a diagnostic signal unless calibrated against validation data.
- Existing `src/gui_demo.py` and `src/web_demo.py` still contain separate TTA implementations; this run did not refactor demo code to avoid broad UI-side changes.
- No dedicated lint/typecheck config was found (`pyproject.toml`, `setup.cfg`, `tox.ini`, `.github` absent), so syntax validation used `py_compile`/`compileall`.
