# Dataset Setup Guide for FADE-Net

This guide explains how to prepare **AFAD-Full**, the dataset required by the current training and evaluation pipeline. AAF and UTKFace are historical or optional references only; the current `src/dataset.py` dataloader uses AFAD.

## 1. Directory Structure

The project expects the following structure after preprocessing:

```
code/
├── datasets/           # Processed & Aligned images (Generated)
│   └── AFAD/
├── data/               # Raw downloaded datasets (Recommended)
│   ├── UTKFace/
│   ├── AFAD-Full/
│   └── All-Age-Faces/
```

## 2. Preparing Raw Data

Please ensure you have the AFAD-Full raw dataset downloaded.

### AFAD (Asian Face Age Dataset)
*   **Source**: [GitHub / Official Site]
*   **Format**: Nested folders (Age -> Gender -> Images).
*   **Default path**: `datasets/AFAD` under the project root.
*   **Override path**: set `FADE_NET_AFAD_DIR` or pass `--afad_dir` to training/evaluation scripts.
*   **Verified local path**: `F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full`

### AAF (All-Age-Faces Dataset, historical/optional)
*   **Source**: [GitHub]
*   **Format**: Flat folder with aligned faces.
*   **Path in Script**: Please check `scripts/preprocess.py`.

### UTKFace (historical/optional)
*   **Source**: [Susanqq/UTKFace]
*   **Path**: Expects `./data/UTKFace/train` and `./data/UTKFace/val`.

## 3. Running Preprocessing

We provide a script to **clean, align, and organize** the datasets into the standard format required by `src/dataset.py`.

1.  **Use an existing AFAD folder directly when it already matches `age/gender/images`**:

    ```powershell
    $env:FADE_NET_AFAD_DIR="F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full"
    .\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
    ```

    Paper-grade reruns should use an isolated split file tag instead of upgrading or overwriting an old split file:

    ```powershell
    .\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --split_file_tag formal_v1 --fresh --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full"
    ```

2.  **Or edit preprocessing paths**: Open `scripts/preprocess.py` and modify the `raw_afad_dir` and `raw_aaf_dir` variables to point to where you downloaded the datasets.
    ```python
    # scripts/preprocess.py
    raw_afad_dir = "path/to/AFAD-Full"
    raw_aaf_dir = "path/to/AAF/aglined_faces"
    ```

3.  **Run the Script**:
    ```bash
    python scripts/preprocess.py
    ```

4.  **Result**:
    The script will create a `datasets/` directory in the project root containing the processed images.

## 4. Verification

After processing, you should verify the `datasets/` folder exists and contains valid images. If you use `--afad_dir` or `FADE_NET_AFAD_DIR`, verify that path instead.

Verified on 2026-05-22 with the local AFAD path above:

- Images scanned: `163374`
- Split: `dataset_split_AFAD_72_8_20.json`
- Train / Val / Test: `117604 / 13039 / 32731`
- Split fingerprint: `03a06cdaad2266bea0072361aa8761350f7247a5c2eb11bf1e326c65fa8fe598`

New split files also store `_metadata.dataset_fingerprint`, a hash of sample order and labels. Training checkpoints copy this value into metadata so later resume/evaluation can reject silently changed sample lists. This is not a per-image content hash. Legacy split files without `_metadata` are refused by default; use `--allow_legacy_split_upgrade` only when you intentionally trust and stamp an old split after size/index validation.

If a metadata-backed split file mismatches the current dataset fingerprint, sample count, or split ratio, training now stops instead of regenerating it. Treat that as a protocol change: inspect the dataset path/order first, then intentionally create a new split file or archive/remove the old one before rerunning.

For formal paper reruns, prefer `--split_file_tag formal_v1`. With `--split 72-8-20`, this writes `dataset_split_AFAD_72_8_20_formal_v1.json` and adds `splitfile-formal_v1` to artifact IDs, so old candidate-triage artifacts stay separate from paper-grade reruns.
