import os
import zipfile
import datetime
import glob
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

SAFE_FILE_PATTERNS = [
    "final_result_*.txt",
    "training_log_*.csv",
    "batch_log_*.csv",
    "docs/paper_result_*.csv",
    "docs/paper_result_summary.md",
    "docs/ablation_plan_v4.md",
    "docs/backbone_screening_runs.csv",
]
SAFE_DIRS = ["plots", "src"]
SAFE_PLOT_NAMES = {
    "1_loss_curve.png",
    "2_mae_curve.png",
    "3_lr_schedule.png",
    "4_generalization_gap.png",
    "4_kl_loss_gap.png",
    "5_batch_stability.png",
    "6_time_efficiency.png",
    "7_batch_loss_dist.png",
    "8_loss_lr_combined.png",
}
WEIGHT_PATTERNS = [
    "best_model_*.pth",
    "swa_model_*.pth",
    "last_checkpoint_*.pth",
    "checkpoint_epoch_*.pth",
]
DENY_NAMES = {".env", ".env.local", ".env.production"}
DENY_SUFFIXES = (
    ".ckpt",
    ".key",
    ".log",
    ".onnx",
    ".pem",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".tar",
    ".tar.gz",
    ".zip",
)
MAX_DEFAULT_FILE_BYTES = 25 * 1024 * 1024


def _safe_to_pack(path, root_dir=ROOT_DIR, allow_weight_file=False):
    path_obj = Path(path)
    if path_obj.is_symlink():
        return False
    try:
        root_path = Path(root_dir).resolve()
        resolved_path = path_obj.resolve(strict=True)
        resolved_path.relative_to(root_path)
    except (OSError, ValueError):
        return False
    if not resolved_path.is_file():
        return False

    name = path_obj.name
    if name in DENY_NAMES or name == ".DS_Store":
        return False
    if name.endswith((".pt", ".pth", ".ckpt")) and allow_weight_file:
        return True
    if name.endswith(DENY_SUFFIXES):
        return False
    if "__pycache__" in path_obj.parts:
        return False
    if os.path.getsize(resolved_path) > MAX_DEFAULT_FILE_BYTES:
        return False
    return True


def _safe_plot_file(path, root_dir=ROOT_DIR):
    return Path(path).name in SAFE_PLOT_NAMES and _safe_to_pack(path, root_dir=root_dir)


def pack_results(output=None, include_weights=False, overwrite=False):
    root_dir = str(ROOT_DIR)

    # Generate timestamped filename
    if output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = os.path.join(root_dir, f"training_results_{timestamp}.zip")
    else:
        zip_filename = os.path.abspath(output)
    if os.path.exists(zip_filename) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing package: {zip_filename}")
    os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
    
    print(f"📦 Start packaging to: {zip_filename}")
    patterns = list(SAFE_FILE_PATTERNS)
    if include_weights:
        patterns.extend(WEIGHT_PATTERNS)
    
    count = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Walk through patterns in ROOT
        for pattern in patterns:
            full_pattern = os.path.join(root_dir, pattern)
            for file_path in glob.glob(full_pattern):
                if not _safe_to_pack(file_path, root_dir=ROOT_DIR, allow_weight_file=pattern in WEIGHT_PATTERNS):
                    continue
                arcname = os.path.relpath(file_path, root_dir)
                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)
                count += 1

        for dirname in SAFE_DIRS:
            target_dir = os.path.join(root_dir, dirname)
            if not os.path.exists(target_dir):
                continue
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for file in files:
                    file_path = os.path.join(root, file)
                    is_safe = (
                        _safe_plot_file(file_path, root_dir=ROOT_DIR)
                        if dirname == "plots"
                        else _safe_to_pack(file_path, root_dir=ROOT_DIR)
                    )
                    if not is_safe:
                        continue
                    arcname = os.path.relpath(file_path, root_dir)
                    print(f"  Adding: {arcname}")
                    zipf.write(file_path, arcname)
                    count += 1
                    
    print(f"\n✅ Packaging complete!")
    print(f"📁 Total files: {count}")
    print(f"💾 Size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")
    print(f"📍 Location: {zip_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package FADE-Net training artifacts")
    parser.add_argument("--output", help="Explicit zip output path; default is timestamped under project root")
    parser.add_argument("--include-weights", action="store_true", help="Include .pth model/checkpoint weights")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing zip package")
    args = parser.parse_args()
    pack_results(args.output, include_weights=args.include_weights, overwrite=args.overwrite)
