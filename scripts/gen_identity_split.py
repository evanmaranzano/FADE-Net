"""
Generate identity-disjoint split for AFAD dataset.
Follows CVPR 2024 protocol (Paplham & Franc):
- 10 folders via round-robin on identity ID
- 5 folds: train=[folders], val=[folders], test=[folders]
- Default uses fold 0: train=[0-5], val=[6,7], test=[8,9]

Usage: python gen_identity_split.py --afad_dir /opt/data/instance_gpu_3/AFAD --fold 0
"""

import os
import json
import argparse
import glob
from collections import defaultdict

# CVPR 2024 5-fold identity-disjoint splits
FOLDS = {
    0: {"train": [0,1,2,3,4,5], "val": [6,7], "test": [8,9]},
    1: {"train": [2,3,4,5,6,7], "val": [8,9], "test": [0,1]},
    2: {"train": [4,5,6,7,8,9], "val": [0,1], "test": [2,3]},
    3: {"train": [5,6,7,8,9,0], "val": [1,2], "test": [3,4]},
    4: {"train": [6,7,8,9,0,1], "val": [2,3], "test": [4,5]},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afad_dir", required=True, help="Path to AFAD dataset root")
    parser.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4], help="Which fold to use")
    parser.add_argument("--output", default=None, help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--nr_folders", type=int, default=10, help="Number of identity folders")
    args = parser.parse_args()

    afad_dir = args.afad_dir
    fold = args.fold
    nr_folders = args.nr_folders

    # Discover all images
    print(f"Scanning {afad_dir} for images...")
    image_paths = []
    for age_dir in sorted(os.listdir(afad_dir)):
        age_path = os.path.join(afad_dir, age_dir)
        if not os.path.isdir(age_path) or not age_dir.isdigit():
            continue
        for gender_dir in os.listdir(age_path):
            gender_path = os.path.join(age_path, gender_dir)
            if not os.path.isdir(gender_path):
                continue
            for fname in os.listdir(gender_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(age_dir, gender_dir, fname))

    print(f"Found {len(image_paths)} images")

    # Extract identity IDs from filenames
    # Format: {id_num}-{index}.jpg
    id2folder = {}
    folder_counter = 0

    # First pass: collect all unique IDs
    unique_ids = set()
    for rel_path in image_paths:
        fname = os.path.basename(rel_path)
        id_str = fname.split('-')[0]
        unique_ids.add(id_str)

    print(f"Found {len(unique_ids)} unique identities")

    # Round-robin folder assignment
    for uid in sorted(unique_ids, key=lambda x: int(x)):
        id2folder[uid] = folder_counter
        folder_counter = (folder_counter + 1) % nr_folders

    # Assign each image to a folder based on its identity
    sample_folders = []
    for rel_path in image_paths:
        fname = os.path.basename(rel_path)
        id_str = fname.split('-')[0]
        sample_folders.append(id2folder[id_str])

    # Get the split definition for this fold
    split_def = FOLDS[fold]
    train_folders = set(split_def["train"])
    val_folders = set(split_def["val"])
    test_folders = set(split_def["test"])

    print(f"Fold {fold}: train folders={sorted(train_folders)}, val folders={sorted(val_folders)}, test folders={sorted(test_folders)}")

    # Build indices
    train_idx = []
    val_idx = []
    test_idx = []

    for idx, folder in enumerate(sample_folders):
        if folder in train_folders:
            train_idx.append(idx)
        elif folder in val_folders:
            val_idx.append(idx)
        elif folder in test_folders:
            test_idx.append(idx)

    print(f"Split result: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Verify no identity leakage
    def get_ids(indices):
        ids = set()
        for idx in indices:
            fname = os.path.basename(image_paths[idx])
            ids.add(fname.split('-')[0])
        return ids

    train_ids = get_ids(train_idx)
    val_ids = get_ids(val_idx)
    test_ids = get_ids(test_idx)

    assert len(train_ids & val_ids) == 0, f"Identity leakage: {len(train_ids & val_ids)} IDs in both train and val"
    assert len(train_ids & test_ids) == 0, f"Identity leakage: {len(train_ids & test_ids)} IDs in both train and test"
    assert len(val_ids & test_ids) == 0, f"Identity leakage: {len(val_ids & test_ids)} IDs in both val and test"
    print("✅ No identity leakage detected")

    # Count per-age distribution
    age_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for idx in train_idx:
        age = int(os.path.basename(os.path.dirname(os.path.dirname(image_paths[idx]))))
        age_counts[age]["train"] += 1
    for idx in val_idx:
        age = int(os.path.basename(os.path.dirname(os.path.dirname(image_paths[idx]))))
        age_counts[age]["val"] += 1
    for idx in test_idx:
        age = int(os.path.basename(os.path.dirname(os.path.dirname(image_paths[idx]))))
        age_counts[age]["test"] += 1

    # Save split file
    if args.output:
        output_path = args.output
    else:
        output_path = f"dataset_split_AFAD_identity_fold{fold}.json"

    split_data = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "_metadata": {
            "protocol": f"identity_disjoint_fold{fold}",
            "source": "CVPR2024 Paplham & Franc",
            "num_samples": len(image_paths),
            "num_identities": len(unique_ids),
            "nr_folders": nr_folders,
            "fold": fold,
            "train_folders": sorted(list(train_folders)),
            "val_folders": sorted(list(val_folders)),
            "test_folders": sorted(list(test_folders)),
            "split_ratios_approx": [
                round(len(train_idx)/len(image_paths), 4),
                round(len(val_idx)/len(image_paths), 4),
                round(len(test_idx)/len(image_paths), 4),
            ],
        }
    }

    with open(output_path, 'w') as f:
        json.dump(split_data, f)

    print(f"💾 Saved to {output_path}")
    print(f"   Train: {len(train_idx)} ({100*len(train_idx)/len(image_paths):.1f}%)")
    print(f"   Val:   {len(val_idx)} ({100*len(val_idx)/len(image_paths):.1f}%)")
    print(f"   Test:  {len(test_idx)} ({100*len(test_idx)/len(image_paths):.1f}%)")

if __name__ == "__main__":
    main()
