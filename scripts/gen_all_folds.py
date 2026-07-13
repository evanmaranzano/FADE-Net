"""
Generate all 5 identity-disjoint splits for AFAD (age 15-40).
Folds 0-4 following CVPR 2024 protocol.
"""

import os
import json
import argparse
from collections import defaultdict

FOLDS = {
    0: {"train": [0,1,2,3,4,5], "val": [6,7], "test": [8,9]},
    1: {"train": [2,3,4,5,6,7], "val": [8,9], "test": [0,1]},
    2: {"train": [4,5,6,7,8,9], "val": [0,1], "test": [2,3]},
    3: {"train": [5,6,7,8,9,0], "val": [1,2], "test": [3,4]},
    4: {"train": [6,7,8,9,0,1], "val": [2,3], "test": [4,5]},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afad_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_age", type=int, default=15)
    parser.add_argument("--max_age", type=int, default=40)
    parser.add_argument("--nr_folders", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all images in age range
    print(f"Scanning {args.afad_dir} for ages {args.min_age}-{args.max_age}...")
    image_paths = []
    for age_dir in sorted(os.listdir(args.afad_dir)):
        age_path = os.path.join(args.afad_dir, age_dir)
        if not os.path.isdir(age_path) or not age_dir.isdigit():
            continue
        age = int(age_dir)
        if age < args.min_age or age > args.max_age:
            continue
        for gender_dir in os.listdir(age_path):
            gender_path = os.path.join(age_path, gender_dir)
            if not os.path.isdir(gender_path):
                continue
            for fname in os.listdir(gender_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(age_dir, gender_dir, fname))

    print(f"Found {len(image_paths)} images")

    # Extract identity IDs
    unique_ids = set()
    for rel_path in image_paths:
        fname = os.path.basename(rel_path)
        unique_ids.add(fname.split('-')[0])

    print(f"Found {len(unique_ids)} unique identities")

    # Round-robin folder assignment
    id2folder = {}
    folder_counter = 0
    for uid in sorted(unique_ids, key=lambda x: int(x)):
        id2folder[uid] = folder_counter
        folder_counter = (folder_counter + 1) % args.nr_folders

    # Assign folders
    sample_folders = []
    for rel_path in image_paths:
        fname = os.path.basename(rel_path)
        sample_folders.append(id2folder[fname.split('-')[0]])

    # Generate each fold
    for fold_id, split_def in FOLDS.items():
        train_folders = set(split_def["train"])
        val_folders = set(split_def["val"])
        test_folders = set(split_def["test"])

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

        # Verify no leakage
        def get_ids(indices):
            return {os.path.basename(image_paths[i]).split('-')[0] for i in indices}

        train_ids = get_ids(train_idx)
        val_ids = get_ids(val_idx)
        test_ids = get_ids(test_idx)

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        total = len(train_idx) + len(val_idx) + len(test_idx)
        print(f"Fold {fold_id}: Train={len(train_idx)} ({100*len(train_idx)/total:.1f}%), "
              f"Val={len(val_idx)} ({100*len(val_idx)/total:.1f}%), "
              f"Test={len(test_idx)} ({100*len(test_idx)/total:.1f}%)")

        # Save
        split_data = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "_metadata": {
                "protocol": f"identity_disjoint_fold{fold_id}",
                "source": "CVPR2024 Paplham & Franc",
                "num_samples": total,
                "num_identities": len(unique_ids),
                "nr_folders": args.nr_folders,
                "fold": fold_id,
                "min_age": args.min_age,
                "max_age": args.max_age,
                "num_classes": args.max_age - args.min_age + 1,
                "train_folders": sorted(list(train_folders)),
                "val_folders": sorted(list(val_folders)),
                "test_folders": sorted(list(test_folders)),
                "split_ratios_approx": [
                    round(len(train_idx)/total, 4),
                    round(len(val_idx)/total, 4),
                    round(len(test_idx)/total, 4),
                ],
            }
        }

        output_path = os.path.join(args.output_dir,
                                    f"dataset_split_AFAD_{args.min_age}_{args.max_age}_iddisjoint_fold{fold_id}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f)

    print(f"\n✅ All 5 folds saved to {args.output_dir}")

if __name__ == "__main__":
    main()
