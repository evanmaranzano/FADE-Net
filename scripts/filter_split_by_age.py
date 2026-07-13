"""
Filter identity-disjoint split to age range 15-40 (26 classes).
Reads the existing split file and removes samples with age outside [15, 40].
"""

import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afad_dir", required=True, help="Path to AFAD dataset root")
    parser.add_argument("--input_split", required=True, help="Input split JSON file")
    parser.add_argument("--output_split", required=True, help="Output split JSON file")
    parser.add_argument("--min_age", type=int, default=15, help="Minimum age")
    parser.add_argument("--max_age", type=int, default=40, help="Maximum age")
    args = parser.parse_args()

    # Load split file
    with open(args.input_split, 'r') as f:
        split_data = json.load(f)

    train_idx = split_data['train']
    val_idx = split_data['val']
    test_idx = split_data['test']
    metadata = split_data.get('_metadata', {})

    print(f"Original split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Discover all images and their ages
    image_paths = []
    for age_dir in sorted(os.listdir(args.afad_dir)):
        age_path = os.path.join(args.afad_dir, age_dir)
        if not os.path.isdir(age_path) or not age_dir.isdigit():
            continue
        for gender_dir in os.listdir(age_path):
            gender_path = os.path.join(age_path, gender_dir)
            if not os.path.isdir(gender_path):
                continue
            for fname in os.listdir(gender_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(age_dir, gender_dir, fname))

    print(f"Total images: {len(image_paths)}")

    # Create index to age mapping
    idx_to_age = {}
    for idx, rel_path in enumerate(image_paths):
        age = int(os.path.basename(os.path.dirname(os.path.dirname(rel_path))))
        idx_to_age[idx] = age

    # Filter indices by age range
    def filter_by_age(indices, min_age, max_age):
        return [idx for idx in indices if min_age <= idx_to_age.get(idx, 999) <= max_age]

    train_filtered = filter_by_age(train_idx, args.min_age, args.max_age)
    val_filtered = filter_by_age(val_idx, args.min_age, args.max_age)
    test_filtered = filter_by_age(test_idx, args.min_age, args.max_age)

    print(f"Filtered split (age {args.min_age}-{args.max_age}):")
    print(f"  Train: {len(train_filtered)} (removed {len(train_idx) - len(train_filtered)})")
    print(f"  Val:   {len(val_filtered)} (removed {len(val_idx) - len(val_filtered)})")
    print(f"  Test:  {len(test_filtered)} (removed {len(test_idx) - len(test_filtered)})")

    # Verify no identity leakage in filtered split
    def get_ids(indices):
        ids = set()
        for idx in indices:
            fname = os.path.basename(image_paths[idx])
            ids.add(fname.split('-')[0])
        return ids

    train_ids = get_ids(train_filtered)
    val_ids = get_ids(val_filtered)
    test_ids = get_ids(test_filtered)

    assert len(train_ids & val_ids) == 0, f"Identity leakage: {len(train_ids & val_ids)} IDs in both train and val"
    assert len(train_ids & test_ids) == 0, f"Identity leakage: {len(train_ids & test_ids)} IDs in both train and test"
    assert len(val_ids & test_ids) == 0, f"Identity leakage: {len(val_ids & test_ids)} IDs in both val and test"
    print("✅ No identity leakage detected")

    # Update metadata
    metadata['num_samples'] = len(train_filtered) + len(val_filtered) + len(test_filtered)
    metadata['min_age'] = args.min_age
    metadata['max_age'] = args.max_age
    metadata['num_classes'] = args.max_age - args.min_age + 1
    metadata['split_ratios_approx'] = [
        round(len(train_filtered) / metadata['num_samples'], 4),
        round(len(val_filtered) / metadata['num_samples'], 4),
        round(len(test_filtered) / metadata['num_samples'], 4),
    ]

    # Save filtered split
    filtered_split = {
        'train': train_filtered,
        'val': val_filtered,
        'test': test_filtered,
        '_metadata': metadata,
    }

    with open(args.output_split, 'w') as f:
        json.dump(filtered_split, f)

    print(f"💾 Saved to {args.output_split}")

if __name__ == "__main__":
    main()
