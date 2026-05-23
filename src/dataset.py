import os
import math
import torch
import numpy as np
import json
import random
import hashlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from utils import DLDLProcessor
from config import Config, ROOT_DIR
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter1d
from experiment import optional_sanitize_token


class CachedSplitMetadataMismatchError(ValueError):
    """Raised when a metadata-backed split no longer matches the current dataset."""


class LegacySplitMetadataError(ValueError):
    """Raised when an unverified legacy split is used without explicit approval."""


# ==========================================
# Collate Function
# ==========================================
def my_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def dataset_fingerprint(dataset):
    """Hash dataset order and labels so cached split indices cannot silently drift."""
    digest = hashlib.sha256()
    digest.update(str(len(dataset)).encode("utf-8"))

    datasets = getattr(dataset, "datasets", [dataset])
    for child in datasets:
        image_paths = getattr(child, "image_paths", None)
        ages = getattr(child, "ages", None)
        if image_paths is None or ages is None:
            digest.update(repr(child).encode("utf-8"))
            continue

        for path, age in zip(image_paths, ages):
            norm_path = os.path.normpath(path).replace("\\", "/")
            relative_tail = "/".join(norm_path.split("/")[-3:])
            digest.update(f"{float(age):.4f}|{relative_tail}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_split_indices(train_idx, val_idx, test_idx, dataset_len):
    all_indices = list(train_idx) + list(val_idx) + list(test_idx)
    if len(all_indices) != dataset_len:
        raise ValueError(f"Dataset size mismatch (Stored: {len(all_indices)} vs Current: {dataset_len})")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Cached split contains duplicate indices")
    if all_indices and (min(all_indices) < 0 or max(all_indices) >= dataset_len):
        raise ValueError("Cached split contains out-of-range indices")


def _split_ratios_match(stored, expected):
    if stored is None:
        return True
    if len(stored) != len(expected):
        return False
    return all(abs(float(a) - float(b)) < 1e-8 for a, b in zip(stored, expected))


def _split_metadata(dataset_len, split_ratios, dataset_hash):
    return {
        'version': 2,
        'num_samples': dataset_len,
        'split_ratios': list(split_ratios),
        'dataset_fingerprint': dataset_hash,
        'legacy_upgraded': False,
    }


def _write_split_json(save_path, train_indices, val_indices, test_indices, dataset_len, split_ratios, dataset_hash):
    save_data = {
        '_metadata': _split_metadata(dataset_len, split_ratios, dataset_hash),
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f)


def split_filename_with_tag(filename, split_file_tag):
    tag = optional_sanitize_token(split_file_tag)
    if not tag:
        return filename
    stem, extension = os.path.splitext(filename)
    return f"{stem}_{tag}{extension or '.json'}"

# ==========================================
# 0. Stratified Split Strategy (The "Platinum" Choice)
# ==========================================
def get_stratified_split(
    dataset,
    all_ages,
    split_ratios=(0.80, 0.10, 0.10),
    save_path=None,
    dataset_hash=None,
    allow_legacy_split_upgrade=False,
):
    """
    Perform Stratified Sampling based on age labels.
    Ensures the specified split ratio holds true for *every single age class*.
    """
    if save_path is None:
        save_path = os.path.join(ROOT_DIR, "dataset_split_stratified.json")        
    assert abs(sum(split_ratios) - 1.0) < 1e-5, "Split ratios must sum to 1"
    
    # Check for existing split
    if os.path.exists(save_path):
        print(f"📄 Loading existing stratified split from {save_path}...")
        metadata_backed_split = False
        try:
            with open(save_path, "r") as f:
                indices_dict = json.load(f)
            train_idx = indices_dict['train']
            val_idx = indices_dict['val']
            test_idx = indices_dict['test']
            metadata = indices_dict.get("_metadata", {})
            metadata_backed_split = bool(metadata)
            
            _validate_split_indices(train_idx, val_idx, test_idx, len(dataset))
            if metadata:
                if metadata.get("dataset_fingerprint") != dataset_hash:
                    raise CachedSplitMetadataMismatchError("Dataset fingerprint mismatch")
                if metadata.get("num_samples") != len(dataset):
                    raise CachedSplitMetadataMismatchError("Dataset metadata sample count mismatch")
                if not _split_ratios_match(metadata.get("split_ratios"), split_ratios):
                    raise CachedSplitMetadataMismatchError("Split ratio metadata mismatch")
            else:
                if not allow_legacy_split_upgrade:
                    raise LegacySplitMetadataError(
                        "Legacy split file has no _metadata; refusing to trust it without "
                        "allow_legacy_split_upgrade=True."
                    )
                print("⚠️ Legacy split file has no dataset fingerprint; using size/index validation only.")
                metadata = _split_metadata(len(dataset), split_ratios, dataset_hash)
                metadata["legacy_upgraded"] = True
                indices_dict["_metadata"] = metadata
                with open(save_path, 'w') as f:
                    json.dump(indices_dict, f)
                print("💾 Upgraded legacy split file with dataset fingerprint metadata.")

            print(f"✅ Loaded: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
            return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
        except (CachedSplitMetadataMismatchError, LegacySplitMetadataError):
            raise
        except Exception as e:
            if metadata_backed_split:
                raise RuntimeError(f"Metadata-backed split file is invalid; refusing to regenerate: {e}") from e
            print(f"⚠️ Load failed ({e}), regenerating...")

    ratios_str = '/'.join([f"{int(r*100)}" for r in split_ratios])
    print(f"⚖️ Performing Stratified Sampling ({ratios_str} per age)...")
    
    # Group indices by age
    indices_by_age = defaultdict(list)
    for idx, age in enumerate(all_ages):
        age_int = int(round(age))
        indices_by_age[age_int].append(idx)
        
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Fixed seed for generation
    rng = random.Random(42)
    
    for age, indices in indices_by_age.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        # Remaining goes to test (handles rounding errors)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
        
    print(f"✅ Stratified Split Done.")
    print(f"   Train: {len(train_indices)}")
    print(f"   Val:   {len(val_indices)}")
    print(f"   Test:  {len(test_indices)}")
    
    # Save to JSON
    _write_split_json(save_path, train_indices, val_indices, test_indices, len(dataset), split_ratios, dataset_hash)
    print(f"💾 Split saved to {save_path}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# ==========================================
# 2. AFAD Dataset
# ==========================================
class AFADDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None):
        self.transform = transform
        self.config = config
        self.dldl_proc = DLDLProcessor(config)
        self.image_paths = []
        self.ages = []
        
        if not os.path.exists(root_dir):
            print(f"⚠️ [AFAD] Path not found: {root_dir}")
        else:
            print("⏳ Scanning AFAD...")
            # Sorted ensures deterministic order before shuffling
            for age_folder in sorted(os.listdir(root_dir)):
                age_path = os.path.join(root_dir, age_folder)
                if os.path.isdir(age_path) and age_folder.isdigit():
                    age = int(age_folder)
                    # Use Strict range
                    if age < config.min_age or age > config.max_age:
                        continue
                    
                    for gender_folder in sorted(os.listdir(age_path)):
                        gender_path = os.path.join(age_path, gender_folder)
                        if os.path.isdir(gender_path):
                            for img_name in sorted(os.listdir(gender_path)):
                                if img_name.lower().endswith(('.jpg', '.png')):
                                    self.image_paths.append(os.path.join(gender_path, img_name))
                                    self.ages.append(float(age))
            print(f"✅ AFAD Loaded: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                img_path = self.image_paths[idx]
                age = self.ages[idx]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                label_dist = self.dldl_proc.generate_label_distribution(age)
                return image, label_dist, torch.tensor(age, dtype=torch.float32)
            except Exception as e:
                if attempt < 2:
                    idx = random.randint(0, len(self.image_paths) - 1)
                else:
                    import warnings
                    warnings.warn(f"Failed to load image after 3 attempts (last: {self.image_paths[idx]}): {e}")
                    return None




# ==========================================
# Subset with Transform
# ==========================================
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None, augment_label=False, config=None):
        self.subset = subset
        self.transform = transform
        self.augment_label = augment_label
        self.config = config
        self.dldl_proc = DLDLProcessor(config) if config else None
        
    def __getitem__(self, idx):
        item = self.subset[idx]
        if item is None: return None
        image, label_dist, age = item
        
        # Apply Logic
        if self.transform:
            image = self.transform(image)
            
        # Label Jitter
        if self.augment_label and self.config and getattr(self.config, 'use_sigma_jitter', False):
            # Uniform noise relative to jitter range
            # e.g. [-0.2, 0.2]
            jitter_range = getattr(self.config, 'sigma_jitter', 0.2)
            offset = np.random.uniform(-jitter_range, jitter_range)
            # Re-generate label distribution
            # Note: 'age' is a tensor, we need scalar for logic or handle tensor in dldl
            label_dist = self.dldl_proc.generate_label_distribution(age, sigma_offset=offset)
            
        return image, label_dist, age
        
    def __len__(self):
        return len(self.subset)

# ==========================================
# LDS Weights
# ==========================================
def calculate_lds_weights(ages, config):
    print("⚖️ Calculating LDS Weights...")
    # Use config.lds_sigma if available, else default to 3
    sigma = getattr(config, 'lds_sigma', 3)
    print(f"   -> Smoothing Window (Sigma): {sigma}")
    
    age_counts = Counter(ages)
    hist = np.zeros(config.num_classes)
    for age, count in age_counts.items():
        idx = int(round(age))
        if 0 <= idx < config.num_classes:
            hist[idx] = count
    
    smooth_hist = gaussian_filter1d(hist, sigma=sigma)
    weights = 1.0 / (smooth_hist + 1e-5)
    
    active_mask = hist > 0
    mean_weight = np.mean(weights[active_mask])

    weights = weights / mean_weight

    # 🛡️ Safety Clip: 防止稀缺样本权重过大导致梯度爆炸
    weights = np.clip(weights, 0.0, 10.0)
    print(f"   -> Max Weight: {np.max(weights):.2f}, Mean (Active): {np.mean(weights[active_mask]):.2f}")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(config.device)
    print("✅ LDS Weights Ready.")
    return weights_tensor

# ==========================================
# Safe Random Erasing (Keypoint-Aware)
# ==========================================
class SafeRandomErasing(object):
    """
    Randomly selects a rectangle region in an image and erases its pixels.
    'Safe' variant: Ensures the erased region does not overlap too much with critical face landmarks.
    Since images are canonically aligned (eyes at 35% height), we can define safe zones.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False, config=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.config = config
        
        # Define approximate landmark zones for 224x224 aligned face
        # Eyes center line is ~35% (0.35 * 224 = 78)
        # Eyes are roughly at x=0.32 and x=0.68? No, alignment centered them.
        # Let's say Critical Zone is the central band.
        # We try to avoid completely covering the "Central T-Zone"
        
    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img

        # img is Tensor [C, H, W]
        if self.inplace:
            _img = img
        else:
            _img = img.clone()
            
        c, img_h, img_w = _img.shape
        area = img_h * img_w

        # Max retries to find a safe spot
        for attempt in range(20):
            target_area = (torch.rand(1).item() * (self.scale[1] - self.scale[0]) + self.scale[0]) * area
            aspect_ratio = torch.rand(1).item() * (self.ratio[1] - self.ratio[0]) + self.ratio[0]

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                i = torch.randint(0, img_h - h + 1, (1,)).item()
                j = torch.randint(0, img_w - w + 1, (1,)).item()
                
                # Check Safety: Avoid obliterating the eyes/mouth completely
                # Simple Heuristic: 
                # Eyes roughly at y=0.35h. Mouth at y=0.75h.
                # Center x=0.5w.
                
                # Let's define "Critical Points"
                # Left Eye ~ (0.32w, 0.35h), Right Eye ~ (0.68w, 0.35h)
                # Nose ~ (0.5w, 0.55h)
                # Mouth ~ (0.5w, 0.75h)
                
                crit_pts = [
                    (0.32 * img_w, 0.35 * img_h), # L Eye
                    (0.68 * img_w, 0.35 * img_h), # R Eye
                    (0.50 * img_w, 0.55 * img_h), # Nose
                    (0.50 * img_w, 0.75 * img_h)  # Mouth
                ]
                
                # Count how many critical points are inside the erase box
                pts_covered = 0
                for (cx, cy) in crit_pts:
                    if i <= cy < i + h and j <= cx < j + w:
                        pts_covered += 1
                        
                # Policy: Don't cover more than 2 critical points? 
                # Or just don't cover BOTH eyes.
                # Let's be strict: Allow max 1 critical point covered.
                if pts_covered <= 1:
                    # Valid spot!
                    if self.value == 'random':
                        v = torch.empty([c, h, w], dtype=torch.float32).normal_()
                    else:
                        v = torch.tensor(self.value, dtype=torch.float32)
                        
                    _img[:, i:i+h, j:j+w] = v
                    return _img

        # If failed to find safe spot after retries, return original (or unsafe fallback)
        print("⚠️ SafeRandomErasing: Failed to find safe spot (20 attempts). Skipping.")
        return _img

# ==========================================
# Main: Get DataLoaders
# ==========================================
def get_dataloaders(config):
    # Transforms
    # Base Transforms list
    train_transforms_list = [
        # V2 Training uses strong augs.
        # Adjusted: Scale 0.8-1.0 to preserve facial features (wrinkles) better than 0.5.
        transforms.RandomResizedCrop(config.img_size, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),
        
        # Added: Affine (Translation + Shear + Rotation)
        # Merged Rotation (15) into Affine to avoid black borders (Artifacts).
        # Fill with approx ImageNet Mean (124, 116, 104)
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.05, 0.05), 
            scale=(0.9, 1.1), 
            shear=5,
            fill=(124, 116, 104) 
        ),
        
        # Added: Blur for quality robustness
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        
        # transforms.RandomRotation(15), # 🗑️ Removed (Merged into Affine)
        
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # ✅ [Modified] Safe Random Erasing (Keypoint-Aware)
    if getattr(config, 'use_random_erasing', False):
        re_scale = (0.02, 0.15)
        print(f"🛡️ [Aug] Safe Random Erasing: ENABLED (p={config.re_prob}, scale={re_scale})")
        # Custom Safe Erasing
        train_transforms_list.append(
            SafeRandomErasing(
                p=config.re_prob, 
                scale=re_scale, 
                ratio=(0.3, 3.3), 
                value='random',
                config=config
            )
        )
    
    train_transform = transforms.Compose(train_transforms_list)
    
    val_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("=" * 60)
    print(f"🚀 Loading Dataset (AFAD)")
    print("=" * 60)
    
    all_datasets = []
    all_ages = []
    
    # 1. AFAD
    if getattr(config, 'use_afad', True) and hasattr(config, 'afad_dir') and os.path.exists(config.afad_dir):
        afad = AFADDataset(config.afad_dir, config=config)
        if len(afad) > 0:
            all_datasets.append(afad)
            all_ages.extend(afad.ages)
            print(f"✅ [Dataset] Added AFAD ({len(afad)} images)")
            
    if not all_datasets:
        raise ValueError("No datasets found! Check config paths.")

    full_dataset = ConcatDataset(all_datasets)
    print(f"\n📦 Total Images: {len(full_dataset)}")
    current_dataset_fingerprint = dataset_fingerprint(full_dataset)
    
    # Stratified Split
    dataset_prefix = "AFAD"
            
    # Determine Ratios based on Protocol
    split_protocol = getattr(config, 'split_protocol', '80-10-10')
    
    if split_protocol == '72-8-20':
        print("⚠️ Using Standard 80-20 Protocol (Train 72% / Val 8% / Test 20%)")
        target_ratios = (0.72, 0.08, 0.20)
        # Use a distinct filename to strictly avoid overwriting the main benchmark split
        split_filename = f"dataset_split_{dataset_prefix}_72_8_20.json"
    elif split_protocol == '80-10-10':
        print("⚖️ Using Balanced 80-10-10 Protocol (Train 80% / Val 10% / Test 10%)")
        target_ratios = (0.80, 0.10, 0.10)
        split_filename = f"dataset_split_{dataset_prefix}_80_10_10.json"
    elif split_protocol == '90-5-5':
        print("📜 Using Legacy 90-5-5 Protocol (Train 90% / Val 5% / Test 5%)")
        target_ratios = (0.90, 0.05, 0.05)
        split_filename = f"dataset_split_{dataset_prefix}_90_5_5.json"
    else:
        # Default 80-10-10
        if split_protocol != '80-10-10':
            print(f"⚠️ Unknown protocol '{split_protocol}', falling back to 80-10-10")
        target_ratios = (0.80, 0.10, 0.10)
        split_filename = f"dataset_split_{dataset_prefix}_80_10_10.json"
        
    split_file_tag = optional_sanitize_token(getattr(config, "split_file_tag", None))
    split_filename = split_filename_with_tag(split_filename, split_file_tag)
    if split_file_tag:
        print(f"🏷️ Split File Tag: {split_file_tag}")
    print(f"📄 Using split file: {split_filename} (Mode: {split_protocol})")
    
    train_subset, val_subset, test_subset = get_stratified_split(
        full_dataset, 
        all_ages, 
        split_ratios=target_ratios,
        save_path=os.path.join(ROOT_DIR, split_filename),
        dataset_hash=current_dataset_fingerprint,
        allow_legacy_split_upgrade=getattr(config, "allow_legacy_split_upgrade", False),
    )
    split_path = os.path.join(ROOT_DIR, split_filename)
    split_metadata = {}
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as f:
            split_metadata = json.load(f).get("_metadata", {})
    config.split_metadata = {
        "split_file": split_filename,
        "split_file_tag": split_file_tag,
        "fingerprint": file_sha256(split_path) if os.path.exists(split_path) else None,
        "dataset_fingerprint": current_dataset_fingerprint,
        "legacy_upgraded": bool(split_metadata.get("legacy_upgraded", False)),
    }

    # LDS Weights based on TRAIN distribution only (avoid val/test leakage)
    class_weights = None
    if getattr(config, 'use_reweighting', False):
        if not hasattr(train_subset, 'indices'):
            raise AttributeError("Train subset must expose 'indices' for LDS weighting.")
        train_ages = [all_ages[i] for i in train_subset.indices]
        print(f"[LDS] Using TRAIN-only age distribution ({len(train_ages)} samples)")
        class_weights = calculate_lds_weights(train_ages, config)
    
    # Apply Transforms
    # Enable Label Augmentation for Train Set
    train_set = SubsetWithTransform(train_subset, transform=train_transform, augment_label=True, config=config)
    val_set = SubsetWithTransform(val_subset, transform=val_transform)
    test_set = SubsetWithTransform(test_subset, transform=val_transform)
    
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.device.type == "cuda",
        "collate_fn": my_collate_fn,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
                              
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
                            
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
                             
    return train_loader, val_loader, test_loader, class_weights
