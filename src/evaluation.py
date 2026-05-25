import torch
import torch.nn.functional as F


TTA_MODES = ("raw", "flip", "multi")
TTA_ALIASES = {
    "none": "raw",
    "raw": "raw",
    "flip": "flip",
    "multi": "multi",
}


def normalize_tta_mode(mode: str) -> str:
    normalized = TTA_ALIASES.get(str(mode).lower())
    if normalized is None:
        raise ValueError(f"Unsupported TTA mode: {mode!r}. Expected one of {sorted(TTA_ALIASES)}")
    return normalized


def _resize_for_scale(images, scale: float, base_size: int):
    if scale == 1.0:
        return images

    new_size = int(base_size * scale)
    resized = F.interpolate(images, size=new_size, mode="bilinear", align_corners=False)
    if new_size > base_size:
        start = (new_size - base_size) // 2
        return resized[:, :, start:start + base_size, start:start + base_size]

    pad = (base_size - new_size) // 2
    return F.pad(resized, (pad, base_size - new_size - pad, pad, base_size - new_size - pad), mode="reflect")


def _tta_views(images, mode: str, base_size: int):
    views = []
    scales = (1.0,) if mode == "flip" else (0.9, 1.0, 1.1)
    for scale in scales:
        resized = _resize_for_scale(images, scale, base_size)
        views.append(resized)
        views.append(torch.flip(resized, dims=[3]))
    return views


def _forward_augmented_probs(model, views, max_augmented_batch_size: int | None = None):
    total_size = sum(v.size(0) for v in views)
    if max_augmented_batch_size is None:
        original_batch_size = views[0].size(0)
        max_augmented_batch_size = min(total_size, max(original_batch_size * 2, 16))

    if not isinstance(max_augmented_batch_size, int):
        raise TypeError(f"max_augmented_batch_size must be int or None, got {type(max_augmented_batch_size).__name__}.")
    chunk_size = max_augmented_batch_size
    if chunk_size <= 0:
        raise ValueError("max_augmented_batch_size must be positive when provided.")

    # Collect all logits first, then apply softmax globally for correct normalization.
    all_logits = []
    augmented = torch.cat(views, dim=0)
    for start in range(0, total_size, chunk_size):
        chunk_logits = model(augmented[start:start + chunk_size])
        all_logits.append(chunk_logits)

    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=1)
    return probs.view(len(views), views[0].size(0), -1)


def predict_probs(
    model,
    images,
    mode: str = "multi",
    base_size: int | None = None,
    max_augmented_batch_size: int | None = None,
):
    mode = normalize_tta_mode(mode)
    base_size = base_size or images.shape[-1]

    if mode == "raw":
        return F.softmax(model(images), dim=1)

    views = _tta_views(images, mode, base_size)
    view_probs = _forward_augmented_probs(model, views, max_augmented_batch_size=max_augmented_batch_size)
    return view_probs.mean(dim=0)


def predict_age_with_uncertainty(
    model,
    images,
    mode: str = "multi",
    base_size: int | None = None,
    max_augmented_batch_size: int | None = None,
):
    mode = normalize_tta_mode(mode)
    base_size = base_size or images.shape[-1]

    if mode == "raw":
        probs = F.softmax(model(images), dim=1)
        ages = probs_to_ages(probs, probs.shape[1])
        return probs, ages, torch.zeros_like(ages)

    views = _tta_views(images, mode, base_size)
    view_probs = _forward_augmented_probs(model, views, max_augmented_batch_size=max_augmented_batch_size)
    mean_probs = view_probs.mean(dim=0)
    view_ages = torch.stack([probs_to_ages(probs, probs.shape[1]) for probs in view_probs], dim=0)
    mean_ages = probs_to_ages(mean_probs, mean_probs.shape[1])
    # NOTE: mean_ages is derived from the averaged probability distribution,
    # while age_std is the std of per-view expected ages. Due to softmax
    # nonlinearity, mean_ages != view_ages.mean(dim=0). The std itself is
    # unaffected by this mean shift and remains a valid uncertainty measure.
    # Uses population std (unbiased=False) for small fixed N (2 or 6 views),
    # treating views as the full population rather than a sample.
    age_std = view_ages.std(dim=0, unbiased=False)
    return mean_probs, mean_ages, age_std


def probs_to_ages(probs, num_classes: int):
    rank_arange = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
    return torch.sum(probs * rank_arange, dim=1)


def mae_from_probs(probs, ages, num_classes: int) -> float:
    """Return total absolute error (sum, not mean). Divide by sample count for MAE."""
    pred_ages = probs_to_ages(probs, num_classes)
    return torch.sum(torch.abs(pred_ages - ages)).item()


def evaluate_mae(model, loader, config, device, modes=TTA_MODES, max_batches=None):
    was_training = model.training
    model.train(mode=False)
    normalized_modes = tuple(normalize_tta_mode(mode) for mode in modes)
    mae_sums = {mode: 0.0 for mode in normalized_modes}
    count = 0
    processed_batches = 0

    with torch.no_grad():
        for images, _labels, ages in loader:
            if images.numel() == 0:
                continue

            images = images.to(device)
            ages = ages.to(device)

            for mode in normalized_modes:
                probs = predict_probs(
                    model,
                    images,
                    mode=mode,
                    base_size=config.img_size,
                    max_augmented_batch_size=getattr(config, "tta_batch_size", None),
                )
                mae_sums[mode] += mae_from_probs(probs, ages, config.num_classes)
            count += images.size(0)
            processed_batches += 1
            if max_batches is not None and processed_batches >= max_batches:
                break

    if count == 0:
        if was_training:
            model.train()
        raise RuntimeError("No valid evaluation samples were loaded.")

    if was_training:
        model.train()
    return {mode: mae_sums[mode] / count for mode in normalized_modes}
