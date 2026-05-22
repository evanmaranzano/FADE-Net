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


def predict_probs(model, images, mode: str = "multi", base_size: int | None = None):
    mode = normalize_tta_mode(mode)
    base_size = base_size or images.shape[-1]

    if mode == "raw":
        return F.softmax(model(images), dim=1)

    all_probs = []
    scales = (1.0,) if mode == "flip" else (0.9, 1.0, 1.1)
    for scale in scales:
        resized = _resize_for_scale(images, scale, base_size)
        all_probs.append(F.softmax(model(resized), dim=1))
        flipped = torch.flip(resized, dims=[3])
        all_probs.append(F.softmax(model(flipped), dim=1))

    return torch.stack(all_probs, dim=0).mean(dim=0)


def probs_to_ages(probs, num_classes: int):
    rank_arange = torch.arange(num_classes, device=probs.device, dtype=probs.dtype)
    return torch.sum(probs * rank_arange, dim=1)


def mae_from_probs(probs, ages, num_classes: int) -> float:
    pred_ages = probs_to_ages(probs, num_classes)
    return torch.sum(torch.abs(pred_ages - ages)).item()


def evaluate_mae(model, loader, config, device, modes=TTA_MODES, max_batches=None):
    model.eval()
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
                probs = predict_probs(model, images, mode=mode, base_size=config.img_size)
                mae_sums[mode] += mae_from_probs(probs, ages, config.num_classes)
            count += images.size(0)
            processed_batches += 1
            if max_batches is not None and processed_batches >= max_batches:
                break

    if count == 0:
        raise RuntimeError("No valid evaluation samples were loaded.")

    return {mode: mae_sums[mode] / count for mode in normalized_modes}
