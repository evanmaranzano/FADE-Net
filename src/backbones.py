import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackboneFeatureSpec:
    shallow_index: int
    mid_index: int
    shallow_channels: int
    mid_channels: int
    out_channels: int
    shallow_spatial: tuple[int, int]
    mid_spatial: tuple[int, int]
    deep_spatial: tuple[int, int]
    feature_count: int


class FeatureBackbone(nn.Module):
    out_channels: int
    pretrained_loaded: bool = False

    def forward_features(self, x, capture_indices=()):
        raise NotImplementedError

    def pool_features(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def replace_late_se_with_attention(self, make_attention: Callable[[int], nn.Module], count: int = 4):
        return []

    def normalize_feature_indices(self, feature_indices: tuple[int, int]) -> tuple[int, int]:
        return feature_indices

    def infer_feature_spec(self, img_size: int, feature_indices: tuple[int, int]) -> BackboneFeatureSpec:
        feature_indices = self.normalize_feature_indices(feature_indices)
        if len(feature_indices) != 2:
            raise ValueError(f"MSFF expects exactly two feature indices, got {feature_indices!r}.")

        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            probe = torch.zeros(1, 3, img_size, img_size, device=device)
            with torch.no_grad():
                deep, captured = self.forward_features(probe, capture_indices=feature_indices)
        finally:
            if was_training:
                self.train()

        missing = [idx for idx in feature_indices if idx not in captured]
        if missing:
            raise ValueError(f"Backbone did not expose requested MSFF features: {missing}")

        shallow = captured[feature_indices[0]]
        mid = captured[feature_indices[1]]
        return BackboneFeatureSpec(
            shallow_index=feature_indices[0],
            mid_index=feature_indices[1],
            shallow_channels=shallow.shape[1],
            mid_channels=mid.shape[1],
            out_channels=deep.shape[1],
            shallow_spatial=tuple(shallow.shape[-2:]),
            mid_spatial=tuple(mid.shape[-2:]),
            deep_spatial=tuple(deep.shape[-2:]),
            feature_count=self.feature_count,
        )


class TorchvisionMobileNetV3Backbone(FeatureBackbone):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.pretrained_loaded = pretrained
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            self.model = mobilenet_v3_large(weights=weights)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Failed to load requested ImageNet weights for MobileNetV3-Large. "
                "Use --no_pretrained for an explicit scratch run."
            ) from exc

        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.feature_count = len(self.features)
        self.out_channels = self.model.classifier[0].in_features
        self.model.classifier = nn.Identity()

    def forward_features(self, x, capture_indices=()):
        capture_indices = set(capture_indices)
        captured = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in capture_indices:
                captured[idx] = x
        return x, captured

    def pool_features(self, x):
        return self.avgpool(x).flatten(1)

    def replace_late_se_with_attention(self, make_attention: Callable[[int], nn.Module], count: int = 4):
        se_blocks = []
        for idx, block in enumerate(self.features):
            if not hasattr(block, "block") or not isinstance(block.block, nn.Sequential):
                continue
            for module in block.block:
                if "SqueezeExcitation" in str(type(module)):
                    se_blocks.append(idx)
                    break

        replaced = []
        for idx in se_blocks[-count:]:
            block = self.features[idx]
            for module_idx, module in enumerate(block.block):
                if "SqueezeExcitation" not in str(type(module)):
                    continue
                if not hasattr(module, "fc1"):
                    continue
                channels = module.fc1.in_channels
                block.block[module_idx] = make_attention(channels)
                replaced.append(idx)
                break
        return replaced


class TimmFeatureBackbone(FeatureBackbone):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "Using backbone_source='timm' requires installing timm. "
                "Run: pip install timm"
            ) from exc

        self.model_name = model_name
        self.pretrained_loaded = pretrained
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load requested timm pretrained weights for {model_name}. "
                "Use --no_pretrained for an explicit scratch run."
            ) from exc
        channels = list(self.model.feature_info.channels())
        if not channels:
            raise ValueError(f"timm backbone {model_name!r} did not expose feature_info channels.")
        self.feature_count = len(channels)
        self.out_channels = channels[-1]

    def forward_features(self, x, capture_indices=()):
        capture_indices = set(capture_indices)
        features = self.model(x)
        captured = {idx: features[idx] for idx in capture_indices if idx < len(features)}
        return features[-1], captured

    def normalize_feature_indices(self, feature_indices: tuple[int, int]) -> tuple[int, int]:
        if len(feature_indices) != 2:
            return feature_indices
        if min(feature_indices) < 0 or max(feature_indices) >= self.feature_count:
            raise ValueError(
                f"Requested MSFF indices {feature_indices} exceed timm "
                f"feature_count={self.feature_count}. Configure explicit valid stage indices."
            )
        return feature_indices


def build_backbone(config) -> FeatureBackbone:
    source = getattr(config, "backbone_source", "torchvision")
    name = getattr(config, "backbone_name", "mobilenet_v3_large")
    pretrained = bool(getattr(config, "backbone_pretrained", True))

    if source == "torchvision" and name == "mobilenet_v3_large":
        return TorchvisionMobileNetV3Backbone(pretrained=pretrained)
    if source == "timm":
        return TimmFeatureBackbone(name, pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: source={source!r}, name={name!r}")
