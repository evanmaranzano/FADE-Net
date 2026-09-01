"""
FaRL ViT-B/16 teacher model (EXP-030).

Minimal CLIP ViT-B/16 visual encoder matching the FaRL-Base-Patch16 checkpoint
(FacePerceiver/FaRL, LAIONFace20M). Module names mirror the checkpoint's
`visual.*` keys so weights load by name. The checkpoint also carries FaRL
auxiliary masked-image-modeling keys (mask_token, ln_lm, lm_transformer,
lm_head) which are not part of feature extraction; they are explicitly
allow-listed as ignorable extras during loading.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# FaRL auxiliary keys that are part of the checkpoint but unused for
# feature extraction (masked image modeling head).
FARL_AUX_KEY_PREFIXES = ("lm_head.", "lm_transformer.", "ln_lm.", "mask_token")


class QuickGELU(nn.Module):
    """OpenAI CLIP MLP activation: x * sigmoid(1.702 * x)."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(width, width * 4)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(width * 4, width)),
        ]))

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + x
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width=768, layers=12, heads=12):
        super().__init__()
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class FaRLVisualEncoder(nn.Module):
    """CLIP ViT-B/16 visual encoder aligned with FaRL `visual.*` keys."""

    def __init__(self, input_resolution=224, patch_size=16, width=768,
                 layers=12, heads=12, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=patch_size, bias=False)
        grid = input_resolution // patch_size
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(
            torch.randn(grid * grid + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim))

    def forward(self, x):
        """Returns the ln_post CLS token (width-dim, before proj)."""
        x = self.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, L, width)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        return x[:, 0]  # CLS token, (B, width)


class FaRLTeacher(nn.Module):
    """FaRL ViT-B/16 visual encoder + linear age-distribution head."""

    def __init__(self, num_classes=81, output_min_age=0):
        super().__init__()
        self.num_classes = num_classes
        self.output_min_age = output_min_age
        self.visual = FaRLVisualEncoder()
        self.head = nn.Linear(768, num_classes)
        self.register_buffer(
            'age_values',
            torch.arange(output_min_age, output_min_age + num_classes,
                         dtype=torch.float32),
        )

    def forward(self, images):
        features = self.visual(images)
        logits = self.head(features)
        probs = F.softmax(logits, dim=1)
        age = (probs * self.age_values.to(probs.device)).sum(dim=1)
        return {
            'main_logits': logits,
            'main_prob': probs,
            'age': age,
            'base_age': age,
        }


def _split_farl_visual_keys(state_dict):
    """Split checkpoint keys into (standard visual keys, aux keys, other)."""
    visual, aux, other = {}, {}, {}
    for key, value in state_dict.items():
        if not key.startswith('visual.'):
            other[key] = value
            continue
        short = key[len('visual.'):]
        if short.startswith(FARL_AUX_KEY_PREFIXES):
            aux[short] = value
        else:
            visual[short] = value
    return visual, aux, other


def load_farl_visual_weights(encoder, state_dict, source='checkpoint'):
    """Load `visual.*` keys into the encoder with strict accounting.

    All encoder parameters must be covered by the checkpoint; every standard
    visual key must be consumed. FaRL auxiliary keys are allow-listed extras.
    Returns a stats dict with consumed/missing/unexpected key counts.
    """
    visual, aux, other = _split_farl_visual_keys(state_dict)
    missing, unexpected = encoder.load_state_dict(visual, strict=False)
    if missing:
        raise KeyError(
            f"FaRL weights from {source} miss encoder keys: {sorted(missing)}"
        )
    if unexpected:
        raise KeyError(
            f"FaRL weights from {source} contain unconsumed visual keys: "
            f"{sorted(unexpected)}"
        )
    stats = {
        'consumed': len(visual),
        'missing': len(missing),
        'aux_ignored': len(aux),
        'non_visual_ignored': len(other),
    }
    logger.info(
        f"FaRL visual weights: consumed={stats['consumed']}, "
        f"missing=0, aux_ignored={stats['aux_ignored']} "
        f"({sorted(aux)}), non_visual_ignored={stats['non_visual_ignored']}"
    )
    return stats


def build_teacher(weights_path=None, num_classes=81, output_min_age=0,
                  map_location='cpu'):
    """Build the FaRL teacher and optionally load pretrained visual weights."""
    model = FaRLTeacher(num_classes=num_classes, output_min_age=output_min_age)
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=map_location,
                                weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        load_farl_visual_weights(model.visual, state_dict,
                                 source=str(weights_path))
    return model
