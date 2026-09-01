"""
FADE-Net: Distribution-Conditioned Scale Routing + Correction-Need Guided Bounded Residual Refinement

MobileNetV4-Conv-Small + DCSR + Age Distribution Learning + CGBR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbones import build_backbone
    from .experiment import set_derived_attrs
    from .dcsr_cgbr import FeatureAdapter, CoarseDistributionHead, DCSR, CGBR
except ImportError:
    from backbones import build_backbone
    from experiment import set_derived_attrs
    from dcsr_cgbr import FeatureAdapter, CoarseDistributionHead, DCSR, CGBR


class FADENet(nn.Module):
    """
    FADE-Net with DCSR and CGBR.

    Architecture:
        MobileNetV4-Conv-Small -> F1, F2, F3 (multi-scale features)
        F3 -> Coarse Distribution Head -> p_coarse
        DCSR(F1, F2, F3, p_coarse) -> F_fuse
        F_fuse -> Main Distribution Head -> p_main -> mu (base age)
        CGBR(F_fuse, p_main) -> gate, residual -> refined age
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Age range
        self.min_age = config.min_age
        self.max_age = config.max_age
        self.num_classes = config.num_classes

        # Build backbone
        self.backbone = build_backbone(config)
        self.backbone_channels = self.backbone.out_channels

        # Get feature info for multi-scale extraction
        self.feature_indices = self._get_feature_indices(config)

        # Feature adapters for alignment
        fusion_channels = getattr(config, 'fusion_channels', 96)
        self.adapters = nn.ModuleList([
            FeatureAdapter(ch, fusion_channels)
            for ch in self.feature_channels
        ])

        # Coarse distribution head
        coarse_head_dim = getattr(config, 'coarse_head_dim', 128)
        self.coarse_head = CoarseDistributionHead(
            self.feature_channels[-1],  # Use deepest features
            self.num_classes,
            coarse_head_dim,
        )

        # DCSR
        route_groups = getattr(config, 'route_groups', 8)
        embed_dim = getattr(config, 'distribution_embed_dim', 16)
        self.dcsr = DCSR(
            fusion_channels, route_groups, self.num_classes,
            embed_dim, self.min_age, self.max_age,
        )

        # Main distribution head
        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fusion_channels, 128),
            nn.Hardswish(),
            nn.Linear(128, self.num_classes),
        )

        # CGBR
        self.use_cgbr = getattr(config, 'use_cgbr', True)
        if self.use_cgbr:
            residual_bound = getattr(config, 'residual_bound', 3.0)
            gate_error_scale = getattr(config, 'gate_error_scale', 3.0)
            self.cgbr = CGBR(
                fusion_channels, self.num_classes, embed_dim,
                residual_bound, gate_error_scale,
                self.min_age, self.max_age,
            )

        # Age values buffer
        self.register_buffer('ages', torch.arange(
            self.min_age, self.max_age + 1, dtype=torch.float32
        ))

        # Print architecture info
        self._print_architecture()

    def _get_feature_indices(self, config):
        """Probe the selected backbone instead of assuming Small-stage channels."""
        configured_indices = tuple(
            getattr(config, 'msff_feature_indices', (1, 3))
        )
        spec = self.backbone.infer_feature_spec(
            getattr(config, 'img_size', 256), configured_indices
        )
        self.feature_channels = [
            spec.shallow_channels,
            spec.mid_channels,
            spec.out_channels,
        ]
        return [spec.shallow_index, spec.mid_index]

    def _print_architecture(self):
        """Print architecture summary."""
        print("=" * 60)
        print("🎯 FADE-Net Architecture")
        print("=" * 60)
        print(
            f"  Backbone: {getattr(self.config, 'backbone_source', 'unknown')}/"
            f"{getattr(self.config, 'backbone_name', type(self.backbone).__name__)}"
        )
        print(f"  Age Range: {self.min_age}-{self.max_age} ({self.num_classes} classes)")
        print(f"  Feature Indices: {self.feature_indices}")
        print(f"  Feature Channels: {self.feature_channels}")
        print(f"  Fusion Channels: {self.adapters[0].conv1x1.out_channels}")
        print(f"  DCSR: ENABLED (route_groups={self.dcsr.route_groups})")
        print(f"  CGBR: {'ENABLED' if self.use_cgbr else 'DISABLED'}")
        if self.use_cgbr:
            print(f"    Residual Bound: {self.cgbr.residual_bound}")
            print(f"    Gate Error Scale: {self.cgbr.gate_error_scale}")
        print("=" * 60)

    def extract_features(self, x):
        """Extract multi-scale features from backbone."""
        # Use backbone's forward_features with capture_indices
        # Returns (deep_feature, captured_dict)
        deep, captured = self.backbone.forward_features(
            x, capture_indices=self.feature_indices
        )
        f1 = captured[self.feature_indices[0]]
        f2 = captured[self.feature_indices[1]]
        f3 = deep
        return [f1, f2, f3]

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input image
            return_features: if True, return intermediate features for analysis
        """
        # 1. Extract multi-scale features
        features = self.extract_features(x)
        f1, f2, f3 = features[0], features[1], features[2]

        # 2. Coarse age distribution
        coarse_logits = self.coarse_head(f3)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        ages = self.ages.to(coarse_probs.device)
        coarse_age = (coarse_probs * ages).sum(dim=1)

        # 3. Feature alignment
        target_size = f2.shape[2:]  # Use mid-level spatial size
        f1_aligned = self.adapters[0](f1, target_size)
        f2_aligned = self.adapters[1](f2, target_size)
        f3_aligned = self.adapters[2](f3, target_size)

        # 4. DCSR: Distribution-Conditioned Scale Routing
        fused, route_weights = self.dcsr(f1_aligned, f2_aligned, f3_aligned, coarse_probs)

        # 5. Main age distribution
        main_logits = self.main_head(fused)
        main_probs = F.softmax(main_logits, dim=1)
        ages = self.ages.to(main_probs.device)
        base_age = (main_probs * ages).sum(dim=1)

        # 6. CGBR: Correction-Need Guided Bounded Residual Refinement
        if self.use_cgbr:
            gate, residual, refined_age, _ = self.cgbr(fused, main_probs, base_age)
        else:
            gate = torch.ones(x.size(0), 1, device=x.device)
            residual = torch.zeros(x.size(0), 1, device=x.device)
            refined_age = base_age

        outputs = {
            'coarse_logits': coarse_logits,
            'coarse_prob': coarse_probs,
            'coarse_age': coarse_age,
            'main_logits': main_logits,
            'main_prob': main_probs,
            'base_age': base_age,
            'gate': gate,
            'residual': residual,
            'age': refined_age,
            'route_weights': route_weights,
        }

        if return_features:
            outputs['features'] = {
                'f1': f1, 'f2': f2, 'f3': f3,
                'f1_aligned': f1_aligned,
                'f2_aligned': f2_aligned,
                'f3_aligned': f3_aligned,
                'fused': fused,
            }

        return outputs

    def get_params_groups(self, backbone_lr, head_lr):
        """Get parameter groups for differential learning rate."""
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]
