import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbones import build_backbone
except ImportError:
    from backbones import build_backbone


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class BottleneckSPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x):
        pooled = [x, self.pool1(x), self.pool2(x), self.pool3(x)]
        return self.project(torch.cat(pooled, dim=1))


class LightweightAgeEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_multi_scale = bool(getattr(config, "use_multi_scale", False))
        self.use_spp = bool(getattr(config, "use_spp", False))
        configured_msff_indices = tuple(getattr(config, "msff_feature_indices", (6, 12)))

        dropout = getattr(config, "dropout", 0.2)
        num_classes = config.num_classes

        self.backbone = build_backbone(config)
        self.last_channel = self.backbone.out_channels

        if getattr(config, "use_hybrid_attention", True):
            print("[Model] Hybrid Attention: ENABLED")
            replaced = self.backbone.replace_late_se_with_attention(
                lambda channels: CoordAtt(channels, channels, reduction=16),
                count=getattr(config, "hybrid_attention_blocks", 4),
            )
            self.hybrid_attention_replaced_blocks = tuple(replaced)
            config.hybrid_attention_replaced_blocks = self.hybrid_attention_replaced_blocks
            if not replaced:
                print("[Model] Backbone has no replaceable SE blocks; HA is skipped for this backbone.")
        else:
            print("[Model] Hybrid Attention: DISABLED")
            self.hybrid_attention_replaced_blocks = ()
            config.hybrid_attention_replaced_blocks = ()

        if self.use_multi_scale:
            print("[Model] Multi-Scale Fusion: ENABLED")
            self.feature_spec = self.backbone.infer_feature_spec(config.img_size, configured_msff_indices)
            self.msff_indices = (self.feature_spec.shallow_index, self.feature_spec.mid_index)
            config.effective_msff_feature_indices = self.msff_indices
            config.effective_msff_channels = (self.feature_spec.shallow_channels, self.feature_spec.mid_channels)
            config.effective_msff_spatial = (self.feature_spec.shallow_spatial, self.feature_spec.mid_spatial)
            config.effective_deep_channels = self.feature_spec.out_channels
            fusion_dim = getattr(config, "fusion_dim", 64)
            fusion_out_dim = getattr(config, "fusion_out_dim", 128)

            self.project_shallow = nn.Sequential(
                nn.Conv2d(self.feature_spec.shallow_channels, fusion_dim, 1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )
            self.project_mid = nn.Sequential(
                nn.Conv2d(self.feature_spec.mid_channels, fusion_dim, 1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True),
            )
            self.fusion_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.fusion_out = nn.Sequential(
                nn.Conv2d(fusion_dim, fusion_out_dim, 1, bias=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:
            print("[Model] Multi-Scale Fusion: DISABLED")
            self.feature_spec = None
            self.msff_indices = ()
            config.effective_msff_feature_indices = ()
            config.effective_msff_channels = ()
            config.effective_msff_spatial = ()
            config.effective_deep_channels = self.last_channel
            fusion_out_dim = 0

        if self.use_spp:
            print("[Model] SPP: ENABLED")
            self.spp_channels = getattr(config, "spp_channels", 512)
            self.spp_module = BottleneckSPP(self.last_channel, self.spp_channels)
            semantic_dim = self.spp_channels
        else:
            print("[Model] SPP: DISABLED")
            semantic_dim = getattr(config, "semantic_dim", 1280)
            self.semantic_projector = nn.Sequential(
                nn.Linear(self.last_channel, semantic_dim),
                nn.Hardswish(),
            )

        classifier_input_dim = semantic_dim + (fusion_out_dim if self.use_multi_scale else 0)
        self.final_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def _fuse_multi_scale(self, captured):
        shallow = captured[self.msff_indices[0]]
        mid = captured[self.msff_indices[1]]

        f_shallow = self.project_shallow(shallow)
        f_mid = self.project_mid(mid)
        f_shallow = F.adaptive_avg_pool2d(f_shallow, f_mid.shape[-2:])

        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * f_shallow + weights[1] * f_mid
        return self.fusion_out(fused)

    def forward(self, x):
        capture_indices = self.msff_indices if self.use_multi_scale else ()
        deep_feature, captured = self.backbone.forward_features(x, capture_indices=capture_indices)

        if self.use_spp:
            semantic = self.spp_module(deep_feature)
            semantic = F.adaptive_avg_pool2d(semantic, (1, 1)).flatten(1)
        else:
            semantic = self.semantic_projector(self.backbone.pool_features(deep_feature))

        if self.use_multi_scale:
            texture = self._fuse_multi_scale(captured)
            semantic = torch.cat([semantic, texture], dim=1)

        return self.final_head(semantic)
