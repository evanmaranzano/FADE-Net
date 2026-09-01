"""
DCSR: Distribution-Conditioned Scale Routing
CGBR: Correction-Need Guided Bounded Residual Refinement

Core innovation modules for FADE-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionStatistics(nn.Module):
    """Extract distribution statistics: expectation, entropy, variance, skewness, boundary mass."""

    def __init__(self, min_age=0, max_age=80):
        super().__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.num_classes = max_age - min_age + 1
        # Register age values as buffer
        ages = torch.arange(min_age, max_age + 1, dtype=torch.float32)
        self.register_buffer('ages', ages)

    def forward(self, probs):
        """
        Args:
            probs: (B, K) probability distribution
        Returns:
            stats: (B, 5) [normalized_expectation, normalized_entropy, normalized_variance, abs_skewness, boundary_mass]
        """
        ages = self.ages.to(probs.device)
        # Expectation
        mu = (probs * ages).sum(dim=1, keepdim=True)  # (B, 1)

        # Normalized expectation
        norm_expectation = (mu - self.min_age) / (self.max_age - self.min_age)

        # Normalized entropy
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.num_classes), device=probs.device))
        norm_entropy = entropy / max_entropy

        # Normalized variance
        variance = ((ages - mu) ** 2 * probs).sum(dim=1, keepdim=True)
        norm_variance = variance / (self.max_age - self.min_age) ** 2

        # Skewness
        std = torch.sqrt(variance + 1e-8)
        skewness = (((ages - mu) / std) ** 3 * probs).sum(dim=1, keepdim=True)
        abs_skewness = torch.abs(skewness)

        # Boundary probability mass (3 bins from each end)
        left_mass = probs[:, :3].sum(dim=1, keepdim=True)
        right_mass = probs[:, -3:].sum(dim=1, keepdim=True)
        boundary_mass = left_mass + right_mass

        return torch.cat([norm_expectation, norm_entropy, norm_variance, abs_skewness, boundary_mass], dim=1)


class DistributionEncoder(nn.Module):
    """Compress full distribution into a compact embedding."""

    def __init__(self, num_classes=81, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.Hardswish(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, probs):
        return self.mlp(probs)


class FeatureAdapter(nn.Module):
    """Align features to unified channel dimension with spatial resize."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish()

    def forward(self, x, target_size):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.dwconv(x)
        x = self.bn2(x)
        return self.act(x)


class CoarseDistributionHead(nn.Module):
    """Generate coarse age distribution from deep features."""

    def __init__(self, in_channels, num_classes=81, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        return logits


class DCSR(nn.Module):
    """Distribution-Conditioned Scale Routing module."""

    def __init__(self, fusion_channels=96, route_groups=8, num_classes=81,
                 embed_dim=16, min_age=0, max_age=80):
        super().__init__()
        self.fusion_channels = fusion_channels
        self.route_groups = route_groups
        self.embed_dim = embed_dim

        # Distribution statistics extractor
        self.dist_stats = DistributionStatistics(min_age, max_age)

        # Distribution encoder
        self.dist_encoder = DistributionEncoder(num_classes, embed_dim)

        # Routing MLP: input = GAP(deep_features) + distribution_descriptor
        route_input_dim = fusion_channels + 5 + embed_dim
        self.route_mlp = nn.Sequential(
            nn.Linear(route_input_dim, 128),
            nn.Hardswish(),
            nn.Linear(128, route_groups * 3),
        )

        # Output fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.Hardswish(),
        )

    def forward(self, f1, f2, f3, coarse_probs):
        """
        Args:
            f1: (B, C, H, W) shallow features (aligned)
            f2: (B, C, H, W) mid features (aligned)
            f3: (B, C, H, W) deep features (aligned)
            coarse_probs: (B, K) coarse age distribution
        Returns:
            fused: (B, C, H, W) fused features
            route_weights: (B, G, 3) routing weights
        """
        B, C, H, W = f2.shape
        target_size = (H, W)

        # Resize f1, f3 to match f2 spatial size
        f1_resized = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f3_resized = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)

        # Stack features: (B, 3, C, H, W)
        features = torch.stack([f1_resized, f2, f3_resized], dim=1)

        # Stop gradients into the coarse head, but keep the encoder trainable.
        coarse_probs = coarse_probs.detach()
        stats = self.dist_stats(coarse_probs)  # (B, 5)
        embed = self.dist_encoder(coarse_probs)  # (B, embed_dim)
        dist_desc = torch.cat([stats, embed], dim=1)  # (B, 5 + embed_dim)

        # GAP of deep features
        gap = F.adaptive_avg_pool2d(f3, 1).flatten(1)  # (B, C)

        # Route input
        route_input = torch.cat([gap, dist_desc], dim=1)  # (B, C + 5 + embed_dim)

        # Generate routing weights
        route_logits = self.route_mlp(route_input)  # (B, G*3)
        route_logits = route_logits.view(B, self.route_groups, 3)  # (B, G, 3)
        route_weights = F.softmax(route_logits, dim=2)  # (B, G, 3)

        # Grouped routing
        # Reshape features for grouped operation
        features_grouped = features.view(B, 3, self.route_groups, C // self.route_groups, H, W)
        # features_grouped: (B, 3, G, C//G, H, W)

        # Weighted sum per group
        # route_weights: (B, G, 3) -> (B, 3, G, 1, 1, 1)
        weights = route_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Fuse: (B, G, C//G, H, W)
        fused_grouped = (features_grouped * weights).sum(dim=1)

        # Concatenate groups: (B, C, H, W)
        fused = fused_grouped.view(B, C, H, W)

        # Output conv
        fused = self.fusion_conv(fused)

        return fused, route_weights


class CGBR(nn.Module):
    """Correction-Need Guided Bounded Residual Refinement module."""

    def __init__(self, in_channels, num_classes=81, embed_dim=16,
                 residual_bound=3.0, gate_error_scale=3.0,
                 min_age=0, max_age=80):
        super().__init__()
        self.residual_bound = residual_bound
        self.gate_error_scale = gate_error_scale
        self.min_age = min_age
        self.max_age = max_age

        # Distribution statistics extractor
        self.dist_stats = DistributionStatistics(min_age, max_age)

        # Gate input: 5 stats + embed_dim
        gate_input_dim = 5 + embed_dim

        # Correction-need gate
        self.gate_head = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.Hardswish(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Residual head: input = GAP(fused_features) + distribution_descriptor
        residual_input_dim = in_channels + gate_input_dim
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, 128),
            nn.Hardswish(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # Distribution encoder for main distribution (separate from coarse)
        self.main_dist_encoder = DistributionEncoder(num_classes, embed_dim)

    def forward(self, fused_features, main_probs, base_age=None):
        """
        Args:
            fused_features: (B, C, H, W) from DCSR
            main_probs: (B, K) main age distribution
            base_age: (B,) base age from distribution expectation (optional, for inference)
        Returns:
            gate: (B, 1) correction gate value
            residual: (B, 1) bounded residual
            refined_age: (B,) final refined age
        """
        # Stop gradients into the main distribution head, but keep the encoder trainable.
        main_probs = main_probs.detach()
        stats = self.dist_stats(main_probs)  # (B, 5)
        embed = self.main_dist_encoder(main_probs)  # (B, embed_dim)
        dist_desc = torch.cat([stats, embed], dim=1)  # (B, 5 + embed_dim)

        # Correction-need gate
        gate = self.gate_head(dist_desc)  # (B, 1)

        # GAP of fused features
        gap = F.adaptive_avg_pool2d(fused_features, 1).flatten(1)  # (B, C)

        # Residual input
        residual_input = torch.cat([gap, dist_desc], dim=1)  # (B, C + 5 + embed_dim)

        # Bounded residual
        residual = self.residual_bound * self.residual_head(residual_input)  # (B, 1)

        # Base age (expectation of main distribution)
        if base_age is None:
            ages = torch.arange(self.min_age, self.max_age + 1,
                               dtype=main_probs.dtype, device=main_probs.device)
            base_age = (main_probs * ages).sum(dim=1)  # (B,)

        # Refined age
        refined_age = base_age + gate.squeeze(1) * residual.squeeze(1)
        refined_age = torch.clamp(refined_age, self.min_age, self.max_age)

        return gate, residual, refined_age, base_age


class FADELoss(nn.Module):
    """Combined loss for FADE-Net with DCSR and CGBR."""

    def __init__(self, min_age=0, max_age=80, label_sigma=2.0,
                 lambda_main_kl=1.0, lambda_main_reg=1.0,
                 lambda_coarse=0.3, lambda_refine=0.5, lambda_gate=0.1,
                 gate_error_scale=3.0, lambda_cdf=0.0,
                 label_sigma_by_age=None):
        super().__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.num_classes = max_age - min_age + 1
        self.label_sigma = label_sigma
        self.lambda_main_kl = lambda_main_kl
        self.lambda_main_reg = lambda_main_reg
        self.lambda_coarse = lambda_coarse
        self.lambda_refine = lambda_refine
        self.lambda_gate = lambda_gate
        self.gate_error_scale = gate_error_scale
        self.lambda_cdf = lambda_cdf

        # Age values
        self.register_buffer('ages', torch.arange(min_age, max_age + 1, dtype=torch.float32))
        if label_sigma_by_age is not None:
            label_sigma_by_age = torch.as_tensor(label_sigma_by_age, dtype=torch.float32)
            if label_sigma_by_age.numel() != self.num_classes:
                raise ValueError("label_sigma_by_age must have one value per output age")
            if torch.any(label_sigma_by_age <= 0):
                raise ValueError("label_sigma_by_age values must be positive")
        self.register_buffer('label_sigma_by_age', label_sigma_by_age)

    def _gaussian_label(self, true_age):
        """Create Gaussian label distribution."""
        # true_age: (B,)
        # ages: (K,)
        ages = self.ages.to(true_age.device)
        diff = ages.unsqueeze(0) - true_age.unsqueeze(1)  # (B, K)
        if self.label_sigma_by_age is None:
            sigma = self.label_sigma
        else:
            age_index = torch.round(true_age).long() - self.min_age
            age_index = torch.clamp(age_index, 0, self.num_classes - 1)
            sigma_table = self.label_sigma_by_age.to(true_age.device)
            sigma = sigma_table[age_index].unsqueeze(1)
        log_weights = -0.5 * (diff / sigma) ** 2
        # Normalize
        probs = F.softmax(log_weights, dim=1)
        return probs

    @staticmethod
    def _cdf_distance(pred_probs, target_probs):
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(target_probs, dim=1)
        return torch.mean(torch.sum((pred_cdf - target_cdf) ** 2, dim=1))

    def forward(self, outputs, true_ages, epoch=0, cgbr_start_epoch=16, cgbr_full_epoch=26):
        """
        Args:
            outputs: dict from model forward pass
            true_ages: (B,) ground truth ages
            epoch: current epoch
            cgbr_start_epoch: epoch to start CGBR
            cgbr_full_epoch: epoch to reach full CGBR weights
        """
        # Gaussian label distribution
        target_dist = self._gaussian_label(true_ages)

        # 1. Coarse distribution loss
        coarse_logits = outputs['coarse_logits']
        coarse_log_probs = F.log_softmax(coarse_logits, dim=1)
        coarse_kl = F.kl_div(coarse_log_probs, target_dist, reduction='batchmean')

        # Coarse age expectation
        coarse_probs = F.softmax(coarse_logits, dim=1)
        ages = self.ages.to(coarse_probs.device)
        coarse_age = (coarse_probs * ages).sum(dim=1)
        coarse_reg = F.smooth_l1_loss(coarse_age, true_ages)

        loss_coarse = coarse_kl + coarse_reg

        # 2. Main distribution loss
        main_logits = outputs['main_logits']
        main_log_probs = F.log_softmax(main_logits, dim=1)
        main_kl = F.kl_div(main_log_probs, target_dist, reduction='batchmean')

        main_probs = outputs['main_prob']
        base_age = outputs['base_age']
        main_reg = F.smooth_l1_loss(base_age, true_ages)
        main_cdf = self._cdf_distance(main_probs, target_dist)

        loss_main = self.lambda_main_kl * main_kl + self.lambda_main_reg * main_reg + \
                    self.lambda_cdf * main_cdf

        # 3. CGBR losses (ramp up)
        if epoch >= cgbr_start_epoch:
            # Linear ramp from cgbr_start_epoch to cgbr_full_epoch
            ramp = min(1.0, (epoch - cgbr_start_epoch) / (cgbr_full_epoch - cgbr_start_epoch))
            current_lambda_refine = self.lambda_refine * ramp
            current_lambda_gate = self.lambda_gate * ramp

            # Gate supervision
            gate = outputs['gate'].squeeze(1)  # (B,)
            with torch.no_grad():
                gate_target = torch.clamp(
                    torch.abs(base_age - true_ages) / self.gate_error_scale,
                    0.0, 1.0
                )
            loss_gate = F.smooth_l1_loss(gate, gate_target)

            # Refinement loss
            refined_age = outputs['age']
            loss_refine = F.smooth_l1_loss(refined_age, true_ages)

            loss_total = loss_main + self.lambda_coarse * loss_coarse + \
                        current_lambda_refine * loss_refine + current_lambda_gate * loss_gate
        else:
            loss_gate = torch.tensor(0.0, device=true_ages.device)
            loss_refine = torch.tensor(0.0, device=true_ages.device)
            loss_total = loss_main + self.lambda_coarse * loss_coarse

        return {
            'total': loss_total,
            'main_kl': main_kl,
            'main_reg': main_reg,
            'main_cdf': main_cdf,
            'coarse_kl': coarse_kl,
            'coarse_reg': coarse_reg,
            'gate': loss_gate,
            'refine': loss_refine,
        }
