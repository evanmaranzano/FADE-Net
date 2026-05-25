import torch
from src.config import Config
from src.model import LightweightAgeEstimator, FrequencyDomainAttention


def test_freq_attention_shape():
    """Freq attention should preserve input shape."""
    attn = FrequencyDomainAttention(512, reduction=16)
    x = torch.randn(2, 512, 7, 7)
    out = attn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_freq_attention_starts_as_identity_gate():
    """Enabling FREQ should not initially rescale pretrained semantic features."""
    attn = FrequencyDomainAttention(8, reduction=16)
    x = torch.randn(2, 8, 7, 7)

    out = attn(x)

    assert torch.allclose(out, x, atol=1e-6)


def test_freq_attention_in_model():
    """Model with freq attention should produce correct output."""
    cfg = Config()
    cfg.backbone_source = "torchvision"
    cfg.backbone_name = "mobilenet_v3_large"
    cfg.backbone_pretrained = False
    cfg.use_freq_attention = True
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)


def test_freq_attention_disabled():
    """Model without freq attention should work normally."""
    cfg = Config()
    cfg.backbone_source = "torchvision"
    cfg.backbone_name = "mobilenet_v3_large"
    cfg.backbone_pretrained = False
    cfg.use_freq_attention = False
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)
