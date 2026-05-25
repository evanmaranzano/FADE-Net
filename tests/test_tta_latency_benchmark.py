import sys
from pathlib import Path
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_tta_latency


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, images):
        self.calls.append(images.size(0))
        return torch.zeros(images.size(0), 3)


def test_tta_latency_benchmark_compares_chunk1_against_batched_default(monkeypatch):
    calls = []

    def fake_measure_latency(**kwargs):
        calls.append(kwargs)
        if kwargs["max_augmented_batch_size"] == 1:
            return 0.06
        if kwargs["max_augmented_batch_size"] is None:
            return 0.02
        raise AssertionError(f"unexpected chunk size: {kwargs['max_augmented_batch_size']!r}")

    monkeypatch.setattr(benchmark_tta_latency, "measure_latency", fake_measure_latency)

    result = benchmark_tta_latency.compare_tta_latency(
        model=FakeModel(),
        images=torch.zeros(1, 3, 4, 4),
        base_size=4,
        warmup=0,
        iters=1,
    )

    assert result["chunk1_latency_ms"] == 60.0
    assert result["batched_latency_ms"] == 20.0
    assert result["latency_improvement_percent"] == 66.66666666666666
    assert result["speedup_x"] == 3.0
    assert [call["max_augmented_batch_size"] for call in calls] == [1, None]
    assert [call["base_size"] for call in calls] == [4, 4]
    assert [call["warmup"] for call in calls] == [0, 0]
    assert [call["iters"] for call in calls] == [1, 1]


def test_tta_latency_cli_uses_no_training_and_reports_improvement(monkeypatch, capsys):
    captured = {}

    class FakeEstimator:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(benchmark_tta_latency, "LightweightAgeEstimator", FakeEstimator)
    def fake_compare_tta_latency(**kwargs):
        captured["compare_kwargs"] = kwargs
        return {
            "chunk1_latency_ms": 60.0,
            "batched_latency_ms": 20.0,
            "latency_improvement_percent": 66.7,
            "speedup_x": 3.0,
        }

    monkeypatch.setattr(benchmark_tta_latency, "compare_tta_latency", fake_compare_tta_latency)
    monkeypatch.setattr(
        benchmark_tta_latency.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_tta_latency.py",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--batch_size",
            "3",
            "--device",
            "cpu",
            "--no_pretrained",
            "--ablation_id",
            "A9",
        ],
    )

    benchmark_tta_latency.main()

    assert captured["cfg"].backbone_pretrained is False
    assert captured["cfg"].use_moe is True
    assert captured["device"].type == "cpu"
    assert captured["compare_kwargs"]["iters"] == 1
    assert captured["compare_kwargs"]["warmup"] == 0
    assert captured["compare_kwargs"]["base_size"] == captured["cfg"].img_size
    assert tuple(captured["compare_kwargs"]["images"].shape) == (3, 3, captured["cfg"].img_size, captured["cfg"].img_size)
    assert captured["compare_kwargs"]["device"].type == "cpu"
    assert "latency_improvement_percent=66.700" in capsys.readouterr().out
