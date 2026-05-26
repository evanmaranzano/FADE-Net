import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pack_results


def test_pack_results_excludes_weights_by_default(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "final_result_FADE-Net_seed42.txt").write_text("MAE: 3.0", encoding="utf-8")
        (root / "best_model_FADE-Net_seed42.pth").write_bytes(b"weights")
        output = root / "pack.zip"

        monkeypatch.setattr(pack_results, "ROOT_DIR", root)
        pack_results.pack_results(output=output)

        names = zipfile.ZipFile(output).namelist()

    assert "final_result_FADE-Net_seed42.txt" in names
    assert "best_model_FADE-Net_seed42.pth" not in names


def test_pack_results_can_include_weights_explicitly(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "best_model_FADE-Net_seed42.pth").write_bytes(b"weights")
        output = root / "pack.zip"

        monkeypatch.setattr(pack_results, "ROOT_DIR", root)
        pack_results.pack_results(output=output, include_weights=True)

        names = zipfile.ZipFile(output).namelist()

    assert "best_model_FADE-Net_seed42.pth" in names


def test_pack_results_denies_secrets_and_binary_artifacts_in_safe_dirs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        src_dir = root / "src"
        plots_dir = root / "plots" / "seed_42"
        src_dir.mkdir()
        plots_dir.mkdir(parents=True)
        (src_dir / "model.py").write_text("print('ok')", encoding="utf-8")
        (src_dir / ".env").write_text("TOKEN=secret", encoding="utf-8")
        (src_dir / "private.key").write_text("secret", encoding="utf-8")
        (src_dir / "weights.pth").write_bytes(b"weights")
        (plots_dir / "debug.log").write_text("debug", encoding="utf-8")
        (plots_dir / "1_loss_curve.png").write_bytes(b"png")
        (plots_dir / "private_sample.png").write_bytes(b"private")
        output = root / "pack.zip"

        monkeypatch.setattr(pack_results, "ROOT_DIR", root)
        pack_results.pack_results(output=output)

        names = zipfile.ZipFile(output).namelist()

    assert "src/model.py" in names
    assert "plots/seed_42/1_loss_curve.png" in names
    assert "src/.env" not in names
    assert "src/private.key" not in names
    assert "src/weights.pth" not in names
    assert "plots/seed_42/debug.log" not in names
    assert "plots/seed_42/private_sample.png" not in names


def test_pack_results_refuses_to_overwrite_existing_zip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output = root / "pack.zip"
        output.write_bytes(b"existing")

        monkeypatch.setattr(pack_results, "ROOT_DIR", root)

        with pytest.raises(FileExistsError):
            pack_results.pack_results(output=output)

        assert output.read_bytes() == b"existing"


def test_pack_results_skips_symlinks_that_match_safe_patterns(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "root"
        outside = Path(tmpdir) / "outside"
        root.mkdir()
        outside.mkdir()
        outside_secret = outside / "secret.txt"
        outside_secret.write_text("secret", encoding="utf-8")
        link = root / "final_result_FADE-Net_seed42.txt"
        try:
            link.symlink_to(outside_secret)
        except OSError:
            pytest.skip("symlink creation is not available on this Windows environment")
        output = root / "pack.zip"

        monkeypatch.setattr(pack_results, "ROOT_DIR", root)
        pack_results.pack_results(output=output)

        names = zipfile.ZipFile(output).namelist()

    assert "final_result_FADE-Net_seed42.txt" not in names
