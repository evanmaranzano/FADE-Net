"""Session-wide fixtures for deterministic tests."""
import warnings
import random

import numpy as np
import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")


@pytest.fixture(autouse=True)
def _seed_everything():
    """Fix all random seeds before each test for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture(autouse=True)
def _suppress_derived_attr_warnings():
    """Suppress Config derived-attr warnings in test code (tests intentionally construct configs)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Config: directly mutating derived attr")
        yield
