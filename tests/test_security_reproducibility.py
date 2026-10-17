"""
Unit tests for security policies, path traversal defense, and seed reproducibility.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
import pandas as pd
from PIL import Image

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset


def test_seed_reproducibility():
    """Verify that set_seed guarantees deterministic numbers in torch and numpy."""
    set_seed(42)
    t1 = torch.randn(10).numpy()
    n1 = np.random.rand(10)

    set_seed(42)
    t2 = torch.randn(10).numpy()
    n2 = np.random.rand(10)

    assert np.allclose(t1, t2), "Torch random generation was not deterministic"
    assert np.allclose(n1, n2), "NumPy random generation was not deterministic"


def test_path_traversal_prevention(tmp_path):
    """Verify that malicious relative paths outside dataset_root raise ValueError."""
    fake_df = pd.DataFrame([{
        "relative_path": "../../Windows/System32/notepad.exe",
        "view_idx": 0,
        "quality_grade": 5.0,
        "view_class": "A4C",
        "filename": "notepad.exe",
        "sequence_group": "A4C_fake",
    }])

    dataset = CardiacUltrasoundDataset(fake_df, dataset_root=tmp_path)
    with pytest.raises(ValueError, match="Path traversal detected"):
        _ = dataset[0]


def test_no_hardcoded_secrets_in_repo():
    """Ensures no hardcoded passwords, tokens, or private keys exist in source code."""
    project_root = default_config.paths.project_root
    suspicious_patterns = ["password=", "secret=", "api_key=", "private_key", "bearer "]

    for py_file in project_root.rglob("*.py"):
        if ".git" in py_file.parts or ".pytest_cache" in py_file.parts or ".venv" in py_file.parts:
            continue
        if py_file.name == "test_security_reproducibility.py":
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        for pat in suspicious_patterns:
            assert pat not in content, f"Potential credential leak found in {py_file}: '{pat}'"
