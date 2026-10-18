"""
Unit tests for security policies, path traversal defense, and seed reproducibility.
"""

import pytest
import numpy as np

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.paths import resolve_safe_path


def test_seed_reproducibility():
    """Verify that set_seed guarantees deterministic numbers in torch and numpy."""
    torch = pytest.importorskip("torch")
    from self_check_ultrasound.utils.seed import set_seed

    set_seed(42)
    t1 = torch.randn(10).numpy()
    n1 = np.random.rand(10)

    set_seed(42)
    t2 = torch.randn(10).numpy()
    n2 = np.random.rand(10)

    assert np.allclose(t1, t2), "Torch random generation was not deterministic"
    assert np.allclose(n1, n2), "NumPy random generation was not deterministic"


def test_resolve_safe_path_allows_inside_and_blocks_escape(tmp_path):
    root = tmp_path / "root"
    (root / "A4C").mkdir(parents=True)
    assert resolve_safe_path(root, "A4C/x.jpg") == (root / "A4C" / "x.jpg").resolve()
    for bad in ["../outside.jpg", "../../Windows/System32/notepad.exe", "A4C/../../outside.jpg"]:
        with pytest.raises(ValueError, match="Path traversal detected"):
            resolve_safe_path(root, bad)


def test_resolve_safe_path_blocks_sibling_with_shared_prefix(tmp_path):
    """Regression: str.startswith() would wrongly accept '<root>_evil' for root '<root>'."""
    root = tmp_path / "root"
    evil = tmp_path / "root_evil"
    root.mkdir()
    evil.mkdir()
    with pytest.raises(ValueError, match="Path traversal detected"):
        resolve_safe_path(root, "../root_evil/secret.jpg")


def test_resolve_safe_path_blocks_absolute_path(tmp_path):
    with pytest.raises(ValueError, match="Path traversal detected"):
        resolve_safe_path(tmp_path, "/etc/passwd")


def test_dataset_rejects_path_traversal(tmp_path):
    pytest.importorskip("torch")
    import pandas as pd
    from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset

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
    """Ensures no hardcoded passwords, tokens, or private keys exist in Python source."""
    project_root = default_config.paths.project_root
    suspicious_patterns = ["password=", "secret=", "api_key=", "private_key", "bearer "]

    for py_file in project_root.rglob("*.py"):
        if any(p in py_file.parts for p in (".git", ".pytest_cache", ".venv", "venv")):
            continue
        if py_file.name == "test_security_reproducibility.py":
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        for pat in suspicious_patterns:
            assert pat not in content, f"Potential credential leak found in {py_file}: '{pat}'"
