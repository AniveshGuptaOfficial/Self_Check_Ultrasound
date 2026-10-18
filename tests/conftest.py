"""
Shared pytest fixtures.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def make_cone(seed: int, brightness: float = 1.0, size: int = 128) -> Image.Image:
    """Synthetic fan-shaped 'ultrasound' image with speckle on a black background."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((size, size), dtype=np.float32)
    for y in range(int(size * 0.15), int(size * 0.88)):
        half_w = int((y - size * 0.15) * 0.45)
        xs, xe = max(0, size // 2 - half_w), min(size, size // 2 + half_w)
        arr[y, xs:xe] = rng.randint(70, 160, size=xe - xs)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


@pytest.fixture
def echo_image():
    return make_cone(seed=0)


@pytest.fixture
def fake_dataset(tmp_path):
    """
    Builds a tiny on-disk dataset + splits DataFrame.
    2 views x 6 sequence groups x 4 frames; groups 0-3 -> train, 4 -> val, 5 -> test.
    Even frames are high grade (8, normal gain); odd frames are low grade (1, very dim).
    """
    rows = []
    for view_idx, view in enumerate(["A4C", "PL"]):
        (tmp_path / view).mkdir()
        for g in range(6):
            split = "train" if g < 4 else ("val" if g == 4 else "test")
            for f in range(4):
                high = f % 2 == 0
                name = f"{8 if high else 1}_D{14 + g}_frame_{f}_v2.jpg"
                make_cone(seed=view_idx * 1000 + g * 10 + f, brightness=1.0 if high else 0.25).save(
                    tmp_path / view / name
                )
                rows.append({
                    "filename": name,
                    "view_class": view,
                    "view_idx": view_idx,
                    "quality_grade": 8 if high else 1,
                    "sequence_group": f"{view}_D{14 + g}_v2",
                    "relative_path": f"{view}/{name}",
                    "split": split,
                })
    return pd.DataFrame(rows), tmp_path
