"""
Unit tests for sequence-aware group-stratified dataset splitting and leakage prevention.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from self_check_ultrasound.config import default_config
from self_check_ultrasound.data.splits import create_sequence_aware_splits, check_split_coverage


def _synthetic_manifest(groups_per_view=8, frames_per_group=20, views=("A4C", "PL")):
    rows = []
    for v in views:
        for g in range(groups_per_view):
            for f in range(frames_per_group):
                rows.append({"view_class": v, "sequence_group": f"{v}_D{g}_v1", "frame_num": f})
    return pd.DataFrame(rows)


def test_synthetic_splits_have_zero_group_leakage_and_full_coverage():
    df = create_sequence_aware_splits(_synthetic_manifest(), random_seed=1)
    groups = {s: set(df[df["split"] == s]["sequence_group"]) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert (df["split"] != "unassigned").all()
    assert check_split_coverage(df)["ok"]


def test_coverage_check_flags_view_with_too_few_groups():
    manifest = pd.concat([_synthetic_manifest(views=("A4C",)), _synthetic_manifest(groups_per_view=2, views=("Random",))])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = create_sequence_aware_splits(manifest, random_seed=1)
    report = check_split_coverage(df)
    assert not report["ok"]
    assert any("Random" in p for p in report["problems"])
    assert any("Split coverage" in str(x.message) for x in w)


def _load_real_splits():
    path = default_config.paths.splits_path
    if not path.exists():
        pytest.skip("artifacts/splits.csv not generated (run scripts/build_manifest.py)")
    return pd.read_csv(path, low_memory=False)


def test_splits_file_exists_and_complete():
    df = _load_real_splits()
    assert len(df) == 37736
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_zero_sequence_group_leakage():
    df = _load_real_splits()
    train_groups = set(df[df["split"] == "train"]["sequence_group"])
    val_groups = set(df[df["split"] == "val"]["sequence_group"])
    test_groups = set(df[df["split"] == "test"]["sequence_group"])
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)


def test_split_proportions_reasonable():
    df = _load_real_splits()
    counts = df["split"].value_counts(normalize=True)
    assert 0.50 <= counts["train"] <= 0.80
    assert 0.10 <= counts["val"] <= 0.30
    assert 0.10 <= counts["test"] <= 0.30


def test_every_view_present_in_every_real_split():
    """If this fails, a view cannot be evaluated on val/test; see check_split_coverage()."""
    report = check_split_coverage(_load_real_splits())
    assert report["ok"], "; ".join(report["problems"])
