"""
Unit tests for sequence-aware group-stratified dataset splitting and leakage prevention.
"""

from pathlib import Path
import pandas as pd
import pytest
from self_check_ultrasound.config import default_config


def test_splits_file_exists_and_complete():
    splits_path = default_config.paths.splits_path
    assert splits_path.exists(), f"Splits file not found at {splits_path}"
    df = pd.read_csv(splits_path, low_memory=False)
    assert len(df) == 37736
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_zero_sequence_group_leakage():
    splits_path = default_config.paths.splits_path
    df = pd.read_csv(splits_path, low_memory=False)

    train_groups = set(df[df["split"] == "train"]["sequence_group"])
    val_groups = set(df[df["split"] == "val"]["sequence_group"])
    test_groups = set(df[df["split"] == "test"]["sequence_group"])

    # Assert completely disjoint sets
    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    assert len(train_val_overlap) == 0, f"Leakage detected between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between val and test: {val_test_overlap}"


def test_split_proportions_reasonable():
    splits_path = default_config.paths.splits_path
    df = pd.read_csv(splits_path, low_memory=False)
    counts = df["split"].value_counts(normalize=True)

    # Train should be around 55-75%, Val around 10-25%, Test around 10-25%
    assert 0.50 <= counts["train"] <= 0.80, f"Train proportion out of expected range: {counts['train']}"
    assert 0.10 <= counts["val"] <= 0.30, f"Val proportion out of expected range: {counts['val']}"
    assert 0.10 <= counts["test"] <= 0.30, f"Test proportion out of expected range: {counts['test']}"
