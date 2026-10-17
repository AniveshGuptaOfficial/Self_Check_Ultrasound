"""
Sequence-aware, group-stratified dataset splitting for CACTUS cardiac ultrasound.
Ensures zero data leakage of adjacent frames from the same acquisition run.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def create_sequence_aware_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Splits the manifest into train, val, and test splits at the sequence_group level.
    Uses balanced greedy bin-packing per view class.
    Guarantees that all frames belonging to a sequence group stay strictly in the same split.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    rng = np.random.RandomState(random_seed)
    df = manifest_df.copy()
    df["split"] = "unassigned"

    # Split within each view class to maintain class balance
    for view_class in sorted(df["view_class"].unique()):
        class_df = df[df["view_class"] == view_class]
        group_counts = class_df.groupby("sequence_group").size().to_dict()
        groups = list(group_counts.keys())

        # Shuffle with seed and sort descending by size to pack largest groups first
        rng.shuffle(groups)
        groups.sort(key=lambda g: group_counts[g], reverse=True)

        n_total = len(class_df)
        targets = {
            "train": n_total * train_ratio,
            "val": n_total * val_ratio,
            "test": n_total * test_ratio,
        }
        current = {"train": 0, "val": 0, "test": 0}

        for g in groups:
            count = group_counts[g]
            # Pick the split whose current count is farthest below its target proportion
            best_split = min(targets.keys(), key=lambda s: current[s] / targets[s])
            current[best_split] += count
            df.loc[df["sequence_group"] == g, "split"] = best_split

    # Validation check: verify strict disjointness of sequence_group between splits
    train_seqs = set(df[df["split"] == "train"]["sequence_group"])
    val_seqs = set(df[df["split"] == "val"]["sequence_group"])
    test_seqs = set(df[df["split"] == "test"]["sequence_group"])

    overlap_train_val = train_seqs.intersection(val_seqs)
    overlap_train_test = train_seqs.intersection(test_seqs)
    overlap_val_test = val_seqs.intersection(test_seqs)

    assert len(overlap_train_val) == 0, f"Leakage detected between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Leakage detected between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Leakage detected between val and test: {overlap_val_test}"

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Splits saved to {output_path}")

    return df
