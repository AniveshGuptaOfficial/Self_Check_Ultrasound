"""
Script to generate the dataset manifest and sequence-aware splits.
Usage:
  python scripts/build_manifest.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.data.manifest import build_manifest
from self_check_ultrasound.data.splits import create_sequence_aware_splits


def main():
    print("=" * 60)
    print("SelfCheckUltrasound: Generating Manifest & Leakage-Free Splits")
    print("=" * 60)

    # 1. Build manifest
    print(f"Scanning dataset from: {default_config.paths.dataset_dir}")
    manifest_df = build_manifest(
        dataset_dir=default_config.paths.dataset_dir,
        output_path=default_config.paths.manifest_path,
        validate_images=False,
    )

    print("\n--- Manifest Summary ---")
    print(f"Total images registered: {len(manifest_df)}")
    print("Class distribution:")
    print(manifest_df["view_class"].value_counts().to_string())
    print("\nQuality grade distribution:")
    print(manifest_df["quality_grade"].value_counts().sort_index().to_string())

    # 2. Create sequence-aware splits
    print("\nCreating sequence-aware group-stratified splits...")
    splits_df = create_sequence_aware_splits(
        manifest_df=manifest_df,
        train_ratio=default_config.data.train_ratio,
        val_ratio=default_config.data.val_ratio,
        test_ratio=default_config.data.test_ratio,
        random_seed=default_config.data.random_seed,
        output_path=default_config.paths.splits_path,
    )

    print("\n--- Split Proportions ---")
    split_counts = splits_df["split"].value_counts()
    for s_name, count in split_counts.items():
        print(f"  {s_name.upper():6s}: {count:6d} images ({count / len(splits_df) * 100:.1f}%)")

    print("\nCross-tabulation (Split vs. View Class):")
    ct = pd_crosstab = splits_df.groupby(["split", "view_class"]).size().unstack(fill_value=0)
    print(ct.to_string())

    print("\nVerifying zero sequence group leakage...")
    train_groups = set(splits_df[splits_df["split"] == "train"]["sequence_group"])
    val_groups = set(splits_df[splits_df["split"] == "val"]["sequence_group"])
    test_groups = set(splits_df[splits_df["split"] == "test"]["sequence_group"])

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0
    print("PASS: Group-level isolation verified across Train, Validation, and Test!")
    print(f"Outputs written to: {default_config.paths.artifacts_dir}")


if __name__ == "__main__":
    main()
