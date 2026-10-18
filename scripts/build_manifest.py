"""
Script to generate the dataset manifest and sequence-aware splits.
Usage:
  python scripts/build_manifest.py            # fast; does NOT open images
  python scripts/build_manifest.py --validate # opens every image: real size/channels/corruption check
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.data.manifest import build_manifest
from self_check_ultrasound.data.splits import create_sequence_aware_splits, check_split_coverage


def main():
    parser = argparse.ArgumentParser(description="Build manifest and leakage-free splits")
    parser.add_argument("--validate", action="store_true", help="Open every image to verify size/channels/corruption")
    args = parser.parse_args()

    print("=" * 60)
    print("SelfCheckUltrasound: Generating Manifest & Leakage-Free Splits")
    print("=" * 60)

    print(f"Scanning dataset from: {default_config.paths.dataset_dir}")
    manifest_df = build_manifest(
        dataset_dir=default_config.paths.dataset_dir,
        output_path=default_config.paths.manifest_path,
        validate_images=args.validate,
    )

    print("\n--- Manifest Summary ---")
    print(f"Total images registered: {len(manifest_df)}")
    print("Class distribution:")
    print(manifest_df["view_class"].value_counts().to_string())
    print("\nQuality grade distribution:")
    print(manifest_df["quality_grade"].value_counts().sort_index().to_string())

    if args.validate:
        n_bad = int((manifest_df["is_valid"] == False).sum())  # noqa: E712
        sizes = manifest_df.groupby(["width", "height", "channels"]).size()
        print(f"\nImage validation: {n_bad} unreadable/corrupt file(s)")
        print("Observed (width, height, channels) combinations:")
        print(sizes.to_string())
    else:
        print("\nImage validation: SKIPPED (dimensions/validity not verified; rerun with --validate)")

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
    print(splits_df.groupby(["split", "view_class"]).size().unstack(fill_value=0).to_string())

    print("\nSequence groups per view and split:")
    cov = check_split_coverage(splits_df)
    for view, per_split in cov["table"].items():
        print("  " + view.ljust(8) + "  ".join(f"{s}: {v['groups']:>3d} grp / {v['images']:>5d} img" for s, v in per_split.items()))
    if cov["problems"]:
        print("\nWARNING - split coverage problems (evaluation on these views is not reliable):")
        for p in cov["problems"]:
            print(f"  - {p}")

    print("\nVerifying zero sequence group leakage...")
    groups = {s: set(splits_df[splits_df["split"] == s]["sequence_group"]) for s in ("train", "val", "test")}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    print("PASS: Group-level isolation verified across Train, Validation, and Test.")
    print(f"Outputs written to: {default_config.paths.artifacts_dir}")


if __name__ == "__main__":
    main()
