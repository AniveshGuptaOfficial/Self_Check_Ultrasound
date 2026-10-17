"""
Unit tests for CACTUS dataset discovery and filename parsing.
"""

from pathlib import Path
import pandas as pd
import pytest
from self_check_ultrasound.data.manifest import parse_image_filename
from self_check_ultrasound.config import default_config


def test_parse_image_filename_depth_format():
    meta = parse_image_filename("1_D14_frame_1088_v2.jpg", "A4C")
    assert meta["view_class"] == "A4C"
    assert meta["quality_grade"] == 1
    assert meta["depth_tag"] == "D14"
    assert meta["frame_num"] == 1088
    assert meta["version_tag"] == "v2"
    assert meta["sequence_group"] == "A4C_D14_v2"


def test_parse_image_filename_dr_format():
    meta = parse_image_filename("2_DR2_frame_1283_v2.jpg", "PL")
    assert meta["view_class"] == "PL"
    assert meta["quality_grade"] == 2
    assert meta["dynamic_range_tag"] == "DR2"
    assert meta["frame_num"] == 1283
    assert meta["version_tag"] == "v2"
    assert meta["sequence_group"] == "PL_DR2_v2"


def test_parse_image_filename_video_format():
    meta = parse_image_filename("2_V1Frame_699.jpg", "PSMV")
    assert meta["view_class"] == "PSMV"
    assert meta["quality_grade"] == 2
    assert meta["video_tag"] == "V1"
    assert meta["frame_num"] == 699
    assert meta["sequence_group"] == "PSMV_V1_v1"


def test_parse_image_filename_random_format():
    meta = parse_image_filename("0_frame_0_v2.jpg", "Random")
    assert meta["view_class"] == "Random"
    assert meta["quality_grade"] == 0
    assert meta["frame_num"] == 0
    assert meta["sequence_group"] == "Random_BASE_v2"


def test_manifest_file_exists_and_schema():
    manifest_path = default_config.paths.manifest_path
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"
    df = pd.read_csv(manifest_path, low_memory=False)
    assert len(df) == 37736
    required_cols = [
        "filename",
        "view_class",
        "quality_grade",
        "sequence_group",
        "relative_path",
        "view_idx",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    # Grade bounds
    assert df["quality_grade"].min() >= 0
    assert df["quality_grade"].max() <= 9
