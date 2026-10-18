"""
Unit tests for CACTUS dataset discovery and filename parsing.
"""

import pandas as pd
import pytest
from PIL import Image

from self_check_ultrasound.data.manifest import parse_image_filename, build_manifest
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
    assert meta["quality_grade"] == 2
    assert meta["dynamic_range_tag"] == "DR2"
    assert meta["frame_num"] == 1283
    assert meta["sequence_group"] == "PL_DR2_v2"


def test_parse_image_filename_video_format():
    meta = parse_image_filename("2_V1Frame_699.jpg", "PSMV")
    assert meta["quality_grade"] == 2
    assert meta["video_tag"] == "V1"
    assert meta["frame_num"] == 699
    assert meta["sequence_group"] == "PSMV_V1_v1"


def test_parse_image_filename_random_format():
    meta = parse_image_filename("0_frame_0_v2.jpg", "Random")
    assert meta["quality_grade"] == 0
    assert meta["frame_num"] == 0
    assert meta["sequence_group"] == "Random_BASE_v2"


def test_build_manifest_does_not_claim_unverified_dimensions(tmp_path):
    (tmp_path / "A4C").mkdir()
    Image.new("RGB", (40, 30)).save(tmp_path / "A4C" / "5_D14_frame_1_v2.jpg")
    (tmp_path / "A4C" / "3_D14_frame_2_v2.jpg").write_bytes(b"not an image")

    lazy = build_manifest(tmp_path, tmp_path / "m_lazy.csv", validate_images=False)
    assert lazy["width"].isna().all() and lazy["is_valid"].isna().all()

    checked = build_manifest(tmp_path, tmp_path / "m_checked.csv", validate_images=True)
    ok = checked[checked["filename"] == "5_D14_frame_1_v2.jpg"].iloc[0]
    bad = checked[checked["filename"] == "3_D14_frame_2_v2.jpg"].iloc[0]
    assert (ok["width"], ok["height"], ok["channels"]) == (40, 30, 3) and bool(ok["is_valid"])
    assert not bool(bad["is_valid"]) and pd.isna(bad["width"])


def test_manifest_file_exists_and_schema():
    manifest_path = default_config.paths.manifest_path
    if not manifest_path.exists():
        pytest.skip("artifacts/dataset_manifest.csv not generated (run scripts/build_manifest.py)")
    df = pd.read_csv(manifest_path, low_memory=False)
    assert len(df) == 37736
    for col in ["filename", "view_class", "quality_grade", "sequence_group", "relative_path", "view_idx"]:
        assert col in df.columns, f"Missing required column: {col}"
    assert df["quality_grade"].min() >= 0
    assert df["quality_grade"].max() <= 9
