"""
Dataset discovery and manifest creation for CACTUS cardiac ultrasound dataset.
Parses view labels, expert quality grades, acquisition parameters, and sequence groups.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from PIL import Image

from ..config import default_config, PathConfig, DataConfig


def parse_image_filename(filename: str, view_class: str) -> Dict[str, Any]:
    """
    Parses CACTUS filename into structured metadata using token extraction.
    Example formats:
      - 1_D14_frame_1088_v2.jpg
      - 2_DR2_frame_1283_v2.jpg
      - 2_V1Frame_699.jpg
      - 1_Folder2_frame_6675_v2.jpg
      - 0_frame_0_v2.jpg
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    # Grade is the first numeric token
    grade = int(parts[0]) if parts[0].isdigit() else 0

    depth_tag = None
    dr_tag = None
    video_tag = None
    version_tag = "v1"
    frame_num = -1

    # Check last token for version (e.g. v2)
    tokens_to_process = parts[1:]
    if tokens_to_process and re.match(r"^v\d+$", tokens_to_process[-1], re.IGNORECASE):
        version_tag = tokens_to_process[-1].lower()
        tokens_to_process = tokens_to_process[:-1]

    for p in tokens_to_process:
        p_up = p.upper()
        if re.match(r"^D\d+$", p_up):
            depth_tag = p_up
        elif re.match(r"^DR\d+$", p_up):
            dr_tag = p_up
        elif m := re.match(r"^(V\d+|FOLDER\d+)", p_up):
            video_tag = m.group(1)
        elif p.isdigit():
            frame_num = int(p)
        elif "frame" in p.lower():
            m_f = re.search(r"frame_?(\d+)", p, re.IGNORECASE)
            if m_f:
                frame_num = int(m_f.group(1))

    if frame_num == -1:
        m_f = re.search(r"frame_?(\d+)", stem, re.IGNORECASE)
        if m_f:
            frame_num = int(m_f.group(1))

    # Form cohesive sequence group key
    tag = depth_tag or dr_tag or video_tag or "BASE"
    sequence_group = f"{view_class}_{tag}_{version_tag}"

    return {
        "filename": filename,
        "view_class": view_class,
        "quality_grade": grade,
        "depth_tag": depth_tag,
        "dynamic_range_tag": dr_tag,
        "video_tag": video_tag,
        "version_tag": version_tag,
        "frame_num": frame_num,
        "sequence_group": sequence_group,
    }


def build_manifest(
    dataset_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    validate_images: bool = False,
) -> pd.DataFrame:
    """
    Scans the dataset directory, parses metadata, and generates a structured manifest DataFrame.
    """
    dataset_dir = Path(dataset_dir or default_config.paths.dataset_dir)
    output_path = Path(output_path or default_config.paths.manifest_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    records = []
    class_to_idx = default_config.data.class_to_idx

    for view_name in sorted(os.listdir(dataset_dir)):
        class_dir = dataset_dir / view_name
        if not class_dir.is_dir() or view_name not in class_to_idx:
            continue

        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            rel_path = f"{view_name}/{filename}"
            full_path = class_dir / filename

            meta = parse_image_filename(filename, view_name)
            meta["relative_path"] = rel_path
            meta["view_idx"] = class_to_idx[view_name]

            if validate_images:
                try:
                    with Image.open(full_path) as img:
                        w, h = img.size
                        meta["width"] = w
                        meta["height"] = h
                        meta["channels"] = len(img.getbands())
                        meta["is_valid"] = True
                except Exception as e:
                    meta["is_valid"] = False
                    meta["error"] = str(e)
            else:
                meta["width"] = default_config.data.raw_size[0]
                meta["height"] = default_config.data.raw_size[1]
                meta["channels"] = 3
                meta["is_valid"] = True

            records.append(meta)

    df = pd.DataFrame(records)

    # Save to artifacts directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Manifest successfully created: {len(df)} images saved to {output_path}")

    return df
