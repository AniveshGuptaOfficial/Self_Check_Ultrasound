"""
PyTorch Dataset implementation for CACTUS cardiac ultrasound images.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..config import default_config
from ..utils.paths import resolve_safe_path


class CardiacUltrasoundDataset(Dataset):
    """
    Loads cardiac ultrasound images from the manifest DataFrame.

    NOTE: when a degradation pipeline is active, `quality_grade` is still the ORIGINAL
    expert grade of the clean frame. Use the returned `degraded` flag to exclude degraded
    samples from quality-regression losses/metrics.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        degradation_pipeline: Optional[Any] = None,
        degradation_prob: float = 0.0,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root or default_config.paths.dataset_dir)
        self.transform = transform
        self.degradation_pipeline = degradation_pipeline
        self.degradation_prob = degradation_prob

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # Raises ValueError("Path traversal detected ...") if the path escapes dataset_root.
        image_path = resolve_safe_path(self.dataset_root, row["relative_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        image = Image.open(image_path).convert("RGB")
        degradation_meta = {"degradation_type": "none", "severity": 0.0}

        if self.degradation_pipeline and self.degradation_prob > 0:
            if torch.rand(1).item() < self.degradation_prob:
                image, degradation_meta = self.degradation_pipeline.apply(image)

        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = image

        return {
            "image": tensor,
            "view_label": int(row["view_idx"]),
            "quality_grade": float(row["quality_grade"]),
            "view_class": str(row["view_class"]),
            "filename": str(row["filename"]),
            "sequence_group": str(row["sequence_group"]),
            "degradation": degradation_meta,
            "degraded": degradation_meta.get("degradation_type", "none") != "none",
        }
