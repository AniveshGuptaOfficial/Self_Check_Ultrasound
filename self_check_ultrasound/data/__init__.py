"""
Data processing package: manifest creation, group-aware splitting,
transforms, synthetic degradations, and PyTorch dataset loaders.
"""

from .manifest import build_manifest, parse_image_filename
from .splits import create_sequence_aware_splits, check_split_coverage
from .transforms import get_transforms
from .degradations import SyntheticDegradationPipeline
from .dataset import CardiacUltrasoundDataset

__all__ = [
    "build_manifest",
    "parse_image_filename",
    "create_sequence_aware_splits",
    "check_split_coverage",
    "get_transforms",
    "SyntheticDegradationPipeline",
    "CardiacUltrasoundDataset",
]
