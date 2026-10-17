"""
Central configuration module for SelfCheckUltrasound.
Defines reproducible parameters, paths, model configs, and operational thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    dataset_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "FRDR_dataset_1479_download_216_202610160047"
        / "Cactus Dataset"
        / "Images Dataset"
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts"
    )
    experiments_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "experiments"
    )
    manifest_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "artifacts"
        / "dataset_manifest.csv"
    )
    splits_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "artifacts"
        / "splits.csv"
    )


@dataclass
class DataConfig:
    classes: Tuple[str, ...] = ("A4C", "PL", "PSAV", "PSMV", "Random", "SC")
    class_to_idx: dict = field(
        default_factory=lambda: {
            "A4C": 0,
            "PL": 1,
            "PSAV": 2,
            "PSMV": 3,
            "Random": 4,
            "SC": 5,
        }
    )
    image_size: Tuple[int, int] = (224, 224)
    raw_size: Tuple[int, int] = (1920, 1080)
    num_classes: int = 6
    quality_grade_min: int = 0
    quality_grade_max: int = 9
    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 0  # 0 is safe for Windows CPU processing
    learning_rate: float = 1e-3
    num_epochs: int = 5
    weight_decay: float = 1e-4
    device: str = "cpu"


@dataclass
class PolicyConfig:
    adequacy_threshold: float = 0.75
    min_confidence_for_action: float = 0.60
    max_reacquisition_attempts: int = 3


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


# Global default instance
default_config = ProjectConfig()
