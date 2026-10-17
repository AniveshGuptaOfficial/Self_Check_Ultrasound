"""
Image transformations and preprocessing for cardiac ultrasound images.
"""

from typing import Tuple
from torchvision import transforms


def get_transforms(
    is_train: bool = True,
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """
    Standard transform pipeline: resize to network input size, convert to tensor, normalize.
    """
    transform_list = [
        transforms.Resize(image_size),
    ]

    if is_train:
        # Subtle data augmentation for training
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # Standard normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)
