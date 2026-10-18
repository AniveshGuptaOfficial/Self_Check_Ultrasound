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

    Training augmentation is deliberately mild and does NOT include horizontal flips:
    mirroring an echocardiogram reverses the anatomical convention of the view (e.g. the
    position of the ventricles in A4C), which can turn a valid view into a non-standard one.
    """
    transform_list = [
        transforms.Resize(image_size),
    ]

    if is_train:
        transform_list.extend([
            transforms.RandomRotation(degrees=10),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)
