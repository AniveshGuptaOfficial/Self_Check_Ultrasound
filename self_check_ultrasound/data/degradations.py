"""
Controlled synthetic degradation pipeline for cardiac ultrasound images.
Simulates common operator and acquisition faults:
- Gain attenuation (under-gain)
- Over-gain saturation
- Acoustic shadowing (rib / calcification drop-out)
- Transducer motion blur
- Out-of-plane / off-axis misplacement
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class SyntheticDegradationPipeline:
    """
    Applies deterministic and stochastic degradations to benchmark quality assessment
    and cause-aware reacquisition guidance.
    """

    SUPPORTED_DEGRADATIONS = (
        "none",
        "under_gain",
        "over_gain",
        "acoustic_shadow",
        "motion_blur",
        "off_axis",
    )

    REACQUISITION_SUGGESTIONS = {
        "under_gain": "Increase 2D overall gain or time-gain compensation (TGC).",
        "over_gain": "Decrease 2D overall gain to resolve acoustic washout.",
        "acoustic_shadow": "Reposition probe to avoid rib acoustic shadowing; adjust intercostal angle.",
        "motion_blur": "Stabilize probe and ask patient to suspend respiration momentarily.",
        "off_axis": "Re-align probe to orthogonal anatomical standard view axis.",
        "none": "Image adequate; no maneuver required.",
    }

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def apply(
        self,
        image: Image.Image,
        degradation_type: Optional[str] = None,
        severity: float = 0.5,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Applies a degradation to a PIL image and returns the degraded image and metadata.
        severity: float between 0.1 (mild) and 1.0 (severe).
        """
        if degradation_type is None:
            degradation_type = self.rng.choice(self.SUPPORTED_DEGRADATIONS)

        if degradation_type not in self.SUPPORTED_DEGRADATIONS:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        severity = float(np.clip(severity, 0.1, 1.0))
        img = image.copy()

        if degradation_type == "none":
            return img, {
                "degradation_type": "none",
                "severity": 0.0,
                "suggestion": self.REACQUISITION_SUGGESTIONS["none"],
            }

        elif degradation_type == "under_gain":
            # Scale brightness down: 1.0 -> 0.1
            factor = max(0.1, 1.0 - (0.85 * severity))
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        elif degradation_type == "over_gain":
            # Scale brightness and contrast up, blowing out speckle
            factor = 1.0 + (1.8 * severity)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.0 + (0.5 * severity))

        elif degradation_type == "motion_blur":
            # Gaussian blur with radius proportional to severity
            radius = 1.0 + (6.0 * severity)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif degradation_type == "acoustic_shadow":
            # Dark vertical stripe or wedge simulating rib acoustic shadow
            arr = np.array(img).astype(np.float32)
            h, w = arr.shape[:2]
            # Center of the shadow band
            center_x = int(w * (0.3 + 0.4 * self.rng.rand()))
            width = int(w * (0.08 + 0.15 * severity))
            x_min = max(0, center_x - width // 2)
            x_max = min(w, center_x + width // 2)

            attenuation = 1.0 - (0.85 * severity)
            arr[:, x_min:x_max] *= attenuation
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        elif degradation_type == "off_axis":
            # Crop away outer structural landmarks and zoom/shift
            w, h = img.size
            crop_ratio = 1.0 - (0.25 * severity)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = int(self.rng.rand() * (w - new_w))
            top = int(self.rng.rand() * (h - new_h))
            img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.BILINEAR)

        meta = {
            "degradation_type": degradation_type,
            "severity": severity,
            "suggestion": self.REACQUISITION_SUGGESTIONS[degradation_type],
        }
        return img, meta
