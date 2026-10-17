"""
Unit tests for synthetic degradation pipeline.
"""

import numpy as np
from PIL import Image
import pytest
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


def test_all_degradations_output_valid_image():
    pipeline = SyntheticDegradationPipeline(random_seed=42)
    dummy_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))

    for deg_type in pipeline.SUPPORTED_DEGRADATIONS:
        out_img, meta = pipeline.apply(dummy_img, degradation_type=deg_type, severity=0.7)
        assert isinstance(out_img, Image.Image)
        assert out_img.size == (224, 224)
        assert meta["degradation_type"] == deg_type
        assert "suggestion" in meta


def test_severity_clipping():
    pipeline = SyntheticDegradationPipeline(random_seed=42)
    dummy_img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))

    _, meta_low = pipeline.apply(dummy_img, degradation_type="under_gain", severity=-1.0)
    assert meta_low["severity"] == 0.1

    _, meta_high = pipeline.apply(dummy_img, degradation_type="under_gain", severity=5.0)
    assert meta_high["severity"] == 1.0


def test_unknown_degradation_raises():
    pipeline = SyntheticDegradationPipeline(random_seed=42)
    dummy_img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError):
        pipeline.apply(dummy_img, degradation_type="unsupported_fault")
