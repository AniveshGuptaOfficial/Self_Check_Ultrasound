"""
Unit tests for model architectures and baseline classifiers.
"""

import torch
import numpy as np
from PIL import Image
from self_check_ultrasound.models.view_classifier import CardiacEchoNet, SimpleFeatureClassifier
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet


def test_cardiac_echo_net_forward_pass():
    model = CardiacEchoNet(num_classes=6)
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        embedding = model.extract_embedding(dummy_input)

    assert output.shape == (4, 6)
    assert embedding.shape == (4, 256)
    assert not torch.isnan(output).any()


def test_quality_assessor_net_forward_pass():
    model = QualityAssessorNet()
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4,)
    assert not torch.isnan(output).any()


def test_simple_feature_classifier_fit_predict():
    clf = SimpleFeatureClassifier(n_bins=8, grid_size=2)
    dummy_img = Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8))
    feats = clf.extract_features(dummy_img)
    # 8 bins + 4 grid cells = 12 features
    assert feats.shape == (12,)

    X_dummy = np.random.randn(20, 12).astype(np.float32)
    y_dummy = np.random.randint(0, 6, size=20)
    clf.fit(X_dummy, y_dummy)
    preds = clf.predict(X_dummy)
    probs = clf.predict_proba(X_dummy)

    assert preds.shape == (20,)
    assert probs.shape == (20, len(np.unique(y_dummy)))
