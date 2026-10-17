"""
Unit tests for Phase 4: Guidance, Policy Triage, and Closed-Loop Simulation.
Verifies Modules M4 (Adequacy), M5 (Cause Classifier), M6/M7 (Policy), and M8 (Simulator).
"""

import pytest
import numpy as np
from PIL import Image

from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy
from self_check_ultrasound.guidance.simulator import ClosedLoopSimulator
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


@pytest.fixture
def dummy_echo_image():
    """Generates a synthetic ultrasound cone pattern on black background."""
    img_arr = np.zeros((128, 128), dtype=np.uint8)
    for y in range(20, 110):
        half_w = int((y - 20) * 0.45)
        x_start = max(0, 64 - half_w)
        x_end = min(128, 64 + half_w)
        img_arr[y, x_start:x_end] = np.random.randint(70, 160, size=(x_end - x_start))
    return Image.fromarray(img_arr).convert("RGB")


def test_adequacy_estimator_features():
    estimator = AdequacyEstimator()
    dummy_probs = np.array([[0.80, 0.10, 0.05, 0.03, 0.01, 0.01]])
    features = estimator.extract_adequacy_features(
        predicted_quality=np.array([7.5]),
        view_probabilities=dummy_probs,
    )
    assert features.shape == (1, 4)
    # Feature 0: quality, Feature 1: top-1 conf, Feature 2: margin, Feature 3: entropy
    assert np.isclose(features[0, 0], 7.5)
    assert np.isclose(features[0, 1], 0.80)
    assert np.isclose(features[0, 2], 0.70)
    assert features[0, 3] > 0.0  # Entropy is positive


def test_adequacy_estimator_predictions():
    estimator = AdequacyEstimator(adequacy_threshold=0.75)
    high_q_feats = np.array([[8.5, 0.95, 0.90, 0.2]])
    low_q_feats = np.array([[2.0, 0.30, 0.05, 1.5]])

    prob_high = estimator.predict_probability(high_q_feats)[0]
    prob_low = estimator.predict_probability(low_q_feats)[0]

    assert 0.0 <= prob_high <= 1.0
    assert 0.0 <= prob_low <= 1.0
    assert prob_high > prob_low
    assert estimator.is_adequate(high_q_feats)[0] == True
    assert estimator.is_adequate(low_q_feats)[0] == False


def test_cause_classifier_feature_extraction(dummy_echo_image):
    classifier = CauseClassifier()
    feats = classifier.extract_physics_features(dummy_echo_image)
    assert len(feats) == 10
    # Mean lum, sat_ratio, p90, sharpness, col_drop, col_min, cov, apex, asym, spread
    assert feats[0] > 0.1  # Non-zero luminosity
    assert feats[6] > 0.05  # Active sector coverage

    # Blank image handling
    blank = Image.new("RGB", (128, 128), color="black")
    blank_feats = classifier.extract_physics_features(blank)
    assert np.all(blank_feats == 0.0)


def test_cause_classifier_rule_fallbacks(dummy_echo_image):
    classifier = CauseClassifier()
    pipeline = SyntheticDegradationPipeline(random_seed=42)

    # Test under-gain detection
    dark_img, _ = pipeline.apply(dummy_echo_image, "under_gain", severity=0.85)
    feats = classifier.extract_physics_features(dark_img)
    preds, confs, can_advise = classifier.predict(feats.reshape(1, -1))
    assert preds[0] == "under_gain"
    assert confs[0] >= 0.60
    assert can_advise[0] == True

    # Test over-gain detection
    bright_img, _ = pipeline.apply(dummy_echo_image, "over_gain", severity=0.85)
    feats_b = classifier.extract_physics_features(bright_img)
    preds_b, confs_b, can_advise_b = classifier.predict(feats_b.reshape(1, -1))
    assert preds_b[0] == "over_gain"
    assert confs_b[0] >= 0.60
    assert can_advise_b[0] == True


def test_policy_accept_high_adequacy():
    policy = ReacquisitionPolicy(adequacy_threshold=0.75)
    decision = policy.evaluate(
        adequacy_prob=0.88,
        diagnosed_fault="none",
        fault_confidence=0.85,
        attempt_count=1,
    )
    assert decision.action == "ACCEPT"
    assert decision.reacquisition_suggestion is None


def test_policy_reacquire_confident_fault():
    policy = ReacquisitionPolicy(adequacy_threshold=0.75, min_confidence_for_action=0.60)
    decision = policy.evaluate(
        adequacy_prob=0.45,
        diagnosed_fault="under_gain",
        fault_confidence=0.85,
        attempt_count=1,
    )
    assert decision.action == "REACQUIRE"
    assert "gain" in decision.reacquisition_suggestion.lower()


def test_policy_refer_attempt_limit():
    policy = ReacquisitionPolicy(max_attempts=3)
    decision = policy.evaluate(
        adequacy_prob=0.45,
        diagnosed_fault="under_gain",
        fault_confidence=0.85,
        attempt_count=4,
    )
    assert decision.action == "REFER"
    assert "limit" in decision.reason.lower()


def test_policy_refer_low_confidence():
    policy = ReacquisitionPolicy(min_confidence_for_action=0.60)
    decision = policy.evaluate(
        adequacy_prob=0.45,
        diagnosed_fault="acoustic_shadow",
        fault_confidence=0.40,  # Below 0.60 threshold
        attempt_count=1,
    )
    assert decision.action == "REFER"
    assert "abstaining" in decision.reason.lower()


def test_simulator_incident_free_trajectory(dummy_echo_image):
    adequacy_estimator = AdequacyEstimator(adequacy_threshold=0.75)
    cause_classifier = CauseClassifier()
    policy = ReacquisitionPolicy(adequacy_threshold=0.75)
    simulator = ClosedLoopSimulator(
        adequacy_estimator=adequacy_estimator,
        cause_classifier=cause_classifier,
        policy=policy,
    )

    res = simulator.run_simulation(
        base_image=dummy_echo_image,
        initial_fault="none",
        case_id="unit_test_nominal",
    )
    assert res.final_action == "ACCEPT"
    assert res.total_attempts == 1
    assert res.success == True


def test_simulator_persistent_fault_referral(dummy_echo_image):
    adequacy_estimator = AdequacyEstimator(adequacy_threshold=0.75)
    cause_classifier = CauseClassifier()
    policy = ReacquisitionPolicy(max_attempts=3)
    simulator = ClosedLoopSimulator(
        adequacy_estimator=adequacy_estimator,
        cause_classifier=cause_classifier,
        policy=policy,
    )

    res = simulator.run_simulation(
        base_image=dummy_echo_image,
        initial_fault="under_gain",
        initial_severity=0.85,
        case_id="unit_test_persistent",
        persistent=True,
    )
    assert res.final_action == "REFER"
    assert res.total_attempts == 4  # Attempts 1, 2, 3 reacquired, attempt 4 referred
    assert res.success == False
