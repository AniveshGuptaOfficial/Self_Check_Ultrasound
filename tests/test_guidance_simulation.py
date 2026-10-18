"""
Unit tests for Phase 4: Guidance, Policy Triage, and Closed-Loop Simulation.
Verifies Modules M4 (Adequacy), M5 (Cause Classifier), M6/M7 (Policy), and M8 (Simulator).
"""

import inspect

import numpy as np
import pytest

from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy
from self_check_ultrasound.guidance.simulator import ClosedLoopSimulator
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


@pytest.fixture
def dummy_echo_image(echo_image):
    return echo_image


def _simulator(max_attempts=3):
    return ClosedLoopSimulator(
        adequacy_estimator=AdequacyEstimator(adequacy_threshold=0.75),
        cause_classifier=CauseClassifier(),
        policy=ReacquisitionPolicy(adequacy_threshold=0.75, max_attempts=max_attempts),
    )


# ------------------------------------------------------------------ M4
def test_adequacy_estimator_features():
    estimator = AdequacyEstimator()
    dummy_probs = np.array([[0.80, 0.10, 0.05, 0.03, 0.01, 0.01]])
    features = estimator.extract_adequacy_features(
        predicted_quality=np.array([7.5]),
        view_probabilities=dummy_probs,
    )
    assert features.shape == (1, 4)
    assert np.isclose(features[0, 0], 7.5)
    assert np.isclose(features[0, 1], 0.80)
    assert np.isclose(features[0, 2], 0.70)
    assert features[0, 3] > 0.0


def test_adequacy_estimator_predictions():
    estimator = AdequacyEstimator(adequacy_threshold=0.75)
    high_q_feats = np.array([[8.5, 0.95, 0.90, 0.2]])
    low_q_feats = np.array([[2.0, 0.30, 0.05, 1.5]])

    prob_high = estimator.predict_probability(high_q_feats)[0]
    prob_low = estimator.predict_probability(low_q_feats)[0]

    assert 0.0 <= prob_high <= 1.0
    assert 0.0 <= prob_low <= 1.0
    assert prob_high > prob_low
    assert estimator.is_adequate(high_q_feats)[0]
    assert not estimator.is_adequate(low_q_feats)[0]


def test_adequacy_estimator_fit_save_load(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y = (X[:, 1] > 0.5).astype(int)
    est = AdequacyEstimator(adequacy_threshold=0.6)
    assert not est.is_fitted
    est.fit(X, y)
    path = tmp_path / "adeq.joblib"
    est.save(path)
    loaded = AdequacyEstimator.load(path)
    assert loaded.is_fitted
    assert loaded.adequacy_threshold == 0.6
    assert np.allclose(est.predict_probability(X), loaded.predict_probability(X))


# ------------------------------------------------------------------ M5
def test_cause_classifier_feature_extraction(dummy_echo_image):
    from PIL import Image
    classifier = CauseClassifier()
    feats = classifier.extract_physics_features(dummy_echo_image)
    assert len(feats) == 10
    assert feats[0] > 0.1
    assert feats[6] > 0.05

    blank = Image.new("RGB", (128, 128), color="black")
    assert np.all(classifier.extract_physics_features(blank) == 0.0)


def test_cause_classifier_rule_fallbacks(dummy_echo_image):
    classifier = CauseClassifier()
    pipeline = SyntheticDegradationPipeline(random_seed=42)

    dark_img, _ = pipeline.apply(dummy_echo_image, "under_gain", severity=0.85)
    preds, confs, can_advise = classifier.predict(classifier.extract_physics_features(dark_img).reshape(1, -1))
    assert preds[0] == "under_gain" and confs[0] >= 0.60 and can_advise[0]

    bright_img, _ = pipeline.apply(dummy_echo_image, "over_gain", severity=0.85)
    preds_b, confs_b, can_b = classifier.predict(classifier.extract_physics_features(bright_img).reshape(1, -1))
    assert preds_b[0] == "over_gain" and confs_b[0] >= 0.60 and can_b[0]


def test_cause_classifier_predict_proba_shape_and_sum(dummy_echo_image):
    classifier = CauseClassifier()
    feats = classifier.extract_physics_features(dummy_echo_image).reshape(1, -1)
    probs = classifier.predict_proba(feats)
    assert probs.shape == (1, len(CauseClassifier.FAULT_CLASSES))
    assert np.isclose(probs.sum(), 1.0)


# ------------------------------------------------------------------ M6/M7
def test_policy_accept_high_adequacy():
    policy = ReacquisitionPolicy(adequacy_threshold=0.75)
    decision = policy.evaluate(adequacy_prob=0.88, diagnosed_fault="none", fault_confidence=0.85, attempt_count=1)
    assert decision.action == "ACCEPT"
    assert decision.reacquisition_suggestion is None


def test_policy_reacquire_confident_fault():
    policy = ReacquisitionPolicy(adequacy_threshold=0.75, min_confidence_for_action=0.60)
    decision = policy.evaluate(adequacy_prob=0.45, diagnosed_fault="under_gain", fault_confidence=0.85, attempt_count=1)
    assert decision.action == "REACQUIRE"
    assert "gain" in decision.reacquisition_suggestion.lower()


def test_policy_refer_attempt_limit():
    policy = ReacquisitionPolicy(max_attempts=3)
    decision = policy.evaluate(adequacy_prob=0.45, diagnosed_fault="under_gain", fault_confidence=0.85, attempt_count=4)
    assert decision.action == "REFER"
    assert "limit" in decision.reason.lower()


def test_policy_accepts_adequate_frame_after_last_allowed_maneuver():
    """Regression: the frame produced by the 3rd allowed maneuver (attempt 4) must be accepted if adequate."""
    policy = ReacquisitionPolicy(adequacy_threshold=0.75, max_attempts=3)
    decision = policy.evaluate(adequacy_prob=0.90, diagnosed_fault="none", fault_confidence=0.85, attempt_count=4)
    assert decision.action == "ACCEPT"


def test_policy_refer_low_confidence():
    policy = ReacquisitionPolicy(min_confidence_for_action=0.60)
    decision = policy.evaluate(adequacy_prob=0.45, diagnosed_fault="acoustic_shadow", fault_confidence=0.40, attempt_count=1)
    assert decision.action == "REFER"
    assert "abstaining" in decision.reason.lower()


# ------------------------------------------------------------------ M8
def test_assessment_has_no_ground_truth_inputs():
    """Policy inputs must be computed from the image only (no injected fault/severity)."""
    params = set(inspect.signature(ClosedLoopSimulator.assess).parameters)
    assert params == {"self", "image"}


def test_assessment_separates_clean_from_degraded(dummy_echo_image):
    sim = _simulator()
    pipe = SyntheticDegradationPipeline(random_seed=42)
    clean = sim.assess(dummy_echo_image)
    dark, _ = pipe.apply(dummy_echo_image, "under_gain", severity=0.85)
    degraded = sim.assess(dark)
    assert clean["adequacy_prob"] >= 0.75
    assert degraded["adequacy_prob"] < clean["adequacy_prob"]
    assert degraded["adequacy_prob"] < 0.75


def test_simulator_incident_free_trajectory(dummy_echo_image):
    res = _simulator().run_simulation(base_image=dummy_echo_image, initial_fault="none", case_id="unit_test_nominal")
    assert res.final_action == "ACCEPT"
    assert res.total_attempts == 1
    assert res.success
    assert res.assessment_mode == "image_proxy+heuristic_adequacy"


def test_simulator_persistent_fault_referral(dummy_echo_image):
    res = _simulator(max_attempts=3).run_simulation(
        base_image=dummy_echo_image,
        initial_fault="under_gain",
        initial_severity=0.85,
        case_id="unit_test_persistent",
        persistent=True,
    )
    assert res.final_action == "REFER"
    assert res.total_attempts == 4  # frames 1-3 -> REACQUIRE, frame 4 -> REFER
    assert not res.success


def test_simulator_recovery_reaches_accept_and_records_ground_truth(dummy_echo_image):
    res = _simulator().run_simulation(
        base_image=dummy_echo_image, initial_fault="under_gain", initial_severity=0.85, case_id="recovery"
    )
    assert res.final_action == "ACCEPT"
    assert 2 <= res.total_attempts <= 4
    assert res.trajectory[0].true_fault == "under_gain"
    assert res.trajectory[-1].true_fault == "none"
    assert res.trajectory[0].decision.action == "REACQUIRE"


def test_simulator_requires_both_models_or_neither():
    with pytest.raises(ValueError):
        ClosedLoopSimulator(
            AdequacyEstimator(), CauseClassifier(), ReacquisitionPolicy(), view_model=object(), quality_model=None
        )
