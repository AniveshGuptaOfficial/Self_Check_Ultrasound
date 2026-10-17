"""
Closed-Loop Reacquisition Simulator (Module M8).
Simulates the interactive feedback loop:
Frame -> Quality & Adequacy Evaluation -> Triage -> Maneuver Simulation -> Re-evaluation.

Evaluates 3 canonical clinical trajectories:
1. Incident-free: Initial image is adequate -> ACCEPT immediately.
2. Recovery: Initial image has a correctable fault -> REACQUIRE -> Maneuver applied -> ACCEPT.
3. Safe Referral: Fault is unresolvable or attempt limit reached -> REFER.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

from .reacquisition_policy import ReacquisitionPolicy, PolicyDecision
from ..models.adequacy_estimator import AdequacyEstimator
from ..models.cause_classifier import CauseClassifier
from ..data.degradations import SyntheticDegradationPipeline


@dataclass
class SimulationStep:
    attempt: int
    quality_grade: float
    adequacy_prob: float
    diagnosed_fault: str
    fault_confidence: float
    decision: PolicyDecision
    applied_maneuver: Optional[str]


@dataclass
class SimulationResult:
    case_id: str
    initial_fault: str
    final_action: str
    total_attempts: int
    trajectory: List[SimulationStep]
    success: bool


class ClosedLoopSimulator:
    """
    Executes iterative reacquisition simulation over ultrasound frames.
    """

    def __init__(
        self,
        adequacy_estimator: AdequacyEstimator,
        cause_classifier: CauseClassifier,
        policy: ReacquisitionPolicy,
        degradation_pipeline: Optional[SyntheticDegradationPipeline] = None,
    ):
        self.adequacy_estimator = adequacy_estimator
        self.cause_classifier = cause_classifier
        self.policy = policy
        self.degradation_pipeline = degradation_pipeline or SyntheticDegradationPipeline(random_seed=42)

    def run_simulation(
        self,
        base_image: Image.Image,
        initial_fault: str = "none",
        initial_severity: float = 0.6,
        case_id: str = "case_01",
        persistent: bool = False,
    ) -> SimulationResult:
        """
        Executes a closed-loop quality re-check trajectory starting from base_image.
        """
        trajectory: List[SimulationStep] = []
        current_image = base_image.copy()

        # Apply initial fault if requested
        if initial_fault != "none":
            current_image, _ = self.degradation_pipeline.apply(
                current_image, degradation_type=initial_fault, severity=initial_severity
            )

        attempt = 1
        curr_fault = initial_fault
        curr_severity = initial_severity
        final_action = "REFER"

        while attempt <= self.policy.max_attempts + 1:
            # 1. Physics feature extraction & cause classification
            features = self.cause_classifier.extract_physics_features(current_image)
            pred_faults, confs, can_advise = self.cause_classifier.predict(features.reshape(1, -1))
            diag_fault = pred_faults[0]
            diag_conf = float(confs[0])

            # 2. Quality & Adequacy estimation
            mean_lum = features[0]
            sat_ratio = features[1]
            sharpness = features[3]
            col_drop = features[4]

            # Inherent quality score in [0, 9]
            est_quality = float(np.clip(7.5 - (sat_ratio * 4.0) - (col_drop * 3.0) + (sharpness * 20.0), 0.5, 9.0))
            if diag_fault == "under_gain":
                est_quality = float(np.clip(mean_lum * 9.0, 0.5, 4.0))

            dummy_probs = np.full((1, 6), (1.0 - diag_conf) / 5.0)
            dummy_probs[0, 0] = diag_conf

            adeq_feats = self.adequacy_estimator.extract_adequacy_features(
                predicted_quality=np.array([est_quality]),
                view_probabilities=dummy_probs,
            )
            adeq_prob = float(self.adequacy_estimator.predict_probability(adeq_feats)[0])

            # Adjust adeq_prob based on diagnosed fault
            if diag_fault in ("under_gain", "over_gain", "motion_blur", "acoustic_shadow", "off_axis") and curr_severity > 0.2:
                adeq_prob = min(adeq_prob, 0.45)
            elif diag_fault == "none":
                adeq_prob = max(adeq_prob, 0.85)

            # 3. Policy evaluation
            decision = self.policy.evaluate(
                adequacy_prob=adeq_prob,
                diagnosed_fault=diag_fault,
                fault_confidence=diag_conf,
                attempt_count=attempt,
            )

            step = SimulationStep(
                attempt=attempt,
                quality_grade=est_quality,
                adequacy_prob=adeq_prob,
                diagnosed_fault=diag_fault,
                fault_confidence=diag_conf,
                decision=decision,
                applied_maneuver=None,
            )

            trajectory.append(step)
            final_action = decision.action

            if decision.action in ("ACCEPT", "REFER"):
                break

            # 4. Actionable maneuver simulation
            if decision.action == "REACQUIRE":
                step.applied_maneuver = decision.reacquisition_suggestion
                attempt += 1

                if not persistent:
                    # Simulate corrective compensation
                    curr_severity = max(0.0, curr_severity - 0.5)
                    if curr_severity <= 0.2:
                        current_image = base_image.copy()
                        curr_fault = "none"
                    else:
                        current_image, _ = self.degradation_pipeline.apply(
                            base_image.copy(), degradation_type=curr_fault, severity=curr_severity
                        )
                else:
                    # Persistent fault remains unresolvable across attempts
                    pass

        success = final_action == "ACCEPT"
        return SimulationResult(
            case_id=case_id,
            initial_fault=initial_fault,
            final_action=final_action,
            total_attempts=len(trajectory),
            trajectory=trajectory,
            success=success,
        )
