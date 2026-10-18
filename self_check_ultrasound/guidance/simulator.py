"""
Closed-Loop Reacquisition Simulator (Module M8).
Simulates the interactive feedback loop:
Frame -> Quality & Adequacy Evaluation -> Triage -> Maneuver Simulation -> Re-evaluation.

What is real and what is simulated
----------------------------------
REAL (computed from the image only, never from the injected ground-truth fault/severity):
  * fault diagnosis (CauseClassifier on sector-masked physics features)
  * quality estimate and adequacy probability
  * policy decision (ACCEPT / REACQUIRE / REFER)
SIMULATED:
  * the injected initial fault (SyntheticDegradationPipeline)
  * the effect of the operator maneuver: severity is reduced by `recovery_step` per
    REACQUIRE (a persistent fault is never reduced). This models an operator who follows
    the advice; it does NOT measure operator behaviour. Outcomes therefore test the
    policy/diagnosis logic, not real-world recovery rates.

Assessment modes
----------------
* "trained_models": view_model (CardiacEchoNet) + quality_model (QualityAssessorNet) supply
  the predicted quality and view probabilities used for adequacy.
* "image_proxy" (no models): quality comes from a physics-feature score and the "view
  confidence" is replaced by the classifier's probability that the frame is fault-free.
  This is a stand-in and must be reported as such.
Independently, the adequacy estimator may be fitted (learned) or heuristic.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
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
    true_fault: str = "none"          # ground truth, for evaluation only
    true_severity: float = 0.0        # ground truth, for evaluation only


@dataclass
class SimulationResult:
    case_id: str
    initial_fault: str
    final_action: str
    total_attempts: int
    trajectory: List[SimulationStep]
    success: bool
    assessment_mode: str = "image_proxy+heuristic_adequacy"


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
        view_model: Optional[Any] = None,
        quality_model: Optional[Any] = None,
        device: str = "cpu",
        recovery_step: float = 0.5,
    ):
        self.adequacy_estimator = adequacy_estimator
        self.cause_classifier = cause_classifier
        self.policy = policy
        self.degradation_pipeline = degradation_pipeline or SyntheticDegradationPipeline(random_seed=42)
        self.view_model = view_model
        self.quality_model = quality_model
        self.device = device
        self.recovery_step = recovery_step
        self._transform = None

        if (view_model is None) != (quality_model is None):
            raise ValueError("Provide both view_model and quality_model, or neither.")
        if view_model is not None:
            from ..data.transforms import get_transforms
            self._transform = get_transforms(is_train=False)
            self.view_model.eval()
            self.quality_model.eval()

    @property
    def assessment_mode(self) -> str:
        base = "trained_models" if self.view_model is not None else "image_proxy"
        adeq = "fitted_adequacy" if self.adequacy_estimator.is_fitted else "heuristic_adequacy"
        return f"{base}+{adeq}"

    # ------------------------------------------------------------------ assessment
    @staticmethod
    def _proxy_quality(features: np.ndarray) -> float:
        """Image-only quality score in [0, 9] from sector-masked physics features."""
        lum, sat, _p90, sharp, col_drop, _col_min, cov = (float(features[i]) for i in range(7))
        brightness_ok = 1.0 - min(1.0, abs(lum - 0.40) / 0.30)
        sharp_ok = float(np.clip(sharp / 0.15, 0.0, 1.0))
        sat_ok = 1.0 - float(np.clip(sat / 0.15, 0.0, 1.0))
        shadow_ok = 1.0 - float(np.clip(col_drop / 0.5, 0.0, 1.0))
        cov_ok = float(np.clip(cov / 0.25, 0.0, 1.0))
        score = 0.30 * brightness_ok + 0.30 * sharp_ok + 0.15 * sat_ok + 0.15 * shadow_ok + 0.10 * cov_ok
        return float(np.clip(9.0 * score, 0.0, 9.0))

    def _model_outputs(self, image: Image.Image):
        import torch
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.view_model(x), dim=1).cpu().numpy()
            quality = float(self.quality_model(x).cpu().numpy().reshape(-1)[0])
        return float(np.clip(quality, 0.0, 9.0)), probs

    def assess(self, image: Image.Image) -> Dict[str, Any]:
        """
        Image-only assessment. Takes NO ground-truth fault or severity, by design.
        """
        features = self.cause_classifier.extract_physics_features(image)
        feats_row = features.reshape(1, -1)
        pred_faults, confs, _ = self.cause_classifier.predict(feats_row)
        diag_fault, diag_conf = str(pred_faults[0]), float(confs[0])

        if self.view_model is not None:
            est_quality, view_probs = self._model_outputs(image)
        else:
            est_quality = self._proxy_quality(features)
            p_none = float(self.cause_classifier.predict_proba(feats_row)[0][
                self.cause_classifier.FAULT_CLASSES.index("none")
            ])
            n_cls = len(self.cause_classifier.FAULT_CLASSES)
            view_probs = np.full((1, n_cls), (1.0 - p_none) / (n_cls - 1))
            view_probs[0, 0] = p_none

        adeq_feats = self.adequacy_estimator.extract_adequacy_features(
            predicted_quality=np.array([est_quality]),
            view_probabilities=view_probs,
        )
        adeq_prob = float(self.adequacy_estimator.predict_probability(adeq_feats)[0])
        return {
            "diagnosed_fault": diag_fault,
            "fault_confidence": diag_conf,
            "quality": est_quality,
            "adequacy_prob": adeq_prob,
        }

    # ------------------------------------------------------------------ simulation
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

        curr_fault = initial_fault
        curr_severity = 0.0 if initial_fault == "none" else float(initial_severity)

        if initial_fault != "none":
            current_image, _ = self.degradation_pipeline.apply(
                current_image, degradation_type=initial_fault, severity=curr_severity
            )

        attempt = 1
        final_action = "REFER"

        while attempt <= self.policy.max_attempts + 1:
            a = self.assess(current_image)

            decision = self.policy.evaluate(
                adequacy_prob=a["adequacy_prob"],
                diagnosed_fault=a["diagnosed_fault"],
                fault_confidence=a["fault_confidence"],
                attempt_count=attempt,
            )

            step = SimulationStep(
                attempt=attempt,
                quality_grade=a["quality"],
                adequacy_prob=a["adequacy_prob"],
                diagnosed_fault=a["diagnosed_fault"],
                fault_confidence=a["fault_confidence"],
                decision=decision,
                applied_maneuver=None,
                true_fault=curr_fault,
                true_severity=curr_severity,
            )
            trajectory.append(step)
            final_action = decision.action

            if decision.action in ("ACCEPT", "REFER"):
                break

            # REACQUIRE: simulate the operator following the advice (simulated, not measured)
            step.applied_maneuver = decision.reacquisition_suggestion
            attempt += 1

            if not persistent:
                curr_severity = max(0.0, curr_severity - self.recovery_step)
                if curr_severity <= 0.2:
                    current_image = base_image.copy()
                    curr_fault = "none"
                    curr_severity = 0.0
                else:
                    current_image, _ = self.degradation_pipeline.apply(
                        base_image.copy(), degradation_type=curr_fault, severity=curr_severity
                    )
            # persistent: fault unchanged across attempts

        return SimulationResult(
            case_id=case_id,
            initial_fault=initial_fault,
            final_action=final_action,
            total_attempts=len(trajectory),
            trajectory=trajectory,
            success=final_action == "ACCEPT",
            assessment_mode=self.assessment_mode,
        )
