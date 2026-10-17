"""
Reacquisition Policy & Guidance Mapper (Modules M6 and M7).
Applies confidence-aware triage:
- ACCEPT: Adequacy probability meets validated threshold.
- REACQUIRE: Adequacy is insufficient, supported cause is diagnosed with high confidence,
             and within attempt limit.
- REFER: Cause is uncertain, image is uninterpretable, attempt limit is reached,
         or system abstains due to low confidence.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PolicyDecision:
    action: str  # "ACCEPT", "REACQUIRE", or "REFER"
    reason: str
    reacquisition_suggestion: Optional[str]
    adequacy_prob: float
    diagnosed_fault: str
    fault_confidence: float
    attempt_count: int


class ReacquisitionPolicy:
    """
    Decision policy for cardiac ultrasound quality control and guidance.
    """

    ACTIONS = ("ACCEPT", "REACQUIRE", "REFER")

    SUGGESTIONS = {
        "under_gain": "Increase 2D overall gain or time-gain compensation (TGC).",
        "over_gain": "Decrease 2D overall gain to resolve acoustic washout.",
        "acoustic_shadow": "Reposition probe to avoid rib acoustic shadowing; adjust intercostal angle.",
        "motion_blur": "Stabilize probe and ask patient to suspend respiration momentarily.",
        "off_axis": "Re-align probe to orthogonal anatomical standard view axis.",
        "none": "No adjustment required.",
    }

    def __init__(
        self,
        adequacy_threshold: float = 0.70,
        min_confidence_for_action: float = 0.60,
        max_attempts: int = 3,
    ):
        self.adequacy_threshold = adequacy_threshold
        self.min_confidence_for_action = min_confidence_for_action
        self.max_attempts = max_attempts

    def evaluate(
        self,
        adequacy_prob: float,
        diagnosed_fault: str,
        fault_confidence: float,
        attempt_count: int = 1,
    ) -> PolicyDecision:
        """
        Evaluates acquisition state and outputs decision.
        """
        # 1. Attempt limit enforcement
        if attempt_count > self.max_attempts:
            return PolicyDecision(
                action="REFER",
                reason=f"Maximum reacquisition attempt limit reached ({self.max_attempts} attempts).",
                reacquisition_suggestion="Refer to senior sonographer or cardiology expert.",
                adequacy_prob=float(adequacy_prob),
                diagnosed_fault=diagnosed_fault,
                fault_confidence=float(fault_confidence),
                attempt_count=attempt_count,
            )

        # 2. Acceptance check
        if adequacy_prob >= self.adequacy_threshold:
            return PolicyDecision(
                action="ACCEPT",
                reason=f"Adequacy probability ({adequacy_prob:.2f}) meets acceptance threshold ({self.adequacy_threshold:.2f}).",
                reacquisition_suggestion=None,
                adequacy_prob=float(adequacy_prob),
                diagnosed_fault=diagnosed_fault,
                fault_confidence=float(fault_confidence),
                attempt_count=attempt_count,
            )

        # 3. Insufficient adequacy: check fault confidence and abstention
        if fault_confidence < self.min_confidence_for_action:
            return PolicyDecision(
                action="REFER",
                reason=f"Abstaining: Fault diagnosis confidence ({fault_confidence:.2f}) is below minimum actionable threshold ({self.min_confidence_for_action:.2f}).",
                reacquisition_suggestion="Refer to expert: artifact cause indeterminate.",
                adequacy_prob=float(adequacy_prob),
                diagnosed_fault=diagnosed_fault,
                fault_confidence=float(fault_confidence),
                attempt_count=attempt_count,
            )

        # If fault is "none" or unknown but adequacy failed
        if diagnosed_fault not in self.SUGGESTIONS or diagnosed_fault == "none":
            return PolicyDecision(
                action="REFER",
                reason="Adequacy failed but no actionable physical artifact was confidently recognized.",
                reacquisition_suggestion="Refer to expert for manual anatomical scan assessment.",
                adequacy_prob=float(adequacy_prob),
                diagnosed_fault=diagnosed_fault,
                fault_confidence=float(fault_confidence),
                attempt_count=attempt_count,
            )

        # 4. Actionable reacquisition
        suggestion = self.SUGGESTIONS[diagnosed_fault]
        return PolicyDecision(
            action="REACQUIRE",
            reason=f"Adequacy insufficient ({adequacy_prob:.2f}). Confidently diagnosed {diagnosed_fault} (conf: {fault_confidence:.2f}).",
            reacquisition_suggestion=suggestion,
            adequacy_prob=float(adequacy_prob),
            diagnosed_fault=diagnosed_fault,
            fault_confidence=float(fault_confidence),
            attempt_count=attempt_count,
        )
