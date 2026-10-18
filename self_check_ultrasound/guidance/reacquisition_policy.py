"""
Reacquisition Policy & Guidance Mapper (Modules M6 and M7).
Applies confidence-aware triage:
- ACCEPT: Adequacy probability meets validated threshold (checked FIRST, so an image that
          is fixed by the final allowed maneuver is still accepted).
- REACQUIRE: Adequacy is insufficient, supported cause is diagnosed with high confidence,
             and within attempt limit.
- REFER: Cause is uncertain, image is uninterpretable, attempt limit is reached,
         or system abstains due to low confidence.

Attempt numbering: attempt_count is the 1-based index of the frame being evaluated.
With max_attempts=N the operator may be advised to reacquire after frames 1..N, and
frame N+1 is the last check: it is accepted if adequate, otherwise referred.
"""

from dataclasses import dataclass
from typing import Optional


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

    def _decision(self, action, reason, suggestion, adequacy_prob, diagnosed_fault, fault_confidence, attempt_count):
        return PolicyDecision(
            action=action,
            reason=reason,
            reacquisition_suggestion=suggestion,
            adequacy_prob=float(adequacy_prob),
            diagnosed_fault=diagnosed_fault,
            fault_confidence=float(fault_confidence),
            attempt_count=attempt_count,
        )

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
        args = (adequacy_prob, diagnosed_fault, fault_confidence, attempt_count)

        # 1. Acceptance check (first: a frame that is adequate is always accepted, even on
        #    the final attempt after the last allowed maneuver).
        if adequacy_prob >= self.adequacy_threshold:
            return self._decision(
                "ACCEPT",
                f"Adequacy probability ({adequacy_prob:.2f}) meets acceptance threshold ({self.adequacy_threshold:.2f}).",
                None,
                *args,
            )

        # 2. Attempt limit enforcement (frame is inadequate)
        if attempt_count > self.max_attempts:
            return self._decision(
                "REFER",
                f"Maximum reacquisition attempt limit reached ({self.max_attempts} attempts).",
                "Refer to senior sonographer or cardiology expert.",
                *args,
            )

        # 3. Insufficient adequacy: check fault confidence and abstention
        if fault_confidence < self.min_confidence_for_action:
            return self._decision(
                "REFER",
                f"Abstaining: Fault diagnosis confidence ({fault_confidence:.2f}) is below minimum actionable threshold ({self.min_confidence_for_action:.2f}).",
                "Refer to expert: artifact cause indeterminate.",
                *args,
            )

        # 4. Fault is "none" or unknown but adequacy failed
        if diagnosed_fault not in self.SUGGESTIONS or diagnosed_fault == "none":
            return self._decision(
                "REFER",
                "Adequacy failed but no actionable physical artifact was confidently recognized.",
                "Refer to expert for manual anatomical scan assessment.",
                *args,
            )

        # 5. Actionable reacquisition
        return self._decision(
            "REACQUIRE",
            f"Adequacy insufficient ({adequacy_prob:.2f}). Confidently diagnosed {diagnosed_fault} (conf: {fault_confidence:.2f}).",
            self.SUGGESTIONS[diagnosed_fault],
            *args,
        )
