"""
Interactive demonstration script for SelfCheckUltrasound.

Shows how to programmatically use the self_check_ultrasound modules
(AdequacyEstimator, CauseClassifier, ReacquisitionPolicy, and ClosedLoopSimulator)
on any given echocardiography image.

Usage:
  python scripts/interactive_demo.py
  python scripts/interactive_demo.py --fault motion_blur --severity 0.65
  python scripts/interactive_demo.py --fault acoustic_shadow --severity 0.80
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy
from self_check_ultrasound.guidance.simulator import ClosedLoopSimulator


def main():
    parser = argparse.ArgumentParser(description="SelfCheckUltrasound: Interactive Python Demonstration")
    parser.add_argument(
        "--image",
        type=str,
        default="FRDR_dataset_1479_download_216_202610160047/Cactus Dataset/Images Dataset/A4C/1_D14_frame_1088_v2.jpg",
        help="Path to the input ultrasound image",
    )
    parser.add_argument(
        "--fault",
        type=str,
        default="under_gain",
        choices=["none", "under_gain", "over_gain", "motion_blur", "acoustic_shadow", "off_axis"],
        help="Initial acoustic fault to simulate",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.70,
        help="Initial severity level (0.1 to 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Task adequacy acceptance threshold",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum reacquisition attempts before safety referral",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = project_root / image_path

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("SelfCheckUltrasound: Interactive Programmatic Demo")
    print("=" * 70)
    print(f"Base Image:      {image_path.name}")
    print(f"Initial Fault:   {args.fault} (Severity: {args.severity:.2f})")
    print(f"Policy Settings: Adequacy Threshold = {args.threshold}, Max Attempts = {args.max_attempts}")
    print("-" * 70)

    # 1. Initialize core guidance modules
    adequacy_estimator = AdequacyEstimator(adequacy_threshold=args.threshold)
    cause_classifier = CauseClassifier(min_confidence=0.60)
    policy = ReacquisitionPolicy(adequacy_threshold=args.threshold, max_attempts=args.max_attempts)
    simulator = ClosedLoopSimulator(adequacy_estimator, cause_classifier, policy)

    # 2. Load the ultrasound image
    sample_img = Image.open(image_path).convert("RGB")

    # 3. Execute closed-loop simulation
    result = simulator.run_simulation(
        base_image=sample_img,
        initial_fault=args.fault,
        initial_severity=args.severity,
        case_id="interactive_demo",
    )

    # 4. Display trajectory results
    print(f"\nFinal Outcome: [{result.final_action}] after {result.total_attempts} attempt(s).\n")
    for step in result.trajectory:
        action_badge = f"[{step.decision.action}]"
        print(f"  Attempt {step.attempt}: Adequacy = {step.adequacy_prob:.2f} -> {action_badge:12s} {step.decision.reason}")
        if step.applied_maneuver:
            print(f"    --> Operator Guidance: {step.applied_maneuver}")

    print("=" * 70)


if __name__ == "__main__":
    main()
