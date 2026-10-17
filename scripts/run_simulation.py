"""
Closed-Loop Reacquisition Simulation Experiment (Phase 4).
Evaluates the full end-to-end quality feedback pipeline:
Frame -> Quality Assessment -> Adequacy Estimation -> Policy Decision -> Guidance -> Re-check.

Benchmarks:
1. Incident-free transitions (nominal images)
2. Correctable recovery transitions (under-gain, motion-blur, acoustic-shadow)
3. Safe failure and referral transitions (attempt limit reached or unresolvable artifacts)
"""

import sys
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy
from self_check_ultrasound.guidance.simulator import ClosedLoopSimulator
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


def main():
    set_seed(42)
    print("=" * 70)
    print("SelfCheckUltrasound: Closed-Loop Reacquisition Simulation (Phase 4)")
    print("=" * 70)

    # 1. Initialize modules
    adequacy_estimator = AdequacyEstimator(
        adequacy_threshold=default_config.policy.adequacy_threshold
    )
    cause_classifier = CauseClassifier(
        min_confidence=default_config.policy.min_confidence_for_action
    )
    policy = ReacquisitionPolicy(
        adequacy_threshold=default_config.policy.adequacy_threshold,
        min_confidence_for_action=default_config.policy.min_confidence_for_action,
        max_attempts=default_config.policy.max_reacquisition_attempts,
    )
    degradation_pipeline = SyntheticDegradationPipeline(random_seed=42)

    simulator = ClosedLoopSimulator(
        adequacy_estimator=adequacy_estimator,
        cause_classifier=cause_classifier,
        policy=policy,
        degradation_pipeline=degradation_pipeline,
    )

    # 2. Load validation sample
    splits_path = default_config.paths.splits_path
    dataset_root = default_config.paths.dataset_dir
    splits_df = pd.read_csv(splits_path, low_memory=False)

    print("Calibrating CauseClassifier physics indicators on reference samples...")
    cause_classifier.train_on_reference_samples(splits_df, dataset_root, n_samples_per_view=5)

    # Find a nominal high-quality frame (grade >= 7)
    high_q_rows = splits_df[
        (splits_df["split"] == "val")
        & (splits_df["quality_grade"] >= 7)
    ]
    if len(high_q_rows) == 0:
        high_q_rows = splits_df[splits_df["quality_grade"] >= 6]
    sample_row = high_q_rows.iloc[0]
    base_img_path = dataset_root / sample_row["relative_path"]
    base_img = Image.open(base_img_path).convert("RGB")

    print(f"Base Test Image: {sample_row['filename']} (View: {sample_row['view_class']}, Ground Truth Grade: {sample_row['quality_grade']})")

    # 3. Execute Canonical Clinical Trajectories
    cases = [
        # Trajectory 1: Incident-Free (Nominal)
        {
            "case_id": "Case_1_IncidentFree",
            "description": "Nominal high-quality echocardiogram without acoustic faults",
            "initial_fault": "none",
            "initial_severity": 0.0,
            "persistent": False,
        },
        # Trajectory 2: Correctable Fault (Under-gain)
        {
            "case_id": "Case_2_Recovery_UnderGain",
            "description": "Acoustic attenuation / insufficient overall gain",
            "initial_fault": "under_gain",
            "initial_severity": 0.70,
            "persistent": False,
        },
        # Trajectory 3: Correctable Fault (Motion Blur)
        {
            "case_id": "Case_3_Recovery_MotionBlur",
            "description": "Transducer motion artifact / respiration blur",
            "initial_fault": "motion_blur",
            "initial_severity": 0.80,
            "persistent": False,
        },
        # Trajectory 4: Correctable Fault (Acoustic Shadow)
        {
            "case_id": "Case_4_Recovery_RibShadow",
            "description": "Rib acoustic occlusion across lateral cardiac sector",
            "initial_fault": "acoustic_shadow",
            "initial_severity": 0.70,
            "persistent": False,
        },
        # Trajectory 5: Safe Failure / Referral (Persistent unresolvable fault)
        {
            "case_id": "Case_5_Referral_PersistentFault",
            "description": "Severe persistent uninterpretable fault exceeding attempt limit",
            "initial_fault": "off_axis",
            "initial_severity": 0.95,
            "persistent": True,
        },
    ]

    all_results = []

    print("\n--- Running Simulation Cases ---")
    for c in cases:
        print(f"\n>> {c['case_id']}: {c['description']}")
        res = simulator.run_simulation(
            base_image=base_img,
            initial_fault=c["initial_fault"],
            initial_severity=c["initial_severity"],
            case_id=c["case_id"],
            persistent=c.get("persistent", False),
        )

        for s in res.trajectory:
            action_badge = f"[{s.decision.action}]"
            print(
                f"   Attempt {s.attempt}: Adequacy={s.adequacy_prob:.2f} | "
                f"Fault={s.diagnosed_fault} (conf={s.fault_confidence:.2f}) -> {action_badge} {s.decision.reason}"
            )
            if s.applied_maneuver:
                print(f"     => Guidance Advice: {s.applied_maneuver}")

        print(f"   Outcome: {res.final_action} in {res.total_attempts} attempts (Success: {res.success})")

        # Record for JSON
        serializable_traj = []
        for s in res.trajectory:
            serializable_traj.append({
                "attempt": s.attempt,
                "quality_grade": s.quality_grade,
                "adequacy_prob": s.adequacy_prob,
                "diagnosed_fault": s.diagnosed_fault,
                "fault_confidence": s.fault_confidence,
                "action": s.decision.action,
                "reason": s.decision.reason,
                "applied_maneuver": s.applied_maneuver,
            })

        all_results.append({
            "case_id": res.case_id,
            "description": c["description"],
            "initial_fault": res.initial_fault,
            "final_action": res.final_action,
            "total_attempts": res.total_attempts,
            "success": res.success,
            "trajectory": serializable_traj,
        })

    # 4. Save results to artifacts
    output_path = default_config.paths.artifacts_dir / "simulation_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SIMULATION TRAJECTORY SUMMARY")
    print("=" * 70)
    print(f"{'Case ID':<30s} | {'Fault':<16s} | {'Final Action':<12s} | {'Attempts':<8s} | {'Status':<8s}")
    print("-" * 70)
    for r in all_results:
        status_str = "PASS" if (r["success"] or r["final_action"] == "REFER") else "FAIL"
        print(f"{r['case_id']:<30s} | {r['initial_fault']:<16s} | {r['final_action']:<12s} | {r['total_attempts']:<8d} | {status_str:<8s}")
    print("=" * 70)
    print(f"Full simulation logs saved to: {output_path}")


if __name__ == "__main__":
    main()
