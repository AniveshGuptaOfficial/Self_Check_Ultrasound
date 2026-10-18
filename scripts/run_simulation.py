"""
Closed-Loop Reacquisition Simulation Experiment (Phase 4).
Frame -> Quality Assessment -> Adequacy Estimation -> Policy Decision -> Guidance -> Re-check.

What is simulated vs measured: see the docstring of guidance/simulator.py. In short, the fault
is injected and the operator's compliance is simulated; diagnosis, quality, adequacy and the
policy decision are computed from the image alone (no ground truth is fed to the policy).

Assessment modes (printed and stored with the results):
  trained : uses artifacts/cardiac_echo_net.pt + quality_assessor_net.pt (and
            artifacts/adequacy_estimator.joblib if present, else the heuristic adequacy score)
  proxy   : image-physics quality score + classifier 'clean' probability; heuristic adequacy
  auto    : trained if both checkpoints exist, otherwise proxy   (default)

Usage:
  python scripts/run_simulation.py [--mode auto|trained|proxy]
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy
from self_check_ultrasound.guidance.simulator import ClosedLoopSimulator
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


CASES = [
    {"case_id": "Case_1_IncidentFree", "description": "Nominal high-quality echocardiogram without acoustic faults",
     "initial_fault": "none", "initial_severity": 0.0, "persistent": False},
    {"case_id": "Case_2_Recovery_UnderGain", "description": "Acoustic attenuation / insufficient overall gain",
     "initial_fault": "under_gain", "initial_severity": 0.70, "persistent": False},
    {"case_id": "Case_3_Recovery_MotionBlur", "description": "Transducer motion artifact / respiration blur",
     "initial_fault": "motion_blur", "initial_severity": 0.80, "persistent": False},
    {"case_id": "Case_4_Recovery_RibShadow", "description": "Rib acoustic occlusion across lateral cardiac sector",
     "initial_fault": "acoustic_shadow", "initial_severity": 0.70, "persistent": False},
    {"case_id": "Case_5_Referral_PersistentFault", "description": "Severe persistent uninterpretable fault exceeding attempt limit",
     "initial_fault": "off_axis", "initial_severity": 0.95, "persistent": True},
]


def load_trained_models(artifacts_dir, device):
    import torch
    from self_check_ultrasound.models.view_classifier import CardiacEchoNet
    from self_check_ultrasound.models.quality_assessor import QualityAssessorNet

    view_model = CardiacEchoNet(num_classes=default_config.data.num_classes).to(device)
    view_model.load_state_dict(torch.load(artifacts_dir / "cardiac_echo_net.pt", map_location=device, weights_only=True))
    quality_model = QualityAssessorNet().to(device)
    quality_model.load_state_dict(torch.load(artifacts_dir / "quality_assessor_net.pt", map_location=device, weights_only=True))
    return view_model, quality_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "trained", "proxy"], default="auto")
    args = ap.parse_args()

    set_seed(42)
    art = default_config.paths.artifacts_dir
    device = default_config.training.device
    print("=" * 70)
    print("SelfCheckUltrasound: Closed-Loop Reacquisition Simulation (Phase 4)")
    print("=" * 70)

    have_ckpt = (art / "cardiac_echo_net.pt").exists() and (art / "quality_assessor_net.pt").exists()
    use_trained = args.mode == "trained" or (args.mode == "auto" and have_ckpt)
    if args.mode == "trained" and not have_ckpt:
        raise FileNotFoundError("Trained mode requested but checkpoints are missing in artifacts/.")

    adequacy_path = art / "adequacy_estimator.joblib"
    if use_trained and adequacy_path.exists():
        adequacy_estimator = AdequacyEstimator.load(adequacy_path)
        adequacy_estimator.adequacy_threshold = default_config.policy.adequacy_threshold
    else:
        adequacy_estimator = AdequacyEstimator(adequacy_threshold=default_config.policy.adequacy_threshold)

    view_model = quality_model = None
    if use_trained:
        view_model, quality_model = load_trained_models(art, device)

    cause_classifier = CauseClassifier(min_confidence=default_config.policy.min_confidence_for_action)
    policy = ReacquisitionPolicy(
        adequacy_threshold=default_config.policy.adequacy_threshold,
        min_confidence_for_action=default_config.policy.min_confidence_for_action,
        max_attempts=default_config.policy.max_reacquisition_attempts,
    )
    simulator = ClosedLoopSimulator(
        adequacy_estimator=adequacy_estimator,
        cause_classifier=cause_classifier,
        policy=policy,
        degradation_pipeline=SyntheticDegradationPipeline(random_seed=42),
        view_model=view_model,
        quality_model=quality_model,
        device=device,
    )
    print(f"Assessment mode: {simulator.assessment_mode}")
    if not adequacy_estimator.is_fitted:
        print("WARNING: adequacy is the HEURISTIC score (not a learned/calibrated probability). "
              "Run scripts/train_adequacy.py to fit it.")

    splits_df = pd.read_csv(default_config.paths.splits_path, low_memory=False)
    dataset_root = default_config.paths.dataset_dir

    print("Calibrating CauseClassifier on TRAIN-split reference frames (diverse sequence groups)...")
    n = cause_classifier.train_on_reference_samples(splits_df, dataset_root, n_samples_per_view=12, split="train")
    print(f"  trained on {n} synthetic-degradation samples")

    high_q_rows = splits_df[(splits_df["split"] == "val") & (splits_df["quality_grade"] >= 7)]
    if len(high_q_rows) == 0:
        high_q_rows = splits_df[splits_df["quality_grade"] >= 6]
    sample_row = high_q_rows.iloc[0]
    base_img = Image.open(dataset_root / sample_row["relative_path"]).convert("RGB")
    print(f"Base Test Image: {sample_row['filename']} (View: {sample_row['view_class']}, Ground Truth Grade: {sample_row['quality_grade']})")

    all_results = []
    print("\n--- Running Simulation Cases ---")
    for c in CASES:
        print(f"\n>> {c['case_id']}: {c['description']}")
        res = simulator.run_simulation(
            base_image=base_img,
            initial_fault=c["initial_fault"],
            initial_severity=c["initial_severity"],
            case_id=c["case_id"],
            persistent=c["persistent"],
        )
        traj = []
        for s in res.trajectory:
            print(f"   Attempt {s.attempt}: Adequacy={s.adequacy_prob:.2f} | "
                  f"Dx={s.diagnosed_fault} (conf={s.fault_confidence:.2f}) [true={s.true_fault}] -> "
                  f"[{s.decision.action}] {s.decision.reason}")
            if s.applied_maneuver:
                print(f"     => Guidance Advice: {s.applied_maneuver}")
            traj.append({
                "attempt": s.attempt,
                "quality_grade": s.quality_grade,
                "adequacy_prob": s.adequacy_prob,
                "diagnosed_fault": s.diagnosed_fault,
                "true_fault": s.true_fault,
                "diagnosis_correct": s.diagnosed_fault == s.true_fault,
                "fault_confidence": s.fault_confidence,
                "action": s.decision.action,
                "reason": s.decision.reason,
                "applied_maneuver": s.applied_maneuver,
            })
        print(f"   Outcome: {res.final_action} in {res.total_attempts} attempts (Success: {res.success})")
        all_results.append({
            "case_id": res.case_id,
            "description": c["description"],
            "initial_fault": res.initial_fault,
            "final_action": res.final_action,
            "total_attempts": res.total_attempts,
            "success": res.success,
            "assessment_mode": res.assessment_mode,
            "trajectory": traj,
        })

    output_path = art / "simulation_trajectories.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    expected = {"Case_1_IncidentFree": "ACCEPT", "Case_5_Referral_PersistentFault": "REFER"}
    print("\n" + "=" * 78)
    print(f"SIMULATION TRAJECTORY SUMMARY  (mode: {simulator.assessment_mode})")
    print("=" * 78)
    print(f"{'Case ID':<30s} | {'Fault':<16s} | {'Final':<9s} | {'Attempts':<8s} | {'Dx correct@1':<12s}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['case_id']:<30s} | {r['initial_fault']:<16s} | {r['final_action']:<9s} | "
              f"{r['total_attempts']:<8d} | {str(r['trajectory'][0]['diagnosis_correct']):<12s}")
    print("=" * 78)
    print("Outcomes are reported as observed; recovery after a maneuver is SIMULATED (see simulator docstring).")
    print(f"Full simulation logs saved to: {output_path}")


if __name__ == "__main__":
    main()
