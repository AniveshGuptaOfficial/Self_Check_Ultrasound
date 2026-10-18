"""
Evaluate the sector-masked CauseClassifier (Module M5) with two kinds of evidence:

  1. SYNTHETIC: held-out sequence groups (val and test) degraded by our own pipeline,
     reported per fault and per severity. This only shows the synthetic faults are separable.
  2. REAL: AUROC of P(none) against genuine expert quality grades (high >= 7 vs low <= 3).

Trains on the TRAIN split only. Writes artifacts/cause_classifier_eval.json.

Usage:
  python scripts/evaluate_cause_classifier.py [--train_per_view 12] [--eval_per_view 8]
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.models.cause_classifier import CauseClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_per_view", type=int, default=12)
    ap.add_argument("--eval_per_view", type=int, default=8)
    ap.add_argument("--real_per_view", type=int, default=25)
    args = ap.parse_args()

    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} not found. Run scripts/build_manifest.py first.")
    splits_df = pd.read_csv(splits_path, low_memory=False)
    root = default_config.paths.dataset_dir

    clf = CauseClassifier(min_confidence=default_config.policy.min_confidence_for_action)
    n = clf.train_on_reference_samples(splits_df, root, n_samples_per_view=args.train_per_view, split="train")
    print(f"Trained on {n} samples from the TRAIN split (diverse sequence groups).")

    results = {"n_train_samples": n, "synthetic": {}, "real_expert_grades": {}}
    for split in ("val", "test"):
        syn = clf.evaluate_on_synthetic(splits_df, root, split=split, n_samples_per_view=args.eval_per_view)
        real = clf.evaluate_against_expert_grades(splits_df, root, split=split, n_per_view=args.real_per_view)
        results["synthetic"][split] = syn
        results["real_expert_grades"][split] = real
        print(f"\n[{split}] SYNTHETIC faults: acc={syn.get('accuracy', float('nan')):.3f} "
              f"(n={syn.get('n')}), abstain={syn.get('abstain_rate', float('nan')):.3f}")
        print(f"        per severity: {syn.get('per_severity_accuracy')}")
        print(f"        per fault   : {syn.get('per_fault_accuracy')}")
        print(f"[{split}] REAL grades: {real}")

    out = default_config.paths.artifacts_dir / "cause_classifier_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
