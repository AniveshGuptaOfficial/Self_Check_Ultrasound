"""
Fit the task-specific AdequacyEstimator (Module M4) on REAL held-out downstream outcomes.

Target:  adequate = 1 if the trained view classifier (CardiacEchoNet) predicts the correct
         standard cardiac view for the frame AND true view is not Random, else 0.
Features: predicted quality (QualityAssessorNet) + view-probability confidence / margin / entropy.

Protocol (no in-sample fitting):
  * the two CNNs were trained on the TRAIN split only;
  * the estimator is fitted on the VAL split and reported with sequence-group-aware
    out-of-fold predictions (GroupKFold by sequence_group);
  * the final estimator is then evaluated once on the untouched TEST split.
Note: checkpoints were chosen by best-epoch on VAL, so VAL features are mildly optimistic;
the TEST numbers are the ones to report.

Requires: artifacts/cardiac_echo_net.pt and artifacts/quality_assessor_net.pt
          (run train_view_baseline.py and train_quality_baseline.py first).
Writes:   artifacts/adequacy_estimator.joblib, artifacts/adequacy_metrics.json

Usage:
  python scripts/train_adequacy.py [--max_val_samples 3000] [--max_test_samples 3000]
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.eval_utils import balanced_subsample
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator


def binary_ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.any():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(ece)


def collect_features(df, view_model, quality_model, estimator, device, batch_size):
    ds = CardiacUltrasoundDataset(df, dataset_root=default_config.paths.dataset_dir,
                                  transform=get_transforms(is_train=False, image_size=default_config.data.image_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_all, qual_all, labels_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            probs_all.append(torch.softmax(view_model(x), dim=1).cpu().numpy())
            qual_all.append(quality_model(x).cpu().numpy())
            labels_all.append(batch["view_label"].numpy())
    probs = np.concatenate(probs_all)
    quality = np.clip(np.concatenate(qual_all), 0.0, 9.0)
    labels = np.concatenate(labels_all)
    X = estimator.extract_adequacy_features(quality, probs)
    
    # Adequate = downstream classifier predicts correct view AND true view is a standard cardiac view (not Random)
    random_idx = default_config.data.class_to_idx.get("Random", 4)
    preds = np.argmax(probs, axis=1)
    y = ((preds == labels) & (labels != random_idx)).astype(int)
    return X, y


def report(y, p, threshold):
    out = {"n": int(len(y)), "base_rate_adequate": float(y.mean())}
    if len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = binary_ece(y, p)
    acc = p >= threshold
    out["threshold"] = threshold
    out["accept_rate"] = float(acc.mean())
    out["view_accuracy_on_accepted"] = float(y[acc].mean()) if acc.any() else None
    out["view_accuracy_on_rejected"] = float(y[~acc].mean()) if (~acc).any() else None
    out["selective_accuracy"] = float(y[acc].mean()) if acc.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_val_samples", type=int, default=3000)
    ap.add_argument("--max_test_samples", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    set_seed(default_config.data.random_seed)
    device = torch.device(default_config.training.device)
    art = default_config.paths.artifacts_dir
    for need in ("cardiac_echo_net.pt", "quality_assessor_net.pt"):
        if not (art / need).exists():
            raise FileNotFoundError(f"{art / need} missing. Train the baselines first.")

    view_model = CardiacEchoNet(num_classes=default_config.data.num_classes).to(device)
    view_model.load_state_dict(torch.load(art / "cardiac_echo_net.pt", map_location=device, weights_only=True))
    quality_model = QualityAssessorNet().to(device)
    quality_model.load_state_dict(torch.load(art / "quality_assessor_net.pt", map_location=device, weights_only=True))
    view_model.eval()
    quality_model.eval()

    splits_df = pd.read_csv(default_config.paths.splits_path, low_memory=False)
    n_cls = len(default_config.data.classes)
    val_df = balanced_subsample(splits_df[splits_df["split"] == "val"], args.max_val_samples, n_cls)
    test_df = balanced_subsample(splits_df[splits_df["split"] == "test"], args.max_test_samples, n_cls)
    print(f"Adequacy fit on {len(val_df)} val frames; held-out report on {len(test_df)} test frames")

    threshold = default_config.policy.adequacy_threshold
    est = AdequacyEstimator(adequacy_threshold=threshold)

    X_val, y_val = collect_features(val_df, view_model, quality_model, est, device, args.batch_size)
    X_test, y_test = collect_features(test_df, view_model, quality_model, est, device, args.batch_size)

    # sequence-group-aware out-of-fold predictions on VAL
    groups = val_df["sequence_group"].values
    n_splits = min(5, len(set(groups)))
    oof = np.full(len(y_val), np.nan)
    if n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(X_val, y_val, groups):
            if len(set(y_val[tr].tolist())) < 2:
                continue
            fold = AdequacyEstimator(adequacy_threshold=threshold)
            fold.fit(X_val[tr], y_val[tr])
            oof[te] = fold.predict_probability(X_val[te])
    have = ~np.isnan(oof)
    results = {"val_out_of_fold": report(y_val[have], oof[have], threshold) if have.any() else None}

    est.fit(X_val, y_val)
    results["test_heldout"] = report(y_test, est.predict_probability(X_test), threshold)
    heur = AdequacyEstimator(adequacy_threshold=threshold)  # unfitted heuristic for comparison
    results["test_heldout_heuristic_baseline"] = report(y_test, heur.predict_probability(X_test), threshold)

    est.save(art / "adequacy_estimator.joblib")
    with open(art / "adequacy_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved estimator to {art / 'adequacy_estimator.joblib'}")


if __name__ == "__main__":
    main()
