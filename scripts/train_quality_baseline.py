"""
Training and evaluation script for Image Quality Assessment baseline (Module M3).
Compares:
1. Mean heuristic baseline (constant = mean TRAIN grade)
2. QualityAssessorNet (CNN regressing the 0-9 expert grade; MSE loss, AdamW)

Protocol: train on TRAIN, select best epoch on VAL (MAE), report VAL (selection set) and the
untouched TEST split with bootstrap 95% CIs.

Usage:
  python scripts/train_quality_baseline.py [--max_train_samples 1200] [--max_val_samples 300] \
         [--max_test_samples 300] [--epochs 3]
"""

import sys
import copy
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.metrics import calculate_regression_metrics, calculate_qwk
from self_check_ultrasound.utils.eval_utils import balanced_subsample, bootstrap_ci
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pearson(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.std() < 1e-8 or p.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality assessment baselines")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_val_samples", type=int, default=300)
    parser.add_argument("--max_test_samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def predict_grades(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(model(batch["image"].to(device)).cpu().numpy())
    return np.concatenate(out)


def full_metrics(y, pred):
    m = calculate_regression_metrics(y, pred)
    m["qwk"] = calculate_qwk(y, pred)
    m["ci95_mae"] = bootstrap_ci(y, pred, mean_absolute_error)
    m["ci95_rmse"] = bootstrap_ci(y, pred, rmse)
    m["ci95_pearson_r"] = bootstrap_ci(y, pred, pearson)
    return m


def main():
    args = parse_args()
    set_seed(default_config.data.random_seed)
    device = torch.device(default_config.training.device)
    n_cls = len(default_config.data.classes)
    dataset_root = default_config.paths.dataset_dir

    print("=" * 65)
    print("SelfCheckUltrasound: Quality Assessment Baseline Evaluation (M3)")
    print("=" * 65)

    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits not found at {splits_path}. Run build_manifest.py first.")
    splits_df = pd.read_csv(splits_path, low_memory=False)

    train_df = balanced_subsample(splits_df[splits_df["split"] == "train"], args.max_train_samples, n_cls, seed=42)
    val_df = balanced_subsample(splits_df[splits_df["split"] == "val"], args.max_val_samples, n_cls, seed=42)
    test_df = balanced_subsample(splits_df[splits_df["split"] == "test"], args.max_test_samples, n_cls, seed=42)
    print(f"Dataset active subset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test images")

    y_train = train_df["quality_grade"].values.astype(np.float32)
    y_val = val_df["quality_grade"].values.astype(np.float32)
    y_test = test_df["quality_grade"].values.astype(np.float32)

    mean_grade = float(np.mean(y_train))
    naive_val = full_metrics(y_val, np.full_like(y_val, mean_grade))
    naive_test = full_metrics(y_test, np.full_like(y_test, mean_grade))
    print(f"\n--- Naive Baseline (Constant Train Mean = {mean_grade:.2f}) ---")
    print(f"TEST MAE: {naive_test['mae']:.3f} | RMSE: {naive_test['rmse']:.3f}")

    train_tf = get_transforms(is_train=True, image_size=default_config.data.image_size)
    eval_tf = get_transforms(is_train=False, image_size=default_config.data.image_size)
    train_loader = DataLoader(CardiacUltrasoundDataset(train_df, dataset_root=dataset_root, transform=train_tf),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CardiacUltrasoundDataset(val_df, dataset_root=dataset_root, transform=eval_tf),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CardiacUltrasoundDataset(test_df, dataset_root=dataset_root, transform=eval_tf),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = QualityAssessorNet(dropout_rate=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"\n--- Training QualityAssessorNet on {device} ({args.epochs} epochs) ---")
    best_mae, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, total = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            grades = batch["quality_grade"].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(images), grades)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total += images.size(0)

        val_pred = predict_grades(model, val_loader, device)
        em = calculate_regression_metrics(y_val, val_pred)
        qwk = calculate_qwk(y_val, val_pred)
        is_best = em["mae"] < best_mae
        if is_best:
            best_mae, best_state = em["mae"], copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train MSE: {train_loss / total:.4f} | Val MAE: {em['mae']:.3f} | "
              f"Val RMSE: {em['rmse']:.3f} | Pearson r: {em['pearson_r']:.3f} | QWK: {qwk:.3f}"
              f"{' (*Best)' if is_best else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = full_metrics(y_val, predict_grades(model, val_loader, device))
    test_metrics = full_metrics(y_test, predict_grades(model, test_loader, device))

    artifacts_dir = default_config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "quality_assessor_net.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nBest model checkpoint saved to: {checkpoint_path} (Best Val MAE: {best_mae:.3f})")

    results = {
        "dataset_split": {"num_train": len(train_df), "num_val": len(val_df), "num_test": len(test_df)},
        "naive_baseline": naive_val,
        "quality_assessor_net": val_metrics,
        "test": {"naive_baseline": naive_test, "quality_assessor_net": test_metrics},
        "note": "val = model-selection set (optimistic); test = untouched held-out split (report this).",
    }
    metrics_path = artifacts_dir / "quality_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation comparison saved to: {metrics_path}")

    def fmt(m, key):
        c = m[key]
        return f"{c['point']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"

    print("\n" + "=" * 78)
    print("QUALITY ASSESSMENT SUMMARY (TEST split, point [95% bootstrap CI])")
    print("=" * 78)
    print(f"{'Metric':<18s} | {'Naive (Mean)':<28s} | {'QualityAssessorNet':<28s}")
    print("-" * 78)
    print(f"{'MAE':<18s} | {fmt(naive_test, 'ci95_mae'):<28s} | {fmt(test_metrics, 'ci95_mae'):<28s}")
    print(f"{'RMSE':<18s} | {fmt(naive_test, 'ci95_rmse'):<28s} | {fmt(test_metrics, 'ci95_rmse'):<28s}")
    print(f"{'Pearson r':<18s} | {fmt(naive_test, 'ci95_pearson_r'):<28s} | {fmt(test_metrics, 'ci95_pearson_r'):<28s}")
    print(f"{'QWK':<18s} | {naive_test['qwk']:<28.3f} | {test_metrics['qwk']:<28.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
