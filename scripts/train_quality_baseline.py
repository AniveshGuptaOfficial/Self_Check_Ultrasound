"""
Training and evaluation script for Image Quality Assessment baseline (Module M3).
Compares:
1. Mean / Median heuristic baseline
2. QualityAssessorNet (Convolutional Neural Network predicting continuous quality grades 0-9)

Usage:
  python scripts/train_quality_baseline.py [--max_samples 1200] [--epochs 3]
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.metrics import calculate_regression_metrics, calculate_qwk
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality assessment baselines")
    parser.add_argument("--max_train_samples", type=int, default=1200, help="Max train samples")
    parser.add_argument("--max_val_samples", type=int, default=300, help="Max val samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for CNN")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(default_config.data.random_seed)
    device = torch.device(default_config.training.device)

    print("=" * 65)
    print("SelfCheckUltrasound: Quality Assessment Baseline Evaluation (M3)")
    print("=" * 65)

    # 1. Load splits
    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits not found at {splits_path}. Run build_manifest.py first.")

    splits_df = pd.read_csv(splits_path, low_memory=False)
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)

    if args.max_train_samples and len(train_df) > args.max_train_samples:
        k = max(1, args.max_train_samples // len(default_config.data.classes))
        train_df = pd.concat(
            [grp.sample(n=min(len(grp), k), random_state=42) for _, grp in train_df.groupby("view_class")]
        ).reset_index(drop=True)

    if args.max_val_samples and len(val_df) > args.max_val_samples:
        k_val = max(1, args.max_val_samples // len(default_config.data.classes))
        val_df = pd.concat(
            [grp.sample(n=min(len(grp), k_val), random_state=42) for _, grp in val_df.groupby("view_class")]
        ).reset_index(drop=True)

    print(f"Dataset active subset: {len(train_df)} train images, {len(val_df)} val images")

    y_train = train_df["quality_grade"].values.astype(np.float32)
    y_val = val_df["quality_grade"].values.astype(np.float32)

    # 2. Naive Baseline (Mean Grade Predictor)
    mean_grade = float(np.mean(y_train))
    naive_preds = np.full_like(y_val, mean_grade)
    naive_metrics = calculate_regression_metrics(y_val, naive_preds)
    naive_metrics["qwk"] = calculate_qwk(y_val, naive_preds)

    print(f"\n--- Naive Baseline (Constant Mean = {mean_grade:.2f}) ---")
    print(f"MAE: {naive_metrics['mae']:.3f} | RMSE: {naive_metrics['rmse']:.3f} | QWK: {naive_metrics['qwk']:.3f}")

    # 3. Deep Learning Baseline (QualityAssessorNet)
    dataset_root = default_config.paths.dataset_dir
    train_transform = get_transforms(is_train=True, image_size=default_config.data.image_size)
    val_transform = get_transforms(is_train=False, image_size=default_config.data.image_size)

    train_dataset = CardiacUltrasoundDataset(train_df, dataset_root=dataset_root, transform=train_transform)
    val_dataset = CardiacUltrasoundDataset(val_df, dataset_root=dataset_root, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = QualityAssessorNet(dropout_rate=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"\n--- Training QualityAssessorNet on {device} ({args.epochs} epochs) ---")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total_train = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            grades = batch["quality_grade"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, grades)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        epoch_loss = train_loss / total_train

        # Validation pass
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                val_preds_list.append(outputs.cpu().numpy())

        epoch_val_preds = np.concatenate(val_preds_list)
        epoch_metrics = calculate_regression_metrics(y_val, epoch_val_preds)
        epoch_qwk = calculate_qwk(y_val, epoch_val_preds)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train MSE: {epoch_loss:.4f} | "
            f"Val MAE: {epoch_metrics['mae']:.3f} | "
            f"Val RMSE: {epoch_metrics['rmse']:.3f} | "
            f"Pearson r: {epoch_metrics['pearson_r']:.3f} | "
            f"QWK: {epoch_qwk:.3f}"
        )

    final_metrics = epoch_metrics
    final_metrics["qwk"] = epoch_qwk

    # 4. Save results and checkpoint
    artifacts_dir = default_config.paths.artifacts_dir
    checkpoint_path = artifacts_dir / "quality_assessor_net.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    comparison = {
        "dataset_split": {
            "num_train": len(train_df),
            "num_val": len(val_df),
        },
        "naive_baseline": naive_metrics,
        "quality_assessor_net": final_metrics,
    }

    metrics_path = artifacts_dir / "quality_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Evaluation comparison saved to: {metrics_path}")

    print("\n" + "=" * 65)
    print("QUALITY ASSESSMENT COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<20s} | {'Naive (Mean)':<20s} | {'QualityAssessorNet':<20s}")
    print("-" * 65)
    print(f"{'MAE':<20s} | {naive_metrics['mae']:<20.3f} | {final_metrics['mae']:<20.3f}")
    print(f"{'RMSE':<20s} | {naive_metrics['rmse']:<20.3f} | {final_metrics['rmse']:<20.3f}")
    print(f"{'Pearson r':<20s} | {naive_metrics['pearson_r']:<20.3f} | {final_metrics['pearson_r']:<20.3f}")
    print(f"{'QWK (Agreement)':<20s} | {naive_metrics['qwk']:<20.3f} | {final_metrics['qwk']:<20.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
