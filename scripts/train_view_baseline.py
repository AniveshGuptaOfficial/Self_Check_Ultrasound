"""
Training and evaluation script for cardiac view classification baselines (Module M1).
Compares:
1. SimpleFeatureClassifier (Classical spatial intensity + Logistic Regression baseline)
2. CardiacEchoNet (Lightweight Convolutional Neural Network baseline)

Usage:
  python scripts/train_view_baseline.py [--max_samples 1200] [--epochs 3]
"""

import sys
import copy
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.metrics import calculate_classification_metrics, calculate_ece
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet, SimpleFeatureClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train view classification baselines")
    parser.add_argument("--max_train_samples", type=int, default=1200, help="Max train samples for fast benchmark")
    parser.add_argument("--max_val_samples", type=int, default=300, help="Max val samples for fast benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for CNN")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def train_cnn(model, train_loader, val_loader, epochs, lr, device, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n--- Training CardiacEchoNet on {device} ({epochs} epochs) ---")
    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    best_preds, best_probs, best_targets = None, None, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["view_label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        epoch_loss = train_loss / total_train
        epoch_acc = train_correct / total_train

        # Validation pass
        model.eval()
        val_preds = []
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["view_label"].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_probs.append(probs.cpu().numpy())
                val_preds.append(preds.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        val_targets = np.concatenate(val_targets)
        val_preds = np.concatenate(val_preds)
        val_probs = np.concatenate(val_probs)

        val_metrics = calculate_classification_metrics(val_targets, val_preds, class_names)
        val_ece = calculate_ece(val_targets, val_probs)
        val_metrics["ece"] = val_ece

        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(val_metrics)
            best_preds, best_probs, best_targets = val_preds, val_probs, val_targets

        best_badge = " (*Best)" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.3f} | "
            f"Val BalAcc: {val_metrics['balanced_accuracy']:.3f} | "
            f"Val MacroF1: {val_metrics['macro_f1']:.3f} | ECE: {val_ece:.3f}{best_badge}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        return best_metrics, best_preds, best_probs, best_targets
    return val_metrics, val_preds, val_probs, val_targets


def main():
    args = parse_args()
    set_seed(default_config.data.random_seed)
    device = torch.device(default_config.training.device)
    class_names = list(default_config.data.classes)

    print("=" * 65)
    print("SelfCheckUltrasound: View Classification Baseline Evaluation (M1)")
    print("=" * 65)

    # 1. Load splits
    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits not found at {splits_path}. Run build_manifest.py first.")

    splits_df = pd.read_csv(splits_path, low_memory=False)
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)

    # Subsample if requested for fast benchmark
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

    # 2. Classical ML Baseline (SimpleFeatureClassifier)
    print("\n--- Fitting Classical ML Baseline (Spatial Features + Logistic Regression) ---")
    feat_clf = SimpleFeatureClassifier(n_bins=16, grid_size=4)

    X_train = []
    y_train = []
    dataset_root = default_config.paths.dataset_dir

    for _, row in train_df.iterrows():
        img_path = dataset_root / row["relative_path"]
        with Image.open(img_path) as img:
            X_train.append(feat_clf.extract_features(img))
            y_train.append(int(row["view_idx"]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    feat_clf.fit(X_train, y_train)

    X_val = []
    y_val = []
    for _, row in val_df.iterrows():
        img_path = dataset_root / row["relative_path"]
        with Image.open(img_path) as img:
            X_val.append(feat_clf.extract_features(img))
            y_val.append(int(row["view_idx"]))

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    classical_preds = feat_clf.predict(X_val)
    classical_probs = feat_clf.predict_proba(X_val)
    classical_metrics = calculate_classification_metrics(y_val, classical_preds, class_names)
    classical_metrics["ece"] = calculate_ece(y_val, classical_probs)

    print(f"Classical Baseline Balanced Accuracy: {classical_metrics['balanced_accuracy']:.3f}")
    print(f"Classical Baseline Macro F1:         {classical_metrics['macro_f1']:.3f}")
    print(f"Classical Baseline ECE:              {classical_metrics['ece']:.3f}")

    # 3. Deep Learning Baseline (CardiacEchoNet)
    train_transform = get_transforms(is_train=True, image_size=default_config.data.image_size)
    val_transform = get_transforms(is_train=False, image_size=default_config.data.image_size)

    train_dataset = CardiacUltrasoundDataset(train_df, dataset_root=dataset_root, transform=train_transform)
    val_dataset = CardiacUltrasoundDataset(val_df, dataset_root=dataset_root, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CardiacEchoNet(num_classes=default_config.data.num_classes, dropout_rate=0.3).to(device)
    cnn_metrics, cnn_preds, cnn_probs, cnn_targets = train_cnn(
        model, train_loader, val_loader, args.epochs, args.lr, device, class_names
    )

    # 4. Save results and checkpoint
    artifacts_dir = default_config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifacts_dir / "cardiac_echo_net.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    comparison = {
        "dataset_split": {
            "num_train": len(train_df),
            "num_val": len(val_df),
        },
        "classical_baseline": classical_metrics,
        "cardiac_echo_net": cnn_metrics,
    }

    metrics_path = artifacts_dir / "view_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Evaluation comparison saved to: {metrics_path}")

    print("\n" + "=" * 65)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<20s} | {'Classical (Logistic)':<20s} | {'CardiacEchoNet (CNN)':<20s}")
    print("-" * 65)
    print(f"{'Accuracy':<20s} | {classical_metrics['accuracy']:<20.3f} | {cnn_metrics['accuracy']:<20.3f}")
    print(f"{'Balanced Accuracy':<20s} | {classical_metrics['balanced_accuracy']:<20.3f} | {cnn_metrics['balanced_accuracy']:<20.3f}")
    print(f"{'Macro F1':<20s} | {classical_metrics['macro_f1']:<20.3f} | {cnn_metrics['macro_f1']:<20.3f}")
    print(f"{'ECE (Calibration)':<20s} | {classical_metrics['ece']:<20.3f} | {cnn_metrics['ece']:<20.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
