"""
Training and evaluation script for cardiac view classification baselines (Module M1).
Compares:
1. SimpleFeatureClassifier (Classical spatial intensity + Logistic Regression baseline)
2. CardiacEchoNet (Lightweight Convolutional Neural Network baseline)

Protocol: models are trained on TRAIN; the CNN's best epoch is selected on VAL; the final
numbers are reported on VAL (selection set, optimistic) AND on the untouched TEST split, each
with bootstrap 95% CIs, plus a paired bootstrap of CNN - classical so that "A beats B" claims
are backed by an interval, not a point estimate.

Usage:
  python scripts/train_view_baseline.py [--max_train_samples 1200] [--max_val_samples 300] \
         [--max_test_samples 300] [--epochs 3]
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.metrics import calculate_classification_metrics, calculate_ece, compute_calibration_curve
from self_check_ultrasound.utils.eval_utils import balanced_subsample, bootstrap_ci, paired_bootstrap_diff
from self_check_ultrasound.data.dataset import CardiacUltrasoundDataset
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet, SimpleFeatureClassifier


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def parse_args():
    parser = argparse.ArgumentParser(description="Train view classification baselines")
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_val_samples", type=int, default=300)
    parser.add_argument("--max_test_samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def predict_loader(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
            targets.append(batch["view_label"].numpy())
    return np.concatenate(targets), np.concatenate(probs)


def split_metrics(y, probs, class_names):
    preds = np.argmax(probs, axis=1)
    m = calculate_classification_metrics(y, preds, class_names)
    m["calibration_curve"] = compute_calibration_curve(y, probs)
    m["ece"] = m["calibration_curve"]["ece"]
    m["ci95_accuracy"] = bootstrap_ci(y, preds, accuracy_score)
    m["ci95_macro_f1"] = bootstrap_ci(y, preds, macro_f1)
    return m, preds


def train_cnn(model, train_loader, val_loader, epochs, lr, device, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n--- Training CardiacEchoNet on {device} ({epochs} epochs) ---")
    best_f1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["view_label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += images.size(0)

        y_val, p_val = predict_loader(model, val_loader, device)
        m = calculate_classification_metrics(y_val, np.argmax(p_val, axis=1), class_names)
        ece = calculate_ece(y_val, p_val)
        is_best = m["macro_f1"] > best_f1
        if is_best:
            best_f1, best_state = m["macro_f1"], copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss / total:.4f} | Train Acc: {correct / total:.3f} | "
              f"Val BalAcc: {m['balanced_accuracy']:.3f} | Val MacroF1: {m['macro_f1']:.3f} | ECE: {ece:.3f}"
              f"{' (*Best)' if is_best else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)


def classical_xy(feat_clf, df, dataset_root):
    X, y = [], []
    for _, row in df.iterrows():
        with Image.open(dataset_root / row["relative_path"]) as img:
            X.append(feat_clf.extract_features(img))
        y.append(int(row["view_idx"]))
    return np.array(X), np.array(y)


def main():
    args = parse_args()
    set_seed(default_config.data.random_seed)
    device = torch.device(default_config.training.device)
    class_names = list(default_config.data.classes)
    n_cls = len(class_names)
    dataset_root = default_config.paths.dataset_dir

    print("=" * 65)
    print("SelfCheckUltrasound: View Classification Baseline Evaluation (M1)")
    print("=" * 65)

    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits not found at {splits_path}. Run build_manifest.py first.")
    splits_df = pd.read_csv(splits_path, low_memory=False)

    train_df = balanced_subsample(splits_df[splits_df["split"] == "train"], args.max_train_samples, n_cls, seed=42)
    val_df = balanced_subsample(splits_df[splits_df["split"] == "val"], args.max_val_samples, n_cls, seed=42)
    test_df = balanced_subsample(splits_df[splits_df["split"] == "test"], args.max_test_samples, n_cls, seed=42)
    print(f"Dataset active subset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test images")
    print("(val/test subsamples are class-balanced, so accuracy == balanced accuracy)")

    # ---- Classical baseline
    print("\n--- Fitting Classical ML Baseline (Spatial Features + Logistic Regression) ---")
    feat_clf = SimpleFeatureClassifier(n_bins=16, grid_size=4)
    X_train, y_train = classical_xy(feat_clf, train_df, dataset_root)
    feat_clf.fit(X_train, y_train)
    X_val, y_val = classical_xy(feat_clf, val_df, dataset_root)
    X_test, y_test = classical_xy(feat_clf, test_df, dataset_root)

    cl_val, cl_val_pred = split_metrics(y_val, feat_clf.predict_proba(X_val), class_names)
    cl_test, cl_test_pred = split_metrics(y_test, feat_clf.predict_proba(X_test), class_names)
    print(f"Classical  val BalAcc {cl_val['balanced_accuracy']:.3f} | test BalAcc {cl_test['balanced_accuracy']:.3f} "
          f"(95% CI {cl_test['ci95_accuracy']['ci_low']:.3f}-{cl_test['ci95_accuracy']['ci_high']:.3f})")

    # ---- CNN
    train_tf = get_transforms(is_train=True, image_size=default_config.data.image_size)
    eval_tf = get_transforms(is_train=False, image_size=default_config.data.image_size)
    train_loader = DataLoader(CardiacUltrasoundDataset(train_df, dataset_root=dataset_root, transform=train_tf),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CardiacUltrasoundDataset(val_df, dataset_root=dataset_root, transform=eval_tf),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CardiacUltrasoundDataset(test_df, dataset_root=dataset_root, transform=eval_tf),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CardiacEchoNet(num_classes=default_config.data.num_classes, dropout_rate=0.3).to(device)
    train_cnn(model, train_loader, val_loader, args.epochs, args.lr, device, class_names)

    yv, pv = predict_loader(model, val_loader, device)
    yt, pt = predict_loader(model, test_loader, device)
    cnn_val, cnn_val_pred = split_metrics(yv, pv, class_names)
    cnn_test, cnn_test_pred = split_metrics(yt, pt, class_names)

    # ---- Paired comparison on TEST (CNN - classical)
    comparison_test = {
        "accuracy_diff_cnn_minus_classical": paired_bootstrap_diff(yt, cnn_test_pred, cl_test_pred, accuracy_score),
        "macro_f1_diff_cnn_minus_classical": paired_bootstrap_diff(yt, cnn_test_pred, cl_test_pred, macro_f1),
    }

    artifacts_dir = default_config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "cardiac_echo_net.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    results = {
        "dataset_split": {"num_train": len(train_df), "num_val": len(val_df), "num_test": len(test_df)},
        # VAL keys kept for backward compatibility with the plotting code
        "classical_baseline": cl_val,
        "cardiac_echo_net": cnn_val,
        "test": {"classical_baseline": cl_test, "cardiac_echo_net": cnn_test, "paired_comparison": comparison_test},
        "note": "val = model-selection set (optimistic); test = untouched held-out split (report this).",
    }
    metrics_path = artifacts_dir / "view_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation saved to: {metrics_path}")

    def ci(m, key):
        c = m[key]
        return f"{c['point']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"

    print("\n" + "=" * 78)
    print("BASELINE COMPARISON SUMMARY (TEST split, point estimate [95% bootstrap CI])")
    print("=" * 78)
    print(f"{'Metric':<20s} | {'Classical (Logistic)':<26s} | {'CardiacEchoNet (CNN)':<26s}")
    print("-" * 78)
    print(f"{'Accuracy':<20s} | {ci(cl_test, 'ci95_accuracy'):<26s} | {ci(cnn_test, 'ci95_accuracy'):<26s}")
    print(f"{'Macro F1':<20s} | {ci(cl_test, 'ci95_macro_f1'):<26s} | {ci(cnn_test, 'ci95_macro_f1'):<26s}")
    print(f"{'ECE':<20s} | {cl_test['ece']:<26.3f} | {cnn_test['ece']:<26.3f}")
    d = comparison_test["accuracy_diff_cnn_minus_classical"]
    print(f"\nCNN - classical accuracy: {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] "
          f"-> {'significant' if d['significant'] else 'NOT significant (CI includes 0)'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
