"""
Multi-Seed Evaluation Runner (SelfCheckUltrasound).
Evaluates baseline models across multiple independent random seeds to report
mean +/- std performance and demonstrate statistical robustness across seed variations.

Usage:
  python scripts/run_multiseed_eval.py --seeds 42 123 456 789 1011 --samples 100
  python scripts/run_multiseed_eval.py --fast  # Runs quick smoke evaluation across 3 seeds
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.metrics import (
    calculate_classification_metrics,
    calculate_ece,
    compute_calibration_curve,
)
from self_check_ultrasound.utils.eval_utils import balanced_subsample
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet, SimpleFeatureClassifier
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from scipy.stats import pearsonr


def evaluate_single_seed(
    seed: int,
    splits_df: pd.DataFrame,
    dataset_root: Path,
    n_samples: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs single-seed evaluation on test split and returns key performance metrics."""
    set_seed(seed)

    # 1. Balanced sample for test and train
    n_classes = len(default_config.data.classes)
    per_class = max(1, n_samples // n_classes)

    test_sub = balanced_subsample(splits_df[splits_df["split"] == "test"], per_class, n_classes, seed=seed)
    train_sub = balanced_subsample(splits_df[splits_df["split"] == "train"], per_class * 2, n_classes, seed=seed)

    # 2. Extract Classical Spatial Features
    feat_clf = SimpleFeatureClassifier()
    X_train, y_train = [], []
    for _, row in train_sub.iterrows():
        img_p = dataset_root / row["relative_path"]
        with Image.open(img_p) as img:
            X_train.append(feat_clf.extract_features(img))
        y_train.append(int(row["view_idx"]))

    feat_clf.fit(np.array(X_train), np.array(y_train))

    X_test, y_test, q_true = [], [], []
    test_images = []
    for _, row in test_sub.iterrows():
        img_p = dataset_root / row["relative_path"]
        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB")
            X_test.append(feat_clf.extract_features(img_rgb))
            test_images.append(img_rgb.copy())
        y_test.append(int(row["view_idx"]))
        q_true.append(float(row["quality_grade"]))

    X_test, y_test = np.array(X_test), np.array(y_test)
    q_true = np.array(q_true)

    # Classical View Predictions
    p_classic = feat_clf.predict_proba(X_test)
    preds_classic = np.argmax(p_classic, axis=1)
    m_classic = calculate_classification_metrics(y_test, preds_classic, list(default_config.data.classes))
    ece_classic = calculate_ece(y_test, p_classic)

    # 3. Neural View Model (CardiacEchoNet)
    view_model = CardiacEchoNet(num_classes=n_classes).to(device)
    view_ckpt = default_config.paths.artifacts_dir / "cardiac_echo_net.pt"
    if view_ckpt.exists():
        view_model.load_state_dict(torch.load(view_ckpt, map_location=device, weights_only=True))
    view_model.eval()

    transform = get_transforms(is_train=False)
    p_cnn = []
    with torch.no_grad():
        for img in test_images:
            tensor = transform(img).unsqueeze(0).to(device)
            out = view_model(tensor)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            p_cnn.append(prob)
    p_cnn = np.array(p_cnn)
    preds_cnn = np.argmax(p_cnn, axis=1)
    m_cnn = calculate_classification_metrics(y_test, preds_cnn, list(default_config.data.classes))
    ece_cnn = calculate_ece(y_test, p_cnn)

    # 4. Neural Quality Model (QualityAssessorNet)
    quality_model = QualityAssessorNet().to(device)
    q_ckpt = default_config.paths.artifacts_dir / "quality_assessor_net.pt"
    if q_ckpt.exists():
        quality_model.load_state_dict(torch.load(q_ckpt, map_location=device, weights_only=True))
    quality_model.eval()

    q_preds = []
    with torch.no_grad():
        for img in test_images:
            tensor = transform(img).unsqueeze(0).to(device)
            pred_score = quality_model(tensor).item()
            q_preds.append(pred_score)
    q_preds = np.array(q_preds)

    mae_q = float(np.mean(np.abs(q_preds - q_true)))
    rmse_q = float(np.sqrt(np.mean((q_preds - q_true) ** 2)))
    r_val, _ = pearsonr(q_preds, q_true) if len(q_preds) > 1 and np.std(q_preds) > 1e-6 else (0.0, 1.0)

    return {
        "seed": seed,
        "n_samples": len(test_sub),
        "classical": {
            "accuracy": float(m_classic["accuracy"]),
            "macro_f1": float(m_classic["macro_f1"]),
            "ece": float(ece_classic),
        },
        "cardiac_echo_net": {
            "accuracy": float(m_cnn["accuracy"]),
            "macro_f1": float(m_cnn["macro_f1"]),
            "ece": float(ece_cnn),
        },
        "quality_assessor_net": {
            "mae": float(mae_q),
            "rmse": float(rmse_q),
            "pearson_r": float(r_val),
        },
    }


def aggregate_seed_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean, std, min, max across multi-seed evaluations."""
    n_runs = len(runs)
    seeds = [r["seed"] for r in runs]

    def stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "num_seeds": n_runs,
        "seeds": seeds,
        "classical_view": {
            "accuracy": stats([r["classical"]["accuracy"] for r in runs]),
            "macro_f1": stats([r["classical"]["macro_f1"] for r in runs]),
            "ece": stats([r["classical"]["ece"] for r in runs]),
        },
        "cardiac_echo_net": {
            "accuracy": stats([r["cardiac_echo_net"]["accuracy"] for r in runs]),
            "macro_f1": stats([r["cardiac_echo_net"]["macro_f1"] for r in runs]),
            "ece": stats([r["cardiac_echo_net"]["ece"] for r in runs]),
        },
        "quality_assessor_net": {
            "mae": stats([r["quality_assessor_net"]["mae"] for r in runs]),
            "rmse": stats([r["quality_assessor_net"]["rmse"] for r in runs]),
            "pearson_r": stats([r["quality_assessor_net"]["pearson_r"] for r in runs]),
        },
        "individual_runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Robustness Evaluation for SelfCheckUltrasound")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011], help="Random seeds to evaluate")
    parser.add_argument("--samples", type=int, default=60, help="Number of test samples per seed")
    parser.add_argument("--fast", action="store_true", help="Fast smoke run with 3 seeds and 30 samples")
    args = parser.parse_args()

    if args.fast:
        seeds = [42, 123, 456]
        n_samples = 30
    else:
        seeds = args.seeds
        n_samples = args.samples

    print("=" * 74)
    print("SelfCheckUltrasound: Multi-Seed Robustness Evaluation")
    print(f"Seeds: {seeds} | Samples per seed: {n_samples}")
    print("=" * 74)

    splits_path = default_config.paths.splits_path
    if not splits_path.exists():
        print(f"[ERROR] Splits CSV not found at {splits_path}")
        sys.exit(1)

    splits_df = pd.read_csv(splits_path, low_memory=False)
    dataset_root = default_config.paths.dataset_dir

    runs = []
    for idx, seed in enumerate(seeds, 1):
        print(f"[{idx}/{len(seeds)}] Evaluating Random Seed {seed}...")
        res = evaluate_single_seed(seed, splits_df, dataset_root, n_samples)
        runs.append(res)
        print(f"      Classic Acc: {res['classical']['accuracy']:.3f} | CNN Acc: {res['cardiac_echo_net']['accuracy']:.3f} | "
              f"Quality MAE: {res['quality_assessor_net']['mae']:.3f}")

    aggregated = aggregate_seed_metrics(runs)

    # Save to artifacts and results
    artifacts_dir = default_config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_art = artifacts_dir / "multiseed_metrics.json"
    out_res = project_root / "results" / "multiseed_metrics.json"

    with open(out_art, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    with open(out_res, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print("\n" + "=" * 74)
    print("MULTI-SEED STATISTICAL SUMMARY (Mean +/- Std Across Seeds)")
    print("=" * 74)
    print(f"{'Component / Model':<30s} | {'Metric':<14s} | {'Multi-Seed (Mean +/- Std)':<22s}")
    print("-" * 74)

    c_acc = aggregated["classical_view"]["accuracy"]
    c_f1 = aggregated["classical_view"]["macro_f1"]
    c_ece = aggregated["classical_view"]["ece"]
    print(f"{'Classical Baseline':<30s} | {'Accuracy':<14s} | {c_acc['mean']:.3f} +/- {c_acc['std']:.3f}")
    print(f"{'Classical Baseline':<30s} | {'Macro F1':<14s} | {c_f1['mean']:.3f} +/- {c_f1['std']:.3f}")
    print(f"{'Classical Baseline':<30s} | {'ECE':<14s} | {c_ece['mean']:.3f} +/- {c_ece['std']:.3f}")

    print("-" * 74)
    cnn_acc = aggregated["cardiac_echo_net"]["accuracy"]
    cnn_f1 = aggregated["cardiac_echo_net"]["macro_f1"]
    cnn_ece = aggregated["cardiac_echo_net"]["ece"]
    print(f"{'CardiacEchoNet (CNN)':<30s} | {'Accuracy':<14s} | {cnn_acc['mean']:.3f} +/- {cnn_acc['std']:.3f}")
    print(f"{'CardiacEchoNet (CNN)':<30s} | {'Macro F1':<14s} | {cnn_f1['mean']:.3f} +/- {cnn_f1['std']:.3f}")
    print(f"{'CardiacEchoNet (CNN)':<30s} | {'ECE':<14s} | {cnn_ece['mean']:.3f} +/- {cnn_ece['std']:.3f}")

    print("-" * 74)
    q_mae = aggregated["quality_assessor_net"]["mae"]
    q_rmse = aggregated["quality_assessor_net"]["rmse"]
    q_r = aggregated["quality_assessor_net"]["pearson_r"]
    print(f"{'QualityAssessorNet':<30s} | {'MAE':<14s} | {q_mae['mean']:.3f} +/- {q_mae['std']:.3f}")
    print(f"{'QualityAssessorNet':<30s} | {'RMSE':<14s} | {q_rmse['mean']:.3f} +/- {q_rmse['std']:.3f}")
    print(f"{'QualityAssessorNet':<30s} | {'Pearson r':<14s} | {q_r['mean']:.3f} +/- {q_r['std']:.3f}")

    print("=" * 74)
    print(f"[OK] Multi-seed evaluation saved to: {out_res.relative_to(project_root)}")


if __name__ == "__main__":
    main()
