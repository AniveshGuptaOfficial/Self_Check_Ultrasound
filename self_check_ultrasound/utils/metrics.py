"""
Evaluation metrics for cardiac view classification, quality grade regression,
and model calibration.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
)
from scipy.stats import pearsonr, spearmanr


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Computes standard and robust classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }

    if class_names:
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        report["per_class_f1"] = {
            name: float(score) for name, score in zip(class_names, per_class_f1)
        }

    return report


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, Pearson r, and Spearman rho for continuous quality prediction.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Pearson and Spearman
    if len(y_true) > 1 and np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        p_corr, _ = pearsonr(y_true, y_pred)
        s_corr, _ = spearmanr(y_true, y_pred)
    else:
        p_corr, s_corr = 0.0, 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(p_corr),
        "spearman_rho": float(s_corr),
    }


def calculate_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: int = 0,
    max_rating: int = 9,
) -> float:
    """
    Computes Quadratic Weighted Kappa for ordinal quality grade agreement.
    """
    # Round predictions to nearest integer and clip
    y_pred_rounded = np.clip(np.round(y_pred).astype(int), min_rating, max_rating)
    y_true_int = np.clip(y_true.astype(int), min_rating, max_rating)

    return float(
        cohen_kappa_score(
            y_true_int,
            y_pred_rounded,
            weights="quadratic",
            labels=list(range(min_rating, max_rating + 1)),
        )
    )


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes calibration curve statistics (accuracy vs. confidence per bin) and ECE.
    Works for multi-class classification (using predicted top-1 confidence) and binary.

    Returns dict with:
        bin_edges: list of bin boundaries (length n_bins + 1)
        bin_centers: list of bin midpoints (length n_bins)
        bin_accuracies: list of empirical accuracies per bin
        bin_confidences: list of mean predicted confidences per bin
        bin_counts: list of sample counts per bin
        bin_proportions: list of sample fractions per bin
        ece: expected calibration error (float)
        mce: maximum calibration error (float)
    """
    if y_prob.ndim == 1:
        predictions = (y_prob >= 0.5).astype(int)
        confidences = np.where(predictions == 1, y_prob, 1.0 - y_prob)
    else:
        predictions = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)

    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    bin_proportions = []
    ece = 0.0
    mce = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        b_low = bin_boundaries[i]
        b_high = bin_boundaries[i + 1]
        bin_centers.append(float((b_low + b_high) / 2.0))

        if i == 0:
            in_bin = (confidences >= b_low) & (confidences <= b_high)
        else:
            in_bin = (confidences > b_low) & (confidences <= b_high)

        count = int(np.sum(in_bin))
        prop = float(count / max(n_total, 1))
        bin_counts.append(count)
        bin_proportions.append(prop)

        if count > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            bin_accuracies.append(acc_in_bin)
            bin_confidences.append(conf_in_bin)
            diff = abs(conf_in_bin - acc_in_bin)
            ece += diff * prop
            mce = max(mce, diff)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float(bin_centers[-1]))

    return {
        "bin_edges": [float(x) for x in bin_boundaries],
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_proportions": bin_proportions,
        "ece": float(ece),
        "mce": float(mce),
    }


def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for predicted class probabilities.
    """
    curve = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(curve["ece"])
