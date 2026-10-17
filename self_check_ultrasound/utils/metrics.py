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


def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) for predicted class probabilities.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
