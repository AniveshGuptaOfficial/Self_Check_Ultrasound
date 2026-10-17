"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest
from self_check_ultrasound.utils.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_qwk,
    calculate_ece,
)


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    class_names = ["A", "B", "C"]

    metrics = calculate_classification_metrics(y_true, y_pred, class_names=class_names)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "per_class_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert metrics["mae"] < 0.3
    assert metrics["rmse"] < 0.3
    assert metrics["pearson_r"] > 0.95


def test_qwk_metric():
    y_true = np.array([1, 2, 3, 4, 5])
    # Perfect agreement
    qwk_perfect = calculate_qwk(y_true, y_true, min_rating=0, max_rating=9)
    assert np.isclose(qwk_perfect, 1.0)

    # Partial agreement
    y_imperfect = np.array([1, 2, 4, 4, 5])
    qwk_imperfect = calculate_qwk(y_true, y_imperfect, min_rating=0, max_rating=9)
    assert 0.8 < qwk_imperfect < 1.0


def test_ece_metric():
    y_true = np.array([0, 1])
    # Overconfident wrong predictions
    y_prob = np.array([[0.1, 0.9], [0.9, 0.1]])
    ece = calculate_ece(y_true, y_prob)
    assert ece > 0.5
