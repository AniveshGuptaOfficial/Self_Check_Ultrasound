"""
Unit tests for the Diagnostic Adequacy Ablation Study module.
Tests scripts/run_ablation_study.py functions, gating metrics, and selective accuracy calculation.
"""

import numpy as np
import pytest

from scripts.run_ablation_study import (
    compute_aurc,
    evaluate_gate_at_coverages,
    run_ablation_evaluation,
    format_ablation_markdown_table,
    generate_synthetic_data,
)
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator


def test_compute_aurc():
    coverages = np.array([0.2, 0.5, 0.8, 1.0])
    risks = np.array([0.1, 0.2, 0.3, 0.4])
    aurc = compute_aurc(coverages, risks)
    assert 0.0 < aurc < 1.0
    # Perfect oracle with zero risk should have AURC == 0
    zero_aurc = compute_aurc(coverages, np.zeros_like(risks))
    assert np.isclose(zero_aurc, 0.0)


def test_evaluate_gate_at_coverages():
    # 100 samples: 80 standard, 20 random
    # High scores (indices 50-99) correspond to adequacy = 1
    y_adequate = np.array([0] * 50 + [1] * 50)
    is_random = np.array([0] * 80 + [1] * 20)
    scores = np.linspace(0.0, 1.0, 100)

    target_coverages = [0.20, 0.50, 0.80]
    results = evaluate_gate_at_coverages(
        y_adequate=y_adequate,
        is_random_mask=is_random,
        scores=scores,
        target_coverages=target_coverages,
    )

    assert "auroc" in results
    assert "average_precision" in results
    assert "aurc" in results
    assert "by_coverage" in results

    # At top 20% coverage (highest scores), all should be adequate (score >= 0.8)
    cov_20 = results["by_coverage"]["20%"]
    assert cov_20["selective_accuracy"] == 1.0
    assert cov_20["selective_risk"] == 0.0


def test_run_ablation_evaluation_synthetic():
    probs, quality, labels = generate_synthetic_data(n_samples=150, n_classes=6, seed=42)
    adeq_est = AdequacyEstimator(adequacy_threshold=0.75)
    random_idx = 4
    y_target = ((np.argmax(probs, axis=1) == labels) & (labels != random_idx)).astype(int)
    X = adeq_est.extract_adequacy_features(quality, probs)
    adeq_est.fit(X, y_target)

    results = run_ablation_evaluation(
        probs=probs,
        quality=quality,
        labels=labels,
        adequacy_estimator=adeq_est,
        target_coverages=[0.50, 0.70],
    )

    assert results["n_total"] == 150
    assert "strategies" in results
    strategies = results["strategies"]

    expected_gates = [
        "fitted_adequacy",
        "quality_only",
        "confidence_only",
        "heuristic_adequacy",
        "negative_entropy",
        "random_baseline",
    ]
    for gate in expected_gates:
        assert gate in strategies
        assert "auroc" in strategies[gate]
        assert "by_coverage" in strategies[gate]
        assert "50%" in strategies[gate]["by_coverage"]
        assert "70%" in strategies[gate]["by_coverage"]


def test_format_ablation_markdown_table():
    probs, quality, labels = generate_synthetic_data(n_samples=50, n_classes=6, seed=42)
    adeq_est = AdequacyEstimator(adequacy_threshold=0.75)
    y_target = ((np.argmax(probs, axis=1) == labels) & (labels != 4)).astype(int)
    X = adeq_est.extract_adequacy_features(quality, probs)
    adeq_est.fit(X, y_target)

    results = run_ablation_evaluation(probs, quality, labels, adeq_est, target_coverages=[0.50, 0.70])
    table = format_ablation_markdown_table(results)

    assert "## Diagnostic Adequacy Gating Ablation Study" in table
    assert "Fitted Adequacy Estimator (M4)" in table
    assert "Quality-Only Gate" in table
    assert "Confidence-Only Gate" in table
    assert "Random Rej @ 70%" in table
