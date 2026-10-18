"""
Tests for CauseClassifier training and the two evaluation routines
(synthetic held-out faults, and real-grade AUROC).
"""

import numpy as np

from self_check_ultrasound.models.cause_classifier import CauseClassifier


def test_reference_sampling_spreads_over_sequence_groups(fake_dataset):
    splits_df, _ = fake_dataset
    rows = CauseClassifier._select_reference_rows(splits_df, "train", n_per_view=4, grade_min=7, seed=1)
    for view in ("A4C", "PL"):
        groups = [r["sequence_group"] for r in rows if r["view_class"] == view]
        assert len(groups) == 4
        assert len(set(groups)) == 4, "picks must come from different sequence groups"


def test_train_and_heldout_synthetic_evaluation(fake_dataset):
    splits_df, root = fake_dataset
    clf = CauseClassifier()
    n = clf.train_on_reference_samples(splits_df, root, n_samples_per_view=4)
    assert n == 2 * 4 * len(CauseClassifier.FAULT_CLASSES)
    assert clf.is_fitted

    probs = clf.predict_proba(np.zeros((1, 10), dtype=np.float32))
    assert probs.shape == (1, 6) and np.isclose(probs.sum(), 1.0)

    res = clf.evaluate_on_synthetic(splits_df, root, split="val", n_samples_per_view=2)
    assert res["n"] > 0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert set(res["per_fault_accuracy"]) <= set(CauseClassifier.FAULT_CLASSES)
    assert "per_severity_accuracy" in res and "confusion" in res
    assert res["split"] == "val"


def test_expert_grade_auroc_on_real_style_frames(fake_dataset):
    splits_df, root = fake_dataset
    clf = CauseClassifier()
    clf.train_on_reference_samples(splits_df, root, n_samples_per_view=4)
    res = clf.evaluate_against_expert_grades(splits_df, root, split="val", n_per_view=2)
    assert res["n_high"] > 0 and res["n_low"] > 0
    # dim frames (grade 1) should score lower P(none) than normal frames (grade 8)
    assert res["auroc_p_none_vs_high_grade"] > 0.8
    assert res["frac_predicted_none_high_grade"] >= res["frac_predicted_none_low_grade"]
