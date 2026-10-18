"""
Tests for bootstrap / subsampling helpers.
"""

import numpy as np
import pandas as pd

from self_check_ultrasound.utils.eval_utils import balanced_subsample, bootstrap_ci, paired_bootstrap_diff


def _acc(y, p):
    return float(np.mean(y == p))


def test_bootstrap_ci_contains_point_and_is_ordered():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    p = np.where(rng.rand(300) < 0.8, y, 1 - y)
    ci = bootstrap_ci(y, p, _acc, n_boot=300)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert 0.7 < ci["point"] < 0.9
    assert ci["n"] == 300


def test_paired_bootstrap_detects_real_difference_but_not_noise():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 400)
    good = np.where(rng.rand(400) < 0.9, y, 1 - y)
    bad = np.where(rng.rand(400) < 0.6, y, 1 - y)
    assert paired_bootstrap_diff(y, good, bad, _acc, n_boot=300)["significant"]

    a = np.where(rng.rand(300) < 0.7, y[:300], 1 - y[:300])
    b = a.copy()
    flip = rng.choice(300, 3, replace=False)  # 3 of 300 predictions differ
    b[flip] = 1 - b[flip]
    assert not paired_bootstrap_diff(y[:300], a, b, _acc, n_boot=300)["significant"]


def test_balanced_subsample():
    df = pd.DataFrame({"view_class": ["A"] * 100 + ["B"] * 100 + ["C"] * 10, "x": range(210)})
    out = balanced_subsample(df, max_samples=60, n_classes=3)
    assert out["view_class"].value_counts().to_dict() == {"A": 20, "B": 20, "C": 10}
    assert len(balanced_subsample(df, max_samples=None, n_classes=3)) == 210
