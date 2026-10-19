"""Tests for the external-validation label mapping and metrics (no torch)."""

import numpy as np
import pandas as pd

from self_check_ultrasound.utils.external import (
    norm_view, map_views, correct_mask, binary_ece, auroc_ci, summarize,
)


def _probs(idx_list, p=0.9):
    out = np.full((len(idx_list), 6), (1 - p) / 5)
    for i, k in enumerate(idx_list):
        out[i, k] = p
    return out


def test_norm_view_and_mapping():
    assert norm_view("PSAX-AV") == "PSAXAV"
    s = pd.Series(["PLAX", "psax", "A4C", "A2C", "Other"])
    g = map_views(s, "tmed2")
    assert list(g[:3]) == ["PL", "PSAX", "A4C"]
    assert pd.isna(g[3]) and pd.isna(g[4])
    c = map_views(pd.Series(["PSAX-AV", "PSAX-MV", "S4C", "A3C"]), "cameo")
    assert list(c[:3]) == ["PSAV", "PSMV", "SC"] and pd.isna(c[3])


def test_correct_mask_handles_coarse_psax_and_unseen():
    groups = pd.Series(["A4C", "PL", "PSAX", "PSAX", None], dtype=object)
    # A4C ok; PL predicted PSAV (wrong); PSAX predicted PSMV (ok); PSAX predicted Random (wrong); unseen -> False
    probs = _probs([0, 2, 3, 4, 0])
    assert list(correct_mask(probs, groups)) == [True, False, True, False, False]


def test_summarize_reports_unseen_rejection_and_auroc():
    rng = np.random.RandomState(0)
    n = 200
    groups = pd.Series(["A4C"] * n + [None] * 50, dtype=object)
    right = rng.rand(n) < 0.7
    probs = np.vstack([_probs([0 if r else 1 for r in right]), _probs([0] * 50, p=0.4)])
    adequacy = np.concatenate([np.where(right, 0.9, 0.3), np.full(50, 0.2)])
    out = summarize(probs, groups, adequacy, threshold=0.75, n_boot=200)
    assert out["n_mapped"] == n and out["n_unseen"] == 50
    assert 0.6 < out["view_accuracy"]["point"] < 0.8
    assert out["adequacy_auroc_predicts_correct_view"]["point"] > 0.95
    assert out["accept_rate_unseen_views"] == 0.0
    assert binary_ece(np.ones(10), np.ones(10)) == 0.0
    assert auroc_ci(np.ones(5), np.arange(5)) is None
