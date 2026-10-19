"""
External-validation helpers (numpy/pandas/sklearn only, no torch).

Maps another echo dataset's view labels onto the CACTUS model's 6-class output space and
computes the metrics that matter for the reacquisition policy on that dataset:
view accuracy, calibration, adequacy AUROC, and rejection of views the model never saw.

Model class indices (config.DataConfig.class_to_idx): A4C=0, PL=1, PSAV=2, PSMV=3, Random=4, SC=5.
A target "group" is the set of model classes that count as correct for that image, so a coarse
label such as TMED-2 'PSAX' (aortic + mitral + others) is correct if the model says PSAV or PSMV.
"""

import re
from typing import Dict, Optional, Set
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GROUP_TO_CLASSES: Dict[str, Set[int]] = {
    "A4C": {0},
    "PL": {1},
    "PSAV": {2},
    "PSMV": {3},
    "SC": {5},
    "PSAX": {2, 3},  # coarse parasternal short axis (level not specified)
}

# normalised source label -> target group. Anything not listed is "unseen" (no CACTUS counterpart).
DATASET_MAPS: Dict[str, Dict[str, str]] = {
    # TMED-2: PLAX / PSAX / A2C / A4C / Other. A2C and Other have no CACTUS class.
    "tmed2": {"PLAX": "PL", "PSAX": "PSAX", "A4C": "A4C"},
    # CAMEO (as described on IEEE DataPort): A2C, A3C, A4C, PLAX, PSAX at aortic/mitral/papillary
    # levels, and S4C. Folder spellings must be checked against the downloaded archive.
    "cameo": {"A4C": "A4C", "PLAX": "PL", "PSAXAV": "PSAV", "PSAXMV": "PSMV", "S4C": "SC"},
    # CAMUS has only 2CH and 4CH.
    "camus": {"4CH": "A4C", "A4C": "A4C"},
}


def norm_view(label: str) -> str:
    """Upper-case and strip every non-alphanumeric character: 'PSAX-AV' -> 'PSAXAV'."""
    return re.sub(r"[^A-Z0-9]", "", str(label).upper())


def map_views(views: pd.Series, dataset: str) -> pd.Series:
    """Returns the target group for each row, or None for unseen views."""
    if dataset not in DATASET_MAPS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {sorted(DATASET_MAPS)}")
    table = DATASET_MAPS[dataset]
    return views.map(lambda v: table.get(norm_view(v)))


def correct_mask(probs: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Boolean per row: argmax class lies in the target group's class set. Unseen rows -> False."""
    pred = np.argmax(probs, axis=1)
    out = np.zeros(len(pred), dtype=bool)
    for i, (p, g) in enumerate(zip(pred, groups)):
        if g is not None and g == g:  # not None / NaN
            out[i] = int(p) in GROUP_TO_CLASSES[g]
    return out


def binary_ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> float:
    correct = np.asarray(correct, dtype=float)
    conf = np.asarray(conf, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
    return float(ece)


def auroc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Optional[Dict[str, float]]:
    """AUROC with percentile bootstrap CI; None when only one class is present."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    rng = np.random.RandomState(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        stats.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": float(roc_auc_score(y, score)), "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def summarize(probs: np.ndarray, groups: pd.Series, adequacy: Optional[np.ndarray],
              threshold: float, n_boot: int = 1000) -> Dict[str, object]:
    """View accuracy, calibration, adequacy AUROC on mapped rows; accept rate on mapped vs unseen rows."""
    mapped = np.array([g is not None and g == g for g in groups])
    correct = correct_mask(probs, groups)
    conf = probs.max(axis=1)
    out: Dict[str, object] = {
        "n_total": int(len(groups)),
        "n_mapped": int(mapped.sum()),
        "n_unseen": int((~mapped).sum()),
    }
    if mapped.any():
        c = correct[mapped]
        rng = np.random.RandomState(42)
        boots = [c[rng.randint(0, len(c), len(c))].mean() for _ in range(n_boot)]
        out["view_accuracy"] = {"point": float(c.mean()), "ci_low": float(np.percentile(boots, 2.5)),
                                "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(c))}
        out["ece_max_prob"] = binary_ece(c, conf[mapped])
        out["mean_confidence_correct"] = float(conf[mapped][c].mean()) if c.any() else None
        out["mean_confidence_wrong"] = float(conf[mapped][~c].mean()) if (~c).any() else None
    if adequacy is not None:
        adequacy = np.asarray(adequacy, dtype=float)
        accept = adequacy >= threshold
        if mapped.any():
            out["adequacy_auroc_predicts_correct_view"] = auroc_ci(correct[mapped], adequacy[mapped], n_boot)
            out["accept_rate_mapped"] = float(accept[mapped].mean())
            out["view_accuracy_on_accepted"] = float(correct[mapped][accept[mapped]].mean()) if accept[mapped].any() else None
        if (~mapped).any():
            out["accept_rate_unseen_views"] = float(accept[~mapped].mean())
    return out
