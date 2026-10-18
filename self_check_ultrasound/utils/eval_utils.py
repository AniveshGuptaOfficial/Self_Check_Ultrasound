"""
Evaluation helpers (no torch dependency):
- class-balanced subsampling of a split DataFrame
- percentile bootstrap confidence intervals
- paired bootstrap for the difference between two models on the same samples
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def balanced_subsample(
    df: pd.DataFrame,
    max_samples: Optional[int],
    n_classes: int,
    group_col: str = "view_class",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns at most `max_samples` rows, drawn as evenly as possible across `group_col`.
    If max_samples is falsy or >= len(df) the frame is returned unchanged.
    """
    if not max_samples or len(df) <= max_samples:
        return df.reset_index(drop=True)
    k = max(1, max_samples // n_classes)
    parts = [grp.sample(n=min(len(grp), k), random_state=seed) for _, grp in df.groupby(group_col)]
    return pd.concat(parts).reset_index(drop=True)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI of metric_fn(y_true, y_pred), resampling samples with replacement."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap of metric(a) - metric(b) on the SAME samples.
    `significant` is True when the CI excludes 0.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))
    return {
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "n": int(n),
    }
