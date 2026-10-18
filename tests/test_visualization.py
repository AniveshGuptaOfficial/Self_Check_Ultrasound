"""
Unit tests for visualization and figure generation module (Phase 5).
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

from self_check_ultrasound.visualization.plots import (
    plot_dataset_splits_distribution,
    plot_quality_distributions,
    plot_degradation_grid,
    plot_confusion_matrices,
    plot_simulation_trajectories,
)
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline


@pytest.fixture
def dummy_splits_df():
    views = ["A4C", "PL", "PSAV", "PSMV", "Random", "SC"]
    splits = ["train", "val", "test"]
    rows = []
    for s in splits:
        for v in views:
            for i in range(10):
                rows.append({
                    "view_class": v,
                    "split": s,
                    "quality_grade": np.random.uniform(2.0, 8.5),
                })
    return pd.DataFrame(rows)


def test_plot_dataset_splits_distribution(dummy_splits_df, tmp_path):
    out_file = tmp_path / "test_fig1.png"
    result = plot_dataset_splits_distribution(dummy_splits_df, out_file)
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_quality_distributions(dummy_splits_df, tmp_path):
    out_file = tmp_path / "test_fig2.png"
    result = plot_quality_distributions(dummy_splits_df, out_file)
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_degradation_grid(tmp_path):
    out_file = tmp_path / "test_fig3.png"
    dummy_img = Image.new("RGB", (64, 64), color="gray")
    pipe = SyntheticDegradationPipeline(random_seed=42)
    result = plot_degradation_grid(dummy_img, pipe, out_file)
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_confusion_matrices(tmp_path):
    import json
    metrics_file = tmp_path / "metrics.json"
    dummy_metrics = {
        "classical_baseline": {
            "accuracy": 0.50,
            "confusion_matrix": [[10] * 6 for _ in range(6)],
        },
        "cardiac_echo_net": {
            "accuracy": 0.55,
            "confusion_matrix": [[12] * 6 for _ in range(6)],
        },
    }
    with open(metrics_file, "w") as f:
        json.dump(dummy_metrics, f)

    out_file = tmp_path / "test_fig4.png"
    result = plot_confusion_matrices(metrics_file, out_file)
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_simulation_trajectories(tmp_path):
    import json
    traj_file = tmp_path / "trajectories.json"
    dummy_traj = [
        {
            "case_id": "Case_1_IncidentFree",
            "final_action": "ACCEPT",
            "trajectory": [
                {"attempt": 1, "adequacy_prob": 0.90, "action": "ACCEPT"},
            ],
        },
        {
            "case_id": "Case_5_Referral_PersistentFault",
            "final_action": "REFER",
            "trajectory": [
                {"attempt": 1, "adequacy_prob": 0.40, "action": "REACQUIRE"},
                {"attempt": 2, "adequacy_prob": 0.40, "action": "REFER"},
            ],
        },
    ]
    with open(traj_file, "w") as f:
        json.dump(dummy_traj, f)

    out_file = tmp_path / "test_fig5.png"
    result = plot_simulation_trajectories(traj_file, out_file)
    assert result.exists()
    assert result.stat().st_size > 1000
