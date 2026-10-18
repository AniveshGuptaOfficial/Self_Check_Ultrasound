"""
Generate Publication-Quality Research Plots (Phase 5: Design Quality).
Produces high-resolution figures for academic reporting and research documentation:
- Figure 1: Dataset Partition Distribution (Sequence-Isolated Splits)
- Figure 2: Expert Quality Grade Distribution by Cardiac View
- Figure 3: Controlled Acoustic Degradation Library & Guidance Mappings
- Figure 4: View Classification Baseline Confusion Matrices
- Figure 5: Closed-Loop Simulation Trajectory Progressions
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.data.degradations import SyntheticDegradationPipeline
from self_check_ultrasound.visualization.plots import (
    plot_dataset_splits_distribution,
    plot_quality_distributions,
    plot_degradation_grid,
    plot_confusion_matrices,
    plot_simulation_trajectories,
)


def main():
    print("=" * 70)
    print("SelfCheckUltrasound: Research Figure Generation (Phase 5)")
    print("=" * 70)

    figures_dir = default_config.paths.artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits_path = default_config.paths.splits_path
    dataset_root = default_config.paths.dataset_dir
    metrics_path = default_config.paths.artifacts_dir / "view_baseline_metrics.json"
    trajectories_path = default_config.paths.artifacts_dir / "simulation_trajectories.json"

    print(f"Loading split metadata from: {splits_path}")
    splits_df = pd.read_csv(splits_path, low_memory=False)

    generated_figures = []

    # 1. Figure 1: Dataset Partition Distribution
    fig1_path = figures_dir / "fig1_dataset_splits_distribution.png"
    print(f"\n[1/5] Generating Figure 1: Dataset Partition Distribution...")
    plot_dataset_splits_distribution(splits_df, fig1_path)
    generated_figures.append(("Figure 1: Dataset Splits", fig1_path))
    print(f"      Saved: {fig1_path} ({fig1_path.stat().st_size / 1024:.1f} KB)")

    # 2. Figure 2: Quality Grade Distribution
    fig2_path = figures_dir / "fig2_quality_distribution_by_view.png"
    print(f"\n[2/5] Generating Figure 2: Quality Distribution by View...")
    plot_quality_distributions(splits_df, fig2_path)
    generated_figures.append(("Figure 2: Quality by View", fig2_path))
    print(f"      Saved: {fig2_path} ({fig2_path.stat().st_size / 1024:.1f} KB)")

    # 3. Figure 3: Synthetic Degradation Library Grid
    fig3_path = figures_dir / "fig3_synthetic_degradations_grid.png"
    print(f"\n[3/5] Generating Figure 3: Controlled Acoustic Degradation Grid...")
    # Find a nominal test image (grade >= 7)
    sample_rows = splits_df[
        (splits_df["split"] == "val")
        & (splits_df["quality_grade"] >= 7)
    ]
    sample_row = sample_rows.iloc[0]
    sample_img_path = dataset_root / sample_row["relative_path"]
    sample_img = Image.open(sample_img_path).convert("RGB")
    pipe = SyntheticDegradationPipeline(random_seed=42)
    plot_degradation_grid(sample_img, pipe, fig3_path)
    generated_figures.append(("Figure 3: Degradation Grid", fig3_path))
    print(f"      Saved: {fig3_path} ({fig3_path.stat().st_size / 1024:.1f} KB)")

    # 4. Figure 4: Baseline Confusion Matrices
    fig4_path = figures_dir / "fig4_confusion_matrices.png"
    print(f"\n[4/5] Generating Figure 4: View Classification Confusion Matrices...")
    if metrics_path.exists():
        plot_confusion_matrices(metrics_path, fig4_path)
        generated_figures.append(("Figure 4: Confusion Matrices", fig4_path))
        print(f"      Saved: {fig4_path} ({fig4_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"      [SKIP] Metrics file not found at {metrics_path}")

    # 5. Figure 5: Closed-Loop Trajectories
    fig5_path = figures_dir / "fig5_closed_loop_trajectories.png"
    print(f"\n[5/5] Generating Figure 5: Closed-Loop Simulation Trajectories...")
    if trajectories_path.exists():
        plot_simulation_trajectories(trajectories_path, fig5_path)
        generated_figures.append(("Figure 5: Simulation Trajectories", fig5_path))
        print(f"      Saved: {fig5_path} ({fig5_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"      [SKIP] Trajectories file not found at {trajectories_path}")

    print("\n" + "=" * 70)
    print("RESEARCH FIGURE GENERATION COMPLETE")
    print("=" * 70)
    for title, path in generated_figures:
        print(f"  * {title:<32s} -> {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
