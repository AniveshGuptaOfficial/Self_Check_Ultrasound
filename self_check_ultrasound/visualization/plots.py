"""
Research plotting and figure generation module (Phase 5: Design Quality).
Produces clean, publication-ready figures for research papers and technical reports:
1. Dataset & view distributions across sequence-isolated partitions
2. Quality grade distributions across anatomical views
3. Controlled synthetic fault grid with physical annotations
4. Comparative confusion matrices for view recognition baselines
5. Closed-loop reacquisition simulation trajectories
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# Cohesive scientific color palette
PALETTE = {
    "primary": "#1f77b4",       # Deep blue
    "secondary": "#2ca02c",     # Forest green
    "accent": "#ff7f0e",        # Warm orange
    "danger": "#d62728",        # Crimson red
    "purple": "#9467bd",        # Royal purple
    "teal": "#17becf",          # Cyan/teal
    "gray": "#7f7f7f",          # Slate gray
    "light_gray": "#e0e0e0",    # Light grid gray
    "dark": "#2b2b2b",          # Dark charcoal
}

VIEW_COLORS = {
    "A4C": "#1f77b4",
    "PL": "#ff7f0e",
    "PSAV": "#2ca02c",
    "PSMV": "#d62728",
    "Random": "#7f7f7f",
    "SC": "#9467bd",
}


def set_publication_style():
    """Applies a clean, readable publication plot style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#eaeaea",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


def plot_dataset_splits_distribution(
    splits_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Figure 1: Stacked bar chart showing cardiac view distribution partitioned
    across sequence-isolated Train, Validation, and Test splits.
    """
    set_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    views = sorted(splits_df["view_class"].unique())
    splits = ["train", "val", "test"]
    split_labels = ["Train (63.8%)", "Validation (19.2%)", "Test (17.1%)"]

    # Pivot view counts by split
    counts = splits_df.groupby(["split", "view_class"]).size().unstack(fill_value=0)
    counts = counts.reindex(splits)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bar_width = 0.5
    indices = np.arange(len(splits))

    bottom = np.zeros(len(splits))
    for v in views:
        vals = counts[v].values
        ax.bar(
            indices,
            vals,
            bar_width,
            bottom=bottom,
            label=v,
            color=VIEW_COLORS.get(v, "#333333"),
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += vals

    ax.set_title("CACTUS Dataset: Sequence-Isolated Partition Distribution", pad=12, fontweight="bold")
    ax.set_xlabel("Partition Split", labelpad=8)
    ax.set_ylabel("Number of Images", labelpad=8)
    ax.set_xticks(indices)
    ax.set_xticklabels(split_labels)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    # Annotate total counts on top of bars
    for i, total in enumerate(bottom):
        ax.text(i, total + 350, f"{int(total):,} frames", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylim(0, max(bottom) * 1.12)
    ax.legend(title="Cardiac View", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_quality_distributions(
    splits_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Figure 2: Quality score distributions across all 6 cardiac views.
    Demonstrates expert completeness and clarity grades (0-9).
    """
    set_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    views = sorted(splits_df["view_class"].unique())
    data = [splits_df[splits_df["view_class"] == v]["quality_grade"].values for v in views]

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = np.arange(len(views))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.0, "color": "#555555"},
        capprops={"linewidth": 1.0, "color": "#555555"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.2, "markeredgecolor": "#888888"},
    )

    for patch, v in zip(box["boxes"], views):
        patch.set_facecolor(VIEW_COLORS.get(v, "#333333"))
        patch.set_alpha(0.8)
        patch.set_edgecolor("#333333")

    ax.set_title("Expert Quality Grade Distribution by Cardiac View (0-9 Scale)", pad=12, fontweight="bold")
    ax.set_xlabel("Cardiac View Class", labelpad=8)
    ax.set_ylabel("Expert Quality Grade", labelpad=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(views)
    ax.set_ylim(-0.5, 9.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # Annotate mean quality above each box
    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.text(i, 9.1, f"Mean:\n{m:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333333")

    # Legend for symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=1.5, label="Median"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black", markerfacecolor="white", markersize=6, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_degradation_grid(
    base_image: Image.Image,
    degradation_pipeline: Any,
    output_path: Path,
) -> Path:
    """
    Figure 3: 2x3 visual grid of controlled acoustic degradations with
    annotated physical descriptions and guidance suggestions.
    """
    set_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fault_modes = [
        ("none", 0.0, "Nominal / Benchmark", "Reference Acquisition\n(No compensation required)"),
        ("under_gain", 0.70, "Under-Gain (Attenuation)", "Advice: Increase 2D overall\ngain or TGC sliders"),
        ("over_gain", 0.70, "Over-Gain (Saturation)", "Advice: Decrease 2D gain\nto resolve acoustic blowout"),
        ("motion_blur", 0.80, "Motion Blur (Jitter)", "Advice: Stabilize transducer;\nsuspend patient respiration"),
        ("acoustic_shadow", 0.70, "Acoustic Shadow (Rib)", "Advice: Reposition probe angle\naround intercostal margin"),
        ("off_axis", 0.90, "Off-Axis (Misaligned)", "Advice: Re-align probe to\northogonal view plane"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    axes = axes.flatten()

    for idx, (fault, sev, title, guidance) in enumerate(fault_modes):
        ax = axes[idx]
        deg_img, _ = degradation_pipeline.apply(base_image.copy(), degradation_type=fault, severity=sev)
        ax.imshow(deg_img)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

        # Overlay text box at bottom with advice
        bg_col = "#e8f4ea" if fault == "none" else "#fff3cd"
        border_col = "#2ca02c" if fault == "none" else "#e0a800"
        ax.text(
            0.5,
            -0.12,
            guidance,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_col, edgecolor=border_col, linewidth=0.8),
        )

    plt.suptitle("Controlled Acoustic Degradation Library & Actionable Guidance Mappings", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    metrics_path: Path,
    output_path: Path,
) -> Path:
    """
    Figure 4: Side-by-side normalized confusion matrices comparing
    Classical Spatial Baseline vs. CardiacEchoNet (CNN).
    """
    set_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    cm_classical = np.array(metrics["classical_baseline"]["confusion_matrix"])
    cm_cnn = np.array(metrics["cardiac_echo_net"]["confusion_matrix"])

    # Row-normalize to percentages
    cm_classical_norm = cm_classical.astype(np.float32) / (cm_classical.sum(axis=1, keepdims=True) + 1e-7) * 100
    cm_cnn_norm = cm_cnn.astype(np.float32) / (cm_cnn.sum(axis=1, keepdims=True) + 1e-7) * 100

    classes = ["A4C", "PL", "PSAV", "PSMV", "Random", "SC"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    for ax, cm, title, acc in zip(
        axes,
        [cm_classical_norm, cm_cnn_norm],
        ["Classical Spatial + Logistic Baseline", "CardiacEchoNet (4-Stage CNN Baseline)"],
        [metrics["classical_baseline"]["accuracy"] * 100, metrics["cardiac_echo_net"]["accuracy"] * 100],
    ):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(f"{title}\n(Overall Accuracy: {acc:.1f}%)", pad=10, fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted View Class", labelpad=8)
        ax.set_ylabel("True View Class", labelpad=8)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right")
        ax.set_yticklabels(classes)

        # Annotate cell numbers
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm[i, j]
                text_col = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=text_col, fontsize=8.5)

    # Common colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recall (%)", rotation=270, labelpad=14)

    plt.suptitle("Cardiac View Recognition Baseline Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_simulation_trajectories(
    trajectories_path: Path,
    output_path: Path,
) -> Path:
    """
    Figure 5: Step-by-step closed-loop feedback trajectories illustrating
    adequacy probability progression and triage gating decisions over attempts.
    """
    set_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    case_colors = {
        "Case_1_IncidentFree": "#2ca02c",
        "Case_2_Recovery_UnderGain": "#1f77b4",
        "Case_3_Recovery_MotionBlur": "#ff7f0e",
        "Case_4_Recovery_RibShadow": "#9467bd",
        "Case_5_Referral_PersistentFault": "#d62728",
    }
    case_markers = {
        "Case_1_IncidentFree": "o",
        "Case_2_Recovery_UnderGain": "s",
        "Case_3_Recovery_MotionBlur": "^",
        "Case_4_Recovery_RibShadow": "D",
        "Case_5_Referral_PersistentFault": "X",
    }

    # Threshold line
    ax.axhline(0.75, color="#d62728", linestyle="--", linewidth=1.5, label="Adequacy Acceptance Threshold (0.75)")
    ax.fill_between([0.8, 4.2], 0.75, 1.05, color="#2ca02c", alpha=0.08, label="Diagnostic Accept Zone")
    ax.fill_between([0.8, 4.2], 0.0, 0.75, color="#ff7f0e", alpha=0.06, label="Reacquire / Refer Zone")

    for case in trajectories:
        c_id = case["case_id"]
        traj = case["trajectory"]
        attempts = [s["attempt"] for s in traj]
        adeq = [s["adequacy_prob"] for s in traj]
        color = case_colors.get(c_id, "#333333")
        marker = case_markers.get(c_id, "o")

        label = f"{c_id.replace('_', ' ')} -> {case['final_action']}"
        ax.plot(attempts, adeq, marker=marker, color=color, linewidth=2.0, markersize=8, label=label)

        # Annotate decisions on points
        for s in traj:
            badge = s["action"]
            badge_color = "#2ca02c" if badge == "ACCEPT" else ("#ff7f0e" if badge == "REACQUIRE" else "#d62728")
            ax.text(
                s["attempt"],
                s["adequacy_prob"] + (0.04 if s["adequacy_prob"] < 0.8 else -0.05),
                f"[{badge}]",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=badge_color,
            )

    ax.set_title("Closed-Loop Quality Reacquisition Simulation Trajectories", pad=12, fontweight="bold")
    ax.set_xlabel("Reacquisition Attempt Index", labelpad=8)
    ax.set_ylabel("Calibrated Adequacy Probability P(Adequate)", labelpad=8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.3)
    ax.set_ylim(0.25, 1.05)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
