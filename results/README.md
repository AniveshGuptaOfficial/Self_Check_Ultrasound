# Final Scientific Results & Evaluation Summaries

This directory contains lightweight, version-controlled JSON summaries of the final empirical evaluations and benchmarks reported in the research paper and repository documentation.

While large model weights (`*.pt`, `*.joblib`) and Full HD dataset manifests (`*.csv`) are kept locally in `artifacts/` (and excluded from Git), these JSON summaries are committed directly to ensure complete traceability, open science reproducibility, and auditability.

---

## Files in this Directory

| File | Originating Script | Contents |
| :--- | :--- | :--- |
| `baseline_metrics.json` | `scripts/train_view_baseline.py`<br>`scripts/train_quality_baseline.py` | Complete test-set evaluation metrics and 95% bootstrap confidence intervals for View Classification, Image Quality Assessment, and Task Adequacy. |
| `view_baseline_metrics.json` | `scripts/train_view_baseline.py` | Standalone view classification results: Accuracy, Macro F1, Per-class F1, Expected Calibration Error (ECE), Confusion Matrix, and 95% bootstrap CIs. |
| `quality_baseline_metrics.json` | `scripts/train_quality_baseline.py` | Standalone quality assessment results: MAE, RMSE, Pearson $r$, Spearman $\rho$, Quadratic Weighted Kappa (QWK), and 95% bootstrap CIs. |
| `adequacy_metrics.json` | `scripts/train_adequacy.py` | Validation out-of-fold and held-out test set accuracy, balanced accuracy, F1, AUROC, Brier score, and ECE for task adequacy prediction. |
| `cause_classifier_eval.json` | `scripts/evaluate_cause_classifier.py` | Diagnostic accuracy and classification report of the physics-informed acoustic cause classifier across 6 fault classes. |
| `performance_benchmarks.json` | `scripts/benchmark_performance.py` | Real-time CPU execution profiles: per-module latencies, end-to-end cycle latency ($120.5\text{ ms}$ / $8.3\text{ FPS}$), memory footprint ($1.5\text{ MB}$), and batch scaling. |
| `simulation_trajectories.json` | `scripts/run_simulation.py` | Closed-loop reacquisition simulation trajectories across 5 clinical scenarios (nominal, under-gain recovery, motion blur, acoustic shadow, persistent fault). |
| `calibration_data.json` | `self_check_ultrasound/utils/metrics.py` | Empirical calibration curve statistics (accuracy vs. confidence per bin) for Reliability Diagram visualization. |
| `multiseed_metrics.json` | `scripts/run_multiseed_eval.py` | Statistical aggregation ($\mu \pm \sigma$) across multi-seed runs (seeds 42, 123, 456, 789, 1011) evaluating robustness against random variation. |

---

## Verifying & Synchronizing with README

To ensure numbers in `README.md` never drift from these JSON outputs, run:

```bash
# Check alignment between JSON outputs and README tables
python scripts/sync_results_table.py --check

# Re-generate markdown summary and update README.md
python scripts/sync_results_table.py --update-readme
```
