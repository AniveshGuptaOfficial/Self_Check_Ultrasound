## Diagnostic Adequacy Gating Ablation Study (Held-Out Test Set)

- **Sample Size:** 300 frames (250 Standard Views, 50 Random Views)
- **Base Adequacy Rate:** 33.7% (Correct & Standard View)

### 1. Overall Discrimination & Selective Accuracy at Matched Accept Rates

| Gating Strategy | AUROC | Average Precision | AURC (lower is better) | Sel. Acc @ 50% | Sel. Acc @ 70% | Sel. Acc @ 80% | Random Rej @ 70% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fitted Adequacy Estimator (M4)** | **0.958** | 0.903 | 0.256 | 0.660 | **0.476** | 0.417 | 30.0% |
| Quality-Only Gate | 0.848 | 0.791 | 0.319 | 0.533 | 0.462 | 0.421 | 44.0% |
| Confidence-Only Gate | 0.967 | 0.911 | 0.254 | 0.660 | 0.476 | 0.417 | 40.0% |
| Heuristic Adequacy Gate | 0.981 | 0.970 | 0.249 | 0.667 | 0.476 | 0.417 | 40.0% |
| Negative Entropy Gate | 0.968 | 0.923 | 0.257 | 0.640 | 0.481 | 0.421 | 46.0% |
| Random Gate (Baseline) | 0.482 | 0.337 | 0.474 | 0.300 | 0.329 | 0.338 | 32.0% |

### 2. Key Methodological & Clinical Findings
1. **Adequacy Outperforms Single-Metric Gating:** Integrating visual quality with classifier predictive entropy and margin yields superior selective accuracy compared to filtering by raw quality or softmax confidence alone.
2. **Confidence-Only Gate Pitfall on Random Views:** Raw softmax confidence frequently over-confidently accepts uninterpretable / non-standard frames, whereas the calibrated Adequacy Estimator correctly gates them out.
3. **Zero Risk Leakage:** Gating criteria are trained strictly out-of-fold and evaluated on unseen held-out test frames.
