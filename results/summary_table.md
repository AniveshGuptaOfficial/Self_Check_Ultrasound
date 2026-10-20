## Empirical Baseline Results (Held-Out TEST Split)

All baseline evaluations are reported on the untouched **held-out TEST split** using **95% percentile bootstrap confidence intervals**:

### View Classification (Module M1)
Comparison between classical spatial feature baseline and `CardiacEchoNet` (4-stage CNN, ~370k parameters):

| Metric | Classical (Spatial + Logistic) | CardiacEchoNet (CNN) | Paired Difference [95% CI] |
| :--- | :---: | :---: | :---: | 
| **Accuracy** | **0.713** [0.663, 0.763] | 0.370 [0.313, 0.420] | -0.343 [-0.407, -0.277] |
| **Macro F1-Score** | **0.688** [0.636, 0.734] | 0.258 [0.223, 0.291] | -0.430 [-0.480, -0.378] |
| **ECE (Calibration)** | **0.107** | 0.292 | — |

*Training uses mild rotation without horizontal flipping, preserving standard anatomical cardiac chamber orientation.*

### Quality Assessment (Module M3)
Continuous regression against expert completeness/clarity grades (0–9 scale):

| Metric | Naive Baseline (Mean = 3.99) | QualityAssessorNet (Best Val Checkpoint) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | 2.250 [2.100, 2.407] | **2.168** [2.033, 2.322] |
| **Root Mean Squared Error (RMSE)** | 2.638 [2.513, 2.769] | **2.555** [2.409, 2.706] |
| **Pearson Correlation (r)** | 0.000 | **0.411** [0.332, 0.485] |
| **Quadratic Weighted Kappa (QWK)** | 0.000 | **0.138** |

### Computational Performance & Runtime Benchmarks (Phase 7)

Evaluated on local CPU hardware (`AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD`, 8 logical cores, PyTorch 2.14.0+cpu, 4 threads; single-frame streaming mode batch size = 1):

| Pipeline Component / Module | Latency (Mean) | Latency (Median) | 95th Percentile | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **CardiacEchoNet** (M1 View / M2 Downstream) | 28.80 ms | 29.07 ms | 30.49 ms | 34.7 FPS |
| **QualityAssessorNet** (M3 Quality Grade) | 29.98 ms | 30.10 ms | 32.06 ms | 33.4 FPS |
| **CauseClassifier** (M5 Sector Physics + RF) | 8.82 ms | 8.90 ms | 10.05 ms | 113.6 FPS |
| **AdequacyEstimator** (M4 Task Adequacy) | 2.17 ms | 2.01 ms | 3.05 ms | 460.5 FPS |
| **ReacquisitionPolicy** (M6 Guidance / M7 Triage) | 9.42 us | 9.10 us | — | 106,157 calls/sec |
| **End-to-End Decision Cycle** (Raw Frame to Triage) | **73.99 ms** | **73.52 ms** | **83.21 ms** | **13.5 FPS** |
| **Full HD Ingestion & RGB Transform** (1920x1080) | 23.62 ms | 22.70 ms | 25.14 ms | 42.3 images/sec |

- **Memory Footprint:** Peak traced heap memory = **1.51 MB** (Model weights in RAM = 0.33 MB).
- **Interactive Point-of-Care Feasibility:** The end-to-end closed-loop decision cycle executes at **13.5 FPS entirely on CPU** (74.0 ms latency), satisfying point-of-care clinical ultrasound guidance latency targets (5–10 Hz) without requiring GPU acceleration.
