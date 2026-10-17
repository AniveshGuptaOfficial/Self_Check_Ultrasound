# SelfCheckUltrasound: ImageQuality-Aware Diagnosis and Reacquisition Guidance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 21 Passing](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)]()
[![Security: pip--audit clean](https://img.shields.io/badge/security-pip--audit%20clean-success.svg)]()

A Python research prototype and machine learning framework for cardiac ultrasound (echocardiography). The system connects **image quality assessment**, **task-specific adequacy estimation**, **cause-aware reacquisition guidance**, and **closed-loop simulation** to support ultrasound acquisition quality control.

---

## 1. Project Scope & Ethical Boundary

- **Organ & Modality:** Exclusively **cardiac ultrasound (echocardiography / echo)**.
- **Data Origin:** Acquired on a **CAE Blue Phantom** echocardiography training phantom with a **GE M4S Matrix Probe** on a **GE Healthcare Vivid-Q ultrasound machine**.
- **Clinical Boundary:** This is an engineering research prototype and decision-support simulation. It is **not** an autonomous medical device, does **not** perform clinical disease diagnosis (the phantom contains anatomical structures, not human pathologies), and contains **no** protected health information (PHI).
- **Environment:** Pure local Python package with reproducible CLI execution, automated unit testing, and structured file outputs. No web servers or web applications.

---

## 2. Dataset Overview: CACTUS

This project utilizes the **CACTUS** open dataset ([Elmekki et al., 2025](https://doi.org/10.1016/j.compbiomed.2025.110003)):
- **Total Images:** 37,736 images (100% Full HD $1920 \times 1080$, 100% RGB, 0 corrupted files).
- **Cardiac Views (6 Classes):**
  1. `A4C` — Apical Four Chamber (7,422 images)
  2. `PL` — Parasternal Long Axis (6,102 images)
  3. `PSAV` — Parasternal Short Axis – Aortic Valve (5,832 images)
  4. `PSMV` — Parasternal Short Axis – Mitral Valve (6,014 images)
  5. `SC` — Subcostal Four Chamber (6,345 images)
  6. `Random` — Non-standard / Uninterpretable / Out-of-plane scans (6,021 images)
- **Expert Quality Grades:** Every image has a ground-truth score (0–9 scale) assigned by cardiovascular imaging experts evaluating **completeness** (visibility of target cardiac structures) and **clarity** (luminosity, speckle purity, SNR).
- **Sequence Metadata:** Filenames encode ultrasound parameters: depth (`D14`–`D17`), dynamic range (`DR2`–`DR7`), video sequence (`V1`–`V5`, `Folder1`–`Folder5`), and sequential frame indices.

---

## 3. Architecture & Core Modules

```
Self_Check_Ultrasound/
├── configs/
│   └── default_config.py          # Central configuration (paths, seeds, thresholds)
├── self_check_ultrasound/
│   ├── data/
│   │   ├── manifest.py            # Dataset discovery & metadata parser (all 37,736 files)
│   │   ├── splits.py              # Sequence-aware group-stratified train/val/test splitting
│   │   ├── transforms.py          # Standardized preprocessing (224x224 resize, normalization)
│   │   ├── degradations.py        # Controlled synthetic fault generator (6 modes)
│   │   └── dataset.py             # PyTorch Dataset loader with path-traversal protection
│   ├── models/
│   │   ├── view_classifier.py     # M1: Cardiac view recognition (CardiacEchoNet & Logistic)
│   │   └── quality_assessor.py    # M3: Expert quality grade regression (QualityAssessorNet)
│   └── utils/
│       ├── seed.py                # Deterministic seed controller (random, numpy, torch)
│       └── metrics.py             # Balanced accuracy, Macro F1, QWK, Pearson r, ECE
├── scripts/
│   ├── build_manifest.py          # Generates dataset_manifest.csv & leakage-free splits.csv
│   ├── train_view_baseline.py     # Trains & evaluates view classification baselines
│   └── train_quality_baseline.py  # Trains & evaluates image quality assessment model
├── tests/                         # Comprehensive 21-test automated suite (100% pass)
├── requirements.txt               # Pinned Python dependencies
└── CITATION.md                    # Formal academic attribution & CC BY 4.0 terms
```

---

## 4. Key Methodological Contributions

### 4.1 Sequence-Aware Leakage Prevention
Consecutive video frames in echocardiography (e.g., `frame_1088` and `frame_1089`) are visually near-identical. A random split results in massive data leakage. Our `splits.py` module groups images by `(view, sequence_group, machine_param)` and applies balanced bin-packing to ensure:
- **Zero Sequence Leakage:** The intersection of sequence groups across Train, Validation, and Test is strictly $\emptyset$.
- **Partition:** Train: 24,070 images ($63.8\%$), Validation: 7,228 images ($19.2\%$), Test: 6,438 images ($17.1\%$).

### 4.2 Controlled Synthetic Degradations
To evaluate cause-aware reacquisition guidance in a controlled setting, `degradations.py` provides 6 calibrated fault modes with severity scaling ($\alpha \in [0.1, 1.0]$):
1. `under_gain`: Acoustic attenuation / insufficient gain ($I' = \alpha I$).
2. `over_gain`: Acoustic blooming / signal blowout.
3. `acoustic_shadow`: Vertical wedge drop-out simulating rib / calcification shadows.
4. `motion_blur`: Transducer jitter and respiratory motion blur.
5. `off_axis`: Anatomical misalignment and landmark clipping.
6. `none`: Reference image.

---

## 5. Empirical Baseline Results

### View Classification (Module M1)
Comparison between classical spatial feature baseline and `CardiacEchoNet` (4-stage CNN, ~370k parameters):

| Metric | Classical (Spatial + Logistic) | CardiacEchoNet (CNN) |
| :--- | :---: | :---: |
| **Accuracy** | 0.497 | **0.503** |
| **Balanced Accuracy** | 0.497 | **0.503** |
| **Macro F1-Score** | **0.445** | 0.360 |
| **Expected Calibration Error (ECE)** | **0.138** | 0.207 |

### Quality Assessment (Module M3)
Continuous regression against expert completeness/clarity grades (0–9 scale):

| Metric | Naive Baseline (Mean $\bar{y} = 3.99$) | QualityAssessorNet |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | 1.910 | **1.738** |
| **Root Mean Squared Error (RMSE)** | 2.415 | **2.258** |
| **Pearson Correlation ($r$)** | 0.000 | **0.495** |
| **Quadratic Weighted Kappa (QWK)** | 0.000 | **0.254** |

---

## 6. Installation & Quick Start

### 1. Clone & Setup Environment
```powershell
git clone https://github.com/AniveshGuptaOfficial/Self_Check_Ultrasound.git
cd Self_Check_Ultrasound

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
python -m pytest -v tests/
```
*(Runs 21 automated unit tests verifying data loaders, splits, degradations, models, metrics, and security).*

### 3. Generate Manifest & Splits
```powershell
python scripts/build_manifest.py
```

### 4. Train Baselines
```powershell
# View recognition baseline (M1)
python scripts/train_view_baseline.py --max_train_samples 1200 --max_val_samples 300 --epochs 3

# Quality assessment baseline (M3)
python scripts/train_quality_baseline.py --max_train_samples 1200 --max_val_samples 300 --epochs 3
```

---

## 7. Security & Reproducibility Audit

- **Dependency Security:** Audited with `pip-audit`: **0 known vulnerabilities**.
- **Path Traversal Defense:** `CardiacUltrasoundDataset` strictly validates that all resolved file paths remain inside `dataset_root`. Malicious relative paths (`../../`) raise explicit `ValueError`.
- **Untrusted Deserialization:** State dicts are saved and loaded safely without unconstrained pickle executions.
- **Reproducibility:** Seed determinism (`seed=42`) verified across PyTorch, NumPy, and Python standard libraries.

---

## 8. Citation & Attribution

If you use this codebase or the CACTUS dataset in your research, please cite:

```bibtex
@article{DBLP:journals/cbm/ElmekkiASSZZBKXPMOSM25,
  author       = {Hanae Elmekki and Ahmed Alagha and Hani Sami and Amanda Spilkin and
                  Antonela Zanuttini and Ehsan Zakeri and Jamal Bentahar and Lyes Kadem and
                  Wen{-}Fang Xie and Philippe Pibarot and Rabeb Mizouni and Hadi Otrok and
                  Shakti Singh and Azzam Mourad},
  title        = {{CACTUS:} An open dataset and framework for automated Cardiac Assessment
                  and Classification of Ultrasound images using deep transfer learning},
  journal      = {Comput. Biol. Medicine},
  volume       = {190},
  pages        = {110003},
  year         = {2025},
  doi          = {10.1016/J.COMPBIOMED.2025.110003}
}
```

---

## 9. License

This repository is distributed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/) in accordance with the underlying CACTUS dataset distribution. See [CITATION.md](CITATION.md) for details.
