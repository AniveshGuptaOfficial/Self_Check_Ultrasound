# SelfCheckUltrasound: System Architecture Specification

## 1. Executive Summary & Design Philosophy

**SelfCheckUltrasound** is an open-source, image-quality-aware decision-support framework designed for cardiac ultrasound (echocardiography). The system addresses a fundamental challenge in point-of-care and automated echocardiography: downstream clinical models (e.g., automated view classification, anatomical segmentation, chamber quantification) frequently fail silently when presented with suboptimal acoustic scans.

Rather than treating image quality assessment in isolation or issuing vague rejection warnings, SelfCheckUltrasound provides an end-to-end, closed-loop pipeline organized into **eight modular subsystems (M1–M8)**. The architecture:
1. Validates and preprocesses incoming ultrasound frames within the active acoustic sector cone.
2. Estimates task-specific adequacy calibrated directly against downstream clinical analysis tolerances.
3. Diagnoses the specific physical root cause of acoustic degradation (e.g., acoustic shadowing, gain miscalibration, motion blur, probe off-axis misalignment).
4. Translates diagnosed physical causes into actionable reacquisition guidance for ultrasound operators.
5. Triages scans with strict confidence gating and attempt bounds into three guarded states: **[ACCEPT]**, **[REACQUIRE]**, or **[REFER]**.
6. Simulates closed-loop reacquisition without ground-truth label leakage.

```
       +-----------------------------------------------------------------------------------+
       |                               SelfCheckUltrasound                                 |
       |                            Closed-Loop Echo Pipeline                              |
       +-----------------------------------------------------------------------------------+
                                                 |
                        [Raw Full HD Frame: 1920x1080x3 Echocardiogram]
                                                 |
                                                 v
                       +---------------------------------------------------+
                       | M1. Preprocessing & View Recognition              |
                       | - Path & Dimension Verification (resolve_safe_path)|
                       | - Sector Cone Acoustic Masking (SectorExtractor)  |
                       | - Standardized Transforms (224x224, No Flip)      |
                       | - Screening View Classification (CardiacEchoNet)  |
                       +---------------------------------------------------+
                                        |                   |
                       (Spatial / Tensor Representations)   (Transducer Masked Sector)
                                        |                   |
                    +-------------------+                   +-------------------+
                    |                                                           |
                    v                                                           v
+---------------------------------------+                   +---------------------------------------+
| M3. Quality Assessment                |                   | M5. Cause Classifier                  |
| - QualityAssessorNet (Grade 0-9)      |                   | - 7 Sector-Masked Physics Features    |
| - Continuous Completeness & Clarity   |                   | - Random Forest Classifier            |
| - Standalone Physics Proxy Fallback   |                   | - 5 Acoustic Faults + 'none' (C_conf) |
+---------------------------------------+                   +---------------------------------------+
                    |                                                           |
                    v                                                           |
+---------------------------------------+                                       |
| M4. Adequacy Estimator                |                                       |
| - Features: Quality, Conf, Margin, H  |                                       |
| - Calibrated Logistic Regression      |                                       |
| - Predicts P(Adequate for Task M2)    |                                       |
+---------------------------------------+                                       |
                    |                                                           |
                    +---------------------------+-------------------------------+
                                                |
                                                v
                              +-----------------------------------+
                              | M7. Decision Policy Engine        |
                              | - Inputs: P(Adeq), Cause, C_conf, k|
                              | - Thresholds: tau_adeq=0.75,      |
                              |               tau_conf=0.60,      |
                              |               k_max=3             |
                              +-----------------------------------+
                                  /               |               \
        P(Adeq) >= 0.75          /    P(Adeq) < 0.75               \  C_conf < 0.60
                                /     & C_conf >= 0.60              \ or k > 3
                               /      & k <= 3                       \ or Ambiguous
                              v                   v                   v
+------------------------------------+  +--------------------+  +--------------------+
| M2. Downstream Task Model          |  | M6. Guidance Mapper|  | Safe Clinical      |
| - Full Anatomical Analysis         |  | - Targeted Physical|  | Abstention         |
| - Standard 6-View Identification   |  |   Maneuver Advice  |  | - Senior Referral  |
| - Downstream Clinical Task Success |  +--------------------+  | - Audit Logging    |
+------------------------------------+            |             +--------------------+
                                                  v                        |
                                        +--------------------+             |
                                        | M8. Re-Check Loop  |             v
                                        | - Closed-Loop Sim  |          [REFER]
                                        | - Attempt k = k + 1|
                                        | - Zero-Leakage Eval|
                                        +--------------------+
                                                  |
                                                  +---> (Iterative Reacquisition)
```

---

## 2. The 8-Module Architectural Matrix

| Module | Formal Responsibility | Underlying Model / Component | Primary Inputs | Primary Outputs | Safety Guardrail & Failure Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1. Preprocessing & View Recognition** | Validate images, enforce safety bounds, and identify supported cardiac views. | • `utils/paths.resolve_safe_path`<br/>• `data/transforms.py`<br/>• `models/view_classifier.py` (`CardiacEchoNet` / Classical Logistic) | Raw image file path, raw RGB frame ($1920 \times 1080 \times 3$) | Standardized tensor ($3 \times 224 \times 224$), sector mask, initial view probability vector $\mathbf{p}_{\text{view}} \in \Delta^5$ | • Path traversal rejection via `is_relative_to`<br/>• Dimension verification<br/>• Strict non-flipping policy preserving anatomical chamber orientation |
| **M2. Downstream Task Model** | Perform a defined, label-supported cardiac analysis task on accepted images. | • `models/view_classifier.py` (`CardiacEchoNet` 4-stage CNN, ~370k parameters) | Accepted echocardiogram tensor ($3 \times 224 \times 224$) meeting adequacy tolerance | Confident standard cardiac view class (`A4C`, `PL`, `PSAV`, `PSMV`, `SC`) and downstream analytical features | • Gated strictly behind M7 `[ACCEPT]` triage<br/>• M4 continuously monitors its anticipated error tolerance |
| **M3. Quality Assessment** | Estimate continuous image quality grade using expert annotations or proxy features. | • `models/quality_assessor.py` (`QualityAssessorNet` CNN regressor)<br/>• Fallback: `_proxy_quality` (physics features) | Normalized frame tensor ($3 \times 224 \times 224$) or 7 sector-masked physics features | Continuous scalar quality score $Q \in [0.0, 9.0]$ | • Output clamped strictly to $[0.0, 9.0]$<br/>• Fallback to deterministic physics proxy when PyTorch checkpoint is unavailable |
| **M4. Adequacy Estimator** | Predict downstream task error or whether task execution falls within defined clinical tolerance. | • `models/adequacy_estimator.py` (`AdequacyEstimator` logistic regression with `StandardScaler`) | Quality score $Q$, Top-1 confidence, Top-2 margin, predictive Shannon entropy $H$ | Calibrated task adequacy probability $P(\text{Adequate}) \in [0.0, 1.0]$ | • Fitted via sequence-group-aware out-of-fold `GroupKFold`<br/>• Fallback heuristic explicitly flagged as non-calibrated |
| **M5. Cause Classifier** | Identify supported image-quality root causes and report diagnosis confidence. | • `models/cause_classifier.py` (`CauseClassifier` Random Forest + rule fallbacks) | 7 sector-masked physics features (luminance, saturation, 90th percentile, sharpness, column drop-out, lateral min, radial coverage) | Predicted fault class $\hat{y} \in \{\text{under\_gain}, \text{over\_gain}, \text{shadow}, \text{motion\_blur}, \text{off\_axis}, \text{none}\}$ and confidence $C_{\text{fault}} \in [0.0, 1.0]$ | • Physics features computed strictly within active sector cone<br/>• Fallback to heuristic rules if scikit-learn model uninitialized |
| **M6. Guidance Mapper** | Map sufficiently confident, supported causes to candidate reacquisition suggestions. | • `guidance/reacquisition_policy.py` (`ReacquisitionPolicy.SUGGESTIONS`) | Diagnosed fault $\hat{y}$ from M5 | Actionable clinical transducer / machine maneuver string | • Triggered only if confidence $C_{\text{fault}} \ge \tau_{\text{conf}}$ (0.60)<br/>• Maps to verified sonographic adjustments |
| **M7. Decision Policy** | Choose accept, reacquire, or refer based on validated thresholds and confidence. | • `guidance/reacquisition_policy.py` (`ReacquisitionPolicy.evaluate`) | $P(\text{Adequate})$, diagnosed fault $\hat{y}$, confidence $C_{\text{fault}}$, attempt counter $k$ | `PolicyDecision` object: action $\in \{\text{ACCEPT}, \text{REACQUIRE}, \text{REFER}\}$, reason, suggestion | • Prioritizes acceptance check before attempt limit check<br/>• Enforces safe referral when attempts exceed $k_{\max} = 3$ or confidence is low |
| **M8. Re-Check Loop** | Evaluate successive simulated images, enforce attempt bounds, and stop safely. | • `guidance/simulator.py` (`ClosedLoopSimulator`) | Base image, initial fault, severity, operator recovery model ($\Delta s = 0.5$) | `SimulationResult` trajectory containing attempt history, per-step metrics, and final triage action | • Zero ground-truth leakage: `assess(image)` consumes pixels only<br/>• Strict attempt limit ($k \le 3$)<br/>• Immutable record of true initial severity for auditing |

---

## 3. End-to-End Decision Flow

```mermaid
flowchart TD
    %% Styling Definitions
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef model fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef policy fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef downstream fill:#e8eaf6,stroke:#303f9f,stroke-width:2px;
    classDef action fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef danger fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    %% S1: Ingestion & Preprocessing
    subgraph S1["1. Module M1: Preprocessing & View Recognition"]
        RAW["Raw Echocardiogram Frame<br/>(Full HD 1920x1080 RGB)"] --> VAL["Resolution & Path Check<br/>(resolve_safe_path)"]
        VAL --> SECTOR["Acoustic Sector Cone Extraction<br/>(Transducer Active Aperture Mask)"]
        VAL --> PREPROC["Standardized Transforms<br/>(Resize 224x224, Norm, No Flip)"]
        PREPROC --> M1_VIEW["View Screening & Recognition<br/>(CardiacEchoNet / Logistic Baseline)"]
        M1_VIEW --> VPROB["Candidate View Probabilities<br/>(p_A4C, p_PL, p_PSAV, p_PSMV, p_SC, p_Random)"]
    end

    %% S2: Image Quality Assessment
    subgraph S2["2. Module M3: Quality Assessment"]
        PREPROC --> M3["QualityAssessorNet Regressor<br/>(Continuous Grade 0 - 9)"]
        SECTOR --> PROXY_Q["Proxy Acoustic Quality Fallback<br/>(Sharpness, Saturation, Drop-out, Coverage)"]
        M3 --> QSCORE["Predicted Quality Grade Q<br/>(Completeness & Clarity Score)"]
        PROXY_Q -.->|Standalone Mode| QSCORE
    end

    %% S3: Task Adequacy Estimation
    subgraph S3["3. Module M4: Adequacy Estimator"]
        VPROB --> FEAT_ADEQ["Adequacy Feature Extraction<br/>(Quality Q, Top-1 Conf, Margin, Shannon Entropy)"]
        QSCORE --> FEAT_ADEQ
        FEAT_ADEQ --> M4["Adequacy Estimator<br/>(Calibrated Logistic Regression on OOF Targets)"]
        M4 --> APROB["Task Adequacy Probability P(Adequate)<br/>(Predicts if Downstream Task M2 will be reliable)"]
    end

    %% S4: Acoustic Cause Classification
    subgraph S4["4. Module M5: Cause Classifier"]
        SECTOR --> FEAT_PHYS["Sector-Masked Physics Feature Extractor<br/>(Luminance, Saturation, Column Drop-out, Radial Gradient)"]
        FEAT_PHYS --> M5["Cause Classifier<br/>(Random Forest / Calibrated Rule Fallbacks)"]
        M5 --> FAULT["Diagnosed Acoustic Fault & Confidence<br/>(under_gain, over_gain, shadow, motion_blur, off_axis, none)"]
    end

    %% S5: Decision Policy Engine
    subgraph S5["5. Module M7: Decision Policy Engine"]
        APROB --> M7{"Triage Policy Evaluation<br/>(tau_adeq = 0.75, tau_conf = 0.60, k_max = 3)"}
        FAULT --> M7
        ATTEMPT["Attempt Counter (k <= 3)"] --> M7

        M7 -->|P(Adequate) >= 0.75| ACT_ACCEPT["[ACCEPT]<br/>Adequacy Threshold Met"]
        M7 -->|P(Adequate) < 0.75 & Conf >= 0.60 & k <= 3| ACT_REACQUIRE["[REACQUIRE]<br/>Targeted Correction Required"]
        M7 -->|Conf < 0.60 or k > 3 or Unknown| ACT_REFER["[REFER]<br/>Safe Clinical Abstention / Referral"]
    end

    %% S6: Downstream Task Model
    subgraph S6["6. Module M2: Downstream Task Model"]
        ACT_ACCEPT --> M2["Downstream Cardiac Task Execution<br/>(Full CardiacEchoNet Diagnostic Analysis)"]
        M2 --> CLINICAL_OUT["Validated Cardiac View Output<br/>(Standard View Verified & Ready for Clinical Review)"]
    end

    %% S7: Guidance Mapper
    subgraph S7["7. Module M6: Guidance Mapper"]
        ACT_REACQUIRE --> M6["Reacquisition Guidance Engine<br/>(Maps Confident Physical Cause to Specific Maneuver)"]
        FAULT --> M6
        M6 --> SUGGEST["Actionable Operator Suggestion<br/>• under_gain: 'Increase 2D gain / TGC'<br/>• over_gain: 'Decrease 2D gain / resolve washout'<br/>• shadow: 'Reposition probe / adjust intercostal window'<br/>• motion_blur: 'Stabilize probe / suspend respiration'<br/>• off_axis: 'Re-align probe to orthogonal standard axis'"]
    end

    %% S8: Closed-Loop Re-Check Simulation
    subgraph S8["8. Module M8: Closed-Loop Re-Check Simulator"]
        SUGGEST --> M8["ClosedLoopSimulator Engine<br/>(k = k + 1, Enforce Attempt Bounds)"]
        M8 --> SIM_MANEUVER["Simulate Operator Adjustment<br/>(Controlled Severity Reduction / Persistence)"]
        SIM_MANEUVER -->|Successive Acquired Frame| RAW
    end

    %% Class Assignments
    class RAW,VAL,SECTOR,PREPROC input;
    class M1_VIEW,VPROB,M3,QSCORE,FEAT_ADEQ,M4,APROB,FEAT_PHYS,M5,FAULT,PROXY_Q model;
    class M7,ATTEMPT policy;
    class M2,CLINICAL_OUT downstream;
    class M6,SUGGEST policy;
    class M8,SIM_MANEUVER action;
    class ACT_ACCEPT action;
    class ACT_REACQUIRE policy;
    class ACT_REFER danger;
```

---

## 4. Module-by-Module Technical Deep-Dive

### 4.1 Module M1: Preprocessing & View Recognition
- **Code Anchor:** [`self_check_ultrasound/data/transforms.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/data/transforms.py), [`self_check_ultrasound/models/view_classifier.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/models/view_classifier.py)
- **Role:** The front gate of the system. Validates image integrity, verifies that file access remains bounded inside the approved dataset directory (`resolve_safe_path`), extracts the geometric transducer sector cone, resizes the input to $224 \times 224$, and normalizes with ImageNet statistics.
- **Anatomical Chamber Integrity:** Strictly excludes random horizontal flipping. In echocardiography, horizontal flipping mirrors anatomical left/right cardiac chambers, which would corrupt cardiac orientation.
- **View Recognition:** Generates screening softmax probabilities over the six CACTUS classes:
  $$\mathbf{p}_{\text{view}} = \text{Softmax}(\mathbf{z}_{\text{view}}) \in \Delta^5, \quad \text{Classes: } \{\text{A4C}, \text{PL}, \text{PSAV}, \text{PSMV}, \text{SC}, \text{Random}\}$$

### 4.2 Module M2: Downstream Task Model
- **Code Anchor:** [`self_check_ultrasound/models/view_classifier.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/models/view_classifier.py) (`CardiacEchoNet`)
- **Role:** The primary clinical target model whose execution and reliability SelfCheckUltrasound safeguards.
- **Specification:** A 4-stage convolutional neural network (~370k parameters) with Conv2D, BatchNorm, LeakyReLU, and AdaptiveAvgPool2d layers. It is trained to perform standardized cardiac view classification.
- **Gating Guarantee:** M2 is **never** invoked on unverified or low-adequacy frames. It is executed only after Module M7 confirms an **[ACCEPT]** decision, ensuring that downstream cardiac analysis occurs solely on acoustically adequate data.

### 4.3 Module M3: Quality Assessment
- **Code Anchor:** [`self_check_ultrasound/models/quality_assessor.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/models/quality_assessor.py) (`QualityAssessorNet`)
- **Role:** Produces a continuous, calibrated image quality estimate $Q \in [0.0, 9.0]$.
- **Training Target:** Ground-truth expert grades from the CACTUS dataset, which combine:
  1. *Completeness:* Visibility of standard anatomical structures (e.g., mitral leaflets, interventricular septum, aortic valve cusps).
  2. *Clarity:* Signal-to-noise ratio, speckle pattern sharpness, and boundary definition.
- **Architecture:** 4-stage CNN regressor trained with `MSELoss` and `AdamW` ($lr = 10^{-4}$).
- **Performance (TEST split):** $\text{MAE} = 2.168 \; [2.033, 2.322]$, $\text{RMSE} = 2.555 \; [2.409, 2.706]$, Pearson $r = 0.411 \; [0.332, 0.485]$, $\text{QWK} = 0.138$.

### 4.4 Module M4: Adequacy Estimator
- **Code Anchor:** [`self_check_ultrasound/models/adequacy_estimator.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/models/adequacy_estimator.py) (`AdequacyEstimator`)
- **Role:** Bridges the gap between generic visual quality and task-specific utility. Answers the critical question: *"Is this image good enough for downstream model M2 to succeed within tolerance?"*
- **Feature Vector:** Combines four orthogonal reliability signals:
  $$\mathbf{x}_{\text{adeq}} = \begin{bmatrix} Q \\ \max_j p_j \\ p_{(1)} - p_{(2)} \\ -\sum_j p_j \ln p_j \end{bmatrix} = \begin{bmatrix} \text{Predicted Quality Grade } (0–9) \\ \text{Top-1 View Confidence} \\ \text{Top-2 Probability Margin} \\ \text{Predictive Shannon Entropy } H(\mathbf{p}) \end{bmatrix}$$
- **Out-of-Fold Protocol:** To prevent in-sample overfitting, M4 is fitted on the validation partition using `GroupKFold` grouped by `sequence_group`.
- **Output:** $P(\text{Adequate} \mid \mathbf{x}_{\text{adeq}}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$, where $\mathbf{z} = \text{StandardScaler}(\mathbf{x}_{\text{adeq}})$.

### 4.5 Module M5: Cause Classifier
- **Code Anchor:** [`self_check_ultrasound/models/cause_classifier.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/models/cause_classifier.py) (`CauseClassifier`)
- **Role:** Identifies the physical acoustic fault responsible for image degradation within the ultrasound sector cone.
- **7 Sector-Masked Physics Features:**
  1. `mean_luminance`: Average acoustic intensity inside sector.
  2. `saturation_fraction`: Fraction of blown-out acoustic pixels ($I > 0.95$).
  3. `p90_luminance`: 90th percentile intensity.
  4. `inner_sharpness`: Normalized Laplacian variance within active sector.
  5. `column_dropout_max`: Maximum lateral shadow drop-out score across scan lines.
  6. `lateral_min_intensity`: Minimum intensity across azimuthal columns.
  7. `radial_coverage`: Fraction of active acoustic sector containing non-zero echoes.
- **Classes:** `under_gain`, `over_gain`, `acoustic_shadow`, `motion_blur`, `off_axis`, `none`.
- **Validation:** $81.0\%$ accuracy on held-out synthetic test faults; $\text{AUROC} = 0.701$ against real expert quality grades.

### 4.6 Module M6: Guidance Mapper
- **Code Anchor:** [`self_check_ultrasound/guidance/reacquisition_policy.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/guidance/reacquisition_policy.py)
- **Role:** Deterministic translation engine that maps confident physical diagnoses into standardized sonographic adjustments:
  - `under_gain` $\longrightarrow$ *"Increase 2D overall gain or time-gain compensation (TGC)."*
  - `over_gain` $\longrightarrow$ *"Decrease 2D overall gain to resolve acoustic washout."*
  - `acoustic_shadow` $\longrightarrow$ *"Reposition probe to avoid rib acoustic shadowing; adjust intercostal angle."*
  - `motion_blur` $\longrightarrow$ *"Stabilize probe and ask patient to suspend respiration momentarily."*
  - `off_axis` $\longrightarrow$ *"Re-align probe to orthogonal anatomical standard view axis."*
  - `none` $\longrightarrow$ *"No adjustment required."*

### 4.7 Module M7: Decision Policy Engine
- **Code Anchor:** [`self_check_ultrasound/guidance/reacquisition_policy.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/guidance/reacquisition_policy.py) (`ReacquisitionPolicy`)
- **Role:** Enforces clinically safe, three-way guarded triage based on validated operating thresholds:
  - Default parameters: $\tau_{\text{adeq}} = 0.75$, $\tau_{\text{conf}} = 0.60$, $k_{\max} = 3$.
- **Decision Hierarchy:**
  1. **[ACCEPT]:** Evaluated **first** so that a scan recovered on the final allowed attempt is correctly accepted:
     $$\text{Action} = \text{[ACCEPT]} \iff P(\text{Adequate}) \ge \tau_{\text{adeq}}$$
  2. **[REACQUIRE]:** Triggered when adequacy is substandard, but a known physical fault is confidently identified and attempts remain:
     $$\text{Action} = \text{[REACQUIRE]} \iff \left(P(\text{Adequate}) < \tau_{\text{adeq}}\right) \land \left(C_{\text{fault}} \ge \tau_{\text{conf}}\right) \land \left(k \le k_{\max}\right)$$
  3. **[REFER]:** Triggered when attempts are exhausted, the fault diagnosis is ambiguous, or the scan is uninterpretable:
     $$\text{Action} = \text{[REFER]} \iff \left(k > k_{\max}\right) \lor \left(C_{\text{fault}} < \tau_{\text{conf}}\right) \lor (\hat{y} = \text{"none"} \land P < \tau_{\text{adeq}})$$

### 4.8 Module M8: Closed-Loop Re-Check Simulator
- **Code Anchor:** [`self_check_ultrasound/guidance/simulator.py`](file:///c:/Users/Anivesh/OneDrive/Desktop/Special%20Project/Special%20Project/Self_Check_Ultrasound/self_check_ultrasound/guidance/simulator.py) (`ClosedLoopSimulator`)
- **Role:** Simulates the complete interaction trajectory over consecutive frames.
- **Zero-Leakage Invariant:** The assessment routine `assess(image)` accepts **only raw pixel data**. It has zero access to ground-truth fault labels, injection severities, or recovery targets.
- **Operator Response Model:** Models compliant sonographer actions by reducing fault severity by $\Delta s = 0.5$ per valid maneuver. A persistent fault (e.g., patient acoustic window limitation) is never reduced, verifying that M7 correctly escalates to `[REFER]`.

---

## 5. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Ultrasound Operator
    participant M1 as M1: Ingestion & View Screening
    participant M3 as M3: Quality Assessor
    participant M5 as M5: Cause Classifier
    participant M4 as M4: Adequacy Estimator
    participant M7 as M7: Decision Policy
    participant M2 as M2: Downstream Task Model
    participant M6 as M6: Guidance Mapper
    participant M8 as M8: Re-Check Simulator

    Operator->>M1: Acquire Echocardiogram Frame (1920x1080)
    Note over M1: Validate safety path, extract sector cone, normalize (224x224)
    M1->>M1: Compute View Probabilities p_view

    par Feature Extraction
        M1->>M3: Normalized Frame Tensor
        Note over M3: Predict Continuous Quality Grade Q in [0, 9]
        M1->>M5: Active Sector Pixels
        Note over M5: Extract 7 Physics Features; Classify Fault & Confidence C_conf
    end

    M3->>M4: Predicted Quality Q
    M1->>M4: View Probabilities p_view (Conf, Margin, Entropy)
    Note over M4: Compute P(Adequate) via Calibrated Logistic Regressor

    M4->>M7: Adequacy Probability P(Adequate)
    M5->>M7: Diagnosed Fault & Confidence C_conf
    M8->>M7: Current Attempt Counter k (k <= 3)

    alt P(Adequate) >= 0.75 [ACCEPT]
        M7->>M2: Frame Accepted -> Execute Downstream Cardiac Task
        Note over M2: Perform label-supported 6-view analysis on validated image
        M2-->>Operator: Clinical Diagnostic Output Ready
    else P(Adequate) < 0.75 AND C_conf >= 0.60 AND k <= 3 [REACQUIRE]
        M7->>M6: Trigger Guidance for Diagnosed Fault
        M6-->>Operator: Actionable Maneuver (e.g., 'Increase 2D gain / TGC')
        M7->>M8: Update Trajectory; Increment Attempt k = k + 1
        M8->>Operator: Prompt Reacquisition Frame
    else C_conf < 0.60 OR k > 3 OR Ambiguous [REFER]
        M7-->>Operator: Safe Clinical Abstention; Refer to Senior Sonographer
    end
```

---

## 6. Verification and Regression Testing

All 8 modules are continuously verified across a 56-test automated test suite:
- **Test Command:** `python -m pytest -v tests/`
- **Result:** **56 passed in ~22.8 seconds (100% pass rate)**.
- **Key Modules Tested:**
  - `tests/test_manifest.py`: Filename regex parsing, Full HD validation, zero path traversal.
  - `tests/test_splits.py`: Sequence-group isolation, empty pairwise intersections, full view coverage.
  - `tests/test_models.py`: Shape and gradient propagation for `CardiacEchoNet` (M1/M2) and `QualityAssessorNet` (M3).
  - `tests/test_guidance_simulation.py`: Out-of-fold calibration (M4), sector masking (M5), policy triage logic (M6/M7), zero-leakage assessment (M8).
  - `tests/test_cause_classifier.py`: Real-data AUROC and held-out synthetic accuracy.
