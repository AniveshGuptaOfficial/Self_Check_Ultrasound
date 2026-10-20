# Clinical Reader & Novice Operator Study Protocol

**Protocol Title:** Evaluation of Real-Time Acoustic Feedback and Reacquisition Guidance on Novice Point-of-Care Echocardiographic Quality: A Randomized Crossover Trial  
**Document Identifier:** `SCU-CLIN-PROT-2026-v1.0`  
**Software Under Evaluation:** SelfCheckUltrasound (Version 1.0)  
**Study Type:** Human-Factors Diagnostic Image Quality Trial (In Silico to Bedside Translation)  

---

## 1. Executive Summary & Clinical Rationale

Point-of-Care Ultrasound (POCUS) has transformed acute, critical, and emergency cardiac assessments. However, the diagnostic utility of echocardiography remains strongly operator-dependent. Non-expert and novice clinicians (e.g., internal medicine residents, emergency physicians, medical students) frequently experience high rates of non-diagnostic, incomplete, or severely artifact-corrupted scans ($25\%–45\%$ uninterpretable rate in unguided point-of-care environments).

Existing AI algorithms predominantly classify views or segment chambers retrospectively on post-acquisition videos. **SelfCheckUltrasound** introduces an active, closed-loop decision framework during the scan itself:
1. Estimating task-specific diagnostic adequacy ($P(\text{Adequate}) \in [0.0, 1.0]$) before downstream measurement.
2. Diagnosing sector-masked physical fault mechanisms (`under_gain`, `over_gain`, `acoustic_shadow`, `motion_blur`, `off_axis`).
3. Formulating actionable, bounded probe repositioning guidance (`[REACQUIRE]`) or safe clinical escalation (`[REFER]`).

This protocol establishes the standardized, prospective clinical trial methodology to validate whether novice operators achieve superior diagnostic yields, lower acquisition latency, and reduced cognitive workload when assisted by SelfCheckUltrasound.

---

## 2. Study Objectives & Primary Hypotheses

### 2.1 Hypotheses
* **Primary Hypothesis ($H_1$):** Novice operators assisted by real-time SelfCheckUltrasound guidance achieve a significantly higher proportion of diagnostic-quality standard echocardiograms ($\text{Expert Quality Grade} \ge 7.0/9.0$) compared to unguided novice operators ($p < 0.05$).
* **Secondary Hypothesis ($H_2$):** Real-time acoustic fault triage decreases the average time required to acquire an interpretable cardiac window by $\ge 25\%$.
* **Secondary Hypothesis ($H_3$):** Guidance feedback reduces operator cognitive burden as measured by the NASA Task Load Index (NASA-TLX).

---

## 3. Study Design & Randomization Scheme

### 3.1 Trial Architecture
The study is structured as a **prospective, randomized, two-arm crossover trial** with an independent expert benchmark reference group:

```
                                [ Enrolled Novices (N = 24) ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
            [ Group A (n = 12) ]                            [ Group B (n = 12) ]
                      │                                               │
             Period 1: GUIDED                                Period 1: UNGUIDED
          (SelfCheckUltrasound)                           (Standard POCUS reference)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                 [ 48-Hour Washout Period ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             Period 2: UNGUIDED                              Period 2: GUIDED
          (Standard POCUS reference)                      (SelfCheckUltrasound)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                         [ Blinded Dual Core Lab Review ]
```

### 3.2 Benchmark Reference Cohort
* **Expert Sonographers ($N = 4$):** Certified cardiovascular sonographers (RDCS) each scan the identical target views under nominal conditions to establish upper-bound technical ceiling benchmarks for quality, duration, and anatomical completeness.

---

## 4. Participant Eligibility & Selection Criteria

### 4.1 Novice Operator Cohort ($N = 24$)
* **Inclusion Criteria:**
  1. Medical students (Years 3–4), internal medicine residents (PGY-1/2), or emergency medicine residents.
  2. Completion of introductory cardiovascular anatomy and basic ultrasound physics coursework.
  3. Less than $5$ lifetime hours of clinical echocardiographic scanning experience.
* **Exclusion Criteria:**
  1. Prior formal ultrasound fellowship or certified RDCS sonography credentials.
  2. Uncorrected visual impairment precluding standard monitor visibility.

---

## 5. Ultrasound Equipment & Scanning Setting

* **Ultrasound Hardware:** Standard cart-based or laptop point-of-care ultrasound system equipped with a **phased-array cardiac probe** (2.0–4.0 MHz) operating in standard cardiac presets (tissue harmonic imaging enabled, default depth 16 cm).
* **Scanning Subjects:**
  * **Phase 1 (Controlled Hardware):** CAE Blue Phantom Heart Ultrasound Training Phantom with physiological chamber morphology and pericardial acoustic windows.
  * **Phase 2 (Physiological Volunteers):** Healthy adult volunteers ($BMI \in [18.5, 27.5]\text{ kg/m}^2$, sinus rhythm) with informed consent.

---

## 6. Target Standard Cardiac Views

Each operator must acquire five standard transthoracic echocardiographic (TTE) acoustic windows per subject:
1. **PLAX:** Parasternal Long Axis
2. **PSAX-AV:** Parasternal Short Axis at Aortic Valve Level
3. **PSAX-MV:** Parasternal Short Axis at Mitral Valve Level
4. **A4C:** Apical Four Chamber
5. **SC:** Subcostal Four Chamber

---

## 7. Experimental Procedure & Task Flow

1. **Orientation (15 minutes):** 
   * Brief tutorial reviewing the 5 standard views and probe handling principles.
   * Hands-on familiarity scan (5 minutes) on a non-study phantom to adjust probe grip and orientation marker.
2. **Acquisition Trials:**
   * In the **Unguided Arm**, operators acquire each view with standard static anatomical cards (showing ideal chamber orientation). Operators press a foot switch to "Freeze and Accept" when they believe an optimal view is achieved.
   * In the **Guided Arm**, operators view the SelfCheckUltrasound real-time HUD:
     * Task Adequacy Meter ($P(\text{Adequate}) \ge 0.75$).
     * Fault notification (e.g., `off_axis: conf 0.88`).
     * Guidance prompt (e.g., *"Rotate probe clockwise to align apical septum"* or *"Increase overall 2D gain"*).
     * Hard stop policy: Maximum 3 reacquisition maneuvers per view.
3. **Time Limit:** Maximum 180 seconds allowed per view attempt. If no acceptable view is saved within 180 seconds, the trial is logged as "Timeout / Incomplete".

---

## 8. Outcome Measures & Endpoints

### 8.1 Primary Endpoint
* **Diagnostic Adequacy Rate (%):** Proportion of acquired views rated $\ge 7.0$ on the 0–9 CACTUS expert clarity and completeness scale by blinded consensus review.

### 8.2 Secondary Endpoints
1. **Time-to-Adequacy (seconds):** Elapsed time from initial skin contact to final saved frame.
2. **Acoustic Fault Elimination Rate (%):** Proportion of initial acoustic artifacts successfully cleared following system guidance cues.
3. **Reacquisition Maneuver Count:** Number of physical probe re-adjustments per view.
4. **Referral Appropriateness (%):** True positive rate of the system correctly triggering `[REFER]` on genuinely uninterpretable acoustic windows.
5. **Cognitive Workload:** NASA-TLX survey (Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration; scale 0–100).
6. **Usability Score:** System Usability Scale (SUS) questionnaire completed post-trial.

---

## 9. Blinded Core Laboratory Review

* **Panel:** Two board-certified cardiologists with Level 3 Echocardiography certification.
* **Blinding:** All identifying metadata, guidance overlays, arm allocation, and operator names are stripped. Frames are presented in randomized, scrambled order.
* **Scoring Rubric (CACTUS 0–9 Standard):**
  * **Completeness (0–4.5):** Clear demarcation of key anatomical structures (valvular leaflets, ventricular apex, interventricular septum, pericardium).
  * **Clarity (0–4.5):** Endocardial border delineation, myocardial speckle texture, absence of shadow blooming or motion blur.
* **Inter-Rater Agreement:** Evaluated via quadratic weighted Cohen's Kappa ($\kappa$). Scores differing by $>1.5$ points trigger blinded third-reviewer adjudication.

---

## 10. Statistical Analysis Plan (SAP)

1. **Power Calculation:**
   * Based on published POCUS pilot data: unguided novice diagnostic yield $\approx 55\%$.
   * To detect an increase to $\ge 75\%$ in the guided arm with $\alpha = 0.05$ (two-tailed) and statistical power $1 - \beta = 0.85$, a minimum sample size of **$N = 220$ acquisition trials** ($22$ operators $\times 10$ views) is required. With an anticipated $10\%$ drop-out / incomplete rate, **$N = 24$ operators ($240$ trials)** are enrolled.
2. **Hypothesis Testing:**
   * Primary binary adequacy rate: McNemar's test for paired binary proportions across crossover arms.
   * Continuous endpoints (Quality grade, Time-to-Adequacy): Linear Mixed-Effects Model (LMM) with operator as random intercept, and intervention arm, view class, and scanning period as fixed effects.
   * NASA-TLX Workload: Wilcoxon signed-rank test for paired non-parametric differences.
   * Significance threshold: $\alpha = 0.05$, two-sided.

---

## 11. Ethical Considerations & Data Safety

* **Institutional Review Board (IRB):** Conducted under clinical investigation protocols compliant with Declaration of Helsinki principles.
* **Acoustic Safety:** Mechanical Index (MI) and Thermal Index (TI) maintained strictly $< 1.0$ at all times in adherence to the ALARA (As Low As Reasonably Achievable) FDA diagnostic ultrasound guidelines.
* **Data Security & Privacy:** No personal health information (PHI) is acquired; video clips are cryptographically hashed (SHA-256) and stored on secure local encrypted storage.
