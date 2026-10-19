# SelfCheckUltrasound: research plan (cardiac framing, draft v1)

Status words: DONE = exists in the repo and tested; BUILT = code exists but not yet run on real data; OPEN = not started.
Everything below that sounds like a result is a plan, not a finding.

## 1. Title and scope
**SelfCheckUltrasound: Image Quality-Aware Diagnosis and Reacquisition Guidance** (working title).
Scope: transthoracic echocardiography. Development data are the CACTUS phantom frames. Because the phantom has no disease,
the "downstream task" is **standard-view recognition**, and "diagnostic adequacy" means *predicted reliability of that task*.
Clinical wording ("diagnosis") is reserved for work on patient data.

## 2. Datasets
| Role | Dataset | What it gives | Access (verified from the project pages) | Caveats |
|---|---|---|---|---|
| Development | CACTUS | 37,736 phantom frames, 6 classes, expert grade 0-9 | open | phantom, one scanner |
| External A (views) | TMED-2 | patient frames, labels PLAX / PSAX / A2C / A4C / Other, 112x112 PNG | apply on tmed.cs.tufts.edu, data-use agreement, non-commercial | no quality grade; PSAX level not given; A2C and Other have no CACTUS class |
| External B (quality) | CAMUS | 500 patients, A2C and A4C, expert quality good / medium / poor (about 19% poor), GE Vivid E95 | download from the CREATIS page, must cite Leclerc et al. TMI 2019 | only two views; .mhd format needs conversion |
| External C (all views + quality) | CAMEO | 100 subjects, 9 views (A2C, A3C, A4C, PLAX, PSAX aortic/mitral/papillary, S4C), per-view quality field, DICOM cine + ED/ES PNG | IEEE DataPort, subscription required | quality scale and folder spellings NOT verified; read the descriptor PDF after download |

Label bridge (implemented in `utils/external.py`): A4C->A4C, PLAX->PL, PSAX-AV->PSAV, PSAX-MV->PSMV, S4C->SC; TMED-2 coarse
PSAX counts as correct for PSAV or PSMV. Views with no counterpart (A2C, A3C, papillary, Other) are "unseen views": the policy
should reject them, and `accept_rate_unseen_views` measures that.

## 3. Prior art that bounds the novelty
- Jeong et al., JBHI 2026: one network does view classification, segmentation and 7-way maneuver prediction on echo, with
  entropy-based structure-level quality scoring. **Entropy quality and maneuver guidance are therefore not claims.**
- Olivier et al., ISBI 2023: behaviour-cloned rotation guidance for A4C/PLAX on a phantom.
- Ma 2025, Lei 2025: feedback-driven probe posture in robotic lung ultrasound. Jiang 2024 (DopUS-Net): bounded
  re-identification loop (at most five 5-degree steps) triggered by Doppler/segmentation disagreement.
- Raina 2023-2024: label-free and supervised quality assessment, quality-driven robot optimisation.
- Lo and Lai 2024: classification plus retrieval to suggest the intended standard plane (knee).

## 4. Gaps (rewritten for cardiac; findings from the reviewed set, not a systematic review)
- RG1. Quality scoring and probe guidance now appear together (Jeong), but no decision policy that chooses accept, reacquire or refer was found.
- RG2. Quality is validated against standard vs non-standard views or expert grades, rarely against the error of the task it gates.
- RG3. Cause feedback is limited to probe maneuvers (tilt, rock, rotate). Acoustic and contact causes (gain, shadowing, motion) are not covered in the reviewed set.
- RG4. Closed-loop re-check evidence with a stopping rule exists for vessels (DopUS-Net) and for lung probe posture, but a cause-aware loop for echo was not found.
- RG5 (validation, not novelty). Public echo sets are single-site and differ in views, labels and scanners; phantom-to-patient transfer is untested.

## 5. Objectives and status
| | Objective | Status |
|---|---|---|
| O1 | Three-way accept / reacquire / refer policy with attempt limit | DONE (policy, tests) |
| O2 | Adequacy = predicted reliability of view recognition, fitted on held-out outcomes | BUILT (`train_adequacy.py`), not run |
| O3a | Cause tier 1: misalignment causes from controlled data | OPEN (current off_axis is a crop) |
| O3b | Cause tier 2: acoustic/contact causes from a small expert-annotated set of real frames; abstain when confidence is low | abstention DONE; annotated set OPEN |
| O4 | Closed-loop re-check, external validation, calibration reporting, multi-reader assessment (Komatsu recommendations) | re-check simulated; external tooling BUILT (`external_validation.py`); reader study OPEN |

## 6. Evaluation plan
1. Internal (CACTUS test split): view accuracy and macro-F1, quality MAE/QWK, adequacy AUROC, Brier, ECE, all with bootstrap CIs.
2. External, zero-shot, no tuning: run `external_validation.py` on TMED-2, CAMUS, CAMEO. Report view accuracy with CI on mapped classes,
   adequacy AUROC for "view predicted correctly", accept rate on unseen views, and for CAMUS/CAMEO Spearman and AUROC of
   predicted quality against expert grade. Expect a large drop (phantom to patients); report it.
3. Ablation: policy vs quality-only gate vs confidence-only gate, on selective accuracy at matched accept rate.
4. Cause classifier: held-out synthetic accuracy (labelled as such) plus AUROC of P(clean) against real expert grades (CAMUS/CAMEO).
5. Closed loop: simulated recovery, always labelled simulated. A real novice or reader study needs a clinical collaborator.
6. Report calibration and failure cases, not only means.

## 7. Novelty statement (draft, to be finalised after results)
A task-reliability-gated, cause-aware accept / reacquire / refer policy with a bounded re-check loop for echocardiography,
evaluated for calibration and external transfer. Not claimed: entropy quality, abstention, uncertainty estimation, maneuver prediction.

## 8. Publication plan
- Q2-style outcome (achievable with current scope): policy + calibration + external zero-shot transfer + honest simulated loop.
- Q1-style outcome (needs more): real expert-annotated acoustic-cause set, at least one external dataset showing useful
  selective accuracy, and a small reader or novice evaluation. Check the target journal's current quartile and scope before committing.
- Milestones: (1) apply patch, rerun baselines; (2) fit adequacy; (3) get TMED-2 and CAMUS access; (4) external runs;
  (5) annotated real-frame subset; (6) ablations; (7) draft; (8) literature swap so the set is cardiac-focused and not dominated by one group.

## 9. Risks
Phantom-to-patient shift may make zero-shot numbers poor; CAMEO quality scale unknown; TMED-2 images are 112x112 (resolution mismatch);
the view classifier is weak, which caps adequacy; annotation needs an expert.
