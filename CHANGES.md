# Changes in this patch (v0.2.0)

Files here are drop-in replacements/additions at the same paths. Unchanged files (view_classifier.py, quality_assessor.py,
metrics.py, seed.py, plots.py, test_degradations/models/metrics/visualization, generate_research_plots.py, interactive_demo.py,
utils/__init__.py, models/__init__.py) are NOT included.

## One required edit to an unchanged file
`self_check_ultrasound/models/__init__.py` and `utils/__init__.py` need no change. `data/__init__.py` IS included (exports `check_split_coverage`).

## README edits to make (README.md not rewritten here)
1. Test badge/text: replace "21 passing" with the number printed by `pytest` after your run (expected 56 collected).
2. Tree: `configs/default_config.py` -> `self_check_ultrasound/config.py`; add `guidance/`, `visualization/`, `utils/paths.py`, `utils/eval_utils.py`,
   and scripts `run_simulation.py`, `interactive_demo.py`, `generate_research_plots.py`, `train_adequacy.py`, `evaluate_cause_classifier.py`.
3. Section 5 tables: replace with the TEST-split values (with CIs) printed by the re-run training scripts; delete the old numbers.
4. Section 4.1: partition percentages are whatever `build_manifest.py` prints; mention the coverage check.
5. Section 4.2: say the degradations are image-space proxies.
6. Section 6: run `python -m pytest` AFTER `build_manifest.py`; add `--validate`.
7. Section 7: "Safe deserialization" now true (`weights_only=True` in loaders); "Dependencies": requirements are lower bounds, not pinned.
8. License: consider MIT/Apache-2.0 for the code and keep CC BY 4.0 attribution for the dataset only.

## Behaviour changes to expect after re-running
- Baseline numbers change (flip augmentation removed; test split now reported).
- Simulation outcomes change (no ground-truth leakage): more attempts or referrals are possible and are reported as observed.
