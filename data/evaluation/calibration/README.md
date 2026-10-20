# Calibration evidence

These files informed a revised detector configuration. They are **not heldout validation of the revised configuration**.

- Original 24 primary and 10 routine runs supplied confidence diagnostics (see `../calibration_diagnostics.json`). The original outputs and metrics remain unchanged.
- `development-*` records the 512/853 resize probe on the same 12 original development frames. At confidence 0.60, all 12 person and 12 forklift boxes matched annotations at IoU ≥ 0.50.
- `routine-resize-*` records 12 previously evaluated frames from R01, R05, R07 and R09. The faster resize reintroduced shelf-as-forklift errors at 0.60. The original end-of-script source-hash assertion failed because an optional processor-size setting was added to the shared vision source while this script was running; the pre-run hash matches the archived baseline source. Predictions were already saved and were not rerun.
- `posthoc-065.json` filters those saved predictions at 0.65. It removes the observed shelf errors and retains the original development matches and R01 parked forklift. It also removes R09's person at frame 299. This small calibration sample cannot establish generalization.

The exact original predictor is preserved in `../baseline-source/`. New prospective evaluation results are reported separately from these calibration observations.

`routine-resize-summary.json` received a wording-only edit on 2026-10-20 after the phase-2 freeze; its original SHA-256 is the one recorded in `phase2/freeze.json`.
