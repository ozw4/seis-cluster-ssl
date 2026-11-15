# Preparation and execution evidence

Repository HEAD at start: `3e3ff5b023080166064fdee96cd2c7d7e229cb4e`.
Existing uncommitted MAE sensitivity changes and reports were retained.
Only the root AGENTS.md applies. Baseline definitions/checkpoints/results and
canonical selections are unchanged.

- The fixed manifest binds three new Random zero-weight runs and three reused
  0.2 sources. All 12 new YAMLs differ from their own Random baseline only in
  distillation, output/model identities and fixed diagnostics. The legacy MAE
  four-coefficient defaults remain unchanged.
- All six actual full/smoke dry-runs passed.
- Three one-step GPU smokes passed, excluded from full-experiment counts.
  Actual initial student/head hashes match the historical Random 0.2
  run-metadata hashes in all three surveys. Weighted distillation is exactly
  zero and weighted prototype loss is positive. Smoke validation and the
  top-block trainability summaries are saved in `smoke_validation.json`.
- Related study, common-test and F3 configuration/lineage/publication tests:
  **81 passed in 6.49 seconds**. They cover legacy four-weight and new two-weight
  complete grids, signed paired differences, separate validation, fixed Random
  test namespace/reference mask, inherited definitions, and exact seven-file
  summary/Random report publication sets. Existing teacher/gradient/RNG tests
  are retained. No broad experiment audit was added.
- Ruff lint/format and ShellCheck passed for their checked changes. The two
  idle H100 NVL GPUs were used without stopping unrelated work. The detached
  run log is `run_all.log`; `launch.json` records the original PID, UTC time
  and GPU assignments.

Operational evidence lives under
`operations/hmm_v2/random_distillation_ablation_v1/` in the artifact root:
`preflight.json`, `dry_run_*.log`, `smoke_*.log`, `smoke_validation.json`,
`final_tests.log`, `launch.json`, `run_all.log`, `execution/events.jsonl`.

## Completed execution

The final `analysis_complete.json` receipt is dated **2026-11-11 11:26:22 UTC**.
All three new trainings completed 25 epochs, followed by three extractions,
all 15 medium cells and all 30 small/large cells. The three 0.2 baseline models
and all 45 baseline decoder cells were reused without retraining. No source or
coefficient was filtered from the prespecified comparison.

The common-test evaluator scored all 30 F3 prediction cells; the 15 reused 0.2
Mean IoU/macro F1 values match the original common-test results exactly. The
summary includes 90 test cells, 18 coefficient/budget groups, 30 separate F3
validation values and 156 epoch diagnostic rows. The nine-panel figure was
visually checked for both coefficients, dataset/budget labels, metric units
and sample-SD error bars. All seven curated files are saved in the
[completed report](../../../reports/hmm_v2_random_distillation_ablation_v1/README.md).

Zero distillation improved all nine survey/budget means and 42 of 45 individual
layout values. This is descriptive evidence from continuation seed 42, not
encoder-seed replication. Intermediate fixed-crop diagnostics for reused 0.2
remain unavailable because intermediate checkpoints were not retained; initial
and final values were recovered and original per-epoch training-crop curves
remain available. No additional retrospective experiment audit, baseline
retraining or canonical selection change was performed.
