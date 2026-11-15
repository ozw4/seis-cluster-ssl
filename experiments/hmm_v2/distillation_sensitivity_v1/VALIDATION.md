# Implementation and execution evidence

Repository: `ozw4/seis-cluster-ssl`. Starting HEAD:
`3e3ff5b023080166064fdee96cd2c7d7e229cb4e`. The worktree was clean at the
initial inspection. Only the root `AGENTS.md` applies to the changed paths.
Existing canonical selections, baseline definitions/checkpoints/metrics and
frozen receipts are retained.

## Implemented change

The explicit `train.feature_diagnostics_samples` option enables fixed-crop
measurement on the unmasked, fixed hard-target multi-head route. In this route
it also permits zero distillation while keeping the configured trainable
encoder blocks. The legacy positive-distillation guard remains active without
the option, including v1. Negative/nonfinite coefficients are rejected.

Student and teacher construction order is unchanged for positive coefficients.
The opted-in zero condition also constructs the frozen teacher before heads,
preserving RNG consumption. Hard multi-head losses calculate the raw cosine
loss whenever a teacher is supplied; multiplying by weight zero contributes
exactly zero gradient. Evaluation caches epoch-zero crops, restores Python,
NumPy and Torch CPU/CUDA RNG states and module train/eval flags, and never
iterates the training dataloader. No teacher parameter is updated.

The existing training runner, rolling checkpoints, extraction runner, survey
source/cell auditors, decoder runners and explicit F3 common-test evaluator
remain the execution paths. The added study modules bind this fixed experiment
and aggregate its results; they do not introduce a generic experiment platform.

## Completed preparation checks

- The matrix validator verifies SHA-256 of all definitions and exact inherited
  training/extraction/decoder values. It checks each separate one-step smoke
  recipe against its full recipe.
- 18 live dry-runs passed: full and smoke for every new condition.
- Nine actual GPU one-step smokes passed on GPU1. They are excluded from counts
  of completed 25-epoch training.
- For all three zero-weight GPU smokes, a tensor comparison against MAE100 found
  updates only in `encoder.layers.7.*`; prototype loss contribution was positive
  and weighted distillation was exactly zero.
- Actual initial student/head/teacher hashes and post-initialization Torch RNG
  hashes match across all four coefficients for every survey. The first two
  batches from the original eight-worker dataloaders match by tensor hash,
  including crops, labels, confidence, boundary weights and validity masks.
- All three current initial student/head states also match the **historical**
  0.2 execution's `run_metadata.json` hashes.
- Baseline preflight validates the three completed 0.2 source checkpoints,
  original MAE100/target bindings, and all 45 completed downstream cells.
- Existing v1 recovery reconstructed and matched 405 frozen cells, 27
  checkpoints and 16 source aggregates against the report manifest.
- Auxiliary recovery completed 135 v1 distillation pairs and 270 v2 consistency
  pairs (K6810 and K468 separately), with no missing files. Historical path
  aliases were followed through the original aggregate configurations.
- Baseline fixed-crop distance at epoch 25: F3 `0.024094506638147026`, Parihaka
  `0.02099792205808626`, Volve `0.01421729966023122`. Only epochs 0 and 25 can be
  recovered for the existing baseline; its intermediate fixed-crop checkpoints
  were not retained. New conditions record every completed epoch.

## Software validation

- Existing config/distillation/shared-prefix focused selection: **90 passed,
  1 skipped**.
- Existing head-only/config/common-test focused selection: **64 passed** before
  the final added study tests.
- Study and F3 inventory/lineage contracts: **54 passed**.
- Initial portable suite: **6754 passed, 2 failed**, 145 deselected. Both failures
  were the explicit F3 experiment inventory/lineage registration for the new
  experiment. The new 12-YAML F3 namespace is now registered under the dedicated
  sensitivity contract suite; both affected tests pass. No existing namespace
  rule was relaxed.
- Ruff lint/format, ShellCheck, package-isolation check and Git whitespace checks
  passed for their checked scopes. The final portable suite passed **6761 tests**, 145 deselected, 11
  warnings in 575.36 seconds. The subsequently added all-medium gate test is
  included in the **13 passed** study-focused run. Final report verification
  is appended after full execution and curation.

Operational logs and machine-specific settings are under
`operations/hmm_v2/distillation_sensitivity_v1/` in the artifact root. Key files:
`preflight.json`, `initial_alignment.json`, `historical_initial_alignment.json`,
`gpu_zero_update_audit.json`, `dry_run.log`, `smoke_driver.log`,
`v1_source_check.log`, `contracts_tests.log`, and `execution/events.jsonl`.
The latter records commands, UTC start/end, exit status and any resumed
checkpoint's SHA-256. Training/extraction/decoder progress remains in each
condition's canonical artifact output directory.

## GPU scheduling record

GPU0 was initially busy (~69 GB, 100% utilization), so the first two full F3
conditions used GPU1. At 06:22 UTC GPU0 was observed idle with ~91 GB free; F3
0.4 and Parihaka 0.0 then started there. Neither initial training was stopped.
Only the task-owned waiting queue manager was replaced to use two lanes per GPU;
`device_assignments.json` and `orchestration_events.jsonl` retain this change.
The fixed training/medium/remaining all-condition barriers remain unchanged.

## Decoder CPU scheduling verification

The real-data three-step probe retained identical initial and final decoder
state hashes and all metrics for 64, 1 and 4 CPU threads. The 64-thread execution
took 1.278 seconds and the 1-thread execution 0.051 seconds in that probe
(including their observed startup/cache effects). The runtime now caps decoder
intra-op CPU threads at 1 without changing any declared scientific setting.
Four medium cells resumed at their own checkpoints: F3 0/0.1/0.4 at global steps
2200/1760/2640, and Parihaka 0 at step 1344. The 17 stopped descendants belonged
only to this task-owned orchestration tree; existing external jobs were retained.
No encoder training was interrupted. The final study-focused tests passed
**14 tests**; the teacher/shared-prefix selection passed **82 tests, 1 skipped**.

## Completed execution and final analysis

- All nine independent MAE100 continuations completed 25 epochs, followed by
  nine feature extractions. The smoke runs are excluded from these counts.
- All 45 medium decoder cells completed before the 90 small/large cells. The
  final experiment completion record is dated **2026-11-06 11:26:54 UTC**:
  135 new decoder cells, plus the three existing 0.2 models and their 45 cells.
  No coefficient/dataset was removed after medium evaluation.
- Volve's existing downstream source bridge had assumed distillation 0.2.
  It now reads the finite nonnegative coefficient from the declared recipe,
  while retaining the exact historical 0.2 expected-metadata contract. All
  **55 adapter/recipe tests** and the actual zero-weight Volve dry-run passed.
- After the original medium orchestration stopped at 36/45 cells, recovery
  resumed the two partial Parihaka cells from their own checkpoints and filled
  the remaining nine medium cells, then all small/large cells. Complete cells
  were skipped. `finish_experiments.log` and the command event records retain
  the history; no completed encoder or decoder was retrained.
- On **2026-11-11**, the existing common-test evaluator scored all **60 F3
  cells** from validation-selected saved predictions. The reused 0.2 results
  match all 15 original common-test Mean IoU/macro F1 values exactly. Test mask:
  `2bac34f4ebf8612d120673ae3dbda1a8720624e27a1701491059abf1c60cfe1d`,
  113,979,624 voxels. No decoder selection or fitting used test scores.
- Final aggregation reused recorded preflight/input evidence and normal runner
  outputs, without additional retrospective experiment audits, as requested.
  The focused summary regression checks this behavior, paired improvement
  signs and the separation of 60 F3 validation values from test scores.
  The final study-focused run passed **15 tests in 5.03 seconds**; Ruff lint
  and format checks passed for the changed aggregation/test files.
- The primary producer saved all 180 cell values, 36 budget/coefficient
  aggregates, 312 epoch diagnostic rows, 60 separate F3 validation values,
  JSON/Markdown summaries and the 3×3 sensitivity figure. The figure was
  visually checked for dataset/budget labels, coefficient zero, metric units
  and sample-SD error bars. Supplemental recovery has 135 v1 distillation pairs
  and 270 v2 consistency pairs (K6810/K468 separate), with no missing files.
- The [curated report](../../../reports/hmm_v2_distillation_sensitivity_v1/README.md)
  contains exactly the ten producer-owned files covered by the publication
  regression. It contains no raw data, checkpoint, embedding, prediction volume
  or machine-specific path. No canonical coefficient/checkpoint was changed.

Only the reused 0.2 models' intermediate fixed-crop diagnostics remain
unavailable: intermediate checkpoints were not retained. Initial/final values
were recovered, while all 25 original training-crop diagnostic rows remain
available. This does not require another baseline training run.
