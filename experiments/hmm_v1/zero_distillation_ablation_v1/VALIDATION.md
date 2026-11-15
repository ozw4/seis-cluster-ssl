# Preparation and execution evidence

Prepared and launched on 2026-11-11. Existing uncommitted v2 sensitivity work
and reports were retained. Frozen v1 definitions and result records are
unchanged. All nine continuations, nine extractions, 135 new decoder cells
and the complete comparison finished at **2026-11-15 00:44:09 UTC**.

## Fixed definitions and checks

- The matrix contains nine independent zero-weight continuations, 36 new
  full/smoke/extraction/decoder YAMLs, and a hash-bound 270-cell numerical
  receipt for existing weights 0.1 and 0.2. Training is single-head K6,
  top-one-block, 25 epochs, seed 42. Target fitting is not repeated.
- The original nine 0.2 checkpoints match the versioned frozen-control hashes.
  All 18 actual full/smoke dry-runs passed. Non-distillation training settings
  and extraction/decoder settings are checked against their bound references.
- All nine full-batch, one-step GPU smokes passed. Distillation loss is exactly
  zero, prototype loss is positive and the trainable names are confined to the
  top encoder block. Seven historical models retain initial-parameter hashes;
  all seven match the new smoke runs. Those historical hashes are unavailable
  for F3/MAE and Parihaka/MAE; their absence is not reported as a match.
- CPU regression coverage verifies unchanged positive-weight defaults, the
  legacy rejection without explicit opt-in, initial student/head/RNG parity,
  frozen teacher and lower blocks, encoder updates at zero weight, complete
  final-checkpoint validation after 25 tiny fixture epochs, correct paired
  difference signs, exact complete result grids, the four-file publication
  set, no overwrite, and resolution of every new survey definition.
- The affected pretraining/Volve adapter selection passed 100 tests; the final
  new-study test file passes 12 tests, including the additional completion and
  resolver coverage. Counts overlap and are not summed. The portable suite
  recorded 6,803 passes and four configuration-inventory/namespace failures.
  The missing explicit study registrations were corrected (including the
  existing Volve v2 Random extension); all four failures and the full owning
  Volve configuration test file then passed in a **40-test follow-up**. The
  successful portable cases were reused rather than rerun. Original and
  follow-up logs are retained in operations.
- Ruff lint/format, ShellCheck, `git diff --check`, and the package isolation
  check passed. Exact commands follow the shared development guide.

## Launch

The detached master process was launched at **2026-11-11 23:42:53 UTC**.
F3/Parihaka run serially on GPU 0; Volve runs on GPU 1. A separate full-batch
Volve smoke verified memory feasibility alongside the existing GPU-1 workload.
Existing unrelated GPU jobs were not stopped. The new training processes
survived the chat-server restart and were confirmed running afterward.

Operations are stored under the artifact root at
`operations/hmm_v1/zero_distillation_ablation_v1/`:

- `preflight.json` and `dry_run_*.log`: baseline bindings and actual dry-runs.
- `smoke_validation.json`: zero loss, trainability and available historical
  initialization comparisons.
- `gpu1_feasibility.log`: the additional resource-feasibility smoke.
- `launch.json`: UTC launch time, PID, assignments and source/config hashes.
- `run_all.log` and `execution/*.log` / `*.json`: exact commands and outcomes.
- `portable_tests.log`: original full portable check; namespace corrections
  are verified separately rather than discarding this original record.
- `registration_followup_tests.log`: 40 passing follow-up tests covering all
  four initial inventory/namespace failures.

The master runs all full continuations, then extraction, then all 135 decoder
cells, and finally the complete 405-cell comparison. It writes
`analysis_complete.json` only after successful aggregation. No downstream
improvement or coefficient preference is established by a successful smoke
or by the fact that training has started.

## Completed execution and result validation

`analysis_complete.json` records successful completion of the nine full runs,
135 new decoder cells and 270 reused controls. All recorded training,
extraction and decoder commands returned status zero. The exact four-file
summary contains 405 unique finite result cells and 81 groups.

Independent recomputation from `cells.csv` reproduced all primary/secondary
means, sample SD and both sets of paired differences (versus 0.1 and 0.2),
within absolute tolerance 1e-12. All 270 control primary/secondary values match
the versioned receipt exactly as floating-point numbers. The file set matches
the publication contract, and the summary digest matches the completion
receipt. The independent check is retained as `final_result_validation.json`
in operations. The summary SHA-256 is
`2cbb36bdd2f7d444e788ad321dd015410676e1b6f4f8c8ab3e16539bfb3878bb`.

The zero-weight primary mean is better than 0.2 in 18/27 source/budget groups.
When the 15 layout/budget cells are averaged descriptively, results favor zero
for F3 in every source family, and for Barlow Twins and Random on Parihaka
and Volve. MAE on Parihaka and Volve favors 0.2. These descriptive results
retain the original F3 validation
and Parihaka/Volve test boundaries; they do not change canonical selections
or existing distillation-trained model identities.
