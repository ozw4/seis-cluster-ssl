# Missing-control completion

Prespecified on 2026-11-15 after inspecting the retained evaluation inventory.
The priority is to complete existing scientific comparisons before adding new
architectures or selecting coefficients from test results.

## Identified gaps and fixed scope

1. Barlow Twins v2 with zero distillation and consistency 0.1 has completed
   training/embeddings on all three surveys, but only layout_000/large decoders.
   MAE and Random already have five layouts and three budgets. Run exactly the
   missing **42 decoder cells**, retaining the three original completed cells.
   Use the original per-survey definitions, checkpoints, fixed HMM targets,
   decoder seed 42000, budgets, training epochs and validation selection.
2. All 45 v1 zero-distillation F3 decoders are complete, but have only validation
   scores. Score their frozen predictions on the established 113,979,624-voxel
   common-test mask. Also score the completed 15 Barlow v2 zero-distillation
   cells and rescore its 15 original 0.2 controls as an identity check.
3. Existing v2 experiments cover the other cells of the 2x2 distillation
   (0/0.2) by consistency (0/0.1) design. Assemble all **540 paired cells**
   across three surveys, three source families, five layouts and three nested
   budgets. Report the four conditional effects and their interaction, without
   pooling different survey metrics or selecting the best combination.

The aligned v1 table contains all **405 cells**, retaining 0/0.1/0.2 on every
survey, source and budget. Original F3 validation results remain unchanged.
The existing v2_dis0_con0 study completed on 2026-11-15 and is reused; it must
not be relaunched. Existing K6810 consistency-zero evaluations are also complete.
The older runbook statement that F3 LocalBT/Random cells were incomplete is
outdated relative to the retained metrics and common-test inventory.

The matrix pins artifact originals and versioned recipes by SHA-256. The
figure-only Barlow matrix remains unchanged and is used only as a recipe/source
binding: this extension has its own orchestrator and never runs its out-of-scope
generic training/test/summary phases. No new encoder pretraining or target fit
is needed. No existing score, checkpoint, image or source receipt is overwritten.
Completing a candidate's previously missing decoder cells in its original
canonical run directory preserves its identity and permits normal strict reuse.

## Execution

Use the existing repository environment and artifact/data environment values.
Preserve the frozen F3 label path spelling. Actual machine paths belong only in
ignored operation records.

```bash
python -m seis_ssl_cluster.hmm.missing_controls --phase plan
python -m seis_ssl_cluster.hmm.missing_controls --phase dry-run
MISSING_CONTROLS_GPU=0 bash experiments/hmm_v2/missing_controls_completion_v1/run_all.sh
```

A separate owned CPU process scores v1 F3 while decoder cells run on the chosen
GPU. All 15 medium cells precede small/large cells; no medium outcome removes
later conditions. Missing decoder cells receive a live dry-run before execution.
Completed cells are identity-audited by the standard runner and never retrained.
A partial cell resumes only its own latest.pt. GPU admission requires at least
12 GiB free and utilization <=10%; busy devices wait 30 seconds without changing
batch sizes or stopping other work. Keep only one wrapper active; its exclusive
study lock rejects duplicate launches. Existing unrelated GPU work is preserved.

Run common-test scoring alone with `--phase v1-test` or `--phase barlow-test`.
The latter requires all 30 selected decoders. Both namespaces have fixed allowed
conditions and require the original reference-mask hash. Inventories, selected
best decoder identities and data supports are frozen before scoring. Resume
checks are handled by the existing common-test evaluator. Test scores never
select checkpoints, coefficients, budgets or layouts.

## Outputs and restart

`operations/hmm_v2/missing_controls_completion_v1/` owns launch/PID/exit records,
`status.json`, per-cell logs, preflight evidence, and a final `complete.json`.
The detached launcher ignores terminal hangups. Queued/running status is not
completion. On error the owned current cell retains its checkpoint; an already
running CPU scorer may finish. Before restarting, check the recorded PID and
CPU child PID so the same work is not launched twice.

F3 scores live in separate canonical evaluation namespaces:
`lithology_common_test_v1_zero_distillation_v1` (45 cells) and
`lithology_common_test_barlow_zero_distillation_v1` (30 cells).
Generated bound inventories/configs stay in
`studies/hmm_v2/missing_controls_completion_v1/test_definitions/`.

The summary producer publishes exactly these ten files under that study's
`summary/`, atomically, after every required result exists:

- barlow_cells.csv and barlow_comparison.csv (90 cells / 18 means)
- v1_cells.csv and v1_comparison.csv (405 cells / 81 means)
- factorial_cells.csv and factorial_comparison.csv (540 cells / 108 means)
- factorial_contrasts.csv (135 within-layout contrasts)
- provenance.json, summary.json and summary.md

All values and source tables remain artifacts. No report directory is an input.
Sample SD has ddof=1; five layouts are not five encoder seeds. The continuation
seed remains 42 and parents are held fixed. Independent pretraining/continuation
seed replication and a genuine spatial amplitude holdout remain substantive
future studies, outside this bounded completion. Historical v1 0.1 vs 0.2 recipes
also differ in worker/execution details, as documented in the original study;
common-test alignment does not remove that historical limitation.

## Completed 2026-11-15

All additional training and common-test evaluation completed. A final CSV path
error was corrected and aggregation alone rerun. The ten-file summary now
contains the full 540-cell factorial comparison and aligned 405-cell v1 table.
See [completion validation](VALIDATION.md) for numerical verification and the
retained failure/recovery records.

## Curated comparison tables

`python -m seis_ssl_cluster.hmm.missing_controls --phase review` copies the
completed artifact summary to `reports/hmm_missing_controls_completion_v1/`.
The exact review set is `README.md`, `provenance.json`, `factorial_cells.csv`,
`factorial_comparison.csv`, `factorial_contrasts.csv`, `barlow_comparison.csv`,
and `v1_comparison.csv`. Publication refuses existing or incomplete outputs;
focused tests verify membership and byte-preserving copies. Review files are
never pipeline inputs. [Open the tracked comparison](../../../reports/hmm_missing_controls_completion_v1/README.md).

## HMM versus no-HMM comparison

The shared `factorial_cells.csv` now contains 560 comparison rows for the
original 540 HMM conditions. It places `without_hmm_primary`,
`with_hmm_primary`, and `primary_improvement` side by side, with the metric
name, unit, and direction. The original artifact summary remains immutable;
the derived comparison is under `hmm_vs_no_hmm_v1/` in the same artifact study.
`no_hmm_comparison.yaml` fixes the original 540-cell checksum, ten no-HMM
source/seed definitions, expected coverage, and HMM source summaries before
joining scores.

All 140 distinct no-HMM baseline cells are included: the original 135 decoder
seed-42000 cells and all five Parihaka MAE/large decoder seed-42001 cells. Their
metrics are reused for four HMM loss settings, yielding 540 same-seed and 20
cross-seed comparisons. Both decoder seeds and comparison type are explicit
per row. For Parihaka MAE/large, seed 42001 is the canonical reference selected
in experiment 47; historical seed 42000 remains in the table. Cross-seed pairing
requires identical scientific support and decoder settings except the seed and
initial decoder weight hash. Same-seed pairing checks the full identity.
The no-HMM benchmarks
retain their established pretraining budgets: MAE reconstruction continuation,
three-epoch Barlow Twins, and an untrained Random encoder.

Positive improvement always favors HMM. For F3 mean IoU and Parihaka channel
IoU it is HMM minus no-HMM; for Volve MAE [samples] it is no-HMM minus HMM.
Secondary improvements (macro F1, channel F1, within-two accuracy) use HMM minus
no-HMM. The 112-row summary keeps baseline seeds separate and reports
means/SDs for both models, the mean/sample
SD of the five within-layout improvements, and HMM wins/ties/losses. It does
not combine different metric units. Layout SDs are not encoder-seed SDs.

Generate the five-file artifact bundle and update the seven-file review:

```bash
python -m seis_ssl_cluster.results.hmm_baseline_comparison \
  --output artifacts/seis_ssl_cluster/studies/hmm_v2/missing_controls_completion_v1/hmm_vs_no_hmm_v1 \
  --review-output reports/hmm_missing_controls_completion_v1
```

For an existing derived bundle, add `--reuse-existing`; its definition hash is
checked before reuse. To explicitly rebuild and overwrite the owned five-file
bundle, use `--overwrite` instead. Unowned files prevent replacement. The exact derived set is `factorial_cells.csv`,
`factorial_comparison.csv`, `no_hmm_cells.csv`, `provenance.json`, and
`README.md`. The review set remains the seven files listed above; the paired
factorial tables replace their HMM-only views. Other comparison tables retain
their original bytes. All numerical inputs come from artifacts, never reports.
Focused coverage tests metric directions, retention of both decoder seeds,
separate seed aggregates, support mismatch rejection, paired SDs, negative
effects, recorded source identity, overwrite ownership and exact publication.

This bounded HMM comparison is accompanied by the refreshed full saved-result
CSV/HTML catalog, including all versions, seeds, checkpoints and archived
outputs; see [catalog documentation](../../../docs/experiment_metrics_catalog.md).
