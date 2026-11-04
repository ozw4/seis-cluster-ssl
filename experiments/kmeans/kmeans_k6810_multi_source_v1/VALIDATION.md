# Implementation and launch validation — 2026-10-24

The unordered control completed at 2026-10-30 12:49:43 UTC. All nine
25-epoch continuations, 135 downstream cells, and the audited comparison
summary are complete. The sections below retain the implementation and
recovery history.

## Successful checks

- All nine source-embedding preflights passed.
- Nine fixed target manifests were built and fully validated at K=[6,8,10],
  including matched edge-excluded support, label ranges, cluster occupancy,
  confidence, and reference hashes.
- All eighteen smoke/full training configuration dry-runs passed.
- All nine original-batch GPU one-step smokes passed on GPU0.
- The focused target/study/source-audit/head/provider/checkpoint tests passed:
  87 tests in the focused selection, followed by 98 existing schema-7 and
  checkpoint-extraction regression tests after the legacy-loader compatibility
  correction. The new CLI help contract also passed.
- CPU integration covers interrupted training, checkpoint resume, incompatible
  loss rejection, unordered extraction metadata, and independent cluster-ID
  permutation invariance with ordered consistency forbidden.
- Ruff lint and formatting, package-isolation validation, compilation of new
  modules, experiment documentation links, and `git diff --check` passed.

The portable suite was run with
`python -m pytest -q -m 'not slow and not requires_segy and not requires_cuda'`.
Its initial run reported 6665 passed and 55 failures. The failures were checked
again after correcting the legacy checkpoint dispatch and registering the new
experiment inventories: 51 of the failed tests passed on recheck. The remaining
four failures concern pre-existing untracked work:

- `test_every_v2_config_resolves_with_its_owning_resolver`: unregistered F3
  experiments 135, 136 and 137.
- `test_candidate_namespace_is_unused_by_other_experiments`: the existing
  `experiments/hmm_v2/canonical_paper_sections_v1/execution.yaml` consumes a
  historical LocalBT source namespace.
- `test_existing_build_parser_functions_construct_argparse_parsers`: the
  existing untracked `evaluate_f3_common_test.py` has no `build_parser`.
- `test_recipe_arm_namespace_has_only_declared_recipe_and_continuation_users`:
  the same canonical-paper-sections execution file adds an undeclared historical
  Volve namespace consumer. After explicitly registering the new k-means
  experiment, a focused recheck confirms that only this existing consumer
  remains in the assertion difference.

These unrelated files were preserved. Their failures do not constitute a
successful portable-suite result.

## Live operation

Operation paths below are relative to `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`:

- Target generation: `operations/kmeans/kmeans_k6810_multi_source_v1/20261024T010309093373Z/`.
- Nine successful GPU smokes: `operations/kmeans/kmeans_k6810_multi_source_v1/20261024T010658338798Z/`.
- Detached driver record and progress: `operations/kmeans/kmeans_k6810_multi_source_v1/launch_20261024T010832Z/`.
- Full execution stage logs: `operations/kmeans/kmeans_k6810_multi_source_v1/20261024T010834381381Z/`.

The driver starts with full training, because target creation and smokes have
already completed. It executes nine full continuations, nine audits, nine
embedding extractions, 135 decoder cells, and one final summary stage in order.
The launch record identifies the process and GPU. A failed stage stops the
operation; use the runbook's explicit recovery commands. Completion must be
established from audited final outputs, rather than this launch record.

## 2026-10-30 Volve downstream recovery

All nine 25-epoch continuations, encoder audits, and embedding extractions
finished. F3 and Parihaka each completed all 45 decoder cells (90/135 total).
The first Volve decoder job stopped at 2026-10-30 00:58 UTC before decoder
training, because its `multi_head_training_config` named nonexistent experiment
`37_kmeans_k6810_multi_source_v1` instead of experiment 40.

The configuration derivation now replaces the full experiment reference before
its namespace substring. Only the three Volve downstream recipe-reference
fields changed. Training configurations, targets, checkpoints, embeddings,
decoder settings, and completed F3/Parihaka outputs were retained. Three
regression cases reproduced the missing-recipe bug before correction; the
updated target/study test selection passed all 14 tests. Ruff lint, format,
and `git diff --check` passed. Real-data downstream dry-runs passed for all
three Volve source families after the correction.

Recovery is recorded under
`operations/kmeans/kmeans_k6810_multi_source_v1/recovery_20261030T051739Z/`.
Its `driver.log` and `status.json` supersede the stopped launch log for current
progress. At 05:18 UTC a detached GPU0 driver was launched for the 45 Volve
cells, followed by the complete three-survey comparison summary. A failed stage
stops recovery and records its exit code. Scientific results were pending at
recovery launch; completion is recorded below.

## Completion confirmed on 2026-10-31

The recovery status is `complete`, with `volve_downstream` and `summary` both
finished at 2026-10-30 12:49:43 UTC. The final summary has status `complete`,
135 unique cells (45 each for F3, Parihaka and Volve), and 27 comparison rows.

Results are in
`studies/kmeans/kmeans_k6810_multi_source_v1/summary/` under the artifact root:
`comparison.csv`, `summary.json`, and `summary.md`. The Markdown and CSV are
the concise review outputs; the JSON retains the full audit evidence.

For the primary metric, the five-layout mean favors canonical HMM v2 in all
27 survey/source/label-budget conditions: higher Mean IoU for F3, higher channel
IoU for Parihaka, and lower horizon MAE for Volve. This is a descriptive paired
comparison of HMM target generation plus ordered cross-head consistency against
unordered k-means; it does not isolate HMM decoding or establish statistical
significance.

## 2026-11-04 commit-scope validation

Before publication, the tracked base plus only the k-means implementation,
experiment definitions, and curated result files were copied to a separate
verification checkout. Other in-progress workspace changes were excluded.

The portable suite completed successfully:

```text
6664 passed, 2 skipped, 145 deselected, 11 warnings in 518.57s (0:08:38)
```

Command: `python -m pytest -q -m "not slow and not requires_segy and not requires_cuda"`.
The earlier four workspace-only failures did not occur in this isolated scope.
Ruff lint and formatting passed for all 19 Python files in scope, as did the
package-isolation check, documentation links, and Git whitespace validation.
The curated report's two focused publication tests also passed after CSV line
endings were normalized to LF; reconstruction of the original CRLF bytes
matches the recorded source hash.

The review output is tracked under
[`reports/kmeans_k6810_multi_source_v1/`](../../../reports/kmeans_k6810_multi_source_v1/).
Its exact published set is `README.md`, `comparison.csv`,
`hmm_advantage_by_source_and_budget.png`, and `provenance.json`. Checkpoints,
embeddings, target arrays, and the large audit JSON remain in artifacts.
