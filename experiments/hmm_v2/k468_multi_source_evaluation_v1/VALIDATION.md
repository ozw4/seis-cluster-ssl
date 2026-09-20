# Preparation validation — 2026-09-16

No clustering, target export, training, embedding extraction or downstream job
was executed during preparation.

- Constructed and verified the F3 MAE K468 reuse receipt against live checkpoint,
  resolved config, embeddings, mask, metadata, source summary and 15 metric files.
- The full study `run_all.sh --plan` passed: eight new continuations, 120 new
  downstream cells, three survey summaries and one aggregate.
- The study `run_all.sh --dry-run` passed: reuse hash checks and all eight
  K4/K8 target-stage dry-runs.
- All eight historical K6 frozen-receipt dry-runs passed, including live
  source and target identity validation; no receipts were written.
- Checked 83 existing input paths, including parent checkpoints, manifests,
  source embedding/mask/metadata files, historical K6 roots and all nine
  control checkpoints. The eight new training outputs and manifests were absent.
- Focused regression command: 379 passed.

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_hmm_v2_k468_multi_source.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_definition.py \
  tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_adapters.py
```

The unchanged K6810 config and driver tests also passed in the
initial focused run. Coverage includes source-family drift rejection for both
head selections, K468 configuration resolution, receipt/candidate isolation,
Volve metadata and source audits, and exact survey/aggregate output file sets.

F3 decoder provenance validation now accepts canonical
`surveys/f3/<series>/embeddings/` paths as well as historical compatibility
paths. Foreign survey, version, model, checkpoint and escaping symlink inputs
remain rejected. The F3 experiment inventory and explicit LocalBT source reuse
allowlist include the new versioned definitions. Follow-up integration checks:
105 passed.

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_f3_facies_benchmark_v2_configs.py \
  tests/seis_ssl_cluster/test_f3_local_bt_noise_rotation_search_configs.py \
  tests/seis_ssl_cluster/test_f3_overlap_subcrop_poc.py \
  tests/seis_ssl_cluster/test_voxel_decoder_runner.py
```

The portable suite was run with:

```bash
python -m pytest -q -m "not slow and not requires_segy and not requires_cuda"
```

It completed with 6,207 passed, 145 deselected and five failures against the
then-loaded code. All five were addressed: F3 inventory/lineage registration,
F3 source-path compatibility, and the declared Volve LocalBT consumer list.
The first four failures are covered by the 105 passing integration checks above;
the last is covered by 37 passing checks in
`python -m pytest -q tests/seis_ssl_cluster/test_volve_local_bt_recipe_arm_configs.py`.
Successful portable checks were retained; the full suite was not repeated.
There are no unresolved failures from this validation run.

Ruff lint and format checks passed for the changed Python paths. ShellCheck
passed for all 12 new shell scripts. `python tools/check_seis_ssl_cluster_isolation.py`
and `git diff --check` passed.

Local preparation records are under
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/k468_multi_source_evaluation_v1/preparation/`:
`environment.sh`, `plan.log`, `dry_run.log`, `frozen_k6_dry_run.log`, and `inputs.json`.
The environment file is machine-specific and ignored by Git.

The GPU feasibility stage and later live pipeline stages remain pending until
explicit experiment execution. Both GPUs were busy at the preparation check;
select an available GPU when starting.

## First-run manifest recovery — 2026-09-16

The first live F3 run completed K4/K8 clustering and export, then stopped with
`frozen K=6 source embedding identity drift`. Receipt creation resolved the
historical compatibility links, whereas manifest creation retained their
lexical paths. Preparation had validated receipt creation separately and did
not catch the subsequent receipt-to-manifest path comparison.

New frozen-reference manifests now resolve source and target roots consistently
with receipt creation. Hash, checkpoint, survey and target validation remain
strict. Existing receipts and manifests are not rewritten. Regression coverage
includes K6810 and K468, source/K6 compatibility links, read-only CLI reuse,
changed files and retargeted links to byte-identical foreign inputs.

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_strat_frozen_k6_reference.py \
  tests/seis_ssl_cluster/test_strat_multi_head_target_manifest.py \
  tests/seis_ssl_cluster/test_strat_hmm_source_audit.py \
  tests/seis_ssl_cluster/test_hmm_v2_k468_multi_source.py
```

All 113 tests passed. The failed command passed a live dry-run, then F3's
manifests stage completed for both new arms. Both smoke configurations passed
dry-run; training was not started during recovery. All 53 pre-existing output
files retained their hashes. Ruff, formatting, ShellCheck and `git diff --check`
passed. Recovery evidence and the machine-local `resume.sh` are in
`operations/hmm_v2/k468_multi_source_evaluation_v1/recovery_manifest_paths/`
under the artifact root. That script continues F3 from smoke, then runs Parihaka,
Volve and the aggregate. Do not repeat the initial targets/export commands.

## F3 summary recovery — 2026-09-18

All 30 new F3 cells completed, then the summary rejected the canonical
`mae_hmm_k6` control through a namespace check intended for new candidate runs.
The completed-cell adapter now accepts canonical models only when checkpoint,
embeddings, runs and summary paths match their canonical definitions. Source,
decoder, evaluation and paired-identity audits remain mandatory. New candidate
execution still rejects canonical IDs and overlapping output directories.

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_hmm_f3_completed_cell.py \
  tests/seis_ssl_cluster/test_hmm_v2_multi_source_adapters.py \
  tests/seis_ssl_cluster/test_hmm_k6810_receipts_aggregate.py \
  tests/seis_ssl_cluster/test_f3_lithology_candidate_benchmark.py \
  tests/seis_ssl_cluster/test_f3_lithology_paired_candidate_results.py
```

All 136 tests passed. Regression coverage includes canonical source/output drift,
model-ID mismatch, tampered embedding provenance, missing completed jobs and the
unchanged new-candidate rejection. Ruff lint and formatting passed. The live F3
summary dry-run passed all 45 candidate cells and their paired controls, and all
three Parihaka target configurations passed dry-run.
The F3 summary subsequently completed with all 45 cells and nine comparison
rows, publishing exactly the three expected files. Parihaka target generation
was then confirmed running in the detached recovery process.

Recovery records are in
`operations/hmm_v2/k468_multi_source_evaluation_v1/recovery_summary_20260918/`
under the artifact root. The detached script starts from F3 `summary`, continues
Parihaka's CPU targets/export/manifests, and waits for GPU 0 capacity before its
GPU stages. Volve follows with the same gate, then the cross-survey aggregate.
The gate observes other jobs without stopping them. Do not repeat completed F3
training or launch a second copy of the recovery script.
