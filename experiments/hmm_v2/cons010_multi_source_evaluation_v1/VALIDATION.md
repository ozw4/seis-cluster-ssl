# Preparation validation — 2026-09-19

Preparation did not execute clustering, target export, GPU smoke training, full
training, embedding extraction or downstream evaluation.

- Both complete command plans passed: 18 new full continuations, 270 new decoder
  cells, six survey summaries and two study aggregates. Neither plan contains
  target-generation commands or reuse of completed F3 MAE results.
- All 36 live smoke/full configuration dry-runs passed, using the original
  target manifests and fixed `consistency_weight: 0.1` / `variant: cons010`.
- Verified the 18 target-manifest digests, parent checkpoint and data-manifest
  paths, nine unique parent checkpoint hashes, all nine frozen K6 control
  checkpoint hashes, and 54 distinct absent full-training/embedding/decoder
  output paths.
- New configuration, inheritance rejection, resume namespace, cross-study
  evidence rejection and exact publication tests: 31 passed.
- Existing multi-source config, driver, receipt, adapter and aggregate tests:
  396 passed.
- The combined launcher passed isolated stub execution checks: read-only
  plan/dry-run, invalid-mode and same-GPU rejection, GPU0/GPU1 assignment,
  retention of both child exit statuses when one fails, and duplicate lock
  rejection. These checks launched no experiment processes.
- Ruff lint/format, ShellCheck for all nine launch scripts, repository isolation
  and whitespace checks passed. Portable YAML paths and all Markdown links
  were also verified.

The repository portable test selection completed successfully: **6,322 passed,
145 deselected, 11 warnings** (475.77 seconds). It includes the migration
regressions and new survey namespace/consumer inventories:

```bash
python -m pytest -q -m "not slow and not requires_segy and not requires_cuda"
```

## Existing checkpoint path migration

The first K6810 F3 LocalBT training dry-run rejected a frozen K6 checkpoint
reference whose historical pathname now resolves through an artifact migration
compatibility link. The source checkpoint bytes and hash were unchanged.
Validation now compares the resolved checkpoint paths and hashes while retaining
the original recorded path in returned evidence. Frozen receipts and manifests
are not rewritten. Regression coverage checks successful relocation, altered
checkpoint bytes, and a byte-identical foreign checkpoint path. The subsequent
K6810 study dry-run passed all 18 configurations.

## Evidence and limits

Machine-local logs and input hashes are under
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/cons010_multi_source_evaluation_v1/preparation/`:
`environment.sh`, `inputs.json`, `k468_plan.log`, `k6810_plan.log`, `dry_run.log`,
`k6810_dry_run.log`, `tests_new.log`, `tests_existing.log`, and
`tests_portable.log`, `plan_counts.json`, and `preparation_status.json`. The original `dry_run.log` retains the first path failure
and the 18 successful K468 dry-runs; `k6810_dry_run.log` records the successful
K6810 follow-up.

Training dry-runs validate configuration and source/target references without
allocating training models on the GPU. GPU memory feasibility is assessed by
the one-step smoke stages on execution. Live embedding and downstream source
audits await their new checkpoints and embeddings. All later source, paired
support and completed-output audits remain required.
