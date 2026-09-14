# Fixed K6810 multi-source evaluation

The matrix fixes three source families (MAE100, asymmetric LocalBT3, random)
across F3, Parihaka, and Volve. There are eight new 25-epoch continuations and
120 new downstream cells. F3 MAE reuses experiment 127 directly, giving nine
arms and 135 total evaluation cells. No files are copied or symlinked from the
reused arm or its matching HMM_v1 controls.

The selected head is `multi_resolution_ordered_prototypes_v1`, with ordered
K `[6, 8, 10]`, projection 128, temperature 0.1, normalization enabled, and
hard Viterbi targets. Loss weights are prototype 1.0, usage 0.005,
distillation 0.2, and consistency 0.0 (`consistency_beta: 0.1` is retained in
the schema). The top encoder block alone is unfrozen. Every full run uses
25 epochs, 10,000 samples per epoch, seed 42, AMP disabled, and head/encoder
learning rates of 1e-5. Each source's frozen K6 configuration supplies its
checkpoint, batch size, workers, survey input, masks, AGC and downstream
settings. Smoke and full runs start independently from the same parent.

F3 uses validation metrics. Parihaka and Volve use held-out test metrics with
no test-dependent tuning or selection. Matching Single-Head K6 controls are
conditions 2b, 4b and 6b for MAE, LocalBT and random respectively.

## Receipts

`hmm_v1_controls_receipt.json` contains nine controls, each with checkpoint
identity and 15 unrounded metric cells. It was explicitly derived from the
three authoritative frozen HMM_v1 records. Runtime code reads this versioned
receipt, never `reports/hmm_v1`.

`f3_mae_k6810_reuse_receipt.json` binds the selected F3 checkpoint, its resolved
training configuration, embeddings, valid-token mask, embedding metadata,
existing paired summary, and 15 metrics files. Identities are relative to
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`; the source artifacts remain in experiment
127's namespace. Receipt construction checks the checkpoint's configuration
against the selected scientific contract and the recorded resolved config.

Check definitions without reading live artifacts:

```bash
python -m seis_ssl_cluster.hmm.multi_source_receipts --check \
  --matrix experiments/hmm_v2/k6810_multi_source_evaluation_v1/matrix.yaml \
  --controls experiments/hmm_v2/k6810_multi_source_evaluation_v1/hmm_v1_controls_receipt.json \
  --reuse experiments/hmm_v2/k6810_multi_source_evaluation_v1/f3_mae_k6810_reuse_receipt.json
```

Add `--artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"` to verify the reused
F3 files read-only. Receipt construction is available only through explicit
`--build-receipts` with `--freeze-root`, `--source-summary` and
`--artifact-root`; it refuses existing receipt outputs.

## Execution and aggregation

Use the per-survey `RUNBOOK.md` and `run_all.sh` under:

- `experiments/f3/facies_benchmark_v2/128_hmm_v2_k6810_multi_source_v1/`
- `experiments/parihaka/facies_benchmark_v1/41_hmm_v2_k6810_multi_source_v1/`
- `experiments/volve/horizon_benchmark_v1/34_hmm_v2_k6810_multi_source_v1/`

Both driver planning and the direct survey summary CLI validate the source
references in each `execution.yaml` before reading live artifacts. Preflight
compares raw definitions without environment expansion: clustering retains its
K6 reference except K/output; full training retains the parent, input, masks,
model and training settings; extraction retains its reference except checkpoint
and output. Smoke differs from full only in output and its one-step budget.
Summary `experiment_root` binds the training/downstream paths to the validated
definition, including the fixed F3 reuse reference.

The default driver mode prints commands. Live execution is sequential and
requires `--execute`; interrupted training can resume only its own latest
checkpoint. Complete sources, embeddings and cells are audited before reuse.
Partial clustering/export/embedding outputs require inspection and are never
silently replaced. Survey summaries require all 45 cells and paired source,
completion, decoder and evaluation-support audits.

After all three survey summaries are complete, export
`SEIS_SSL_CLUSTER_WORKSPACE` as the absolute checkout path and run:

```bash
python -m seis_ssl_cluster.hmm.multi_source_aggregate \
  --config experiments/hmm_v2/k6810_multi_source_evaluation_v1/aggregate.yaml \
  --dry-run
python -m seis_ssl_cluster.hmm.multi_source_aggregate \
  --config experiments/hmm_v2/k6810_multi_source_evaluation_v1/aggregate.yaml
```

The aggregate writes exactly `comparison.csv`, `summary.json` and
`summary.md` beneath the dedicated artifact namespace. There are 27 rows
(survey × source × size), 135 underlying cells, all-15 descriptives and five
layout clusters per arm. Positive improvement means K6810 minus K6 for
classification and K6 minus K6810 for Volve error. Metrics are never averaged
across surveys, and no winner or automatic promotion is produced. All summary
outputs are immutable. Optional source-summary SHA-256 values in
`aggregate.yaml` can additionally pin a particular set of completed summaries.
