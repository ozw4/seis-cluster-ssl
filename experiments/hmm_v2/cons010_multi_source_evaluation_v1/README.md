# Multi-source consistency-weight 0.1 experiment

This is a paired loss-weight ablation of the existing K468 and K6810 studies.
The requested `[8,6,10]` head set is represented in the established sorted order
`[6,8,10]`. Both head sets are fixed before execution.

| Property | Fixed setting |
| --- | --- |
| Heads | `[4,6,8]` and `[6,8,10]` |
| Sources | F3, Parihaka, Volve × MAE100, asymmetric LocalBT3, random initialization |
| Consistency | `loss.consistency_weight: 0.1`, `variant: cons010` |
| Consistency policy | `normalized_order_smooth_l1_v1`, beta 0.1 |
| Other losses | Prototype 1.0, usage 0.005, distillation 0.2 |
| Continuation | 25 epochs, 10,000 samples/epoch, seed 42, top encoder block |
| Optimization | Encoder/head LR 1e-5; source-family batch size and workers; AMP off |
| Downstream | Five layouts × small/medium/large; original decoder and label supports |
| Evaluation | F3 validation; Parihaka and Volve test; no result-dependent selection |
| New work | 18 continuations and 270 downstream cells, plus 18 one-step smoke runs |

`execution.yaml` in each survey experiment binds the SHA-256 of the three
inherited zero-consistency definitions (training, embeddings, downstream) and
the existing target manifest. The validator rejects source, decoder, loss,
output or target-reference changes. Training binds the manifest's recorded
SHA-256, including on resume. Existing pseudo-label arrays, confidence, masks,
parent checkpoints and teacher checkpoints remain the original inputs.
No clustering, target export or new target manifest is needed.

All 18 candidates, including both F3 MAE candidates, train from their original
MAE100/LocalBT3/random parent; they do not continue a completed zero-consistency
HMM model. Smoke and full training start independently. Candidate IDs end in
`_cons010`, and outputs use `hmm_v2_<K>_cons010_multi_source_v1` namespaces.
Embedding extraction inherits each source's batch size and prefetch settings.
The existing MAE F3 screening configurations use prefetch 0; the other sources
use 2. This scheduling setting does not change the source data or weights.

## Preparation and start

Activate the repository environment and set absolute paths for this machine:
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `F3_ROOT`, and
`SEIS_SSL_CLUSTER_VOLVE_ROOT`. `F3_ROOT` must match the absolute label SEG-Y path
in the frozen F3 supervision metadata; a different hard-link spelling is not
interchangeable for evaluation provenance. The launchers set the workspace and
Python import path. Machine-specific values are kept in ignored
`operations/hmm_v2/cons010_multi_source_evaluation_v1/preparation/environment.sh`.

From the repository root:

```bash
# Read-only command plan for all 18 candidates and both aggregates.
bash experiments/hmm_v2/cons010_multi_source_evaluation_v1/run_all.sh --plan

# Read-only live validation of all 36 smoke/full training configurations.
bash experiments/hmm_v2/cons010_multi_source_evaluation_v1/run_all.sh --dry-run

# Start both studies: K468 on GPU0 and K6810 on GPU1.
bash experiments/hmm_v2/cons010_multi_source_evaluation_v1/run_all.sh --execute
```

The two studies run concurrently. Within each study, F3, Parihaka and Volve run
sequentially, followed by its aggregate. `K468_GPU` and `K6810_GPU` can override
the device assignments and must differ. There is no utilization or free-memory
admission gate. The launcher holds a lock against duplicate launches, stores
separate logs for each study, waits for both children, and records both exit
statuses. A failed study does not terminate the other study.

To run one study on a selected GPU, use its own wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/run_all.sh --execute
CUDA_VISIBLE_DEVICES=1 bash experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/run_all.sh --execute
```

GPU feasibility, embeddings and downstream execution are not run during
preparation. Downstream source audits require newly completed training and
embeddings, so live downstream dry-runs become available after those stages.
The training dry-run resolves the actual parent and target configuration; it
does not establish GPU memory feasibility.

## Restart and outputs

Survey drivers retain the existing single-stage `--candidate`, `--layout`,
`--size`, and `--resume` interfaces. The active sequence is `smoke`, `full`,
`audit`, `embeddings`, `downstream`, `summary`; the original target stages have
no commands in these studies. Use the survey runbooks to resume an interrupted
stage. Do not repeat the entire wrapper after partial execution: completed
summary directories are immutable, and partial runs require their own resume
checkpoint. Full training cannot resume a zero-consistency checkpoint.

- [K468 definitions](../k468_cons010_multi_source_evaluation_v1/README.md)
- [K6810 definitions](../k6810_cons010_multi_source_evaluation_v1/README.md)

Survey outputs use
`surveys/<survey>/<storage-series>/<kind>/hmm_v2_<K>_cons010_multi_source_v1/`.
F3 pretraining retains v1 storage; embeddings and evaluation use v2. Existing
historical input paths retain their migration compatibility links. Frozen K6
checkpoint evidence accepts such a link only when the resolved path and hash
match; receipt and manifest bytes are preserved.

Cross-survey summaries use
`studies/hmm_v2/<K>_cons010_multi_source_evaluation_v1/summary/`.
Every survey and aggregate producer publishes exactly `comparison.csv`,
`summary.json`, and `summary.md` and never writes tracked reports. Each study
has 135 new cells and 27 per-size comparison rows against the existing matching
K6 controls. The two studies remain separate; incompatible survey metrics are
never averaged. The five layouts are the statistical units; the three nested
label budgets are not independent replicates.

The zero-consistency results can later be joined by survey, source family,
layout and size for the direct consistency ablation. F3 LocalBT/random K6810
zero-consistency downstream evaluation is currently incomplete, so those
baseline comparisons must remain unavailable until separately completed.
See [VALIDATION.md](VALIDATION.md) for preparation evidence.
