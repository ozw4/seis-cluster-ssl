# HMM v3: trace masking after canonical v02

The study runs F3, Parihaka and Volve × MAE100, asymmetric LocalBT3 and random
initialization with hard HMM targets at K `[6, 8, 10]`.
There are nine new 25-epoch continuations and 135 new downstream cells
(three sources × five layouts × three sizes per survey).

The reference is the explicitly selected
[`k6810_cons010_multi_source_evaluation_v1`](../../hmm_v2/k6810_cons010_multi_source_evaluation_v1/README.md)
v02 recipe. The authoritative
[selection](../../hmm_v2/canonical_multi_source_v1/selection.yaml) binds the
completed 135-cell artifact summary with SHA-256
`76af6c4bfc0a3a07051b5622b191f74b61bd55a84a6613f9123568af84087666`.
Every survey execution definition pins that selection, its v02 recipes,
completed reference checkpoint hashes and immutable target-manifest hashes.

| Survey | v3 definition | Matched canonical v02 | Evaluation |
|---|---|---|---|
| F3 | [132](../../f3/facies_benchmark_v2/132_hmm_v3_k6810_trace_mask_multi_source_v1/README.md) | 131 | Validation |
| Parihaka | [45](../../parihaka/facies_benchmark_v1/45_hmm_v3_k6810_trace_mask_multi_source_v1/README.md) | 44 | Held-out test |
| Volve | [38](../../volve/horizon_benchmark_v1/38_hmm_v3_k6810_trace_mask_multi_source_v1/README.md) | 37 | Held-out test |

The treatment masks 10% of supervised-valid XY token columns across Z in
the student input representation, while preserving all HMM labels. A learned
mask token replaces each selected column. The teacher observes the same crop
unmasked. Prototype reconstruction assigns weight 0.5 to masked tokens and
0.5 to visible tokens, and distillation is visible-only at weight 0.2.
Inference and embedding extraction use the unmasked encoder.

The canonical v02 consistency weight of 0.1 and normalized-order smooth-L1
policy are retained. The remaining losses, source-specific parent checkpoints,
input/target lineage, architecture, AGC, invalid-data masks, optimizer,
25-epoch budget, 10,000 samples per epoch, seed 42, top-block unfreezing,
embedding geometry and downstream conditions match the corresponding v02 arm.
This controlled comparison measures the added effect of trace-column masking.
Targets remain fixed throughout training.

The mask mechanism originates in
[experiment 104](../../f3/facies_benchmark_v1/104_strat_hmm_multi_head_k6810_center_trace_masked_v1/README.md).
Its old F3 training recipe is not the matched reference for this study:
v3 follows the selected multi-survey v02 conditions, including consistency.
External test results do not select mask settings, losses, heads or training
duration. Metrics are compared within each survey and are not pooled across
different tasks.

Each survey's runbook and `run_all.sh` owns smoke, full, audit, embeddings,
downstream and paired-summary execution. The default is `--plan`;
`--dry-run` validates live inputs and `--execute` runs the selected stages.
The [matrix](matrix.yaml) records the fixed nine-arm scope and execution
definitions. Completed v02 results are read-only comparison inputs.
The [validation and launch record](VALIDATION.md) records the readiness
checks and the artifact locations for monitoring the running matrix.

New outputs follow
`surveys/<survey>/<storage-series>/<kind>/hmm_v3_k6810_trace_mask_multi_source_v1/<identity>/`.
Execution records live under
`operations/hmm_v3/k6810_trace_mask_multi_source_v1/<survey>/`.
Survey comparison summaries publish exactly `comparison.csv`,
`summary.json` and `summary.md` in their declared artifact destinations.
Curated reports may be produced from that exact set after review; pipeline
inputs always remain in artifacts.
