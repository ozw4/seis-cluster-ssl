# K468 volve execution

This survey belongs to the [K468 multi-source study](../../../hmm_v2/k468_multi_source_evaluation_v1/README.md).
Use its environment setup and fixed scientific recipe. Export an available GPU
through `CUDA_VISIBLE_DEVICES`; the driver sets the workspace and Python import path.

```bash
bash experiments/volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1/run_all.sh --plan
bash experiments/volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1/run_all.sh --dry-run --from targets --to targets
bash experiments/volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1/run_all.sh --execute
```

Stages run sequentially: `targets`, `export`, `manifests`, `smoke`, `full`,
`audit`, `embeddings`, `downstream`, `summary`. K4 and K8 are newly generated;
K6 retains the historical source-family targets and receives a frozen-reference
receipt. There is no replay stage. Full training starts from the original parent,
independently of smoke. Embeddings use the source reference with prefetch depth 2.

Use `--from STAGE --to STAGE` for a contiguous stage range. Candidate filtering
requires a single stage. For interrupted training, select its own latest checkpoint:

```bash
bash experiments/volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1/run_all.sh \
  --execute --from full --to full \
  --candidate random_init_hmm_v2_mh_k468_distill020 --resume
```

For one decoder cell, use `--execute --from downstream --to downstream
--candidate ID --layout layout_000 --size small --resume`. Resume is restricted
to the selected run's own latest checkpoint. Complete sources, embeddings and
cells are audited before reuse; partial target or embedding outputs require
inspection. Existing clustering and summary outputs are never overwritten.
Source audits run before embeddings, downstream and summary stages.

Logs and locks live below
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/k468_multi_source_evaluation_v1/volve/logs/`.
A survey summary requires all 45 cells and matching K6 controls, and publishes
exactly `comparison.csv`, `summary.json` and `summary.md` in the artifact namespace
specified by `60_summary/01_paired_comparison.yaml`.

All three source families run here. Held-out test metrics are evaluation only;
do not tune the head selection, losses or epochs using these results.
