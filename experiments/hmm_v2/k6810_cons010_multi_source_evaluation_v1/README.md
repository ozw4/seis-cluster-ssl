# K6810 consistency-weight 0.1 study

This completed study is the canonical HMM v2 multi-source protocol. The
versioned decision and bound evidence are recorded in
[`../canonical_multi_source_v1/selection.yaml`](../canonical_multi_source_v1/selection.yaml).

The [shared runbook](../cons010_multi_source_evaluation_v1/README.md) defines
scientific controls, environment setup, GPU0/GPU1 launch, outputs and restart.
This study runs nine new 25-epoch continuations and 135 downstream cells.
The unchanged inputs come from `k6810_multi_source_evaluation_v1`, including
the F3 experiment 127 MAE source; every new candidate has consistency weight
0.1 and a `_cons010` suffix. No completed F3 model is reused.

- [f3](../../f3/facies_benchmark_v2/131_hmm_v2_k6810_cons010_multi_source_v1/RUNBOOK.md)
- [parihaka](../../parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/RUNBOOK.md)
- [volve](../../volve/horizon_benchmark_v1/37_hmm_v2_k6810_cons010_multi_source_v1/RUNBOOK.md)

```bash
bash experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/run_all.sh --plan
bash experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/run_all.sh --dry-run
bash experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/run_all.sh --execute
```

`matrix.yaml` is the fixed study contract. `aggregate.yaml` consumes all three
completed survey summaries and the frozen K6 control receipt. It deliberately
has no completed-MAE selection receipt. Run the aggregate alone after recovery:

```bash
python -m seis_ssl_cluster.hmm.multi_source_aggregate --config experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/aggregate.yaml --dry-run
python -m seis_ssl_cluster.hmm.multi_source_aggregate --config experiments/hmm_v2/k6810_cons010_multi_source_evaluation_v1/aggregate.yaml
```
