# K468 consistency-weight 0.1 study

The [shared runbook](../cons010_multi_source_evaluation_v1/README.md) defines
scientific controls, environment setup, GPU0/GPU1 launch, outputs and restart.
This study runs nine new 25-epoch continuations and 135 downstream cells.
The unchanged inputs come from `k468_multi_source_evaluation_v1`, including
the F3 experiment 127 MAE source; every new candidate has consistency weight
0.1 and a `_cons010` suffix. No completed F3 model is reused.

- [f3](../../f3/facies_benchmark_v2/130_hmm_v2_k468_cons010_multi_source_v1/RUNBOOK.md)
- [parihaka](../../parihaka/facies_benchmark_v1/43_hmm_v2_k468_cons010_multi_source_v1/RUNBOOK.md)
- [volve](../../volve/horizon_benchmark_v1/36_hmm_v2_k468_cons010_multi_source_v1/RUNBOOK.md)

```bash
bash experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/run_all.sh --plan
bash experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/run_all.sh --dry-run
bash experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/run_all.sh --execute
```

`matrix.yaml` is the fixed study contract. `aggregate.yaml` consumes all three
completed survey summaries and the frozen K6 control receipt. It deliberately
has no completed-MAE selection receipt. Run the aggregate alone after recovery:

```bash
python -m seis_ssl_cluster.hmm.multi_source_aggregate --config experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/aggregate.yaml --dry-run
python -m seis_ssl_cluster.hmm.multi_source_aggregate --config experiments/hmm_v2/k468_cons010_multi_source_evaluation_v1/aggregate.yaml
```
