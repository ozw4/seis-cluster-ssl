# parihaka K6810 consistency-weight 0.1

Follow the [shared study runbook](../../../hmm_v2/cons010_multi_source_evaluation_v1/README.md)
for the fixed recipe, environment and output contract. All three source families
run new training. Existing targets are read-only inputs bound by manifest hash.
Set `CUDA_VISIBLE_DEVICES` to the GPU assigned to this study.

```bash
bash experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/run_all.sh --plan
bash experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/run_all.sh --dry-run --from smoke --to full
bash experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/run_all.sh --execute --from smoke
```

Stages: `smoke`, `full`, `audit`, `embeddings`, `downstream`, `summary`.
Completed full training and decoder cells are audited before reuse. Resume one
interrupted run from its own checkpoint:

```bash
bash experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/run_all.sh --execute --from full --to full --candidate random_init_hmm_v2_mh_k6810_distill020_cons010 --resume
bash experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/run_all.sh --execute --from downstream --to downstream --candidate random_init_hmm_v2_mh_k6810_distill020_cons010 --layout layout_000 --size small --resume
```

After all 45 cells finish, retry only a failed summary with
`--execute --from summary --to summary`. Existing summary outputs are immutable.
Logs and locks are under
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/k6810_cons010_multi_source_evaluation_v1/parihaka/logs/`.
