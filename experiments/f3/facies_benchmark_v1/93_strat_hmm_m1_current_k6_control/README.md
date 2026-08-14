# Current-code single-head K=6 artifact producers

This directory retains only the current-code K=6 smoke/full pretraining and
embedding extraction configs. They produce reusable artifacts with the
`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1` identity.

```bash
export EXP_CONTROL=experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/01_train_current_k6_smoke.yaml" \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP_CONTROL/02_train_current_k6_full.yaml" --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP_CONTROL/03_extract_current_k6_embeddings.yaml" --dry-run
```

The former token probe, voxel-count label-budget comparison, seed aggregation,
and report publication are retired.
