# F3 soft-posterior pretraining

This condition replaces the matched hard HMM targets with ordered-path state
posteriors while keeping the rest of the pretraining comparison fixed. Exact
settings, identity hashes, and output paths belong to the YAML and validators.
Results are recorded only in the
[`soft_posterior_results_summary.md`](../../../../reports/f3/legacy/facies_benchmark_v1/strat_hmm_multi_head_k6810_soft_posterior_v1/soft_posterior_results_summary.md)
report.

## Execution order

Export the posterior artifact first. Populate the hash variables referenced by
the training YAML from the completed handoff, then validate targets before
starting the isolated smoke or full run.

```bash
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the artifact root}"
export EXP=experiments/f3/facies_benchmark_v1/97_strat_hmm_multi_head_k6810_soft_posterior_v1

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_state_posteriors.py \
  --config "$EXP/01_export_posteriors.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_state_posteriors.py \
  --config "$EXP/01_export_posteriors.yaml" --only-missing

python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase targets
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_soft_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase smoke

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/03_train_soft_full.yaml"
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/04_extract_soft_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase complete
```

The smoke and full outputs are independent. Resume a full run only from its own
checkpoint; use the validator's quarantine option before replacing invalid
prior evidence.
