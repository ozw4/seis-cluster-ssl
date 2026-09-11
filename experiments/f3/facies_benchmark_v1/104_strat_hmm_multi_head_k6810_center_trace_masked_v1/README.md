# F3 center-trace masked pretraining

The scientific question and comparison boundary are recorded in
[`docs/f3_center_trace_masked_pretraining_plan.md`](../../../../docs/f3_center_trace_masked_pretraining_plan.md).
The YAML and resolver define settings, identities, and artifacts; the validator
enforces those contracts.

Set the artifact root and immutable target-manifest digest, then validate
inputs, run and validate the isolated smoke, train the full condition, extract
embeddings, and validate completion in that order.

```bash
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the artifact root}"
export EXP=experiments/f3/facies_benchmark_v1/104_strat_hmm_multi_head_k6810_center_trace_masked_v1
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  | awk '{print $1}')"

python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" --phase inputs
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/01_train_center_trace_masked_smoke.yaml" \
  --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" --phase smoke

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_center_trace_masked_full.yaml"
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" \
  --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/03_extract_center_trace_masked_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_center_trace_masked_pretraining.py \
  --config "$EXP/04_validate_center_trace_masked_pretraining.yaml" --phase complete
```

Resume a full run only from its own checkpoint. Use the validator's
`--quarantine-invalid` option before replacing a foreign or invalid smoke or
handoff artifact.
