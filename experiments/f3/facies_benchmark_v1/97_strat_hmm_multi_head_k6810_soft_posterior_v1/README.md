# F3 M5-U soft-posterior pretraining

M5-U changes only M4 `mh_nocons` target representation: hard Viterbi labels
become exact ordered-path state posteriors. K=6/8/10, model and heads,
initialization, data order, crop/AGC/zero mask, optimization, unfreeze depth,
and loss weights remain fixed. Consistency remains disabled.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/97_strat_hmm_multi_head_k6810_soft_posterior_v1

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_state_posteriors.py \
  --config "$EXP/01_export_posteriors.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_state_posteriors.py \
  --config "$EXP/01_export_posteriors.yaml" --only-missing
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_state_posteriors.py \
  --config "$EXP/01_export_posteriors.yaml" --dry-run --only-missing

export SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_state_posteriors/multi_head_state_posterior_handoff.json" \
  | awk '{print $1}')"
eval "$(python - "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_state_posteriors/multi_head_state_posterior_handoff.json" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding='utf-8').read())
for k in (6, 8, 10):
    survey = manifest['heads'][str(k)]['surveys']['f3_facies_benchmark']
    for name in ('posterior', 'valid_tokens', 'metadata'):
        print(
            f"export SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_K{k}_{name.upper()}_SHA256="
            f"{survey[name]['sha256']}"
        )
PY
)"
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  | awk '{print $1}')"

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_soft_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_soft_smoke.yaml" --device cpu --max-steps 2
```

The exporter validates K6/K8/K10, exact common valid masks, source hashes,
Viterbi replay, and posterior structure before publishing its manifest. A
second `--only-missing` pass must report `REUSE` for all heads.

After the smoke, validate targets and the isolated two-step checkpoint before the full run:

```bash
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase targets
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase smoke

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/03_train_soft_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/03_train_soft_full.yaml"
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/04_extract_soft_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_m5_soft_posterior_pretraining.py \
  --config "$EXP/05_validate_soft_pretraining.yaml" --phase complete
```

The smoke root is separate from the full root and cannot be resumed by the
full job. The final phase atomically publishes
`preflight/soft_posterior_handoff.json`; use `--quarantine-invalid` only to
preserve and replace stale or partial prior evidence. Do not mix temperature
sweeps, hard/soft interpolation, smoothing, boundary auxiliaries, refreshes,
or six-split jobs into this initial run.
