# K=6/8/10 HMM multi-head target bundle

This stage replays the fixed MAE embedding condition once for K=6/8/10. The
prepared feature cache is shared by the HMM backend; K is the only changed
scientific variable. The immutable historical K=6 target remains the training
reference. The replayed K=6 output is parity evidence only.

The subsequent multi-head pretext configurations must use the explicit
`multi_resolution_ordered_prototypes_v1` head spec, the generated manifest,
and a scientific identity whose `target_manifest_sha256` matches that exact
manifest file. The no-consistency and main runs differ only in
`loss.consistency_weight` (0.0 and 0.1 respectively) and their model/output
identity. Prototype and usage weights apply to the mean across heads, while
the consistency weight applies to the mean over head pairs. K=6/8/10 denotes
pretext ordered-state cardinality; downstream F3 lithology `class_count`
remains 6.

The resolved scientific identity binds the head projection/temperature/
normalization, all loss weights, teacher/student initialization, and effective
model, preprocessing, and scientific training settings. Device, workers,
timing, cache location, and resume path remain runtime-only identity.

Set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` to the absolute artifact-registry root for
the current environment before running this stage. The replay config resolves
that variable at load time.

First run the replay config, then export K=6, K=8, and K=10 as schema-v1,
bootstrap-semantics pseudo-targets with no boundary-weight artifact. The K=6
export is replay parity evidence only; the historical K=6 target remains the
manifest's training reference:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/absolute/path/to/artifacts/seis_ssl_cluster

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1/01_replay_hmm_k6810.yaml --dry-run

python proc/seis_ssl_cluster/export_strat_hmm_pseudo_targets.py \
  --clustering-output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/clustering/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --pseudo-target-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --k 6 --confidence 1.0 --schema-version 1 --no-boundary-weight
```

Repeat the export for K=8 and K=10. Publish the manifest only after both required
preflight decisions are positive. `--only-missing` revalidates a complete
manifest only when its embedding, head roots, and replay root match the requested
inputs; `--quarantine-invalid` moves an invalid prior manifest aside.

```bash
python proc/seis_ssl_cluster/build_strat_hmm_multi_head_targets.py \
  --source-embedding-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16" \
  --head-root "6=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap" \
  --head-root "8=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --head-root "10=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --replay-k6-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --migration-decision "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/migration_validation/f3/facies_benchmark_v1/main_332478be/performance_migration_decision.json" \
  --control-summary "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/lithology/f3/facies_benchmark_v1/voxel_label_budget_current_k6_control_v1/original_split/reports/current_k6_control_summary.json" \
  --manifest "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  --only-missing --quarantine-invalid
```

The paired pretext configs are `02_train_multi_head_no_consistency.yaml` and
`03_train_multi_head_consistency.yaml`. Before either is resolved, bind its
identity to the generated manifest's actual digest; the resolver rejects a
stale or incorrect value.

```bash
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  | awk '{print $1}')"

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1/02_train_multi_head_no_consistency.yaml --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1/03_train_multi_head_consistency.yaml --dry-run
```

The configs are intentionally paired: their only scientific setting difference
is `loss.consistency_weight` (`0.0` versus `0.1`); their no-consistency and
main model/output identities are distinct. Both keep the downstream lithology
`class_count` at 6.
