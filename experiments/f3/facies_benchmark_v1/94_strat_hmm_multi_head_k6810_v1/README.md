# F3 K=6/8/10 multi-head pretraining

This stage trains two otherwise identical ordered-prototype encoders using the
same F3 inputs, MAE initialization, and K=6/8/10 target manifest:

- `strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1` uses
  `consistency_weight: 0.0`.
- `strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1` uses
  `consistency_weight: 0.1`.

The current-code single-head control,
`strat_hmm_pretext_m1_current_k6_topblock1_distill_v1`, remains the primary
baseline. This stage does not run voxel-decoder evaluation or tune the
consistency weight.

`01_export_multi_head_pseudo_targets.yaml` is the canonical, strict K=6/8/10
export input. It creates the replay K=6 target separately from the immutable
historical K=6 training target and verifies that the common target-valid mask
is a subset of the source embedding mask. Its `clustering_config` path and
hash are recorded with each complete export handoff alongside per-K clustering
metadata, per-K pseudo-target roots and hash sets, and the prepared-feature
identity.
`01_build_multi_head_targets.yaml` supplies the manifest publication paths
after replay parity has passed. New manifests use schema v2 to record that
subset evidence; legacy schema-v1 manifests remain loadable only when their
source and target valid-token masks are exactly equal.

Run the target and pretraining stages in this order. The build command both
checks the K=6 replay parity and publishes the immutable K=6/8/10 manifest;
the following load check revalidates its hashes before either training config
is accepted.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1

python proc/seis_ssl_cluster/cluster_embeddings.py --config "$EXP/01_replay_hmm_k6810.yaml" --dry-run
python proc/seis_ssl_cluster/cluster_embeddings.py --config "$EXP/01_replay_hmm_k6810.yaml"

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_pseudo_targets.py \
  --config "$EXP/01_export_multi_head_pseudo_targets.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_pseudo_targets.py \
  --config "$EXP/01_export_multi_head_pseudo_targets.yaml" --only-missing

# Revalidate the complete schema-v1 pseudo-target bundle (including source
# hashes) without writing arrays before the K=6 parity and manifest-publication
# stage.
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_pseudo_targets.py \
  --config "$EXP/01_export_multi_head_pseudo_targets.yaml" --only-missing --dry-run

python proc/seis_ssl_cluster/build_strat_hmm_multi_head_targets.py \
  --source-embedding-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16" \
  --head-root "6=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap" \
  --head-root "8=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --head-root "10=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --replay-k6-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --migration-decision "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/migration_validation/f3/facies_benchmark_v1/main_332478be/reports/performance_migration_decision.json" \
  --control-summary "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/lithology/f3/facies_benchmark_v1/voxel_label_budget_current_k6_control_v1/original_split/reports/current_k6_control_summary.json" \
  --manifest "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  --dry-run

python proc/seis_ssl_cluster/build_strat_hmm_multi_head_targets.py \
  --source-embedding-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/embeddings/f3/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16" \
  --head-root "6=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap" \
  --head-root "8=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --head-root "10=$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --replay-k6-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1" \
  --migration-decision "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/migration_validation/f3/facies_benchmark_v1/main_332478be/reports/performance_migration_decision.json" \
  --control-summary "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/lithology/f3/facies_benchmark_v1/voxel_label_budget_current_k6_control_v1/original_split/reports/current_k6_control_summary.json" \
  --manifest "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  --only-missing

export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json \
  | awk '{print $1}')"

python -c "from seis_ssl_cluster.stratigraphy import load_multi_head_target_manifest; load_multi_head_target_manifest('$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json'); print('target manifest: PASS')"

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/02_train_nocons_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/02_train_nocons_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/03_train_cons010_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/03_train_cons010_smoke.yaml" --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/04_train_nocons_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/04_train_nocons_full.yaml"
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/05_train_cons010_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP/05_train_cons010_full.yaml"

# Validate checkpoint, freeze, initialization, and paired scientific identity
# before extracting either embedding. Embedding evidence is intentionally
# deferred until both extraction commands complete.
python proc/seis_ssl_cluster/validate_f3_multi_head_pretraining.py \
  --config "$EXP/08_validate_multi_head_runs.yaml" --phase checkpoints

python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/06_extract_nocons_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/06_extract_nocons_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/07_extract_cons010_embeddings.yaml" --dry-run
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP/07_extract_cons010_embeddings.yaml" --skip-existing

# Require the extracted-array and metadata bindings before downstream planning.
python proc/seis_ssl_cluster/validate_f3_multi_head_pretraining.py \
  --config "$EXP/08_validate_multi_head_runs.yaml" --phase complete

# If a prior handoff is stale or partial, preserve it under a timestamped
# quarantine name before publishing new PASS evidence.
python proc/seis_ssl_cluster/validate_f3_multi_head_pretraining.py \
  --config "$EXP/08_validate_multi_head_runs.yaml" --phase complete --quarantine-invalid

# Then run/reuse the 30 downstream jobs and aggregate them:
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/01_run_multi_head_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/01_run_multi_head_voxel_label_budget.yaml --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/02_summarize_multi_head_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/02_summarize_multi_head_voxel_label_budget.yaml
python proc/seis_ssl_cluster/validate_results_artifacts.py --root results --max-file-size-mb 10
```

Smoke output roots are intentionally separate and must never be resumed by a
full run. A full run may resume only from its own `latest.pt`; the checkpoint
identity rejects cross-variant resumes. The four scientific differences between
the full configs are `loss.consistency_weight`, `identity.model_tag`,
`identity.scientific_identity.variant`, and `paths.output_root`.

Each full-run checkpoint root also records the canonical, versioned rolling
selection state in `latest.pt`, with derived
`checkpoint_selection_history.csv` and `checkpoint_selection_summary.json`
for inspection. Both 500-step and epoch-end saves are selection candidates;
therefore checkpoint validation can correctly select a step `best.pt` even
when the final `latest.pt` is an epoch checkpoint. Resume restores and
continues this history without duplicate events. Keep the required order:
full pretraining, checkpoint validation, embedding extraction, then complete
validation.

For a failed downstream decoder job, use the selected output's valid
`latest.pt` only through the runner's explicit restart command in experiment
95: `--candidate <id> --budget <id> --subsample-seed <n> --resume`. Invalid or
partial outputs instead require `--only-missing`, which quarantines them before
starting a fresh job.

## Completed status

- Target bundle: complete.
- K6 replay parity: exact.
- Pretraining: `mh_nocons` complete and `mh_cons010` complete.
- Embeddings: 2/2 complete.
- Original-split decision: `M4_MH_GO_NOCONS`.
- Six-split formal result: `M4_MH_SPLIT_HOLD`.
- Project decision: `ADOPT_MH_NOCONS_FOR_M5` (adopt `mh_nocons` for M5).
- Next milestone: M5-U posterior-aware soft multi-resolution HMM pretraining.

The 2026-07-20 blocked preflight is a historical archive only, retained under
`results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_v1_historical_preflight_20260720/`.
It is not the current execution status.
