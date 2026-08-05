# F3 periodic-refresh original-split low-label screen

This experiment evaluates the PASSed periodic-refresh encoder on the original
F3 split. It owns exactly one candidate, three budgets, five subsample seeds,
and 15 possible scientific decoder jobs. The fixed center-trace, hard
`mh_nocons`, current-K6, and MAE outputs are read-only references.

Set the repository paths before resolving the configs:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE=/workspace
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/workspace/artifacts/seis_ssl_cluster
export F3_ROOT=/home/dcuser/data/public_data/field/F3
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256="$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT"/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json \
  | awk '{print $1}')"
export EXP=experiments/f3/facies_benchmark_v1/108_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_low_label_v1
```

Run the stages in this order. The first audit command is read-only; the
write/reuse and decoder commands are the authorized follow-up commands for a
completed artifact set and are not run as part of this issue.

```bash
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_periodic_refresh_screening.py \
  --config "$EXP/00_audit_periodic_refresh_screening.yaml" --dry-run
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_periodic_refresh_screening.py \
  --config "$EXP/00_audit_periodic_refresh_screening.yaml" --only-missing

python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget.py \
  --config "$EXP/01_run_periodic_refresh_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget.py \
  --config "$EXP/01_run_periodic_refresh_voxel_label_budget.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget.py \
  --config "$EXP/01_run_periodic_refresh_voxel_label_budget.yaml" --resume
python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_periodic_refresh_voxel_label_budget.py \
  --config "$EXP/01_run_periodic_refresh_voxel_label_budget.yaml" --only-missing

python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh.py \
  --config "$EXP/02_summarize_periodic_refresh_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_center_trace_masked_periodic_refresh.py \
  --config "$EXP/02_summarize_periodic_refresh_voxel_label_budget.yaml"
```

`--resume` is valid only when every selected owned job has a valid latest
checkpoint with the same identity. `--only-missing` reuses complete jobs and
quarantines partial or invalid owned outputs only when the runner explicitly
classifies them for recovery. Historical manifests and results are never
written by this route.

The summary publishes the 75-row matrix, the periodic-refresh-versus-fixed
center-trace primary comparison, the three diagnostics, and the formal
`CTMASK_REFRESH_ORIGINAL_GO`, `CTMASK_REFRESH_ORIGINAL_HOLD`, or
`CTMASK_REFRESH_ORIGINAL_STOP` handoff. `six_split_jobs_executed` and
`six_split_scientific_jobs_executed` are recorded as zero for every status;
only GO sets `six_split_follow_up.ready` to true.

This issue runs no real decoder jobs and adds no six-split implementation.
