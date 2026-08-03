# Center-trace masked original-split screening

This experiment evaluates the frozen center-trace restoration encoder
`mh_ctmask010_nocons` on the original F3 split. It owns exactly 15 new decoder
jobs: `cap25`, `cap50`, and `cap100`, each paired over subsample seeds `0..4`
with decoder seed `42000 + subsample_seed`.

The read-only references are MAE, current K6, and hard `mh_nocons`. The shared
nearest-voxel decoder contract is retained: 384-dimensional embeddings,
hidden channels `[128, 64, 32]`, nearest upsampling, voxelwise layer norm, 50
epochs, 440 steps per epoch, balanced classes, and uniform tile sampling with
replacement. Historical multi-head, soft, and XY manifests are not modified.

The audit binds the schema-1 PASS center-trace handoff, schema-7 selected
checkpoint, unmasked embedding extraction, hard target hashes, canonical
valid-token masks, and the read-only original-split references. It does not use
masked accuracy, pretraining loss, or XY shuffle metrics as gate inputs.

## Commands

Run these commands from the repository workspace root:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$SEIS_SSL_CLUSTER_WORKSPACE/artifacts/seis_ssl_cluster"
export F3_ROOT="/path/to/F3"
export SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256=$(sha256sum \
  "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/pseudo_targets/f3/facies_benchmark_v1/strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/multi_head_target_manifest.json" \
  | cut -d' ' -f1)
export EXP_LOW=experiments/f3/facies_benchmark_v1/105_strat_hmm_multi_head_k6810_center_trace_masked_low_label_v1

python proc/seis_ssl_cluster/audit_f3_center_trace_masked_screening.py \
  --config "$EXP_LOW/00_audit_center_trace_masked_screening.yaml" --dry-run
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_screening.py \
  --config "$EXP_LOW/00_audit_center_trace_masked_screening.yaml"
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_screening.py \
  --config "$EXP_LOW/00_audit_center_trace_masked_screening.yaml" --only-missing

python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_voxel_label_budget.py \
  --config "$EXP_LOW/01_run_center_trace_masked_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_voxel_label_budget.py \
  --config "$EXP_LOW/01_run_center_trace_masked_voxel_label_budget.yaml" \
  --only-missing --device auto
python proc/seis_ssl_cluster/run_f3_lithology_center_trace_masked_voxel_label_budget.py \
  --config "$EXP_LOW/01_run_center_trace_masked_voxel_label_budget.yaml" \
  --only-missing --device auto

python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_center_trace_masked.py \
  --config "$EXP_LOW/02_summarize_center_trace_masked_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_center_trace_masked.py \
  --config "$EXP_LOW/02_summarize_center_trace_masked_voxel_label_budget.yaml"
```

The fixed primary comparison is
`mh_ctmask010_nocons - mh_nocons`. A budget is positive only when both Macro
F1 and Mean IoU have mean delta strictly above zero and at least four wins out
of five. It is negative only when both means are strictly below zero and have
at most one win. Classes 3 and 5 are guarded on F1, IoU, boundary recall at
tolerances 2 and 4; a mean delta at or below `-0.05` in the same class/metric
for two budgets is systematic degradation.

The published status is `CTMASK_ORIGINAL_GO` only for at least two positive
budgets without systematic degradation, `CTMASK_ORIGINAL_STOP` for at least
two negative budgets or degradation, and `CTMASK_ORIGINAL_HOLD` otherwise.
`six_split_follow_up.ready` is true only for GO. This issue executes no
six-split jobs; both six-split counters remain zero.

Only lightweight CSV, JSON, and Markdown evidence is published under
`results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_center_trace_masked_original_split_v1/`.
Raw predictions, checkpoints, embeddings, and logs remain under the ignored
artifact root.
