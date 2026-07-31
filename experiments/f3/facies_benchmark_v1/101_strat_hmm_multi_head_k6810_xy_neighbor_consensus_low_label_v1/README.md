# F3 XY-neighbour consensus original-split low-label screening

This experiment evaluates the frozen `mh_xycons1_nocons` checkpoint with
facies labels on the original split. Exactly 15 XY-consensus candidate decoder
jobs are new: three label budgets (`cap25`, `cap50`, `cap100`) by five paired
subsample seeds. The MAE, current-K6, and hard `mh_nocons` reference jobs are
read-only reused evidence. Hard `mh_nocons` is the primary baseline.

The XY spatial audit is descriptive evidence only, not a retroactive target
acceptance gate. Its temporal transition count is also descriptive only.
Target generation, pretraining, and embedding extraction are not rerun.

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$SEIS_SSL_CLUSTER_WORKSPACE/artifacts/seis_ssl_cluster"
export F3_ROOT="/path/to/F3"
export EXP=experiments/f3/facies_benchmark_v1/101_strat_hmm_multi_head_k6810_xy_neighbor_consensus_low_label_v1

python proc/seis_ssl_cluster/audit_f3_xy_neighbor_consensus_screening.py \
  --config "$EXP/00_audit_xy_neighbor_consensus_screening.yaml" --dry-run
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_consensus_screening.py \
  --config "$EXP/00_audit_xy_neighbor_consensus_screening.yaml"
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_consensus_screening.py \
  --config "$EXP/00_audit_xy_neighbor_consensus_screening.yaml" --only-missing

python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_consensus_voxel_label_budget.py \
  --config "$EXP/01_run_xy_neighbor_consensus_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_consensus_voxel_label_budget.py \
  --config "$EXP/01_run_xy_neighbor_consensus_voxel_label_budget.yaml" \
  --only-missing --device auto
python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_consensus_voxel_label_budget.py \
  --config "$EXP/01_run_xy_neighbor_consensus_voxel_label_budget.yaml" \
  --only-missing --device auto

python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus.py \
  --config "$EXP/02_summarize_xy_neighbor_consensus_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus.py \
  --config "$EXP/02_summarize_xy_neighbor_consensus_voxel_label_budget.yaml"
```

The stop rule is fixed before execution. A budget is positive only when both
Macro F1 and Mean IoU have positive mean paired deltas and at least four wins
of five; it is negative only when both have negative mean deltas and at most
one win. `XYCONS_ORIGINAL_GO` requires at least two positive budgets and no
systematic class-3/5 major degradation. `XYCONS_ORIGINAL_STOP` follows from at
least two negative budgets or a systematic major degradation; otherwise the
status is `XYCONS_ORIGINAL_HOLD`.

Only `XYCONS_ORIGINAL_GO` makes a six-split follow-up ready. Six-split jobs are
not executed in this issue.
