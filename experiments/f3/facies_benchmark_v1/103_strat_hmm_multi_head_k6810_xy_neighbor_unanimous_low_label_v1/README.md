# F3 unanimous XY-neighbour original-split screening

This root runs exactly 15 new original-split decoder jobs: cap25, cap50, and
cap100 across subsample seeds 0–4. It reuses MAE, current K=6, hard
`mh_nocons`, and the existing 3-of-4 `mh_xycons1_nocons` rows; the 3-of-4
comparison is diagnostic only.

Run from the workspace root after schema-6 complete validation and lightweight
pretraining publication:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$SEIS_SSL_CLUSTER_WORKSPACE/artifacts/seis_ssl_cluster"
export F3_ROOT="/path/to/F3"
export EXP_LOW="experiments/f3/facies_benchmark_v1/103_strat_hmm_multi_head_k6810_xy_neighbor_unanimous_low_label_v1"
```

The clean screening audit binds the unanimous target audit, schema-6 handoff,
checkpoint, embeddings, hard-baseline parity, and all read-only reference run
manifests. Its final `--only-missing` pass must reuse unchanged evidence.

```bash
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_screening.py --config "$EXP_LOW/00_audit_xy_neighbor_unanimous_screening.yaml" --dry-run
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_screening.py --config "$EXP_LOW/00_audit_xy_neighbor_unanimous_screening.yaml"
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_screening.py --config "$EXP_LOW/00_audit_xy_neighbor_unanimous_screening.yaml" --only-missing
```

Inspect the candidate-only plan, execute the 15 missing jobs, and run the same
command again. The second pass must classify every new job as
`REUSE_COMPLETED`; reference jobs must never execute.

```bash
python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_unanimous_voxel_label_budget.py --config "$EXP_LOW/01_run_xy_neighbor_unanimous_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_unanimous_voxel_label_budget.py --config "$EXP_LOW/01_run_xy_neighbor_unanimous_voxel_label_budget.yaml" --only-missing --device auto
python proc/seis_ssl_cluster/run_f3_lithology_xy_neighbor_unanimous_voxel_label_budget.py --config "$EXP_LOW/01_run_xy_neighbor_unanimous_voxel_label_budget.yaml" --only-missing --device auto
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous.py --config "$EXP_LOW/02_summarize_xy_neighbor_unanimous_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous.py --config "$EXP_LOW/02_summarize_xy_neighbor_unanimous_voxel_label_budget.yaml"
```

The fixed original-split gate may make six-split work ready only on
`XYUNANIM_ORIGINAL_GO`. This experiment always stops after publication and
executes zero six-split jobs.
