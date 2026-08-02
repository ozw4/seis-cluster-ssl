# F3 unanimous XY-neighbour hard-label successor

This experiment applies one synchronous source-label correction only where all
valid same-z XY neighbours agree on a different label: 4/4 with four valid
neighbours and 3/3 with three. It is independent of the existing 3-of-4
publication and uses schema-6 checkpoints.

Set the execution environment from the workspace root:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$SEIS_SSL_CLUSTER_WORKSPACE/artifacts/seis_ssl_cluster"
export F3_ROOT="/path/to/F3"
export EXP_UNANIM="experiments/f3/facies_benchmark_v1/102_strat_hmm_multi_head_k6810_xy_neighbor_unanimous_v1"
```

Publish and audit the immutable target first. The final `--only-missing` pass
must report exact reuse. Stop here if the audit status is
`XYUNANIM_TARGET_HOLD`.

```bash
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/01_export_xy_neighbor_unanimous_targets.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/01_export_xy_neighbor_unanimous_targets.yaml"
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/01_export_xy_neighbor_unanimous_targets.yaml" --only-missing
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/02_audit_xy_neighbor_unanimous_targets.yaml" --dry-run
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/02_audit_xy_neighbor_unanimous_targets.yaml"
python proc/seis_ssl_cluster/audit_f3_xy_neighbor_unanimous_targets.py --config "$EXP_UNANIM/02_audit_xy_neighbor_unanimous_targets.yaml" --only-missing
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_unanimous_pretraining.py --config "$EXP_UNANIM/06_validate_xy_neighbor_unanimous_pretraining.yaml" --phase targets
```

On `XYUNANIM_TARGET_GO`, run the isolated two-step CPU smoke and stop before
full training if smoke validation fails:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP_UNANIM/03_train_xy_neighbor_unanimous_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP_UNANIM/03_train_xy_neighbor_unanimous_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_unanimous_pretraining.py --config "$EXP_UNANIM/06_validate_xy_neighbor_unanimous_pretraining.yaml" --phase smoke
```

Run the fixed 25-epoch pretraining, extract from the selected `best.pt`, then
publish only lightweight review evidence:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP_UNANIM/04_train_xy_neighbor_unanimous_full.yaml" --dry-run
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py --config "$EXP_UNANIM/04_train_xy_neighbor_unanimous_full.yaml"
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_unanimous_pretraining.py --config "$EXP_UNANIM/06_validate_xy_neighbor_unanimous_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py --config "$EXP_UNANIM/05_extract_xy_neighbor_unanimous_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_unanimous_pretraining.py --config "$EXP_UNANIM/06_validate_xy_neighbor_unanimous_pretraining.yaml" --phase complete
python proc/seis_ssl_cluster/publish_f3_xy_neighbor_unanimous_results.py --config "$EXP_UNANIM/07_review_xy_neighbor_unanimous_results.yaml" --dry-run
python proc/seis_ssl_cluster/publish_f3_xy_neighbor_unanimous_results.py --config "$EXP_UNANIM/07_review_xy_neighbor_unanimous_results.yaml"
```

No six-split decoder job belongs to this root.
