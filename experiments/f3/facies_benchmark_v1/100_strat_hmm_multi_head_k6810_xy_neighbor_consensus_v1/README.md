# F3 XY-neighbour consensus hard-label pretraining

This successor experiment creates exactly one immutable K=6/8/10 hard-target
publication from the frozen source hard manifest. It is a single synchronous
XY four-neighbour consensus pass with an ordered-trace safety guard. Training
uses only the existing hard-label multi-head data/collate/loss route.

Target export and training do not read embeddings or state posteriors, compute
affinities, update emissions, re-decode with Viterbi, sweep or calibrate beta,
refresh targets, or use facies/lithology labels or downstream metrics. The
final review only verifies extraction file digests and recorded checkpoint
identity; it never uses embedding values as target inputs.

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/100_strat_hmm_multi_head_k6810_xy_neighbor_consensus_v1

python proc/seis_ssl_cluster/export_strat_hmm_multi_head_xy_neighbor_consensus_targets.py \
  --config "$EXP/01_export_xy_neighbor_consensus_targets.yaml" --dry-run
python proc/seis_ssl_cluster/export_strat_hmm_multi_head_xy_neighbor_consensus_targets.py \
  --config "$EXP/01_export_xy_neighbor_consensus_targets.yaml"
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_consensus_pretraining.py \
  --config "$EXP/05_validate_xy_neighbor_consensus_pretraining.yaml" --phase targets
```

The smoke root is separate from the full root. Run it before the full job:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_xy_neighbor_consensus_smoke.yaml" --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/02_train_xy_neighbor_consensus_smoke.yaml" --device cpu --max-steps 2
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_consensus_pretraining.py \
  --config "$EXP/05_validate_xy_neighbor_consensus_pretraining.yaml" --phase smoke
```

Then validate the full training and extraction handoff:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config "$EXP/03_train_xy_neighbor_consensus_full.yaml"
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_consensus_pretraining.py \
  --config "$EXP/05_validate_xy_neighbor_consensus_pretraining.yaml" --phase checkpoints
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config "$EXP/04_extract_xy_neighbor_consensus_embeddings.yaml" --skip-existing
python proc/seis_ssl_cluster/validate_f3_xy_neighbor_consensus_pretraining.py \
  --config "$EXP/05_validate_xy_neighbor_consensus_pretraining.yaml" --phase complete
python proc/seis_ssl_cluster/publish_f3_xy_neighbor_consensus_results.py \
  --config "$EXP/06_review_xy_neighbor_consensus_results.yaml"
```

The source hard, M5-U posterior, M5-LS lateral, and XY-consensus checkpoint
identities are intentionally incompatible for resume.
