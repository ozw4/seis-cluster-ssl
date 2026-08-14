# F3 Strat-HMM pretraining milestone 1 producers

This experiment retains the target-generation, pretraining, embedding
extraction, and smoke-validation stages for the single-head K=6 structured
pretext model. HMM labels are pseudo-targets, not final lithology labels or
evaluation output.

Run the producer stages in order:

```bash
bash experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/01_export_bootstrap_hmm_pseudo_targets.sh

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/02_train_single_head_topblock_distill_smoke.yaml \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/03_train_single_head_topblock_distill_full.yaml \
  --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/04_extract_student_embeddings.yaml

python proc/seis_ssl_cluster/build_strat_hmm_pseudo_targets.py \
  --config experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/08_refresh_pseudo_targets_from_logits_smoke.yaml \
  --dry-run --device cpu --overwrite
```

Complete outputs remain under `artifacts/seis_ssl_cluster/`. No active
downstream lithology probe, voxel-count budget, robustness, result summary, or
report publication belongs to this directory.
