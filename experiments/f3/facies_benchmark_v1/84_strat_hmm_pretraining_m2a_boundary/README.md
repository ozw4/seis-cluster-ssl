# F3 Strat-HMM M2-A boundary-weighted pretraining producers

This experiment changes only the pseudo-target boundary weight from the M1
pretraining condition. The fixed candidate uses `alpha=0.5` and `tau=2.0`
tokens; no parameter sweep is part of this stage.

Run the retained producer stages in order:

```bash
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/01_export_alpha0_parity_bootstrap.sh --dry-run
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/02_export_boundary_weighted_bootstrap.sh --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/03_train_boundary_smoke.yaml \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/04_train_boundary_full.yaml \
  --dry-run

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/05_extract_student_embeddings.yaml
```

Complete outputs remain under `artifacts/seis_ssl_cluster/`. This directory
contains only target export, pretraining, and embedding extraction stages.
