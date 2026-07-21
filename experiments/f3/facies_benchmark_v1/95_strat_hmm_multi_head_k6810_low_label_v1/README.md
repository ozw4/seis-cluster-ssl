# M4-MH 009: paired multi-head low-label voxel matrix

This experiment reuses the original label-budget datasets and the current K=6
control read-only. It runs only `mh_nocons` and `mh_cons010`: 3 label budgets ×
5 paired seeds each, for 30 decoder jobs. Every candidate job must share its
decoder, sampling, tile, token, and coverage identities with the paired current
K=6, original MAE, and historical M1 rows. Summary and publication are
intentionally deferred.

Each candidate's `pretraining_handoff` is a PASS
`f3_multi_head_pretraining_handoff` JSON artifact. Its best-checkpoint and
multi-head scientific identity must match the extracted embedding metadata,
whose SHA-256 must be recorded in the handoff. This rejects swapped or
unrelated #275 pretraining provenance before a decoder job is planned.

```bash
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/01_run_multi_head_voxel_label_budget.yaml --dry-run
```
