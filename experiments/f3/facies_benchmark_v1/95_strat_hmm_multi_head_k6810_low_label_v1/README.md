# M4-MH 009: paired multi-head low-label voxel matrix

This experiment reuses the original label-budget datasets and the current K=6
control read-only. It runs only `mh_nocons` and `mh_cons010`: 3 label budgets ×
5 paired seeds each, for 30 decoder jobs. Every candidate job must share its
decoder, sampling, tile, token, and coverage identities with the paired current
K=6 and original MAE rows. A complete, pairing-valid historical M1 source is
reported only as an optional reference. The summary independently
recomputes paired deltas, applies the fixed effect gates, and publishes only
lightweight report files; it never selects an individual K or trains per-head
downstream models.

Each candidate's `pretraining_handoff` is a PASS
`f3_multi_head_pretraining_handoff` JSON artifact. Its best-checkpoint and
multi-head scientific identity must match the extracted embedding metadata,
whose SHA-256 must be recorded in the handoff. This rejects swapped or
unrelated #275 pretraining provenance before a decoder job is planned.

```bash
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/01_run_multi_head_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/01_run_multi_head_voxel_label_budget.yaml --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/02_summarize_multi_head_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1/02_summarize_multi_head_voxel_label_budget.yaml
```

`--only-missing` reuses complete rows, resumes valid incomplete rows, and
quarantines invalid partial outputs. If the source or identity contract is
incomplete, the summary is blocked rather than substituting a report or raw
artifact.
