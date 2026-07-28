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

After the target, pretraining, checkpoint/freeze/initialization validation, and
two embedding extractions in experiment 94 have passed, use this ordered
downstream sequence:

```bash
export EXP=experiments/f3/facies_benchmark_v1/95_strat_hmm_multi_head_k6810_low_label_v1

# 30-job plan and pre-run identity gate.
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config "$EXP/01_run_multi_head_voxel_label_budget.yaml" --dry-run

# Run or revalidate the complete 2 candidates × 3 budgets × 5 seeds matrix.
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config "$EXP/01_run_multi_head_voxel_label_budget.yaml" --only-missing

# A completed rerun must show all 30 rows as REUSED without training.
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py --config "$EXP/01_run_multi_head_voxel_label_budget.yaml" --only-missing

# Recompute paired metrics and scientific decisions, then write/publish reports.
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config "$EXP/02_summarize_multi_head_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_multi_head.py --config "$EXP/02_summarize_multi_head_voxel_label_budget.yaml"

# Validate the exact lightweight publish tree and 10 MiB ceiling.
python proc/seis_ssl_cluster/validate_results_artifacts.py --root results --max-file-size-mb 10
```

`--only-missing` reuses complete rows, resumes valid incomplete rows, and
quarantines invalid partial outputs. If the source or identity contract is
incomplete, the summary is blocked rather than substituting a report or raw
artifact.

To restart one failed job with its own validated `latest.pt`, select exactly
that job and use `--resume`:

```bash
python proc/seis_ssl_cluster/run_f3_lithology_multi_head_voxel_label_budget.py \
  --config "$EXP/01_run_multi_head_voxel_label_budget.yaml" \
  --candidate mh_cons010 --budget cap50 --subsample-seed 3 --resume
```

Never use `--resume` for a partial or identity-invalid job: invoke
`--only-missing` instead so it is timestamp-quarantined and restarted cleanly.

## Completed result

- Original-split overall decision: `M4_MH_GO_NOCONS`.
- Multi-task value: `POSITIVE`.
- Consistency increment: `HOLD`.
- Selected candidate: `mh_nocons`.

`mh_cons010` is not the selected primary candidate.

## Confirmatory follow-up

- The six-split formal result is `M4_MH_SPLIT_HOLD`.
- The original split overestimated the cap50 multi-head incremental effect.
- The project decision is `ADOPT_MH_NOCONS_FOR_M5`: adopt `mh_nocons` as the
  hard-target baseline for M5 without carrying the consistency model forward as
  a primary candidate.
