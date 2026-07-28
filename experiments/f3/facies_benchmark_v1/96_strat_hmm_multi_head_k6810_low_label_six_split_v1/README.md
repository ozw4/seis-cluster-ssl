# M4 selected multi-head six-split low-label confirmation

This suite fixes `mh_nocons`, the current K6 control, and MAE over six existing
F3 splits and cap25/cap50. It never reselects K, consistency, or a model.

1. Validate inputs and plan the 12 datasets:

   `python proc/seis_ssl_cluster/build_f3_lithology_voxel_label_budget_split_datasets.py --config experiments/f3/facies_benchmark_v1/96_strat_hmm_multi_head_k6810_low_label_six_split_v1/01_build_low_label_split_datasets.yaml --dry-run`

2. Build/revalidate datasets and the split_000 parity gate:

   `python proc/seis_ssl_cluster/build_f3_lithology_voxel_label_budget_split_datasets.py --config experiments/f3/facies_benchmark_v1/96_strat_hmm_multi_head_k6810_low_label_six_split_v1/01_build_low_label_split_datasets.yaml --only-missing`

3. Validate the exact 36-job provenance plan:

   `python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_split_suite.py --config experiments/f3/facies_benchmark_v1/96_strat_hmm_multi_head_k6810_low_label_six_split_v1/02_run_low_label_split_decoders.yaml --dry-run`

4. Before the full matrix, run the non-scientific CPU two-step smoke triplet for
   `split_000/cap25`; full execution must use `--only-missing`, then revalidate
   36/36 and summarize only completed jobs. Raw checkpoints, arrays, and
   predictions must not be published.

## Completed result

- 36/36 jobs completed.
- Formal status: `M4_MH_SPLIT_HOLD`.
- Systematic major degradation: `false`.

## Interpretation

- cap25 is robust across the six splits.
- cap50 is split-dependent.
- The comparison with MAE is positive.
- The original split overestimated the multi-head incremental effect.

The formal HOLD is retained: these results do not establish robust superiority
of `mh_nocons` across all splits.

## Project decision

- `ADOPT_MH_NOCONS_FOR_M5`: adopt `mh_nocons` as the M5 hard-target baseline.
- Additional decoder seeds are not a required gate for M5.
- Do not carry `mh_cons010` forward as a primary candidate.

This adoption is a project development decision; it does not reinterpret the
formal `M4_MH_SPLIT_HOLD` result as `CONFIRMED`.

## Next stage

Proceed to [M5-U — Posterior-aware soft multi-resolution HMM
pretraining](../../../../docs/f3_m5_soft_posterior_pretraining_plan.md).
