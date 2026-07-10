# F3 Strat-HMM M2-A result decision

`summarize_f3_strat_hmm_m2_results.py` consolidates the original full split,
label-budget, split/index, and monitored-class evidence for M2-A versus M1.
The YAML config explicitly identifies both model tags, both full-split metrics
JSON files, the paired robustness suite roots, and monitored class IDs.
The label-budget root must contain `suite_manifest.json`; the split/index root
must contain `split_dataset_manifest.json`. Their condition inventories and
baseline/candidate model tags must exactly match the paired-delta CSVs.

Run:

```bash
PYTHONPATH=src python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m2_results.py \
  --config experiments/f3/facies_benchmark_v1/86_strat_hmm_m2a_results/01_summarize_m2a_results.yaml
```

The summary JSON records every Go and Stop predicate plus stable reason codes.
Go requires positive cap25/cap50/cap100 macro-F1 and mean-IoU means, a strict
majority of joint split wins, nonnegative full-split balanced-accuracy delta,
and a Pareto improvement for at least one monitored class. Explicitly negative
robustness evidence produces Stop; all other complete evidence produces Hold.
Missing evidence or a model identity mismatch is an error and produces no
decision report.

Only the summary Markdown/JSON, four CSV tables, and optionally four PNG
figures are published. The common results publisher enforces its suffix
allowlist, forbidden artifact types, and configured size limit.
