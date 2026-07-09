# F3 Strat-HMM M1 Results

This page documents the lightweight milestone-1 summary artifacts for the F3
strat-HMM pretext experiment. HMM maps are structured pretext labels for encoder
adaptation; they are not final task output.

## Regenerate

Regenerate the summary from existing local artifacts:

```bash
python proc/seis_ssl_cluster/summarize_f3_strat_hmm_m1_results.py \
  --config experiments/f3/facies_benchmark_v1/82_strat_hmm_m1_results/01_summarize_m1_results.yaml
```

The command reads:

- the single-run baseline comparison CSV
- `label_budget_m1_v1/reports/paired_deltas.csv`
- `split_index_m1_v1/reports/split_paired_deltas.csv`

It writes the complete local report under the configured artifact
`outputs.output_dir`:

```text
m1_results_summary.md
m1_results_summary.json
figures/
tables/
```

## Publish

The summary config can publish lightweight review files into `results/`:

```yaml
publish:
  enabled: true
  output_dir: results/f3/facies_benchmark_v1/strat_hmm_pretext_m1
  include_figures: true
  max_file_size_mb: 10
```

Publishing copies only Markdown, JSON, CSV tables, and selected PNG figures. It
refuses prohibited suffixes such as `.pt`, `.npy`, `.npz`, and `.joblib`, and it
refuses files larger than `publish.max_file_size_mb`.

Expected review files:

```text
results/f3/facies_benchmark_v1/strat_hmm_pretext_m1/
  m1_results_summary.md
  m1_results_summary.json
  figures/label_budget_delta_curves.png
  figures/split_index_deltas.png
  figures/single_run_metric_comparison.png
  tables/single_split_comparison.csv
  tables/label_budget_summary.csv
  tables/split_index_deltas.csv
```

## Decision

`go` means the strat-HMM M1 candidate is positive on the single-run macro F1 and
mean IoU deltas, label-budget robustness, and split/index macro F1 and mean IoU
deltas.

`hold` means at least one required evidence layer is mixed or negative. Inspect
the generated JSON, tables, and warnings before publishing or expanding the
experiment.

`stop` is the milestone interpretation when downstream lithology metrics do not
improve over the existing baselines. Do not promote visually plausible HMM maps
to final lithology predictions.
