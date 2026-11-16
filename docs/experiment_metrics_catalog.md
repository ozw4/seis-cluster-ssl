# Saved experiment metrics catalog

Export retained F3, Parihaka and Volve evaluations without training, opening
model weights, or recomputing predictions:

```bash
python proc/seis_ssl_cluster/export_experiment_metrics.py \
  --workspace /path/to/checkout \
  --output artifacts/seis_ssl_cluster/studies/all_experiment_metrics_DATE/catalog \
  --dry-run
```

Remove `--dry-run` and add `--zip` to create an offline HTML and CSV bundle. The
output directory and optional sibling ZIP must not exist. `--artifact-root`
overrides the default `<workspace>/artifacts/seis_ssl_cluster`.

The exporter scans canonical survey directories, archives and cross-survey
studies. It resolves file aliases once, does not traverse directory aliases,
and excludes generated catalog/figure bundles. It never reads `reports/`.
Versioned YAML, recorded decoder/probe configs and encoder training sidecars
supply condition identities. `hmm_version` records v1/v2/v3 or an explicit
legacy category. Early single-head v1 models are identified from their actual
training schema and evaluation manifests even when the tag lacks `hmm_v1`;
they appear under the HMM v1 filter in HTML. Explicit model identities take priority over
shared parent directory names: a plain MAE checkpoint under
`ssl_hmm_continuation_v1/mae_continue/` remains no-HMM. Every saved decoder
seed remains a separate row and a separate layout-summary group; a canonical
reference designation never removes historical seed results. Unavailable
coefficients and split identities remain empty or unknown. Probe validation is recognized only when its recorded
config identifies validation tokens.

The exact publication contract is `FIXED_FILES` in
`src/seis_ssl_cluster/results/experiment_catalog.py`, plus the metric CSVs
listed individually in `source_tables/`. `manifest.json` lists this exact set
and hashes every member except itself. The fixed files are:

- `index.html`: standalone filterable report; no network requests. Encoder and
  decoder seeds appear in the evaluation and summary tables, with a decoder-seed
  filter; filtered CSV downloads keep these columns.
- `experiment_metrics.csv`: one row per saved evaluation split/support, with
  principal metrics and full experiment/source identities.
- `metrics_long.csv`: all scalar endpoint metrics, including class, horizon,
  confusion matrix and coverage values, with units and direction.
- `layout_summary.csv`: descriptive means, sample SD and per-metric n within
  the same cohort/model/seed/budget/split/support/checkpoint selection. Conflicting
  values for one replicate prevent averaging. These are not encoder-seed SDs.
- `published_statistics.csv`: scalar results in summary and supplementary JSON.
  Original CSV tables retain the associated row labels and context.
- `experiment_catalog.csv`: versioned survey definitions and matched evaluation
  counts. An unmatched definition does not imply failed execution.
- `source_files.csv`: original paths, sizes, SHA-256, inclusion/parse status,
  endpoint counts, and links to copied metric CSVs. Identical CSVs share a copy.
- `metric_dictionary.csv`: original metric paths, units, directions and counts.
- `validation.json`: export scope, counts and source-read status.
- `manifest.json`: exact publication membership and checksums.

CSV exports use UTF-8 with BOM. Source tables preserve numeric strings and
replace workspace prefixes in text paths with repository-relative paths.
Null/nonfinite endpoint values become empty CSV fields, never zero. Validation,
test and common-test results remain distinct; Volve primary-common and
secondary-per-horizon support remain distinct. Archive and smoke results are
explicitly flagged. Saved values do not certify completion of the entire plan.
No overall ranking combines incompatible surveys or evaluation protocols.

Validate this isolated producer with:

```bash
pytest -q tests/seis_ssl_cluster/test_experiment_metrics_catalog.py
python -m ruff check --no-fix src/seis_ssl_cluster/results \
  proc/seis_ssl_cluster/export_experiment_metrics.py \
  tests/seis_ssl_cluster/test_experiment_metrics_catalog.py
python tools/check_seis_ssl_cluster_isolation.py
```
