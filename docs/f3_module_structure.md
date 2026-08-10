# F3 Module Structure

F3 code is organized by data access, lithology tasks, and experiment-specific
orchestration. Proc entrypoints call the public library API and do not own F3
processing logic.

## Package boundaries

`seis_ssl_cluster.f3.io` owns reusable F3 input and prepared-volume operations:

- `labels.py` parses class metadata and label-file identities.
- `segy.py` locates and reads F3 SEGY data and exposes geometry and value
  inspection helpers.
- `prepare_volume.py` builds the prepared seismic and label volumes used by
  downstream stages.

`seis_ssl_cluster.f3.lithology` owns downstream lithology data and evaluation
logic. Its focused modules cover baseline features, guardrails, milestone
summaries, metrics, prediction, probes, robustness suites, token datasets,
tokenization, visualization, and report generation.

Top-level modules under `seis_ssl_cluster.f3` own inspection utilities and
experiment-specific validation, auditing, result aggregation, and execution
coordination that does not belong in the reusable IO or lithology packages.

## Public import surface

`seis_ssl_cluster.f3` is the aggregate public facade for the core inspection
and lithology APIs used by proc entrypoints and downstream code.

Compatibility modules such as `seis_ssl_cluster.f3.labels`,
`seis_ssl_cluster.f3.prepare_volume`, `seis_ssl_cluster.f3.segy`,
`seis_ssl_cluster.f3.lithology_tokens`, and
`seis_ssl_cluster.f3.lithology_report` re-export their canonical package
implementations. These modules provide one-hop imports only. They do not add
fallback path discovery, data conversion, alternate behavior, or artifact
transformation.

New internal code should import the focused canonical module that owns the
operation. Public callers may use the aggregate facade or an existing
compatibility module.

## Token dataset schema

`seis_ssl_cluster.f3.lithology.token_dataset` owns the shared NPZ schema for F3
lithology token datasets. Required fields are:

- `features`
- `labels`
- `survey_id`
- `split`
- `slice_type`
- `slice_index`
- `token_xyz`
- `voxel_center_xyz`
- `majority_fraction`
- `labeled_fraction`

Optional JSON metadata is stored in `metadata`.

All required arrays share one row count. `features` is a finite two-dimensional
floating-point matrix, `labels` is a one-dimensional integer vector,
`token_xyz` is integer typed, and `voxel_center_xyz`, `majority_fraction`, and
`labeled_fraction` contain finite values. Class `0` is a valid label.

The schema API is:

- `F3LithologyTokenDataset`
- `F3LithologyTokenDatasetSummary`
- `load_f3_lithology_token_dataset`
- `load_f3_lithology_token_dataset_summary`
- `save_f3_lithology_token_dataset`
- `validate_f3_lithology_token_dataset`
- `replace_token_features`

## Lithology report package

`seis_ssl_cluster.f3.lithology.report` separates report responsibilities:

- `_core.py` builds a single-run report.
- `metrics_loader.py` loads report inputs.
- `markdown.py` renders Markdown.
- `figures.py` renders comparison figures.
- `comparison.py` aggregates comparison reports.
- `publish.py` writes the producer-owned lightweight review file set.
- `_common.py` contains report-internal shared types and helpers.

The package `__init__.py` is the public report facade.
`seis_ssl_cluster.f3.lithology_report` preserves the direct import surface by
delegating to this package.

## Proc boundary

Files under `proc/seis_ssl_cluster/` parse arguments, resolve YAML, and call F3
library functions. Moving or splitting an implementation module must not
silently change command names, YAML keys, artifact paths, metric definitions,
or result-file contracts.
