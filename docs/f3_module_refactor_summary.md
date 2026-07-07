# F3 Module Refactor Summary

This is the current post-refactor module contract for F3 inspection,
lithology probe, and baseline comparison code. The refactor preserves existing
CLI commands, YAML keys, artifact paths, results publishing, label-source
policy, class `0` handling, report metrics, and figure outputs.

## Final Module Structure

```text
src/seis_ssl_cluster/f3/
|-- __init__.py
|-- io/
|   |-- __init__.py
|   |-- labels.py
|   |-- prepare_volume.py
|   `-- segy.py
|-- lithology/
|   |-- __init__.py
|   |-- token_dataset.py
|   `-- report/
|       |-- __init__.py
|       |-- _common.py
|       |-- _core.py
|       |-- comparison.py
|       |-- figures.py
|       |-- markdown.py
|       |-- metrics_loader.py
|       `-- publish.py
|-- baseline_features.py
|-- consistency.py
|-- inspection.py
|-- labels.py
|-- lithology_prediction.py
|-- lithology_probe.py
|-- lithology_report.py
|-- lithology_tokens.py
|-- lithology_visualization.py
|-- metrics.py
|-- png_labels.py
|-- prepare_volume.py
|-- report.py
|-- segy.py
|-- splits.py
|-- tokenization.py
`-- visualization.py
```

The implemented new public import paths are:

- `seis_ssl_cluster.f3.io.labels`
- `seis_ssl_cluster.f3.io.prepare_volume`
- `seis_ssl_cluster.f3.io.segy`
- `seis_ssl_cluster.f3.lithology.token_dataset`
- `seis_ssl_cluster.f3.lithology.report`
- `seis_ssl_cluster.f3.lithology.report.comparison`
- `seis_ssl_cluster.f3.lithology.report.figures`
- `seis_ssl_cluster.f3.lithology.report.markdown`
- `seis_ssl_cluster.f3.lithology.report.metrics_loader`
- `seis_ssl_cluster.f3.lithology.report.publish`

`seis_ssl_cluster.f3.inspection` remains a module in this refactor state, not a
subpackage. Inspection code still exposes its public API through
`seis_ssl_cluster.f3`, `seis_ssl_cluster.f3.inspection`,
`seis_ssl_cluster.f3.png_labels`, `seis_ssl_cluster.f3.consistency`,
`seis_ssl_cluster.f3.tokenization`, `seis_ssl_cluster.f3.visualization`, and
`seis_ssl_cluster.f3.report`.

## Compatibility Wrapper Policy

Existing public imports remain valid. Moved modules use one-hop wrappers or
facades only; they do not add fallback path discovery, alternate behavior, data
format migrations, or compatibility transformations.

Compatibility imports covered by tests:

- `seis_ssl_cluster.f3.segy`
- `seis_ssl_cluster.f3.png_labels`
- `seis_ssl_cluster.f3.consistency`
- `seis_ssl_cluster.f3.lithology_tokens`
- `seis_ssl_cluster.f3.baseline_features`
- `seis_ssl_cluster.f3.lithology_probe`
- `seis_ssl_cluster.f3.lithology_report`

The aggregate `seis_ssl_cluster.f3` import surface remains the main public
facade for proc entrypoints and existing downstream code.

## TokenDataset Schema

`seis_ssl_cluster.f3.lithology.token_dataset` owns the shared NPZ schema used by
pretrained token datasets and baseline token datasets. Required NPZ fields are:

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

Optional metadata is stored under `metadata`. The helper API is
`F3LithologyTokenDataset`, `load_f3_lithology_token_dataset`,
`save_f3_lithology_token_dataset`, `validate_f3_lithology_token_dataset`, and
`replace_token_features`. Class `0` is a valid class label and is not treated as
unlabeled by the schema helpers.

## Report Subpackage

`seis_ssl_cluster.f3.lithology.report` is split by responsibility:

- `_core.py`: single-run lithology report builder.
- `metrics_loader.py`: loading probe metrics, prediction metadata, token
  datasets, and related JSON components.
- `markdown.py`: single-run and comparison Markdown rendering.
- `figures.py`: comparison figure styles and rendering.
- `comparison.py`: baseline comparison aggregation and output writing.
- `publish.py`: results publish manifests for single-run and comparison
  reports.

`seis_ssl_cluster.f3.lithology_report` remains the compatibility import and
re-exports the package facade.

## Runbooks

The F3 runbooks keep the same CLI commands and config paths:

- `experiments/f3/facies_benchmark_v1/README.md`
- `experiments/f3/facies_benchmark_v1/50_lithology/README.md`
- `experiments/f3/facies_benchmark_v1/50_lithology_baselines/README.md`

They describe proc entrypoints and artifact layouts rather than direct library
module imports, so no CLI command changes are required.

## Tests And Validation

Issue #92 regression pass completed successfully with these commands:

```bash
python -m compileall -q src proc tests

PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_f3_module_import_compat.py \
  tests/seis_ssl_cluster/test_f3_lithology_report_modules.py

PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_f3_file_inventory.py \
  tests/seis_ssl_cluster/test_f3_segy_inspection.py \
  tests/seis_ssl_cluster/test_f3_png_labels.py \
  tests/seis_ssl_cluster/test_f3_quicklook_visualization.py \
  tests/seis_ssl_cluster/test_f3_label_consistency.py \
  tests/seis_ssl_cluster/test_f3_tokenization_preview.py \
  tests/seis_ssl_cluster/test_f3_inspection_report.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_f3_lithology_token_dataset.py \
  tests/seis_ssl_cluster/test_f3_lithology_probe.py \
  tests/seis_ssl_cluster/test_f3_lithology_prediction.py \
  tests/seis_ssl_cluster/test_f3_lithology_visualization.py \
  tests/seis_ssl_cluster/test_f3_lithology_report.py \
  tests/seis_ssl_cluster/test_f3_lithology_baseline_features.py \
  tests/seis_ssl_cluster/test_f3_lithology_baseline_comparison.py

PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_config.py \
  tests/seis_ssl_cluster/test_active_experiment_configs.py \
  tests/seis_ssl_cluster/test_results_publish.py \
  tests/seis_ssl_cluster/test_results_validation.py

PYTHONPATH=src python proc/seis_ssl_cluster/validate_artifact_paths.py \
  --root /workspace/artifacts/seis_ssl_cluster \
  --scan experiments proc docs README.md results \
  --fail-on-runs

PYTHONPATH=src python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10
```

`tests/seis_ssl_cluster/test_proc_dry_run.py` is intentionally excluded.

The validation CLIs reported `error_count: 0`. They also reported existing
warnings for legacy documentation mentions, legacy embedding path shape, local
artifact paths recorded in publish manifests, and local absolute path markers in
results files; those warnings do not change the error-zero validation result.

## Future Work

- Keep old direct module imports until a separately planned breaking change.
- If inspection modules are later moved under an `f3.inspection` package, do it
  as a dedicated compatibility-preserving step because `f3.inspection` is
  currently a module.
- Keep report content, metric definitions, artifact names, and publish targets
  stable when making internal cleanup changes.
