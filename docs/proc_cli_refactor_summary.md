# Proc CLI Refactor Summary

> Historical record: this document describes the repository state for issue #82.
> The repository-wide results validator named below was removed by issue #329;
> commands and inventory entries in this document are not current instructions.

This summarizes the proc entrypoint commonization and regression checks for
issue #82. The proc scripts remain in `proc/seis_ssl_cluster/`; no
`console_scripts` packaging, command rename, YAML rewrite, or path-policy change
was made.

## Commonized Helpers

Shared CLI helpers live in `src/seis_ssl_cluster/cli.py`:

- `build_config_parser`, `add_config_argument`, and `parse_config_path` preserve
  the existing `--config` YAML workflow.
- `add_dry_run_argument`, `add_device_argument`, `add_skip_existing_argument`,
  `add_overwrite_argument`, `add_path_argument`, `add_append_path_argument`, and
  `add_store_true_argument` preserve existing flag names and argparse shapes.
- `load_config_for_cli` and `resolve_config_for_cli` centralize config file
  loading and resolver invocation without changing resolver behavior.
- `print_cli_summary` provides stable `key: value` output for simple summaries.

`src/seis_ssl_cluster/utils/cli.py` remains in use for existing config-summary
printing and older `parse_config_args` call sites.

## Historical Migrated Script Coverage

The following proc entrypoints used the shared `seis_ssl_cluster.cli` helpers
in the issue #82 repository state:

- `build_f3_inspection_report.py`
- `build_f3_lithology_baseline_token_dataset.py`
- `build_f3_lithology_comparison_report.py`
- `build_f3_lithology_report.py`
- `build_f3_lithology_token_dataset.py`
- `check_f3_label_consistency.py`
- `cluster_embeddings.py`
- `create_random_mae_checkpoint.py`
- `extract_embeddings.py`
- `inspect_f3_files.py`
- `inspect_f3_png_labels.py`
- `inspect_f3_segy_geometry.py`
- `predict_f3_lithology_tokens.py`
- `prepare_f3_facies_volume.py`
- `preview_f3_tokenization.py`
- `train_amp_mae.py`
- `train_f3_lithology_probe.py`
- `validate_results_artifacts.py`
- `visualize_clusters.py`
- `visualize_f3_lithology_predictions.py`
- `visualize_f3_quicklook.py`

`build_f3_lithology_baseline_features.py` was a thin compatibility wrapper
that delegated to `build_f3_lithology_baseline_token_dataset.main`.

## Historical CLI Compatibility

The entrypoint contract test added in issue #82 verified:

- proc modules imported without running `main`;
- existing `build_parser()` functions constructed `argparse.ArgumentParser`
  instances;
- primary workflow help output contained `--config` and `--dry-run`;
- major options then present included `--device`, `--skip-existing`,
  `--overwrite`, `--max-steps`, `--output-root`, `--resume`, comparison-report
  overrides, and the then-existing results-validator options;
- proc `main()` functions stayed small enough to act as entrypoints rather than
  stage implementations.

At that time the results validator was a non-YAML CLI and did not use
`--config`.

## Historical Verification

The following commands were executed for issue #82 at that time:

```bash
python -m compileall -q src proc tests
pytest -q tests/seis_ssl_cluster/test_cli_helpers.py tests/seis_ssl_cluster/test_proc_entrypoints.py
pytest -q tests/seis_ssl_cluster/test_config.py tests/seis_ssl_cluster/test_active_experiment_configs.py tests/seis_ssl_cluster/test_results_publish.py tests/seis_ssl_cluster/test_results_validation.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 pytest -q tests/seis_ssl_cluster/test_f3_lithology_probe.py tests/seis_ssl_cluster/test_f3_lithology_baseline_comparison.py tests/seis_ssl_cluster/test_embedding_extractor.py tests/seis_ssl_cluster/test_training_smoke.py
python proc/seis_ssl_cluster/train_amp_mae.py --help
python proc/seis_ssl_cluster/extract_embeddings.py --help
python proc/seis_ssl_cluster/build_f3_lithology_report.py --help
python proc/seis_ssl_cluster/validate_results_artifacts.py --help
```

Those commands completed successfully in that historical repository state.
`tests/seis_ssl_cluster/test_proc_dry_run.py` was not executed.

## Open Items At That Time

No compatibility gaps were known for issue #82. Stage-specific summaries,
validation command exit-code handling, and implementation logic remained owned
by their existing modules or proc scripts where they differed from the standard
YAML workflow.
