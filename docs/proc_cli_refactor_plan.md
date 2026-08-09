# Proc CLI Refactor Plan

> Historical planning snapshot: this document records the proc inventory used by
> the earlier CLI-refactor work. Issue #329 later removed the repository-wide
> results validator named below. Its inventory entry and planning references are
> retained only as historical context and are not current instructions.

This plan inventories the `proc/seis_ssl_cluster/*.py` entrypoints as they
existed for that refactor and defines the narrow path that was considered for
shared CLI handling. It is a planning document only: no proc script migration,
console script addition, YAML rewrite, or behavior change is included here.

Constraints for that historical refactor were:

- Keep existing command names, required arguments, major options, YAML workflow,
  artifact paths, results publishing behavior, and downstream F3 workflows.
- Keep complete generated outputs under `artifacts/` and selected review outputs
  under `results/`; do not reintroduce `runs/` as a standard path.
- Keep stage processing logic and config resolver behavior in their then-current
  implementation modules.
- Leave existing proc scripts in place.

`proc/seis_ssl_cluster/publish_results.py` was mentioned in the issue example
but was not present in that repository state, so it was not included in the
inventory.

## Historical Inventory

Common column meanings:

- `config`: `required`, `default`, `optional`, or `none`.
- `flags`: notable shared-style options beyond `--config`.
- `loader/resolver`: how the entrypoint loads and normalizes configuration.
- `main work`: the stage-specific implementation called after CLI handling.
- `stdout`: dry-run and completion output style.
- `exit`: process exit behavior.

| Script | argparse shape | config | flags | loader/resolver | main work | stdout | logging | exit |
|---|---|---:|---|---|---|---|---|---|
| `build_f3_inspection_report.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_INSPECTION_REPORT)` + local typed config builders | `build_f3_inspection_report` | local dry-run summary; report and publish paths on completion | none | exception/default |
| `build_f3_lithology_baseline_features.py` | wrapper only | inherited | inherited | delegates to `build_f3_lithology_baseline_token_dataset.main` | same delegated entrypoint | same delegated entrypoint | none | delegated |
| `build_f3_lithology_baseline_token_dataset.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_baseline_token_dataset_config_from_mapping` | `build_f3_lithology_baseline_token_dataset` | local dry-run summary; token/output paths and counts on completion | none | exception/default |
| `build_f3_lithology_comparison_report.py` | local `ArgumentParser` with config or direct path overrides | optional | `--search-root`, `--output-dir`, `--output-csv`, `--output-markdown`, repeated `--metrics-json`, `--figure-dpi`, `--dry-run` | optional `load_config` + `f3_lithology_comparison_report_config_from_mapping`; CLI overrides merged locally; publish config from mapping | `build_f3_lithology_comparison_report` | local dry-run summary; rows, report paths and figures on completion | none | exception/default |
| `build_f3_lithology_report.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_report_config_from_mapping`; publish config from mapping | `build_f3_lithology_report` | local dry-run summary; report and publish paths on completion | none | exception/default |
| `build_f3_lithology_token_dataset.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_token_dataset_config_from_mapping` | `build_f3_lithology_token_dataset` | local dry-run summary; token/output paths and counts on completion | none | exception/default |
| `build_nopims_manifests.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_manifest_build_config` | `scan_nopims_amplitude_manifests_from_path_list` + `write_manifest_json` | shared config summary plus local target lines; manifest counts and output path on completion | none | exception/default |
| `check_f3_label_consistency.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_LABEL_CONSISTENCY)` + local typed config builders | `check_f3_label_consistency` and output writers | local dry-run summary; consistency/report paths on completion | none | exception/default |
| `cluster_embeddings.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `resolve_clustering_config` | `cluster_embeddings` | shared config summary on dry-run; output directory on completion | none | exception/default |
| `create_random_mae_checkpoint.py` | local `ArgumentParser` | required | `--dry-run` | `load_config` + `random_mae_checkpoint_config_from_mapping` for dry-run; stage function also consumes raw config | `create_random_mae_checkpoint_from_config` | local dry-run summary; checkpoint path on completion | none | exception/default |
| `extract_embeddings.py` | local `ArgumentParser` | default | `--dry-run`, `--device`, `--skip-existing` | `load_config` + `resolve_embedding_extraction_config` | `extract_embeddings` | shared config summary with device override on dry-run; output directory on completion | none | exception/default |
| `filter_manifest_by_normalization_qc.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_normalization_qc_config` | `filter_manifests_by_stats_qc` + JSON/manifest/path-list writers | shared config summary plus local existence/target lines; QC counts and outputs on completion | none | exception/default |
| `inspect_f3_files.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_INSPECT_FILES)` + local typed output config | `scan_f3_file_inventory` + output writers | local dry-run summary; inventory counts and output paths on completion | none | exception/default |
| `inspect_f3_png_labels.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_PNG_LABELS)` + local typed output config | `inspect_f3_png_labels` + output writers | local dry-run summary; PNG label counts and output paths on completion | none | exception/default |
| `inspect_f3_segy_geometry.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_SEGY_GEOMETRY)` + local typed output config | `inspect_f3_segy_geometry` | local dry-run summary; geometry/stat paths on completion | none | exception/default |
| `predict_f3_lithology_tokens.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_prediction_config_from_mapping` | `predict_f3_lithology_tokens` | local dry-run summary; prediction paths/counts on completion | none | exception/default |
| `prepare_f3_facies_volume.py` | local `ArgumentParser` | default | `--dry-run`, `--overwrite` | `load_config` + `f3_prepare_volume_config_from_mapping` | `prepare_f3_facies_volume` | local dry-run summary; prepared artifact paths on completion | none | exception/default |
| `prepare_nopims_normalization_stats.py` | local `_parse_args` | default | `--dry-run`, `--overwrite` | `load_config` + `resolve_normalization_stats_config` | normalization-stat computation and JSON writers | shared config summary plus local missing/existing/target lines; output paths on completion | none | exception/default |
| `preview_f3_tokenization.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_TOKENIZATION_PREVIEW)` + local typed config builders | `preview_f3_tokenization` | local dry-run summary; tokenization summaries and paths on completion | none | exception/default |
| `train_amp_mae.py` | local `ArgumentParser` with CLI overrides | default | `--dry-run`, `--device`, `--max-steps`, `--output-root`, `--resume` | `load_config`, local raw-config overrides, then `resolve_mae_training_config` | `run_mae_pretraining` | shared config summary on dry-run; optional resume line; checkpoint path on completion | none | exception/default |
| `train_f3_lithology_probe.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_probe_config_from_mapping` | `train_f3_lithology_probe` | local dry-run summary; metrics/report paths on completion | none | exception/default |
| `validate_results_artifacts.py` | local `ArgumentParser`; no YAML | none | `--root`, `--max-file-size-mb`, repeated `--required-file`, `--local-path-policy` | direct CLI values only | `validate_results_artifacts` | validation status, counts, findings | none | returns `0`/`1` via `SystemExit(main())` |
| `visualize_clusters.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `resolve_cluster_visualization_config` | `run_cluster_visualization` | shared config summary on dry-run; figure/voxel/summary counts on completion | none | exception/default |
| `visualize_f3_lithology_predictions.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_visualization_config_from_mapping` | `visualize_f3_lithology_predictions` | local dry-run summary; figure and report paths on completion | none | exception/default |
| `visualize_f3_quicklook.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_QUICKLOOK)` + local typed config builders | `write_f3_quicklook_figures` | local dry-run summary; quicklook paths on completion | none | exception/default |

## Shared Patterns At That Time

Shared helpers then lived in `src/seis_ssl_cluster/utils/cli.py`:

- `parse_config_args(description, default_config)` handled `--config` and
  `--dry-run`.
- `print_config_summary(cfg, device_override=None)` printed stage-aware summaries
  for NOPIMS manifest, normalization, training, embedding, clustering, and
  cluster visualization configs.
- `run_pending_entrypoint(...)` was available but was not part of that proc
  entrypoint inventory.

Observed repeated patterns included parser construction, config loading,
resolver invocation, compact dry-run output, typed path/value coercion, and
validation command exit-code handling. The CLI refactor intentionally left
stage-specific scientific processing in the owning implementation modules.

## Historical Commonization Boundary

The plan considered small parser/load/print helpers safe to commonize while
keeping training, checkpoint/resume, embedding extraction, clustering,
prediction, F3 scientific algorithms, artifact semantics, and downstream
workflow policy out of the CLI commonization layer.

## Historical Migration Order

The plan was to add focused CLI-contract tests, extract low-risk helpers, then
migrate standard config-driven proc wrappers before irregular no-config or
validator commands. Duplicated local helpers were to be deleted only after
their callers had migrated and been tested.

## Historical Test Policy

Minimum checks for that planning issue were:

```bash
python -m compileall -q src proc tests
pytest -q tests/seis_ssl_cluster/test_config.py
```

It also explicitly excluded:

```bash
pytest tests/seis_ssl_cluster/test_proc_dry_run.py
```
