# Proc CLI Refactor Plan

> Historical planning snapshot: this document records the proc inventory used by
> the earlier CLI-refactor work. Issue #329 later removed the repository-wide
> results validator named below. Its inventory entry and planning references are
> retained only as historical context and are not current instructions.

This plan inventories the `proc/seis_ssl_cluster/*.py` entrypoints as they
existed for that refactor and defines the narrow path that was considered for
shared CLI handling. It is a planning document only: no proc script migration,
console script addition, YAML rewrite, or behavior change is included here.

Normative constraints for that historical refactor were:

- Keep existing command names, required arguments, major options, YAML workflow,
  artifact paths, results publishing behavior, and downstream F3 workflows.
- Keep complete generated outputs under `artifacts/` and selected review outputs
  under `results/`; do not reintroduce `runs/` as a standard path.
- Keep stage processing logic and config resolver behavior in their current
  implementation modules.
- Leave existing proc scripts in place.

`proc/seis_ssl_cluster/publish_results.py` is mentioned in the issue example but
was not present in that repository state, so it was not included in the active
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
| `inspect_f3_segy_geometry.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_SEGY_GEOMETRY)` + local typed output config | `inspect_f3_segy_geometry` + output writers | local dry-run summary; geometry/stat paths on completion | none | exception/default |
| `predict_f3_lithology_tokens.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_prediction_config_from_mapping` | `predict_f3_lithology_tokens` | local dry-run summary; prediction paths/counts on completion | none | exception/default |
| `prepare_f3_facies_volume.py` | local `ArgumentParser` | default | `--dry-run`, `--overwrite` | `load_config` + `f3_prepare_volume_config_from_mapping` | `prepare_f3_facies_volume` | local dry-run summary; prepared artifact paths on completion | none | exception/default |
| `prepare_nopims_normalization_stats.py` | local `_parse_args` | default | `--dry-run`, `--overwrite` | `load_config` + `resolve_normalization_stats_config` | normalization-stat computation and JSON writers | shared config summary plus local missing/existing/target lines; output paths on completion | none | exception/default |
| `preview_f3_tokenization.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_TOKENIZATION_PREVIEW)` + local typed config builders | `preview_f3_tokenization` + output writers | local dry-run summary; tokenization summaries and paths on completion | none | exception/default |
| `train_amp_mae.py` | local `ArgumentParser` with CLI overrides | default | `--dry-run`, `--device`, `--max-steps`, `--output-root`, `--resume` | `load_config`, local raw-config overrides, then `resolve_mae_training_config` | `run_mae_pretraining` | shared config summary on dry-run; optional resume line; checkpoint path on completion | none | exception/default |
| `train_f3_lithology_probe.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_probe_config_from_mapping` | `train_f3_lithology_probe` | local dry-run summary; metrics/report paths on completion | none | exception/default |
| `validate_results_artifacts.py` | local `ArgumentParser`; no YAML | none | `--root`, `--max-file-size-mb`, repeated `--required-file`, `--local-path-policy` | direct CLI values only | `validate_results_artifacts` | validation status, counts, findings | none | returns `0`/`1` via `SystemExit(main())` |
| `visualize_clusters.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `resolve_cluster_visualization_config` | `run_cluster_visualization` | shared config summary on dry-run; figure/voxel/summary counts on completion | none | exception/default |
| `visualize_f3_lithology_predictions.py` | local `ArgumentParser` | default | `--dry-run` | `load_config` + `f3_lithology_visualization_config_from_mapping` | `visualize_f3_lithology_predictions` | local dry-run summary; figure and report paths on completion | none | exception/default |
| `visualize_f3_quicklook.py` | shared `parse_config_args` | default | `--dry-run` | `load_config` + `resolve_f3_facies_inspection_config(stage=STAGE_F3_QUICKLOOK)` + local typed config builders | `write_f3_quicklook_figures` | local dry-run summary; quicklook paths on completion | none | exception/default |

## Shared Patterns

Existing shared helpers at that time lived in `src/seis_ssl_cluster/utils/cli.py`:

- `parse_config_args(description, default_config)` handles `--config` and
  `--dry-run`.
- `print_config_summary(cfg, device_override=None)` prints stage-aware summaries
  for NOPIMS manifest, normalization, training, embedding, clustering, and
  cluster visualization configs.
- `run_pending_entrypoint(...)` was available but was not part of the active proc
  entrypoint inventory.

Observed repeated patterns that were candidates for future extraction:

- Construction of `ArgumentParser` with `--config` and `--dry-run`.
- `load_config(args.config)` followed by one resolver or typed
  `*_config_from_mapping` conversion.
- Passing an explicit F3 inspection `stage=` to
  `resolve_f3_facies_inspection_config`.
- Printing a compact dry-run summary and one final `execution: dry-run; ...`
  line.
- Printing output paths and count fields as `key: value` lines.
- Repeated `Path(...)`, mapping, sequence, optional boolean, optional integer,
  optional fraction, and file-size coercion helpers in F3 proc scripts.
- Store-true boolean flags such as `--dry-run`, `--overwrite`,
  `--skip-existing`, `--fail-on-runs`, and `--allow-test-fixtures`.
- Validation command exit-code handling through `return 0 if report.ok else 1`
  and `raise SystemExit(main())`.
- Visualization commands import matplotlib lazily in implementation modules;
  any warning-avoidance guidance should be centralized as a short CLI/developer
  note rather than repeated per script.

Observed non-shared behavior that should remain script-owned:

- `train_amp_mae.py` raw-config overrides for `--device`, `--max-steps`,
  `--output-root`, and `--resume`.
- `extract_embeddings.py` device override and `--skip-existing` pass-through.
- `prepare_f3_facies_volume.py` and
  `prepare_nopims_normalization_stats.py` overwrite semantics.
- `build_f3_lithology_comparison_report.py` optional no-config workflow and
  direct output/search/metrics CLI overrides.
- Validator commands, because they are not YAML/config-resolver workflows.
- Stage-specific summaries where the field names are part of the established
  stdout contract.

## Commonization Targets

Safe to commonize in a later implementation issue at that time:

- A `--config`/`--dry-run` parser builder that preserves each script's current
  default vs required config behavior.
- A config loading wrapper around `load_config` that does not alter resolver
  behavior or exception messages beyond unavoidable call-site formatting.
- A resolver invocation wrapper for the repeated
  `load_config(args.config) -> resolve_*` shape, including explicit F3
  inspection stages.
- Small print helpers for run summaries and output path/count lines, provided
  they preserve the existing `key: value` stdout style.
- A boolean CLI flag helper for store-true options, without changing flag names
  or defaults.
- Safe `pathlib` and typed-value helpers currently repeated across F3 scripts:
  required mapping/string, string sequence, optional bool, optional positive
  integer, optional non-negative integer, optional fraction, and file-size
  conversion.
- A central developer note for matplotlib warning avoidance in visualization
  entrypoints.

Not safe to commonize in that refactor track:

- Training loops, checkpoint/resume behavior, optimizer control, or MAE debug
  visualization internals.
- Embedding extraction, sliding-window inference, artifact writing, or
  skip-existing behavior.
- Clustering, PCA, sampling, prediction, or voxel reconstruction logic.
- F3 inspection, preparation, tokenization, lithology probe, prediction,
  visualization, report, or baseline-comparison algorithms.
- Config resolver internals or YAML schema semantics.
- Results publishing rules, artifact path policy, or F3 downstream workflow
  conventions.
- CLI command names, existing script locations, or console-script packaging.

## Migration Order

The historical plan was:

1. Add tests around the existing CLI contracts before moving code. Focus on
   parser defaults, required/optional `--config`, dry-run exit behavior, and
   validation command exit codes. Do not use `tests/test_proc_dry_run.py` for
   this refactor track unless a later issue explicitly changes that constraint.
2. Extract low-risk helpers that are already conceptually shared:
   `--config`/`--dry-run` parser creation, output-line formatting, and common
   typed-value/path helpers.
3. Move the F3 inspection parser/load/resolve wrapper next because those scripts
   already share `parse_config_args` and differ mainly by explicit stage and
   local typed output builders.
4. Migrate NOPIMS manifest, normalization stats, normalization QC, MAE training,
   embedding, clustering, and cluster visualization entrypoints while preserving
   their CLI override points.
5. Migrate F3 lithology and baseline scripts after their typed config builders
   and stdout summaries have focused contract tests.
6. Leave validator commands and the comparison report no-config workflow until
   last, because their CLI shape differs most from the standard YAML stage
   commands.
7. Delete duplicated local helpers only after their callers have been migrated
   and tested; keep proc scripts as thin wrappers around `src/` code.

## Compatibility Policy

The historical compatibility policy was:

- Existing command names and file paths under `proc/seis_ssl_cluster/` stay
  valid.
- Existing `--config` workflows stay valid, including scripts with default
  configs, the required-config random checkpoint command, and the optional-config
  comparison report command.
- Existing major options stay valid: `--device`, `--skip-existing`,
  `--overwrite`, `--max-steps`, `--output-root`, `--resume`, validation options,
  and comparison-report overrides.
- YAML files are not rewritten as part of CLI commonization.
- Resolver output and validation behavior remain source-of-truth behavior.
- stdout remains a stable `key: value` style. Minor formatting cleanup is
  acceptable only when it does not remove existing information or materially
  change error/debug workflows.
- Exceptions continue to propagate unless the current script already returns an
  explicit process code.

## Test Policy

Minimum checks for that planning issue were:

```bash
python -m compileall -q src proc tests
pytest -q tests/seis_ssl_cluster/test_config.py
```

It also explicitly excluded:

```bash
pytest tests/seis_ssl_cluster/test_proc_dry_run.py
```

Later implementation issues were expected to add focused tests around parser
compatibility and config loading wrappers before each migration step.
