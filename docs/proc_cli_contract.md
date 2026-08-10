# Proc CLI Contract

`proc/seis_ssl_cluster/*.py` contains command-line entrypoints. Reusable
processing, scientific logic, artifact handling, and configuration validation
belong under `src/seis_ssl_cluster/`.

## Entrypoint boundary

A proc entrypoint may:

1. build an `argparse.ArgumentParser`;
2. load a YAML mapping;
3. resolve and validate the stage configuration;
4. apply declared runtime-only CLI overrides;
5. call one library stage function; and
6. print a compact completion summary.

`main()` remains orchestration code. It must not contain training, inference,
clustering, evaluation, publishing, or artifact-format implementations.

A wrapper entrypoint may delegate to another proc module when both command names
must remain available. The wrapper must not alter arguments, configuration, or
stage behavior.

## Shared CLI helpers

`seis_ssl_cluster.cli` owns the generic helpers used by proc modules:

- `build_config_parser` builds the standard `--config` and `--dry-run` parser.
- `add_config_argument`, `add_dry_run_argument`, `add_device_argument`,
  `add_skip_existing_argument`, and `add_overwrite_argument` add common options.
- `add_path_argument`, `add_append_path_argument`, and
  `add_store_true_argument` preserve stage-specific option names while sharing
  parser construction.
- `parse_config_path` returns the required configuration path.
- `load_config_for_cli` verifies that the configuration path is an existing
  file before calling the repository loader.
- `resolve_config_for_cli` adds the source configuration path to resolver
  failures.
- `print_cli_summary` writes stable key/value summaries.

`seis_ssl_cluster.utils.cli` owns stage-aware configuration summaries and the
`parse_config_args` helper used by proc modules that require those summaries.
Generic parser construction stays in `seis_ssl_cluster.cli`; stage-specific
summary formatting stays in `seis_ssl_cluster.utils.cli`.

## Configuration and override rules

The selected proc command owns the stage identity. Raw YAML must not contain a
top-level `stage` field.

Resolvers receive the raw mapping and return the complete runtime
configuration. CLI overrides are limited to arguments declared by the selected
entrypoint. Scientific identity fields, artifact identities, and input
contracts must still pass the owning resolver or stage validation.

When an entrypoint exposes `--dry-run`, it validates and reports the planned
operation without executing the stage or writing stage outputs.

Stage-specific options remain local when their semantics are not shared. A
generic CLI helper must not encode scientific policy, artifact recovery policy,
checkpoint selection, or workflow ordering.

## Error and output behavior

Invalid arguments use normal `argparse` errors. Missing configuration files and
resolver failures stop before stage execution. Validation commands may map
validated findings to an explicit process exit status; other entrypoints allow
exceptions to propagate.

Completion output should identify the primary output path and a small number of
stage results. Detailed diagnostics belong in generated artifacts rather than
stdout.
