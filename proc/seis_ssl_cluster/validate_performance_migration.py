# ruff: noqa: ANN001, ANN202, C901, E501, PLR0911
"""Run isolated real-data performance migration validation stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.performance_migration_validation import (
	performance_migration_validation_config_from_mapping,
)
from seis_ssl_cluster.migration.performance_validation import (
	build_input_inventory,
	checkpoint_compatibility_smoke,
	compare_embedding_artifacts,
	compare_hmm_replay,
	compare_probe_predictions,
	compare_pseudo_targets,
	export_legacy_m1_pseudo_targets,
	reconstruct_historical_hmm_config,
	run_m1_embedding_extraction,
	run_m1_hmm_replay,
	run_performance_benchmark,
	summarize_performance_migration,
)

_STAGES = (
	'preflight',
	'checkpoint-smoke',
	'extract-m1',
	'embedding-parity',
	'probe-parity',
	'reconstruct-hmm-config',
	'replay-hmm',
	'hmm-parity',
	'export-pseudo-targets',
	'pseudo-target-parity',
	'benchmark',
	'summarize',
	'publish',
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the migration validation CLI parser."""
	parser = argparse.ArgumentParser(
		description='Validate historical F3 artifacts against the current implementation.',
	)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--stage', choices=_STAGES, required=True)
	parser.add_argument('--embedding-config', type=Path)
	parser.add_argument('--hmm-config', type=Path)
	parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	return parser


def main() -> int:
	"""Dispatch one strictly isolated migration validation stage."""
	args = build_parser().parse_args()
	config = _load_migration_config(args.config)
	result = _run_stage(args, config)
	print(f"stage={args.stage} status={result.get('status', result.get('completion_status', 'PASS'))}")
	return 0


def _load_migration_config(config_path: Path):
	raw = load_config(config_path)
	if 'export' not in raw:
		return performance_migration_validation_config_from_mapping(raw)
	_validate_export_config(raw)
	base_path = config_path.with_name('01_checkpoint_smoke.yaml')
	if not base_path.is_file():
		raise FileNotFoundError(f'base migration config required beside export config: {base_path}')
	base = performance_migration_validation_config_from_mapping(load_config(base_path))
	if raw['migration'] != {
		'current_git_sha': base.current_git_sha,
		'historical_baseline_sha': base.historical_baseline_sha,
	}:
		raise ValueError('export config migration identity does not equal base config')
	return base


def _validate_export_config(raw: dict[str, object]) -> None:
	if set(raw) != {'paths', 'migration', 'inputs', 'export'}:
		raise ValueError('pseudo-target export config has unknown or missing top-level keys')
	export = raw.get('export')
	if not isinstance(export, dict):
		raise TypeError('export must be a mapping')
	expected = {
		'output_root',
		'k',
		'survey_id',
		'confidence',
		'schema_version',
		'write_boundary_weight',
		'only_missing',
		'overwrite',
	}
	if set(export) != expected:
		raise ValueError('pseudo-target export config keys do not match the legacy-v1 contract')
	if export['k'] != 6 or export['confidence'] != 1.0 or export['schema_version'] != 1:
		raise ValueError('pseudo-target export config must use historical K=6/schema-v1/confidence=1')
	if export['write_boundary_weight'] is not False or export['overwrite'] is not False:
		raise ValueError('pseudo-target export must not write a boundary field or overwrite')


def _run_stage(args: argparse.Namespace, config):
	if args.stage == 'preflight':
		return build_input_inventory(config, only_missing=args.only_missing)
	if args.stage == 'checkpoint-smoke':
		return checkpoint_compatibility_smoke(config, only_missing=args.only_missing)
	if args.stage == 'extract-m1':
		return _run_embedding_stage(args, config)
	if args.stage == 'embedding-parity':
		return compare_embedding_artifacts(config, only_missing=args.only_missing)
	if args.stage == 'probe-parity':
		return compare_probe_predictions(config, only_missing=args.only_missing)
	if args.stage == 'reconstruct-hmm-config':
		return reconstruct_historical_hmm_config(config, only_missing=args.only_missing)
	if args.stage == 'replay-hmm':
		return _run_hmm_stage(args, config)
	if args.stage == 'hmm-parity':
		return compare_hmm_replay(config, only_missing=args.only_missing)
	if args.stage == 'export-pseudo-targets':
		return export_legacy_m1_pseudo_targets(
			config,
			dry_run=args.dry_run,
			only_missing=args.only_missing,
		)
	if args.stage == 'pseudo-target-parity':
		return compare_pseudo_targets(config, only_missing=args.only_missing)
	if args.stage == 'benchmark':
		return run_performance_benchmark(
			config,
			dry_run=args.dry_run,
			only_missing=args.only_missing,
		)
	if args.stage in {'summarize', 'publish'}:
		return summarize_performance_migration(
			config,
			only_missing=args.only_missing,
			publish=args.stage == 'publish',
			dry_run=args.dry_run,
		)
	raise AssertionError(f'unhandled migration stage: {args.stage}')


def _run_embedding_stage(args: argparse.Namespace, config):
	if args.embedding_config is None:
		raise ValueError('--embedding-config is required for --stage extract-m1')
	mode = 'cache_memmap' if 'memmap' in args.embedding_config.name else 'cache_off'
	device = None if args.device == 'auto' else args.device
	return run_m1_embedding_extraction(
		config,
		embedding_config_path=args.embedding_config,
		mode=mode,
		device=device,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
	)


def _run_hmm_stage(args: argparse.Namespace, config):
	if args.hmm_config is None:
		raise ValueError('--hmm-config is required for --stage replay-hmm')
	return run_m1_hmm_replay(
		config,
		hmm_config_path=args.hmm_config,
		dry_run=args.dry_run,
		only_missing=args.only_missing,
	)


if __name__ == '__main__':
	raise SystemExit(main())
