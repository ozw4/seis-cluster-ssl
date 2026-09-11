"""Thin entrypoint for the read-only F3 prepared-volume parity gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from seis_ssl_cluster.cli import (
	add_dry_run_argument,
	add_path_argument,
	load_config_for_cli,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	NORMALIZATION_PARITY_FIELDS,
	F3PreparedVolumeIdentity,
	F3PrepareVolumeConfig,
	check_f3_prepared_volume_parity,
	f3_prepare_volume_config_from_mapping,
)

EXPERIMENTS = Path(__file__).resolve().parents[2] / 'experiments' / 'f3'
DEFAULT_REFERENCE_CONFIG = (
	EXPERIMENTS / 'facies_benchmark_v1' / '10_prepare' / '01_prepare_f3_volume.yaml'
)
DEFAULT_CANDIDATE_CONFIG = (
	EXPERIMENTS / 'facies_benchmark_v2' / '10_prepare' / '01_prepare_f3_volume.yaml'
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the prepared-volume parity gate."""
	parser = argparse.ArgumentParser(
		description=(
			'Check that two prepared F3 facies registry versions hold the same '
			'volumes (NPY SHA-256, shape, dtype, grid order, normalization '
			'semantics, class order). Read-only.'
		),
	)
	add_path_argument(
		parser,
		'--reference-config',
		default=DEFAULT_REFERENCE_CONFIG,
		help_text='Prepare config of the version the checkpoints were trained on.',
	)
	add_path_argument(
		parser,
		'--candidate-config',
		default=DEFAULT_CANDIDATE_CONFIG,
		help_text='Prepare config of the version whose volumes will be embedded.',
	)
	add_dry_run_argument(
		parser,
		help_text='Resolve both configs and print the compared paths without hashing.',
	)
	return parser


def main() -> None:
	"""Compare two prepared F3 versions and exit non-zero on any mismatch."""
	parser = build_parser()
	args = parser.parse_args()
	reference = _resolve(args.reference_config)
	candidate = _resolve(args.candidate_config)
	_print_paths('reference', reference)
	_print_paths('candidate', candidate)
	if args.dry_run:
		print('f3_prepared_parity.execution: dry-run; comparison skipped')
		return

	parity = check_f3_prepared_volume_parity(reference, candidate)
	_print_identity('reference', parity.reference)
	_print_identity('candidate', parity.candidate)
	for mismatch in parity.mismatches:
		print(f'f3_prepared_parity.mismatch: {mismatch}')
	status = 'PASS' if parity.passed else 'FAIL'
	print(f'f3_prepared_parity.status: {status}')
	if not parity.passed:
		msg = (
			f'{len(parity.mismatches)} prepared-volume field(s) differ; do not '
			'reuse the reference checkpoints on the candidate volumes'
		)
		raise SystemExit(msg)


def _resolve(config_path: Path) -> F3PrepareVolumeConfig:
	raw_config = load_config_for_cli(config_path, loader=load_config)
	return resolve_config_for_cli(
		raw_config,
		resolver=f3_prepare_volume_config_from_mapping,
		config_path=config_path,
	)


def _print_paths(role: str, config: F3PrepareVolumeConfig) -> None:
	print(f'{role}.dataset.version: {config.dataset.version}')
	print(f'{role}.seismic_npy: {config.outputs.seismic_npy}')
	print(f'{role}.label_npy: {config.outputs.label_npy}')
	print(f'{role}.metadata_path: {config.outputs.metadata_path}')
	print(f'{role}.normalization_stats_path: {config.outputs.normalization_stats_path}')


def _print_identity(role: str, identity: F3PreparedVolumeIdentity) -> None:
	print(f'{role}.seismic_sha256: {identity.seismic_sha256}')
	print(f'{role}.seismic_shape_xyz: {identity.seismic_shape_xyz}')
	print(f'{role}.seismic_dtype: {identity.seismic_dtype}')
	print(f'{role}.label_sha256: {identity.label_sha256}')
	print(f'{role}.label_shape_xyz: {identity.label_shape_xyz}')
	print(f'{role}.label_dtype: {identity.label_dtype}')
	print(f'{role}.grid_order: {identity.grid_order}')
	for field in NORMALIZATION_PARITY_FIELDS:
		print(f'{role}.normalization.{field}: {identity.normalization[field]!r}')
	class_order = ', '.join(
		f'{entry["class_id"]}={entry["class_name"]}' for entry in identity.class_order
	)
	print(f'{role}.class_order: {class_order}')


if __name__ == '__main__':
	main()
