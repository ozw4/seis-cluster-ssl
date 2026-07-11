"""Run F3 lithology probes across a label-budget suite manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import TYPE_CHECKING

from seis_ssl_cluster.cli import (
	add_config_argument,
	add_dry_run_argument,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_robustness import (
	F3LabelBudgetProbeRunConfig,
	f3_lithology_label_budget_probe_config_from_mapping,
)
from seis_ssl_cluster.f3 import (
	F3LithologyProbeConfig,
	F3LithologyProbeOutputs,
	train_and_evaluate_f3_lithology_probe,
)

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping, Sequence
	from pathlib import Path

STAGE = 'run_f3_lithology_label_budget_probes'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the label-budget probe runner."""
	parser = argparse.ArgumentParser(
		description='Run F3 lithology probes across a label-budget suite manifest.',
	)
	add_config_argument(parser, required=True)
	add_dry_run_argument(
		parser,
		help_text='Validate the config and print planned probe outputs.',
	)
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help='Skip conditions whose metrics.json already exists.',
	)
	return parser


def main() -> None:
	"""Run label-budget probes or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_label_budget_probe_config_from_mapping,
		config_path=config_path,
	)
	validate_paired_hashes(config.rows)
	if args.dry_run:
		_print_dry_run_summary(config)
		return

	result = run_f3_lithology_label_budget_probes(
		config,
		only_missing=bool(args.only_missing),
	)
	print(f'f3_lithology_label_budget_probes.manifest: {result}')
	print(f'f3_lithology_label_budget_probes.condition_count: {len(config.rows)}')


def run_f3_lithology_label_budget_probes(
	config: F3LabelBudgetProbeRunConfig,
	*,
	only_missing: bool = False,
) -> Path:
	"""Train configured probes and write the probe run manifest."""
	validate_paired_hashes(config.rows)
	planned = list(zip(config.rows, config.probe_configs, strict=True))
	_refuse_existing_probe_outputs(
		(
			config
			for _row, config in planned
			if not _skip_config(config, only_missing=only_missing)
		),
		overwrite=config.overwrite,
	)
	manifest_rows: list[dict[str, object]] = []
	for row, probe_config in planned:
		if _skip_config(probe_config, only_missing=only_missing):
			manifest_rows.append(_probe_run_manifest_row(row, probe_config))
			continue
		result = train_and_evaluate_f3_lithology_probe(probe_config)
		manifest_rows.append(
			_probe_run_manifest_row(
				row,
				probe_config,
				train_token_count=result.train_token_count,
				validation_token_count=result.validation_token_count,
			),
		)
	manifest_path = config.output_root / 'probe_run_manifest.json'
	_write_json(
		manifest_path,
		{
			'artifact_type': 'f3_lithology_label_budget_probe_run_manifest',
			'suite_manifest': str(config.manifest),
			'probe': config.probe.to_dict(),
			'rows': manifest_rows,
		},
	)
	return manifest_path


def validate_paired_hashes(rows: Sequence[Mapping[str, object]]) -> None:
	"""Validate baseline/candidate hash pairing before any probe training."""
	by_condition: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
	for row in rows:
		key = (str(row['budget_id']), int(row['subsample_seed']))
		role = str(row['model_role'])
		by_condition[key][role] = str(row['paired_identity_hash'])
	for (budget_id, subsample_seed), hashes in by_condition.items():
		if sorted(hashes) != ['baseline', 'candidate']:
			msg = (
				'label-budget probe condition requires baseline and candidate rows; '
				f'budget_id={budget_id!r}, subsample_seed={subsample_seed}, '
				f'roles={sorted(hashes)!r}'
			)
			raise ValueError(msg)
		if hashes['baseline'] != hashes['candidate']:
			msg = (
				'paired_identity_hash mismatch for label-budget probe condition; '
				f'budget_id={budget_id!r}, subsample_seed={subsample_seed}, '
				f'baseline={hashes["baseline"]}, candidate={hashes["candidate"]}'
			)
			raise ValueError(msg)


def _skip_config(
	probe_config: F3LithologyProbeConfig,
	*,
	only_missing: bool,
) -> bool:
	return only_missing and probe_config.outputs.metrics_json.exists()


def _refuse_existing_probe_outputs(
	configs: Iterable[F3LithologyProbeConfig],
	*,
	overwrite: bool,
) -> None:
	if overwrite:
		return
	for config in configs:
		existing = [
			path for path in _probe_output_files(config.outputs) if path.exists()
		]
		if existing:
			msg = (
				'refusing to overwrite existing probe output(s); '
				f'first existing path: {existing[0]}'
			)
			raise FileExistsError(msg)


def _probe_output_files(outputs: F3LithologyProbeOutputs) -> tuple[Path, ...]:
	return (
		outputs.probe_joblib,
		outputs.scaler_joblib,
		outputs.config_json,
		outputs.metrics_json,
		outputs.metrics_csv,
		outputs.confusion_matrix_csv,
		outputs.classification_report_md,
		outputs.confusion_matrix_png,
		outputs.per_class_f1_png,
	)


def _probe_run_manifest_row(
	row: Mapping[str, object],
	probe_config: F3LithologyProbeConfig,
	*,
	train_token_count: int | None = None,
	validation_token_count: int | None = None,
) -> dict[str, object]:
	return {
		'model_role': row['model_role'],
		'model_tag': row['model_tag'],
		'budget_id': row['budget_id'],
		'per_class_cap': row['per_class_cap'],
		'subsample_seed': row['subsample_seed'],
		'token_dataset_root': row['token_dataset_root'],
		'probe_output_dir': str(probe_config.outputs.output_dir),
		'metrics_json': str(probe_config.outputs.metrics_json),
		'train_token_count': (
			int(row['selected_train_token_count'])
			if train_token_count is None
			else int(train_token_count)
		),
		'validation_token_count': (
			int(row['validation_token_count'])
			if validation_token_count is None
			else int(validation_token_count)
		),
		'paired_identity_hash': row['paired_identity_hash'],
	}


def _print_dry_run_summary(config: F3LabelBudgetProbeRunConfig) -> None:
	print(f'suite manifest: {config.manifest}')
	print(f'condition count: {len(config.rows)}')
	print(f'probe type: {config.probe.probe_type}')
	print(f'probe random_state: {config.probe.random_state}')
	print('expected probe outputs:')
	for probe_config in config.probe_configs:
		print(f'- {probe_config.outputs.output_dir}')
	print('execution: dry-run; probe training skipped')


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


if __name__ == '__main__':
	main()
