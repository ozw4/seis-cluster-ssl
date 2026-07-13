"""Run F3 lithology probes across split/index token datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.cli import (
	add_config_argument,
	add_dry_run_argument,
	load_config_for_cli,
	parse_config_path,
	resolve_config_for_cli,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_common import (
	_hidden_dims,
	_optional_fraction,
	_optional_int,
	_optional_mapping,
	_optional_nonnegative_float,
	_optional_nullable_str,
	_optional_positive_float,
	_optional_positive_int,
	_optional_str,
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_string_item,
	_validate_allowed_keys,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3 import (
	DEFAULT_EVALUATION_METRICS,
	F3ClassInfo,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
	train_and_evaluate_f3_lithology_probe,
)
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info

STAGE = 'run_f3_lithology_split_sweep_probes'
MANIFEST_ARTIFACT_TYPE = 'f3_lithology_split_probe_run_manifest'


@dataclass(frozen=True)
class F3SplitSweepProbeRunConfig:
	"""Resolved config for running probes across split/index datasets."""

	dataset_manifest: Path
	output_root: Path
	probe: F3LithologyProbeSettings
	labels_class_info: Path
	evaluation_metrics: tuple[str, ...]
	figure_dpi: int
	overwrite: bool
	rows: tuple[Mapping[str, object], ...]
	probe_configs: tuple[F3LithologyProbeConfig, ...]


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI parser for the split/index probe runner."""
	parser = argparse.ArgumentParser(
		description='Run F3 lithology probes across split/index token datasets.',
	)
	add_config_argument(parser, required=True)
	add_dry_run_argument(
		parser,
		help_text='Validate the config and print planned probe outputs.',
	)
	parser.add_argument(
		'--only-missing',
		action='store_true',
		help='Skip split/model rows whose metrics.json already exists.',
	)
	return parser


def main() -> None:
	"""Run split/index probes or print a dry-run summary."""
	parser = build_parser()
	args = parser.parse_args()

	config_path = parse_config_path(args)
	raw_config = load_config_for_cli(config_path, loader=load_config)
	config = resolve_config_for_cli(
		raw_config,
		resolver=f3_lithology_split_sweep_probe_config_from_mapping,
		config_path=config_path,
	)
	validate_paired_hashes(config.rows)
	if args.dry_run:
		_print_dry_run_summary(config, only_missing=bool(args.only_missing))
		return

	result = run_f3_lithology_split_sweep_probes(
		config,
		only_missing=bool(args.only_missing),
	)
	print(f'f3_lithology_split_sweep_probes.manifest: {result}')
	print(f'f3_lithology_split_sweep_probes.row_count: {len(config.rows)}')


def run_f3_lithology_split_sweep_probes(
	config: F3SplitSweepProbeRunConfig,
	*,
	only_missing: bool = False,
) -> Path:
	"""Train configured split/index probes and write the run manifest."""
	validate_paired_hashes(config.rows)
	planned = list(zip(config.rows, config.probe_configs, strict=True))
	configs_to_run = [
		probe_config
		for _row, probe_config in planned
		if not _skip_config(probe_config, only_missing=only_missing)
	]
	manifest_path = config.output_root / 'split_probe_run_manifest.json'
	if _skip_manifest_write(
		manifest_path,
		configs_to_run=configs_to_run,
		only_missing=only_missing,
		overwrite=config.overwrite,
	):
		return manifest_path
	_refuse_existing_probe_outputs(
		configs_to_run,
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
	_write_json(
		manifest_path,
		{
			'artifact_type': MANIFEST_ARTIFACT_TYPE,
			'dataset_manifest': str(config.dataset_manifest),
			'probe': config.probe.to_dict(),
			'rows': manifest_rows,
		},
	)
	return manifest_path


def f3_lithology_split_sweep_probe_config_from_mapping(
	config: Mapping[str, object],
) -> F3SplitSweepProbeRunConfig:
	"""Validate and normalize a split/index probe runner config."""
	_validate_allowed_keys(
		config,
		frozenset({'suite', 'probe', 'labels', 'evaluation', 'outputs'}),
		prefix='config',
	)
	suite = _required_mapping(config, 'suite')
	probe_mapping = _required_mapping(config, 'probe')
	labels = _required_mapping(config, 'labels')
	evaluation = _optional_mapping(config, 'evaluation')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		suite,
		frozenset({'dataset_manifest', 'output_root'}),
		prefix='suite',
	)
	_validate_allowed_keys(labels, frozenset({'class_info'}), prefix='labels')
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')

	dataset_manifest = _required_absolute_path(
		suite,
		'dataset_manifest',
		prefix='suite',
	)
	output_root = _required_absolute_path(suite, 'output_root', prefix='suite')
	class_info = _required_absolute_path(labels, 'class_info', prefix='labels')
	manifest_payload = _read_json(dataset_manifest)
	rows = _split_manifest_rows(manifest_payload, manifest_path=dataset_manifest)
	probe = _probe_settings_from_mapping(probe_mapping)
	classes = read_f3_lithology_class_info(class_info)
	evaluation_metrics = _evaluation_metrics(evaluation)
	figure_dpi = _figure_dpi(evaluation)
	probe_configs = tuple(
		_probe_config_for_split_row(
			row,
			output_root=output_root,
			probe=probe,
			class_info=class_info,
			classes=classes,
			evaluation_metrics=evaluation_metrics,
			figure_dpi=figure_dpi,
		)
		for row in rows
	)
	return F3SplitSweepProbeRunConfig(
		dataset_manifest=dataset_manifest,
		output_root=output_root,
		probe=probe,
		labels_class_info=class_info,
		evaluation_metrics=evaluation_metrics,
		figure_dpi=figure_dpi,
		overwrite=_optional_bool(outputs, 'overwrite', default=False, prefix='outputs'),
		rows=rows,
		probe_configs=probe_configs,
	)


def validate_paired_hashes(rows: Sequence[Mapping[str, object]]) -> None:
	"""Validate baseline/candidate hash pairing for every split."""
	by_split: dict[str, dict[str, list[str]]] = defaultdict(
		lambda: defaultdict(list),
	)
	for row in rows:
		split_id = str(row['split_id'])
		role = str(row['model_role'])
		by_split[split_id][role].append(str(row['paired_identity_hash']))
	for split_id, hashes in by_split.items():
		if sorted(hashes) != ['baseline', 'candidate'] or any(
			len(values) != 1 for values in hashes.values()
		):
			msg = (
				'split/index probe condition requires baseline and candidate rows; '
				f'split_id={split_id!r}, roles={sorted(hashes)!r}'
			)
			raise ValueError(msg)
		baseline_hash = hashes['baseline'][0]
		candidate_hash = hashes['candidate'][0]
		if baseline_hash != candidate_hash:
			msg = (
				'paired_identity_hash mismatch for split/index probe condition; '
				f'split_id={split_id!r}, baseline={baseline_hash}, '
				f'candidate={candidate_hash}'
			)
			raise ValueError(msg)


def _split_manifest_rows(
	payload: Mapping[str, object],
	*,
	manifest_path: Path,
) -> tuple[Mapping[str, object], ...]:
	value = payload.get('rows')
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'split dataset manifest rows must be a list: {manifest_path}'
		raise TypeError(msg)
	rows: list[Mapping[str, object]] = []
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			msg = f'split dataset manifest row {index} must be a mapping'
			raise TypeError(msg)
		_validate_split_manifest_row(item, index=index)
		rows.append(item)
	if not rows:
		msg = f'split dataset manifest contains no rows: {manifest_path}'
		raise ValueError(msg)
	return tuple(rows)


def _validate_split_manifest_row(
	row: Mapping[str, object],
	*,
	index: int,
) -> None:
	required = (
		'split_id',
		'model_role',
		'model_tag',
		'token_dataset_root',
		'train_tokens',
		'validation_tokens',
		'metadata_json',
		'train_token_count',
		'validation_token_count',
		'paired_identity_hash',
	)
	missing = [key for key in required if key not in row]
	if missing:
		msg = f'split dataset manifest row {index} missing key(s): {missing!r}'
		raise ValueError(msg)
	for key in ('split_id', 'model_role', 'model_tag', 'paired_identity_hash'):
		_required_str(row, key, prefix=f'rows[{index}]')
	for key in ('train_token_count', 'validation_token_count'):
		_validate_nonnegative_count(row[key], f'rows[{index}].{key}')
	for key in (
		'token_dataset_root',
		'train_tokens',
		'validation_tokens',
		'metadata_json',
	):
		value = Path(_required_str(row, key, prefix=f'rows[{index}]'))
		if not value.is_absolute():
			msg = f'rows[{index}].{key} must be an absolute path; got {value}'
			raise ValueError(msg)


def _probe_config_for_split_row(  # noqa: PLR0913
	row: Mapping[str, object],
	*,
	output_root: Path,
	probe: F3LithologyProbeSettings,
	class_info: Path,
	classes: tuple[F3ClassInfo, ...],
	evaluation_metrics: tuple[str, ...],
	figure_dpi: int,
) -> F3LithologyProbeConfig:
	token_dataset_root = Path(str(row['token_dataset_root']))
	model_tag = str(row['model_tag'])
	split_id = str(row['split_id'])
	output_dir = (
		output_root
		/ 'probes'
		/ f'split={split_id}'
		/ f'model={model_tag}'
		/ probe.spec
	)
	return F3LithologyProbeConfig(
		inputs=F3LithologyProbeInputs(
			train_tokens=Path(str(row['train_tokens'])),
			validation_tokens=Path(str(row['validation_tokens'])),
			class_info=class_info,
			token_dataset_metadata_json=Path(str(row['metadata_json'])),
		),
		outputs=F3LithologyProbeOutputs(output_dir=output_dir),
		classes=classes,
		probe=probe,
		dataset={'name': 'f3_facies_benchmark'},
		model={
			'tag': model_tag,
			'role': str(row['model_role']),
			'freeze_encoder': True,
		},
		embeddings={
			'feature_source': {
				'kind': 'split_index_token_dataset',
				'reference_model_tag': model_tag,
			},
		},
		labels={'class_info': str(class_info)},
		token_dataset={
			'input_dir': str(token_dataset_root),
			'metadata_json': str(row['metadata_json']),
			'split_id': split_id,
			'paired_identity_hash': row['paired_identity_hash'],
		},
		lithology={'suite_output_root': str(output_root)},
		evaluation_metrics=evaluation_metrics,
		figure_dpi=figure_dpi,
	)


def _probe_settings_from_mapping(
	probe: Mapping[str, object],
) -> F3LithologyProbeSettings:
	_validate_allowed_keys(
		probe,
		frozenset(
			{
				'spec',
				'type',
				'feature_scaling',
				'class_weight',
				'max_iter',
				'random_state',
				'hidden_dims',
				'dropout',
				'max_epochs',
				'early_stopping_patience',
				'batch_size',
				'learning_rate',
				'weight_decay',
			},
		),
		prefix='probe',
	)
	return F3LithologyProbeSettings(
		spec=_required_str(probe, 'spec', prefix='probe'),
		probe_type=_required_str(probe, 'type', prefix='probe'),
		feature_scaling=_optional_str(
			probe,
			'feature_scaling',
			default='standard',
			prefix='probe',
		),
		class_weight=_optional_nullable_str(
			probe,
			'class_weight',
			default='balanced',
			prefix='probe',
		),
		max_iter=_optional_positive_int(probe.get('max_iter', 2000), 'probe.max_iter'),
		hidden_dims=_hidden_dims(probe.get('hidden_dims', (256, 128))),
		dropout=_optional_fraction(probe.get('dropout', 0.2), 'probe.dropout'),
		max_epochs=_optional_positive_int(
			probe.get('max_epochs', 200),
			'probe.max_epochs',
		),
		early_stopping_patience=_optional_positive_int(
			probe.get('early_stopping_patience', 20),
			'probe.early_stopping_patience',
		),
		batch_size=_optional_positive_int(
			probe.get('batch_size', 1024),
			'probe.batch_size',
		),
		learning_rate=_optional_positive_float(
			probe.get('learning_rate', 1.0e-3),
			'probe.learning_rate',
		),
		weight_decay=_optional_nonnegative_float(
			probe.get('weight_decay', 0.0),
			'probe.weight_decay',
		),
		random_state=_optional_int(
			probe.get('random_state', 42),
			'probe.random_state',
		),
	)


def _evaluation_metrics(evaluation: Mapping[str, object]) -> tuple[str, ...]:
	value = evaluation.get('metrics', DEFAULT_EVALUATION_METRICS)
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'evaluation.metrics must be a list of metric names; got {value!r}'
		raise TypeError(msg)
	metrics = tuple(_string_item(item, 'evaluation.metrics') for item in value)
	if not metrics:
		msg = 'evaluation.metrics must contain at least one metric name'
		raise ValueError(msg)
	return metrics


def _figure_dpi(evaluation: Mapping[str, object]) -> int:
	figure = evaluation.get('figure')
	if figure is None:
		return 300
	if not isinstance(figure, Mapping):
		msg = f'evaluation.figure must be a mapping; got {figure!r}'
		raise TypeError(msg)
	_validate_allowed_keys(figure, frozenset({'dpi'}), prefix='evaluation.figure')
	return _optional_positive_int(figure.get('dpi', 300), 'evaluation.figure.dpi')


def _optional_bool(
	mapping: Mapping[str, object],
	key: str,
	*,
	default: bool,
	prefix: str,
) -> bool:
	value = mapping.get(key, default)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_nonnegative_count(value: object, label: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{label} must be a nonnegative integer; got {value!r}'
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


def _skip_manifest_write(
	manifest_path: Path,
	*,
	configs_to_run: Sequence[F3LithologyProbeConfig],
	only_missing: bool,
	overwrite: bool,
) -> bool:
	if overwrite or not manifest_path.exists():
		return False
	if not configs_to_run:
		return True
	if only_missing:
		return False
	msg = (
		'refusing to overwrite existing split probe run manifest; '
		f'path: {manifest_path}'
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
	probe_joblib = probe_config.outputs.probe_joblib
	scaler_joblib = probe_config.outputs.scaler_joblib
	return {
		'split_id': row['split_id'],
		'model_role': row['model_role'],
		'model_tag': row['model_tag'],
		'token_dataset_root': row['token_dataset_root'],
		'probe_output_dir': str(probe_config.outputs.output_dir),
		'probe_spec': probe_config.probe.spec,
		'probe_joblib': {
			'path': str(probe_joblib),
			'sha256': file_sha256(probe_joblib),
		},
		'scaler_joblib': {
			'path': str(scaler_joblib),
			'sha256': file_sha256(scaler_joblib),
		},
		'metrics_json': str(probe_config.outputs.metrics_json),
		'train_token_count': (
			int(row['train_token_count'])
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


def _print_dry_run_summary(
	config: F3SplitSweepProbeRunConfig,
	*,
	only_missing: bool,
) -> None:
	expected_run_count = sum(
		1
		for probe_config in config.probe_configs
		if not _skip_config(probe_config, only_missing=only_missing)
	)
	print(f'stage: {STAGE}')
	print(f'split dataset manifest: {config.dataset_manifest}')
	print(f'row count: {len(config.rows)}')
	print(f'expected run count: {expected_run_count}')
	print(f'probe type: {config.probe.probe_type}')
	print(f'probe random_state: {config.probe.random_state}')
	print('expected probe outputs:')
	for probe_config in config.probe_configs:
		print(f'- {probe_config.outputs.output_dir}')
	print('execution: dry-run; probe training skipped')


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		msg = f'JSON document must be a mapping: {path}'
		raise TypeError(msg)
	return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


if __name__ == '__main__':
	main()
