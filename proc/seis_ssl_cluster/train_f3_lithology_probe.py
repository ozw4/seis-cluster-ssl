"""Train and evaluate an F3 token-level lithology probe."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	DEFAULT_EVALUATION_METRICS,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
	read_f3_lithology_class_info,
	train_and_evaluate_f3_lithology_probe,
)

STAGE = 'train_f3_lithology_probe'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '02_train_linear_probe.yaml'
)


def main() -> None:
	"""Train an F3 lithology probe or print a dry-run summary."""
	parser = ArgumentParser(description='Train an F3 token-level lithology probe.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without training.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	config = f3_lithology_probe_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology probe training skipped')
		return

	result = train_and_evaluate_f3_lithology_probe(config)
	print(f'f3_lithology_probe.probe_joblib: {result.probe_joblib}')
	print(f'f3_lithology_probe.scaler_joblib: {result.scaler_joblib}')
	print(f'f3_lithology_probe.config_json: {result.config_json}')
	print(f'f3_lithology_probe.metrics_json: {result.metrics_json}')
	print(f'f3_lithology_probe.metrics_csv: {result.metrics_csv}')
	print(f'f3_lithology_probe.confusion_matrix_csv: {result.confusion_matrix_csv}')
	print(
		'f3_lithology_probe.classification_report_md: '
		f'{result.classification_report_md}',
	)
	print(f'f3_lithology_probe.confusion_matrix_png: {result.confusion_matrix_png}')
	print(f'f3_lithology_probe.per_class_f1_png: {result.per_class_f1_png}')
	print(f'f3_lithology_probe.train_token_count: {result.train_token_count}')
	print(
		f'f3_lithology_probe.validation_token_count: {result.validation_token_count}',
	)


def f3_lithology_probe_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyProbeConfig:
	"""Validate and normalize the F3 lithology probe config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'embeddings',
				'labels',
				'lithology',
				'token_dataset',
				'probe',
				'evaluation',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	_validate_frozen_encoder(model)
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	token_dataset = _required_mapping(config, 'token_dataset')
	probe = _required_mapping(config, 'probe')
	evaluation = _optional_mapping(config, 'evaluation')
	token_dataset_dir = _required_absolute_path(
		token_dataset,
		'input_dir',
		prefix='token_dataset',
	)
	class_info_path = _required_absolute_path(labels, 'class_info', prefix='labels')
	outputs = F3LithologyProbeOutputs(
		output_dir=_required_absolute_path(probe, 'output_dir', prefix='probe'),
	)
	for label, path in _all_artifact_paths(
		token_dataset_dir=token_dataset_dir,
		class_info_path=class_info_path,
		token_dataset_metadata_json=_optional_absolute_path(
			token_dataset,
			'metadata_json',
			prefix='token_dataset',
		),
		outputs=outputs,
	):
		_validate_artifact_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	return F3LithologyProbeConfig(
		inputs=F3LithologyProbeInputs(
			train_tokens=token_dataset_dir / 'train_tokens.npz',
			validation_tokens=token_dataset_dir / 'validation_tokens.npz',
			class_info=class_info_path,
			token_dataset_metadata_json=_optional_absolute_path(
				token_dataset,
				'metadata_json',
				prefix='token_dataset',
			),
		),
		outputs=outputs,
		classes=read_f3_lithology_class_info(class_info_path),
		probe=_probe_settings_from_mapping(probe),
		dataset=dataset,
		model=model,
		embeddings=embeddings,
		labels=labels,
		token_dataset=token_dataset,
		lithology=lithology,
		evaluation_metrics=_evaluation_metrics(evaluation),
		figure_dpi=_figure_dpi(evaluation),
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
				'output_dir',
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
	return _optional_positive_int(figure.get('dpi', 300), 'evaluation.figure.dpi')


def _all_artifact_paths(
	*,
	token_dataset_dir: Path,
	class_info_path: Path,
	token_dataset_metadata_json: Path | None,
	outputs: F3LithologyProbeOutputs,
) -> tuple[tuple[str, Path], ...]:
	paths: list[tuple[str, Path]] = [
		('token_dataset.input_dir', token_dataset_dir),
		('labels.class_info', class_info_path),
		('probe.output_dir', outputs.output_dir),
		('probe.probe_joblib', outputs.probe_joblib),
		('probe.scaler_joblib', outputs.scaler_joblib),
		('probe.probe_config_resolved_json', outputs.config_json),
		('probe.metrics_json', outputs.metrics_json),
		('probe.metrics_csv', outputs.metrics_csv),
		('probe.confusion_matrix_csv', outputs.confusion_matrix_csv),
		('probe.classification_report_md', outputs.classification_report_md),
		('probe.confusion_matrix_png', outputs.confusion_matrix_png),
		('probe.per_class_f1_png', outputs.per_class_f1_png),
	]
	if token_dataset_metadata_json is not None:
		paths.append(('token_dataset.metadata_json', token_dataset_metadata_json))
	return tuple(paths)


def _print_summary(config: F3LithologyProbeConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'token_dataset.train_tokens: {config.inputs.train_tokens}')
	print(f'token_dataset.validation_tokens: {config.inputs.validation_tokens}')
	print(f'token_dataset.metadata_json: {config.inputs.token_dataset_metadata_json}')
	feature_source = config.token_dataset.get('feature_source')
	if isinstance(feature_source, Mapping):
		print(f'token_dataset.feature_source: {dict(feature_source)}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.checkpoint: {config.model.get("checkpoint")}')
	print(f'model.freeze_encoder: {config.model.get("freeze_encoder")}')
	print(f'probe.spec: {config.probe.spec}')
	print(f'probe.type: {config.probe.probe_type}')
	print(f'probe.feature_scaling: {config.probe.feature_scaling}')
	print(f'probe.class_weight: {config.probe.class_weight}')
	print(f'probe.random_state: {config.probe.random_state}')
	if config.probe.probe_type == 'logistic_regression':
		print(f'probe.max_iter: {config.probe.max_iter}')
	else:
		print(f'probe.hidden_dims: {list(config.probe.hidden_dims)}')
		print(f'probe.dropout: {config.probe.dropout}')
		print(f'probe.max_epochs: {config.probe.max_epochs}')
		print(
			f'probe.early_stopping_patience: {config.probe.early_stopping_patience}',
		)
		print(f'probe.batch_size: {config.probe.batch_size}')
		print(f'probe.learning_rate: {config.probe.learning_rate}')
		print(f'probe.weight_decay: {config.probe.weight_decay}')
	print(f'evaluation.metrics: {list(config.evaluation_metrics)}')
	print(f'evaluation.figure.dpi: {config.figure_dpi}')
	print(f'probe.output_dir: {config.outputs.output_dir}')
	print(f'probe.probe_joblib: {config.outputs.probe_joblib}')
	print(f'probe.scaler_joblib: {config.outputs.scaler_joblib}')
	print(f'probe.probe_config_resolved_json: {config.outputs.config_json}')
	print(f'probe.metrics_json: {config.outputs.metrics_json}')
	print(f'probe.metrics_csv: {config.outputs.metrics_csv}')
	print(f'probe.confusion_matrix_csv: {config.outputs.confusion_matrix_csv}')
	print(f'probe.classification_report_md: {config.outputs.classification_report_md}')
	print(f'probe.confusion_matrix_png: {config.outputs.confusion_matrix_png}')
	print(f'probe.per_class_f1_png: {config.outputs.per_class_f1_png}')
	print(f'classes.count: {len(config.classes)}')


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_str(
	parent: Mapping[str, object],
	key: str,
	*,
	default: str,
	prefix: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_nullable_str(
	parent: Mapping[str, object],
	key: str,
	*,
	default: str | None,
	prefix: str,
) -> str | None:
	value = parent.get(key, default)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string or null; got {value!r}'
		raise TypeError(msg)
	return value


def _string_item(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} entries must be non-empty strings; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{label} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1); got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction < 1.0:
		msg = f'{label} must be in [0, 1); got {value!r}'
		raise ValueError(msg)
	return fraction


def _optional_positive_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0.0:
		msg = f'{label} must be a positive number; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _optional_nonnegative_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool) or value < 0.0:
		msg = f'{label} must be a nonnegative number; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _hidden_dims(value: object) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'probe.hidden_dims must be a list of positive integers; got {value!r}'
		raise TypeError(msg)
	dims = tuple(_optional_positive_int(item, 'probe.hidden_dims') for item in value)
	if not dims:
		msg = 'probe.hidden_dims must contain at least one layer width'
		raise ValueError(msg)
	return dims


def _validate_frozen_encoder(model: Mapping[str, object]) -> None:
	if model.get('freeze_encoder') is not True:
		msg = 'model.freeze_encoder must be true for F3 lithology probe training'
		raise ValueError(msg)


def _validate_artifact_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, f3_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


if __name__ == '__main__':
	main()
