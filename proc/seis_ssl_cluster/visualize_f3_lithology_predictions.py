"""Visualize F3 lithology token predictions on seismic slices."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3LithologyVisualizationConfig,
	F3LithologyVisualizationFigureConfig,
	F3LithologyVisualizationInputs,
	F3LithologyVisualizationOutputs,
	read_f3_lithology_visualization_classes,
	visualize_f3_lithology_predictions,
)

STAGE = 'visualize_f3_lithology_predictions'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '05_visualize_predictions.yaml'
)


def main() -> None:
	"""Write F3 lithology prediction figures or print a dry-run summary."""
	parser = ArgumentParser(description='Visualize F3 lithology predictions.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing figures.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	config = f3_lithology_visualization_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology prediction visualization skipped')
		return

	result = visualize_f3_lithology_predictions(config)
	print(f'f3_lithology_visualization.metadata_json: {result.metadata_json}')
	print(f'f3_lithology_visualization.figure_count: {len(result.png_paths)}')


def f3_lithology_visualization_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVisualizationConfig:
	"""Validate and normalize the F3 lithology visualization config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'registry',
				'lithology',
				'probe',
				'predictions',
				'visualizations',
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
	labels = _required_mapping(config, 'labels')
	registry = _required_mapping(config, 'registry')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _required_mapping(config, 'predictions')
	visualizations = _required_mapping(config, 'visualizations')
	inputs = F3LithologyVisualizationInputs(
		seismic_volume=_required_absolute_path(
			registry,
			'seismic_volume',
			prefix='registry',
		),
		label_volume=_required_absolute_path(
			labels,
			'source_label_volume',
			prefix='labels',
		),
		class_info=_required_absolute_path(labels, 'class_info', prefix='labels'),
		png_label_inventory=_required_absolute_path(
			labels,
			'png_label_inventory',
			prefix='labels',
		),
		segy_geometry_json=_required_absolute_path(
			labels,
			'segy_geometry_json',
			prefix='labels',
		),
		token_predictions=_required_absolute_path(
			predictions,
			'token_predictions',
			prefix='predictions',
		),
		probability_volume=_required_absolute_path(
			predictions,
			'probability_volume',
			prefix='predictions',
		),
		prediction_metadata_json=_required_absolute_path(
			predictions,
			'metadata_json',
			prefix='predictions',
		),
		validation_slice_metrics_csv=_required_absolute_path(
			predictions,
			'validation_slice_metrics_csv',
			prefix='predictions',
		),
	)
	outputs = F3LithologyVisualizationOutputs(
		output_dir=_required_absolute_path(
			visualizations,
			'output_dir',
			prefix='visualizations',
		),
		metadata_json=_required_absolute_path(
			visualizations,
			'metadata_json',
			prefix='visualizations',
		),
		selected_slices_dir=_required_absolute_path(
			visualizations,
			'selected_slices_dir',
			prefix='visualizations',
		),
	)
	for label, path in _all_paths(inputs, outputs, labels):
		_validate_path(path, label, artifact_root=artifact_root, f3_root=f3_root)
	return F3LithologyVisualizationConfig(
		inputs=inputs,
		outputs=outputs,
		classes=read_f3_lithology_visualization_classes(inputs.class_info),
		dataset=dataset,
		model=model,
		labels=labels,
		lithology=lithology,
		probe=probe,
		predictions=predictions,
		selected_slices=_selected_slices_from_mapping(
			_required_mapping(visualizations, 'slices'),
		),
		figure=_figure_config_from_mapping(
			_optional_mapping(visualizations, 'figure'),
		),
	)


def _selected_slices_from_mapping(
	slices: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
	_validate_allowed_keys(
		slices,
		frozenset({'inline', 'crossline', 'z'}),
		prefix='visualizations.slices',
	)
	return {
		key: _int_tuple(slices.get(key, ()), f'visualizations.slices.{key}')
		for key in ('inline', 'crossline', 'z')
	}


def _figure_config_from_mapping(
	figure: Mapping[str, object],
) -> F3LithologyVisualizationFigureConfig:
	output_formats = figure.get('output_formats', ['png'])
	if output_formats != ['png']:
		msg = 'visualizations.figure.output_formats must be ["png"]'
		raise ValueError(msg)
	return F3LithologyVisualizationFigureConfig(
		dpi=_optional_positive_int(
			figure.get('dpi', 300),
			'visualizations.figure.dpi',
		),
		background=_optional_str(
			figure,
			'background',
			default='white',
			prefix='visualizations.figure',
		),
		z_axis=_optional_str(
			figure,
			'z_axis',
			default='down',
			prefix='visualizations.figure',
		),
		include_legend=_optional_bool(
			figure.get('include_legend', True),
			'visualizations.figure.include_legend',
		),
		include_confidence=_optional_bool(
			figure.get('include_confidence', False),
			'visualizations.figure.include_confidence',
		),
		amplitude_clip_percentiles=_percentiles(
			figure.get('amplitude_clip_percentiles', (1.0, 99.0)),
		),
	)


def _all_paths(
	inputs: F3LithologyVisualizationInputs,
	outputs: F3LithologyVisualizationOutputs,
	labels: Mapping[str, object],
) -> tuple[tuple[str, Path], ...]:
	paths: list[tuple[str, Path]] = [
		('registry.seismic_volume', inputs.seismic_volume),
		('labels.source_label_volume', inputs.label_volume),
		('labels.class_info', inputs.class_info),
		('labels.png_label_inventory', inputs.png_label_inventory),
		('labels.segy_geometry_json', inputs.segy_geometry_json),
		('predictions.token_predictions', inputs.token_predictions),
		('predictions.probability_volume', inputs.probability_volume),
		('predictions.metadata_json', inputs.prediction_metadata_json),
		(
			'predictions.validation_slice_metrics_csv',
			inputs.validation_slice_metrics_csv,
		),
		('visualizations.output_dir', outputs.output_dir),
		('visualizations.metadata_json', outputs.metadata_json),
		('visualizations.selected_slices_dir', outputs.selected_slices_dir),
	]
	source_label = _optional_absolute_path(
		labels,
		'source_label_segy',
		prefix='labels',
	)
	if source_label is not None:
		paths.append(('labels.source_label_segy', source_label))
	return tuple(paths)


def _print_summary(config: F3LithologyVisualizationConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'registry.seismic_volume: {config.inputs.seismic_volume}')
	print(f'labels.source_label_volume: {config.inputs.label_volume}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'labels.png_label_inventory: {config.inputs.png_label_inventory}')
	print(f'labels.segy_geometry_json: {config.inputs.segy_geometry_json}')
	print(f'predictions.token_predictions: {config.inputs.token_predictions}')
	print(f'predictions.probability_volume: {config.inputs.probability_volume}')
	print(f'predictions.metadata_json: {config.inputs.prediction_metadata_json}')
	print(
		'predictions.validation_slice_metrics_csv: '
		f'{config.inputs.validation_slice_metrics_csv}',
	)
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.freeze_encoder: {config.model.get("freeze_encoder")}')
	print(f'probe.spec: {config.probe.get("spec")}')
	print(f'visualizations.output_dir: {config.outputs.output_dir}')
	print(f'visualizations.metadata_json: {config.outputs.metadata_json}')
	print(f'visualizations.selected_slices_dir: {config.outputs.selected_slices_dir}')
	print(f'visualizations.slices: {config.selected_slices}')
	print(f'visualizations.figure.dpi: {config.figure.dpi}')
	print(f'visualizations.figure.z_axis: {config.figure.z_axis}')
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


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{label} must be boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a list of integer slice indices; got {value!r}'
		raise TypeError(msg)
	items = tuple(value)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integer slice indices; got {value!r}'
		raise TypeError(msg)
	return items


def _percentiles(value: object) -> tuple[float, float]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
	):
		msg = (
			'visualizations.figure.amplitude_clip_percentiles must contain two '
			f'values; got {value!r}'
		)
		raise TypeError(msg)
	low = float(value[0])
	high = float(value[1])
	return (low, high)


def _validate_frozen_encoder(model: Mapping[str, object]) -> None:
	if model.get('freeze_encoder') is not True:
		msg = 'model.freeze_encoder must be true for F3 lithology visualization'
		raise ValueError(msg)


def _validate_path(
	path: Path | None,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if path is None:
		return
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if label == 'labels.source_label_segy':
		if not _is_relative_to(path, f3_root):
			msg = f'{label} must be under paths.f3_root ({f3_root}); got {path}'
			raise ValueError(msg)
		return
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
