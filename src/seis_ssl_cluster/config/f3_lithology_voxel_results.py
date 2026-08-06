"""Strict configuration for the F3 original-split voxel result summary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_max_file_size_bytes,
	_publish_optional_bool,
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
	_validate_output_not_under_f3_root,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	REQUIRED_MODELS,
	REQUIRED_VERSIONS,
	F3LithologyVoxelResultsConfig,
	F3LithologyVoxelResultsPublishConfig,
	F3LithologyVoxelResultsRun,
)


def f3_lithology_voxel_results_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelResultsConfig:
	"""Validate and resolve the six-run summary configuration."""
	_validate_allowed_keys(
		config, frozenset({'paths', 'runs', 'outputs', 'publish'}), prefix='config'
	)
	paths = _required_mapping(config, 'paths')
	runs = _required_mapping(config, 'runs')
	outputs = _required_mapping(config, 'outputs')
	publish = config.get('publish', {})
	if not isinstance(publish, Mapping):
		raise TypeError('publish must be a mapping')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root', 'results_root'}), prefix='paths'
	)
	_validate_allowed_keys(
		runs,
		frozenset(model.lower().replace('-', '') for model in REQUIRED_MODELS),
		prefix='runs',
	)
	_validate_allowed_keys(
		outputs, frozenset({'output_dir', 'overwrite'}), prefix='outputs'
	)
	_validate_allowed_keys(
		publish,
		frozenset({'enabled', 'output_dir', 'max_file_size_mb', 'overwrite'}),
		prefix='publish',
	)
	_required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	results_root = Path(_required_string(paths, 'results_root', prefix='paths'))
	resolved_runs = []
	for model in REQUIRED_MODELS:
		model_key = model.lower().replace('-', '')
		versions = _required_mapping(runs, model_key)
		_validate_allowed_keys(
			versions,
			frozenset(version.lower() for version in REQUIRED_VERSIONS),
			prefix=f'runs.{model_key}',
		)
		for version in REQUIRED_VERSIONS:
			input_dir = _required_absolute_path(
				versions, version.lower(), prefix=f'runs.{model_key}'
			)
			_validate_output_not_under_f3_root(
				input_dir,
				f'runs.{model_key}.{version.lower()}',
				f3_root=f3_root,
			)
			resolved_runs.append(F3LithologyVoxelResultsRun(model, version, input_dir))
	output_dir = _required_absolute_path(outputs, 'output_dir', prefix='outputs')
	_validate_output_not_under_f3_root(
		output_dir,
		'outputs.output_dir',
		f3_root=f3_root,
	)
	overwrite = _bool(outputs.get('overwrite', False), 'outputs.overwrite')
	publish_enabled = _publish_optional_bool(publish, 'enabled', default=False)
	publish_output = publish.get('output_dir')
	if publish_output is not None:
		if not isinstance(publish_output, str) or not publish_output:
			raise TypeError('publish.output_dir must be a non-empty string')
		publish_output = Path(publish_output)
	if publish_enabled and publish_output is None:
		raise ValueError('publish.output_dir is required when publishing is enabled')
	return F3LithologyVoxelResultsConfig(
		runs=tuple(resolved_runs),
		output_dir=output_dir,
		overwrite=overwrite,
		publish=F3LithologyVoxelResultsPublishConfig(
			enabled=publish_enabled,
			results_root=results_root,
			output_dir=publish_output,
			max_file_size_bytes=_max_file_size_bytes(publish),
			overwrite=_bool(publish.get('overwrite', True), 'publish.overwrite'),
		),
	)


def _required_string(parent: Mapping[str, object], key: str, *, prefix: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{prefix}.{key} must be a non-empty string')
	return value


def _bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value


__all__ = ['f3_lithology_voxel_results_config_from_mapping']
