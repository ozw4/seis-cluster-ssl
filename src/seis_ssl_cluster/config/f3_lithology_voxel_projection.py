"""Strict config resolution for nearest F3 token-to-voxel projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArtifactPaths,
	f3_voxel_prediction_artifact_paths,
)
from seis_ssl_cluster.f3.lithology.voxel_projection import (
	F3VoxelProjectionSourceInfo,
	inspect_f3_lithology_token_projection_source,
)

PROJECTION_MODE_NEAREST = 'nearest'


@dataclass(frozen=True)
class F3LithologyVoxelProjectionConfig:
	"""Resolved inputs, outputs, and policy for the V0 voxel projection."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	model: Mapping[str, object]
	class_info: Path
	source: F3VoxelProjectionSourceInfo
	output_paths: F3VoxelPredictionArtifactPaths
	mode: str
	write_probabilities: bool
	overwrite: bool

	@property
	def output_dir(self) -> Path:
		"""Return the projection artifact directory."""
		return self.output_paths.output_dir


def f3_lithology_voxel_projection_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelProjectionConfig:
	"""Strictly validate and resolve one V0 voxel-projection config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'token_predictions',
				'voxel_projection',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	labels = _required_mapping(config, 'labels')
	token_predictions = _required_mapping(config, 'token_predictions')
	projection = _required_mapping(config, 'voxel_projection')
	_validate_sections(
		{
			'paths': paths,
			'dataset': dataset,
			'model': model,
			'labels': labels,
			'token_predictions': token_predictions,
			'voxel_projection': projection,
		}
	)

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	_validate_frozen_encoder(model, stage='F3 lithology V0 voxel projection')
	resolved_dataset = {
		'name': _required_str(dataset, 'name', prefix='dataset'),
		'version': _required_str(dataset, 'version', prefix='dataset'),
	}
	model_tag = _required_str(model, 'tag', prefix='model')
	class_info = _required_absolute_path(labels, 'class_info', prefix='labels')
	input_dir = _required_absolute_path(
		token_predictions, 'input_dir', prefix='token_predictions'
	)
	configured_sources = {
		'predictions': _required_absolute_path(
			token_predictions, 'predictions', prefix='token_predictions'
		),
		'probabilities': _required_absolute_path(
			token_predictions, 'probabilities', prefix='token_predictions'
		),
		'valid_tokens': _required_absolute_path(
			token_predictions, 'valid_tokens', prefix='token_predictions'
		),
		'metadata_json': _required_absolute_path(
			token_predictions, 'metadata_json', prefix='token_predictions'
		),
	}
	source = inspect_f3_lithology_token_projection_source(input_dir)
	_validate_source_binding(
		source,
		configured_sources=configured_sources,
		class_info=class_info,
		dataset=resolved_dataset,
		model_tag=model_tag,
	)
	output_dir = _required_absolute_path(
		projection, 'output_dir', prefix='voxel_projection'
	)
	_validate_artifact_path_not_f3(
		output_dir,
		'voxel_projection.output_dir',
		artifact_root=artifact_root,
		f3_root=f3_root,
	)
	_validate_output_collision(
		input_dir,
		class_info=class_info,
		output_dir=output_dir,
	)
	mode = projection.get('mode', PROJECTION_MODE_NEAREST)
	if not isinstance(mode, str):
		raise TypeError(f'voxel_projection.mode must be a string; got {mode!r}')
	if mode != PROJECTION_MODE_NEAREST:
		raise ValueError(
			'voxel_projection.mode must be "nearest"; '
			f'got {mode!r}'
		)
	write_probabilities = _optional_bool(
		projection, 'write_probabilities', default=False
	)
	overwrite = _optional_bool(projection, 'overwrite', default=False)
	if output_dir.exists() and not overwrite:
		raise FileExistsError(
			f'refusing to overwrite existing output: {output_dir}'
		)
	return F3LithologyVoxelProjectionConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=resolved_dataset,
		model=dict(model),
		class_info=class_info,
		source=source,
		output_paths=f3_voxel_prediction_artifact_paths(output_dir),
		mode=mode,
		write_probabilities=write_probabilities,
		overwrite=overwrite,
	)


def _validate_sections(
	sections: Mapping[str, Mapping[str, object]],
) -> None:
	_validate_allowed_keys(
		sections['paths'], frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(
		sections['dataset'], frozenset({'name', 'version'}), prefix='dataset'
	)
	_validate_allowed_keys(
		sections['model'], frozenset({'tag', 'freeze_encoder'}), prefix='model'
	)
	_validate_allowed_keys(
		sections['labels'], frozenset({'class_info'}), prefix='labels'
	)
	_validate_allowed_keys(
		sections['token_predictions'],
		frozenset(
			{
				'input_dir',
				'predictions',
				'probabilities',
				'valid_tokens',
				'metadata_json',
			}
		),
		prefix='token_predictions',
	)
	_validate_allowed_keys(
		sections['voxel_projection'],
		frozenset({'mode', 'output_dir', 'write_probabilities', 'overwrite'}),
		prefix='voxel_projection',
	)


def _validate_source_binding(
	source: F3VoxelProjectionSourceInfo,
	*,
	configured_sources: Mapping[str, Path],
	class_info: Path,
	dataset: Mapping[str, str],
	model_tag: str,
) -> None:
	actual_sources = {
		'predictions': source.predictions,
		'probabilities': source.probabilities,
		'valid_tokens': source.valid_tokens,
		'metadata_json': source.metadata_json,
	}
	for key, configured in configured_sources.items():
		actual = actual_sources[key]
		if configured.resolve(strict=False) != actual.resolve(strict=False):
			raise ValueError(
				f'token_predictions.{key} must identify {actual}; got {configured}'
			)
		if configured.parent.resolve(strict=False) != source.input_dir.resolve(
			strict=False
		):
			raise ValueError(
				'all token prediction source files must be in '
				f'token_predictions.input_dir ({source.input_dir})'
			)

	_validate_metadata_output_binding(source, actual_sources)
	_validate_metadata_config_identity(
		source,
		class_info=class_info,
		dataset=dataset,
		model_tag=model_tag,
	)


def _validate_metadata_output_binding(
	source: F3VoxelProjectionSourceInfo,
	actual_sources: Mapping[str, Path],
) -> None:
	metadata_outputs = _metadata_mapping(source.metadata, 'outputs')
	identity_keys = {
		'predictions': 'token_predictions',
		'probabilities': 'probability_volume',
		'valid_tokens': 'valid_token_grid',
		'metadata_json': 'metadata_json',
	}
	for source_key, metadata_key in identity_keys.items():
		value = metadata_outputs.get(metadata_key)
		if not isinstance(value, str) or not value:
			raise TypeError(
				f'token prediction metadata outputs.{metadata_key} '
				'must be a non-empty path string'
			)
		if Path(value).resolve(strict=False) != actual_sources[source_key].resolve(
			strict=False
		):
			raise ValueError(
				f'token_predictions.{source_key} does not match token metadata '
				f'outputs.{metadata_key}'
			)



def _validate_metadata_config_identity(
	source: F3VoxelProjectionSourceInfo,
	*,
	class_info: Path,
	dataset: Mapping[str, str],
	model_tag: str,
) -> None:
	metadata_dataset = _metadata_mapping(source.metadata, 'dataset')
	for key, expected in dataset.items():
		if metadata_dataset.get(key) != expected:
			raise ValueError(
				f'dataset.{key} does not match token metadata dataset.{key}'
			)
	if source.model_tag != model_tag:
		raise ValueError(
			'model.tag does not match token metadata model.tag; '
			f'config={model_tag!r}, metadata={source.model_tag!r}'
		)
	class_order = tuple(
		item.class_id for item in read_f3_lithology_class_info(class_info)
	)
	if class_order != source.class_probability_order:
		raise ValueError(
			'class_info class order must match token metadata '
			'class_probability_order; '
			f'class_info={class_order!r}, metadata={source.class_probability_order!r}'
		)


def _metadata_mapping(
	metadata: Mapping[str, object], key: str
) -> Mapping[str, object]:
	value = metadata.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'token prediction metadata {key} must be a mapping')
	return cast('Mapping[str, object]', value)


def _validate_output_collision(
	input_dir: Path,
	*,
	class_info: Path,
	output_dir: Path,
) -> None:
	source = input_dir.resolve(strict=False)
	classes = class_info.resolve(strict=False)
	output = output_dir.resolve(strict=False)
	if source == output or source.is_relative_to(output):
		raise ValueError(
			'voxel_projection.output_dir must differ from and must not contain '
			'token_predictions.input_dir'
		)
	if output.is_relative_to(source):
		raise ValueError(
			'voxel_projection.output_dir must not be inside '
			'token_predictions.input_dir'
		)
	if classes == output or classes.is_relative_to(output) or output.is_relative_to(
		classes
	):
		raise ValueError(
			'voxel_projection.output_dir must not overlap labels.class_info'
		)


def _optional_bool(
	parent: Mapping[str, object], key: str, *, default: bool
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		raise TypeError(f'voxel_projection.{key} must be boolean; got {value!r}')
	return value


__all__ = [
	'PROJECTION_MODE_NEAREST',
	'F3LithologyVoxelProjectionConfig',
	'f3_lithology_voxel_projection_config_from_mapping',
]
