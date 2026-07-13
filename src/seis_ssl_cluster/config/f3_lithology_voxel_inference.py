"""Strict config for chunked frozen-decoder voxel inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
	_validate_artifact_or_f3_source_path,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderTileSettings,
	_triplet,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArtifactPaths,
	f3_voxel_prediction_artifact_paths,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


@dataclass(frozen=True)
class F3LithologyVoxelInferenceConfig:
	"""Resolved sources, geometry, and output policy for decoder inference."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	model: Mapping[str, object]
	class_info: Path
	embeddings_input_dir: Path
	checkpoint: Path
	tiles: VoxelDecoderTileSettings
	output_paths: F3VoxelPredictionArtifactPaths
	write_probabilities: bool
	overwrite: bool

	@property
	def output_dir(self) -> Path:
		"""Return the final prediction artifact directory."""
		return self.output_paths.output_dir


def f3_lithology_voxel_inference_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelInferenceConfig:
	"""Strictly validate and resolve one chunked voxel-inference job."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'embeddings',
				'decoder',
				'tiles',
				'inference',
				'outputs',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	labels = _required_mapping(config, 'labels')
	embeddings = _required_mapping(config, 'embeddings')
	decoder = _required_mapping(config, 'decoder')
	tiles = _required_mapping(config, 'tiles')
	inference = _required_mapping(config, 'inference')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(model, frozenset({'tag', 'freeze_encoder'}), prefix='model')
	_validate_allowed_keys(labels, frozenset({'class_info'}), prefix='labels')
	_validate_allowed_keys(
		embeddings, frozenset({'input_dir', 'spec'}), prefix='embeddings'
	)
	_validate_allowed_keys(decoder, frozenset({'checkpoint'}), prefix='decoder')
	_validate_allowed_keys(
		tiles,
		frozenset({'core_size_tokens', 'context_halo_tokens'}),
		prefix='tiles',
	)
	_validate_allowed_keys(
		inference,
		frozenset({'write_probabilities', 'overwrite'}),
		prefix='inference',
	)
	_validate_allowed_keys(outputs, frozenset({'output_dir'}), prefix='outputs')
	_validate_frozen_encoder(model, stage='F3 voxel decoder inference')

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	class_info = _required_absolute_path(labels, 'class_info', prefix='labels')
	embeddings_input_dir = _required_absolute_path(
		embeddings, 'input_dir', prefix='embeddings'
	)
	checkpoint = _required_absolute_path(decoder, 'checkpoint', prefix='decoder')
	output_dir = _required_absolute_path(outputs, 'output_dir', prefix='outputs')
	for label, path in (
		('labels.class_info', class_info),
		('embeddings.input_dir', embeddings_input_dir),
		('decoder.checkpoint', checkpoint),
	):
		_validate_artifact_or_f3_source_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	_validate_artifact_path_not_f3(
		output_dir,
		'outputs.output_dir',
		artifact_root=artifact_root,
		f3_root=f3_root,
	)
	_validate_no_output_overlap(
		output_dir,
		sources=(class_info, embeddings_input_dir, checkpoint.parent),
	)
	return F3LithologyVoxelInferenceConfig(
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		model={
			'tag': _required_str(model, 'tag', prefix='model'),
			'freeze_encoder': True,
		},
		class_info=class_info,
		embeddings_input_dir=embeddings_input_dir,
		checkpoint=checkpoint,
		tiles=VoxelDecoderTileSettings(
			core_size_tokens=_triplet(
				tiles.get('core_size_tokens'), 'tiles.core_size_tokens', positive=True
			),
			context_halo_tokens=_triplet(
				tiles.get('context_halo_tokens'),
				'tiles.context_halo_tokens',
				positive=False,
			),
		),
		output_paths=f3_voxel_prediction_artifact_paths(output_dir),
		write_probabilities=_optional_bool(
			inference, 'write_probabilities', default=False
		),
		overwrite=_optional_bool(inference, 'overwrite', default=False),
	)


def _optional_bool(
	parent: Mapping[str, object], key: str, *, default: bool
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		raise TypeError(f'inference.{key} must be boolean; got {value!r}')
	return value


def _validate_no_output_overlap(
	output_dir: Path, *, sources: tuple[Path, ...]
) -> None:
	output = output_dir.resolve(strict=False)
	for source_path in sources:
		source = source_path.resolve(strict=False)
		if (
			output == source
			or output.is_relative_to(source)
			or source.is_relative_to(output)
		):
			raise ValueError('outputs.output_dir must not overlap an inference source')


__all__ = [
	'F3LithologyVoxelInferenceConfig',
	'f3_lithology_voxel_inference_config_from_mapping',
]
