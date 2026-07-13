"""Strict config for frozen-embedding F3 voxel-decoder training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_nonnegative_float,
	_optional_positive_float,
	_optional_positive_int,
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
	_validate_frozen_encoder,
)

if TYPE_CHECKING:
	from pathlib import Path


@dataclass(frozen=True)
class VoxelDecoderSpec:
	"""Architecture of the V1 dense decoder."""

	spec: str
	embedding_dim: int
	class_count: int
	hidden_channels: tuple[int, ...]
	upsample_factors: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class VoxelDecoderTileSettings:
	"""Deterministic core and context geometry."""

	core_size_tokens: tuple[int, int, int]
	context_halo_tokens: tuple[int, int, int]


@dataclass(frozen=True)
class VoxelDecoderTrainSettings:
	"""Fixed-epoch training settings."""

	epochs: int
	batch_size: int
	learning_rate: float
	weight_decay: float
	class_weight: str
	seed: int
	num_workers: int
	amp: bool
	gradient_clip_norm: float


@dataclass(frozen=True)
class F3LithologyVoxelDecoderConfig:
	"""Resolved paths and settings for one decoder job."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	model: Mapping[str, object]
	embeddings_input_dir: Path
	voxel_dataset_input_dir: Path
	decoder: VoxelDecoderSpec
	tiles: VoxelDecoderTileSettings
	train: VoxelDecoderTrainSettings
	output_dir: Path
	embeddings: Mapping[str, object]

	@property
	def survey_id(self) -> str:
		"""Return the embedding artifact prefix."""
		return self.dataset['name']

	def to_dict(self) -> dict[str, object]:
		"""Return the canonical JSON-compatible resolved config."""
		return {
			'paths': {
				'artifact_root': str(self.artifact_root),
				'f3_root': str(self.f3_root),
			},
			'dataset': dict(self.dataset),
			'model': dict(self.model),
			'embeddings': {
				**dict(self.embeddings),
				'input_dir': str(self.embeddings_input_dir),
			},
			'voxel_dataset': {'input_dir': str(self.voxel_dataset_input_dir)},
			'decoder': _plain_dataclass(self.decoder),
			'tiles': _plain_dataclass(self.tiles),
			'train': _plain_dataclass(self.train),
			'outputs': {'output_dir': str(self.output_dir)},
		}


def f3_lithology_voxel_decoder_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyVoxelDecoderConfig:
	"""Strictly validate and resolve a voxel-decoder job config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'embeddings',
				'voxel_dataset',
				'decoder',
				'tiles',
				'train',
				'outputs',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	embeddings = _required_mapping(config, 'embeddings')
	voxel_dataset = _required_mapping(config, 'voxel_dataset')
	decoder = _required_mapping(config, 'decoder')
	tiles = _required_mapping(config, 'tiles')
	train = _required_mapping(config, 'train')
	outputs = _required_mapping(config, 'outputs')

	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(model, frozenset({'tag', 'freeze_encoder'}), prefix='model')
	_validate_allowed_keys(
		embeddings, frozenset({'input_dir', 'spec'}), prefix='embeddings'
	)
	_validate_allowed_keys(
		voxel_dataset, frozenset({'input_dir'}), prefix='voxel_dataset'
	)
	_validate_allowed_keys(
		decoder,
		frozenset(
			{
				'spec',
				'embedding_dim',
				'class_count',
				'hidden_channels',
				'upsample_factors',
			}
		),
		prefix='decoder',
	)
	_validate_allowed_keys(
		tiles,
		frozenset({'core_size_tokens', 'context_halo_tokens'}),
		prefix='tiles',
	)
	_validate_allowed_keys(
		train,
		frozenset(
			{
				'epochs',
				'batch_size',
				'learning_rate',
				'weight_decay',
				'class_weight',
				'seed',
				'num_workers',
				'amp',
				'gradient_clip_norm',
			}
		),
		prefix='train',
	)
	_validate_allowed_keys(outputs, frozenset({'output_dir'}), prefix='outputs')
	_validate_frozen_encoder(model, stage='F3 voxel decoder training')

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	embeddings_dir = _required_absolute_path(
		embeddings, 'input_dir', prefix='embeddings'
	)
	voxel_dir = _required_absolute_path(
		voxel_dataset, 'input_dir', prefix='voxel_dataset'
	)
	output_dir = _required_absolute_path(outputs, 'output_dir', prefix='outputs')
	for label, path in (
		('embeddings.input_dir', embeddings_dir),
		('voxel_dataset.input_dir', voxel_dir),
		('outputs.output_dir', output_dir),
	):
		_validate_artifact_path_not_f3(
			path, label, artifact_root=artifact_root, f3_root=f3_root
		)

	spec = _required_str(decoder, 'spec', prefix='decoder')
	if spec != 'frozen_embedding_decoder_v1':
		raise ValueError(
			f'decoder.spec must be frozen_embedding_decoder_v1; got {spec!r}'
		)
	class_weight = _required_str(train, 'class_weight', prefix='train')
	if class_weight != 'balanced':
		raise ValueError("train.class_weight must be 'balanced'")
	amp = train.get('amp')
	if not isinstance(amp, bool):
		raise TypeError(f'train.amp must be a boolean; got {amp!r}')
	seed = train.get('seed')
	if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
		raise ValueError(f'train.seed must be a non-negative integer; got {seed!r}')
	num_workers = train.get('num_workers')
	if (
		not isinstance(num_workers, int)
		or isinstance(num_workers, bool)
		or num_workers < 0
	):
		raise ValueError(
			f'train.num_workers must be a non-negative integer; got {num_workers!r}'
		)

	return F3LithologyVoxelDecoderConfig(
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
		embeddings_input_dir=embeddings_dir,
		voxel_dataset_input_dir=voxel_dir,
		decoder=VoxelDecoderSpec(
			spec=spec,
			embedding_dim=_optional_positive_int(
				decoder.get('embedding_dim'), 'decoder.embedding_dim'
			),
			class_count=_optional_positive_int(
				decoder.get('class_count'), 'decoder.class_count'
			),
			hidden_channels=_positive_sequence(
				decoder.get('hidden_channels'), 'decoder.hidden_channels'
			),
			upsample_factors=_factor_sequence(decoder.get('upsample_factors')),
		),
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
		train=VoxelDecoderTrainSettings(
			epochs=_optional_positive_int(train.get('epochs'), 'train.epochs'),
			batch_size=_optional_positive_int(
				train.get('batch_size'), 'train.batch_size'
			),
			learning_rate=_optional_positive_float(
				train.get('learning_rate'), 'train.learning_rate'
			),
			weight_decay=_optional_nonnegative_float(
				train.get('weight_decay'), 'train.weight_decay'
			),
			class_weight=class_weight,
			seed=seed,
			num_workers=num_workers,
			amp=amp,
			gradient_clip_norm=_optional_positive_float(
				train.get('gradient_clip_norm'), 'train.gradient_clip_norm'
			),
		),
		output_dir=output_dir,
		embeddings={
			**(
				{'spec': _required_str(embeddings, 'spec', prefix='embeddings')}
				if 'spec' in embeddings
				else {}
			)
		},
	)


def _positive_sequence(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError(f'{label} must be a non-empty list of positive integers')
	return tuple(_optional_positive_int(item, label) for item in value)


def _triplet(value: object, label: str, *, positive: bool) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		raise TypeError(f'{label} must be an integer triple')
	items = tuple(value)
	minimum = 1 if positive else 0
	if any(
		not isinstance(item, int) or isinstance(item, bool) or item < minimum
		for item in items
	):
		qualifier = 'positive' if positive else 'non-negative'
		raise ValueError(f'{label} must contain {qualifier} integers')
	return (items[0], items[1], items[2])


def _factor_sequence(value: object) -> tuple[tuple[int, int, int], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError('decoder.upsample_factors must be a non-empty list')
	return tuple(
		_triplet(item, 'decoder.upsample_factors', positive=True) for item in value
	)


def _plain_dataclass(value: object) -> dict[str, object]:
	payload = asdict(value)  # type: ignore[arg-type]
	return {
		key: [list(child) if isinstance(child, tuple) else child for child in item]
		if isinstance(item, tuple)
		else item
		for key, item in payload.items()
	}


__all__ = [
	'F3LithologyVoxelDecoderConfig',
	'VoxelDecoderSpec',
	'VoxelDecoderTileSettings',
	'VoxelDecoderTrainSettings',
	'f3_lithology_voxel_decoder_config_from_mapping',
]
