"""Context-halo chunked inference for frozen F3 voxel decoders."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	ARTIFACT_TYPE,
	INVALID_CONFIDENCE_VALUE,
	INVALID_PREDICTION_CLASS_ID,
	SCHEMA_VERSION,
	F3VoxelPredictionArrays,
	F3VoxelPredictionArtifactPaths,
	commit_f3_voxel_prediction_artifact,
	create_f3_voxel_prediction_staging_paths,
	open_f3_voxel_prediction_memmaps,
	validate_f3_voxel_prediction_arrays,
	write_f3_voxel_prediction_metadata,
)
from seis_ssl_cluster.f3.lithology.voxel_tiles import read_voxel_tile_manifest
from seis_ssl_cluster.models.voxel_decoder import (
	VoxelDecoder3D,
	validate_context_halo_tokens,
	validate_voxel_decoder_architecture,
	validate_voxel_decoder_architecture_mapping,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
		F3LithologyVoxelInferenceConfig,
	)
	from seis_ssl_cluster.f3.io.labels import F3ClassInfo


@dataclass(frozen=True)
class VoxelDecoderInferencePlan:
	"""Identity-checked decoder sources and full-volume geometry."""

	embeddings: Path
	valid_tokens: Path
	embedding_metadata: Path
	checkpoint: Path
	resolved_decoder_config: Path
	train_tile_manifest: Path
	validation_tile_manifest: Path
	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	embedding_dim: int
	class_ids: tuple[int, ...]
	classes: tuple[F3ClassInfo, ...]
	decoder_spec: Mapping[str, object]
	checkpoint_payload: Mapping[str, object]
	artifact_identities: Mapping[str, object]


@dataclass(frozen=True)
class VoxelDecoderInferenceResult:
	"""Committed output paths and inference coverage counts."""

	paths: F3VoxelPredictionArtifactPaths
	volume_shape_xyz: tuple[int, int, int]
	valid_voxel_count: int
	invalid_voxel_count: int
	tile_count: int
	probabilities_written: bool


@dataclass(frozen=True)
class _InferenceTile:
	core_start: tuple[int, int, int]
	core_stop: tuple[int, int, int]
	input_start: tuple[int, int, int]
	input_stop: tuple[int, int, int]
	padding_before: tuple[int, int, int]
	padding_after: tuple[int, int, int]


def inspect_f3_lithology_voxel_inference(
	config: F3LithologyVoxelInferenceConfig,
	*,
	verify_array_hashes: bool = False,
) -> VoxelDecoderInferencePlan:
	"""Validate small source artifacts and checkpoint-bound identities."""
	if config.checkpoint.name != 'best.pt':
		raise ValueError('decoder.checkpoint must identify the selected best.pt')
	files = output_paths(config.embeddings_input_dir, config.dataset['name'])
	resolved_config_path = config.checkpoint.parent / 'resolved_config.json'
	train_manifest_path = config.checkpoint.parent / 'train_tile_manifest.json'
	validation_manifest_path = (
		config.checkpoint.parent / 'validation_tile_manifest.json'
	)
	for path in (
		files.embeddings,
		files.valid_tokens,
		files.metadata,
		config.class_info,
		config.checkpoint,
		resolved_config_path,
		train_manifest_path,
		validation_manifest_path,
	):
		if not path.is_file():
			raise FileNotFoundError(f'missing voxel inference input: {path}')

	checkpoint = load_voxel_decoder_checkpoint(config.checkpoint, map_location='cpu')
	resolved_config = _read_json_object(resolved_config_path)
	checkpoint_config = _mapping(
		checkpoint.get('resolved_config'), 'checkpoint resolved_config'
	)
	checkpoint_architecture = validate_voxel_decoder_architecture_mapping(
		checkpoint.get('decoder_architecture'),
		field_prefix='checkpoint decoder_architecture',
	)
	checkpoint_config_architecture = validate_voxel_decoder_architecture_mapping(
		checkpoint_config.get('decoder'),
		field_prefix='checkpoint resolved_config.decoder',
	)
	resolved_architecture = validate_voxel_decoder_architecture_mapping(
		resolved_config.get('decoder'),
		field_prefix='resolved_config.json.decoder',
	)
	if not (
		checkpoint_architecture
		== checkpoint_config_architecture
		== resolved_architecture
	):
		raise ValueError('decoder architecture identity mismatch')
	if checkpoint_config != resolved_config:
		raise ValueError('decoder resolved config does not match checkpoint')
	_validate_config_binding(config, resolved_config)
	identities = _mapping(
		checkpoint.get('artifact_identities'), 'checkpoint artifact_identities'
	)
	_validate_artifact_identities(
		identities,
		embeddings=files.embeddings,
		valid_tokens=files.valid_tokens,
		embedding_metadata=files.metadata,
		verify_array_hashes=verify_array_hashes,
	)
	_validate_manifest_identities(
		checkpoint,
		train_path=train_manifest_path,
		validation_path=validation_manifest_path,
	)

	embedding_metadata = _read_json_object(files.metadata)
	volume_shape = _positive_triplet(
		embedding_metadata.get('volume_shape_xyz'), 'embedding volume_shape_xyz'
	)
	token_shape = _positive_triplet(
		embedding_metadata.get('token_grid_shape'), 'embedding token_grid_shape'
	)
	patch_size = _positive_triplet(
		embedding_metadata.get('patch_size'), 'embedding patch_size'
	)
	embedding_dim = _embedding_dim(embedding_metadata)
	decoder_spec = resolved_architecture
	if decoder_spec.get('embedding_dim') != embedding_dim:
		raise ValueError('decoder embedding_dim does not match selected embeddings')
	validate_voxel_decoder_architecture(
		hidden_channels=_positive_sequence(
			decoder_spec.get('hidden_channels'), 'decoder.hidden_channels'
		),
		upsample_factors=_factor_sequence(decoder_spec.get('upsample_factors')),
		patch_size_xyz=patch_size,
	)
	validate_context_halo_tokens(
		context_halo_tokens=config.tiles.context_halo_tokens,
		core_size_tokens=config.tiles.core_size_tokens,
		token_grid_shape_xyz=token_shape,
		upsample_factors=_factor_sequence(decoder_spec.get('upsample_factors')),
	)
	classes = read_f3_lithology_class_info(config.class_info)
	class_ids = tuple(item.class_id for item in classes)
	if len(class_ids) != decoder_spec.get('class_count'):
		raise ValueError('class_info count does not match decoder class_count')
	_validate_checkpoint_classes(identities, classes=classes)
	return VoxelDecoderInferencePlan(
		embeddings=files.embeddings,
		valid_tokens=files.valid_tokens,
		embedding_metadata=files.metadata,
		checkpoint=config.checkpoint,
		resolved_decoder_config=resolved_config_path,
		train_tile_manifest=train_manifest_path,
		validation_tile_manifest=validation_manifest_path,
		volume_shape_xyz=volume_shape,
		token_grid_shape_xyz=token_shape,
		patch_size_xyz=patch_size,
		embedding_dim=embedding_dim,
		class_ids=class_ids,
		classes=classes,
		decoder_spec=decoder_spec,
		checkpoint_payload=checkpoint,
		artifact_identities=identities,
	)


def predict_f3_lithology_voxels(
	config: F3LithologyVoxelInferenceConfig,
	*,
	device: str | torch.device = 'auto',
	write_probabilities: bool | None = None,
	overwrite: bool | None = None,
) -> VoxelDecoderInferenceResult:
	"""Run exact-once core-tile inference and atomically publish its artifact."""
	plan = inspect_f3_lithology_voxel_inference(config, verify_array_hashes=True)
	probabilities_enabled = (
		config.write_probabilities
		if write_probabilities is None
		else _bool(write_probabilities, 'write_probabilities')
	)
	overwrite_enabled = (
		config.overwrite if overwrite is None else _bool(overwrite, 'overwrite')
	)
	run_device = _resolve_device(device)
	embeddings, valid_tokens = _load_source_arrays(plan)
	model = _load_decoder(plan, device=run_device)
	staging = create_f3_voxel_prediction_staging_paths(
		config.output_dir, overwrite=overwrite_enabled
	)
	try:
		arrays = open_f3_voxel_prediction_memmaps(
			staging,
			volume_shape_xyz=plan.volume_shape_xyz,
			class_count=len(plan.class_ids),
			include_probabilities=probabilities_enabled,
		)
		coverage = _write_inference_tiles(
			model,
			embeddings=embeddings,
			valid_tokens=valid_tokens,
			arrays=arrays,
			plan=plan,
			core_size_tokens=config.tiles.core_size_tokens,
			context_halo_tokens=config.tiles.context_halo_tokens,
			device=run_device,
		)
		_flush_arrays(arrays)
		summary = validate_f3_voxel_prediction_arrays(
			arrays,
			volume_shape_xyz=plan.volume_shape_xyz,
			class_probability_order=plan.class_ids,
		)
		_validate_written_coverage(summary, coverage)
		metadata = _prediction_metadata(
			config,
			plan=plan,
			write_probabilities=probabilities_enabled,
			summary=summary,
			coverage=coverage,
		)
		write_f3_voxel_prediction_metadata(staging.metadata, metadata)
		paths = commit_f3_voxel_prediction_artifact(
			staging, config.output_dir, overwrite=overwrite_enabled
		)
	except BaseException:
		shutil.rmtree(staging.output_dir, ignore_errors=True)
		raise
	return VoxelDecoderInferenceResult(
		paths=paths,
		volume_shape_xyz=plan.volume_shape_xyz,
		valid_voxel_count=cast('int', summary['valid_voxel_count']),
		invalid_voxel_count=cast('int', summary['invalid_voxel_count']),
		tile_count=cast('int', coverage['core_tile_count']),
		probabilities_written=probabilities_enabled,
	)


def _write_inference_tiles(  # noqa: PLR0913
	model: torch.nn.Module,
	*,
	embeddings: np.ndarray,
	valid_tokens: NDArray[np.bool_],
	arrays: F3VoxelPredictionArrays,
	plan: VoxelDecoderInferencePlan,
	core_size_tokens: tuple[int, int, int],
	context_halo_tokens: tuple[int, int, int],
	device: torch.device,
) -> dict[str, object]:
	total_written = 0
	valid_written = 0
	tile_count = 0
	class_ids = np.asarray(plan.class_ids, dtype=np.int16)
	input_shape = tuple(
		core_size_tokens[axis] + 2 * context_halo_tokens[axis]
		for axis in range(3)
	)
	with torch.inference_mode():
		for tile in _inference_tiles(
			plan.token_grid_shape_xyz,
			core_size=core_size_tokens,
			halo=context_halo_tokens,
		):
			embedding_crop = np.zeros(
				(*input_shape, plan.embedding_dim), dtype=np.float32
			)
			mask_crop = np.zeros(input_shape, dtype=np.bool_)
			source = _slices(tile.input_start, tile.input_stop)
			destination_stop = tuple(
				tile.padding_before[axis]
				+ tile.input_stop[axis]
				- tile.input_start[axis]
				for axis in range(3)
			)
			destination = _slices(tile.padding_before, destination_stop)
			embedding_crop[destination] = np.asarray(
				embeddings[source], dtype=np.float32
			)
			mask_crop[destination] = valid_tokens[source]
			input_tensor = torch.from_numpy(
				np.ascontiguousarray(np.moveaxis(embedding_crop, -1, 0)[None])
			).to(device)
			mask_tensor = torch.from_numpy(mask_crop[None]).to(device)
			logits = model(input_tensor, mask_tensor)
			core_voxel_shape = tuple(
				min(
					tile.core_stop[axis] * plan.patch_size_xyz[axis],
					plan.volume_shape_xyz[axis],
				)
				- tile.core_start[axis] * plan.patch_size_xyz[axis]
				for axis in range(3)
			)
			crop_start = tuple(
				context_halo_tokens[axis] * plan.patch_size_xyz[axis]
				for axis in range(3)
			)
			crop_stop = tuple(
				crop_start[axis] + core_voxel_shape[axis] for axis in range(3)
			)
			core_logits = logits[(0, slice(None), *_slices(crop_start, crop_stop))]
			probabilities = torch.softmax(core_logits.float(), dim=0)
			confidence, class_indices = probabilities.max(dim=0)
			probability_values = probabilities.movedim(0, -1).cpu().numpy()
			confidence_values = confidence.cpu().numpy()
			class_index_values = class_indices.cpu().numpy()

			core_token_source = _slices(tile.core_start, tile.core_stop)
			valid_core = np.asarray(valid_tokens[core_token_source])
			for axis, repeat in enumerate(plan.patch_size_xyz):
				valid_core = np.repeat(valid_core, repeat, axis=axis)
			valid_core = valid_core[_slices((0, 0, 0), core_voxel_shape)]
			voxel_start = tuple(
				tile.core_start[axis] * plan.patch_size_xyz[axis]
				for axis in range(3)
			)
			voxel_stop = tuple(
				voxel_start[axis] + core_voxel_shape[axis] for axis in range(3)
			)
			voxel_destination = _slices(voxel_start, voxel_stop)
			predictions = class_ids[class_index_values]
			arrays.valid_mask[voxel_destination] = valid_core
			arrays.predictions[voxel_destination] = np.where(
				valid_core, predictions, INVALID_PREDICTION_CLASS_ID
			).astype(np.int16, copy=False)
			arrays.confidence[voxel_destination] = np.where(
				valid_core, confidence_values, np.nan
			).astype(np.float16, copy=False)
			if arrays.probabilities is not None:
				arrays.probabilities[voxel_destination] = np.where(
					valid_core[..., None], probability_values, np.nan
				).astype(np.float16, copy=False)
			written = int(np.prod(core_voxel_shape))
			total_written += written
			valid_written += int(np.count_nonzero(valid_core))
			tile_count += 1

	expected = int(np.prod(plan.volume_shape_xyz))
	if total_written != expected:
		raise AssertionError(
			'core tiles must write every original-volume voxel exactly once; '
			f'wrote {total_written}, expected {expected}'
		)
	return {
		'core_tile_count': tile_count,
		'original_voxel_count': expected,
		'written_voxel_count': total_written,
		'duplicate_write_count': 0,
		'missing_write_count': 0,
		'valid_voxel_count': valid_written,
		'exact_once': True,
	}


def _inference_tiles(
	token_shape: tuple[int, int, int],
	*,
	core_size: tuple[int, int, int],
	halo: tuple[int, int, int],
) -> Iterator[_InferenceTile]:
	for x in range(0, token_shape[0], core_size[0]):
		for y in range(0, token_shape[1], core_size[1]):
			for z in range(0, token_shape[2], core_size[2]):
				start = (x, y, z)
				stop = tuple(
					min(start[axis] + core_size[axis], token_shape[axis])
					for axis in range(3)
				)
				input_start = tuple(
					max(0, start[axis] - halo[axis]) for axis in range(3)
				)
				desired_stop = tuple(
					start[axis] + core_size[axis] + halo[axis]
					for axis in range(3)
				)
				input_stop = tuple(
					min(token_shape[axis], desired_stop[axis]) for axis in range(3)
				)
				yield _InferenceTile(
					core_start=start,
					core_stop=cast('tuple[int, int, int]', stop),
					input_start=cast('tuple[int, int, int]', input_start),
					input_stop=cast('tuple[int, int, int]', input_stop),
					padding_before=cast(
						'tuple[int, int, int]',
						tuple(
							input_start[axis] - (start[axis] - halo[axis])
							for axis in range(3)
						),
					),
					padding_after=cast(
						'tuple[int, int, int]',
						tuple(
							desired_stop[axis] - input_stop[axis]
							for axis in range(3)
						),
					),
				)


def _load_source_arrays(
	plan: VoxelDecoderInferencePlan,
) -> tuple[np.ndarray, NDArray[np.bool_]]:
	embeddings = np.load(plan.embeddings, mmap_mode='r', allow_pickle=False)
	valid_tokens = np.load(plan.valid_tokens, mmap_mode='r', allow_pickle=False)
	if embeddings.dtype.kind != 'f' or embeddings.ndim != 4:
		raise TypeError('embeddings must be floating [TX,TY,TZ,D]')
	if tuple(embeddings.shape) != (*plan.token_grid_shape_xyz, plan.embedding_dim):
		raise ValueError('embedding array shape does not match metadata')
	if (
		valid_tokens.dtype != np.bool_
		or tuple(valid_tokens.shape) != plan.token_grid_shape_xyz
	):
		raise TypeError('valid_tokens must be bool with the metadata token-grid shape')
	return embeddings, valid_tokens


def _load_decoder(
	plan: VoxelDecoderInferencePlan, *, device: torch.device
) -> VoxelDecoder3D:
	spec = plan.decoder_spec
	model = VoxelDecoder3D(
		spec=_nonempty_str(spec.get('spec'), 'decoder.spec'),
		embedding_dim=_positive_int(spec.get('embedding_dim'), 'decoder.embedding_dim'),
		class_count=_positive_int(spec.get('class_count'), 'decoder.class_count'),
		hidden_channels=_positive_sequence(
			spec.get('hidden_channels'), 'decoder.hidden_channels'
		),
		upsample_factors=_factor_sequence(spec.get('upsample_factors')),
		upsample_mode=_nonempty_str(
			spec.get('upsample_mode'), 'decoder.upsample_mode'
		),
		normalization=_nonempty_str(
			spec.get('normalization'), 'decoder.normalization'
		),
		patch_size_xyz=plan.patch_size_xyz,
	).to(device)
	state = plan.checkpoint_payload.get('model_state_dict')
	if not isinstance(state, Mapping):
		raise TypeError('checkpoint model_state_dict must be a mapping')
	model.load_state_dict(state)
	model.eval()
	return model


def _validate_config_binding(
	config: F3LithologyVoxelInferenceConfig,
	resolved: Mapping[str, object],
) -> None:
	if resolved.get('dataset') != dict(config.dataset):
		raise ValueError('inference dataset does not match decoder dataset')
	if resolved.get('model') != dict(config.model):
		raise ValueError('inference model does not match decoder source model')
	embeddings = _mapping(resolved.get('embeddings'), 'resolved embeddings config')
	configured_embeddings = Path(cast('str', embeddings.get('input_dir'))).resolve(
		strict=False
	)
	if configured_embeddings != config.embeddings_input_dir.resolve(strict=False):
		raise ValueError('inference embeddings do not match decoder embeddings')
	tiles = _mapping(resolved.get('tiles'), 'resolved tile config')
	expected_tiles = {
		'core_size_tokens': list(config.tiles.core_size_tokens),
		'context_halo_tokens': list(config.tiles.context_halo_tokens),
	}
	if tiles != expected_tiles:
		raise ValueError('inference tile geometry does not match decoder training')


def _validate_artifact_identities(
	identities: Mapping[str, object],
	*,
	embeddings: Path,
	valid_tokens: Path,
	embedding_metadata: Path,
	verify_array_hashes: bool,
) -> None:
	selected = {
		'embeddings': embeddings,
		'valid_tokens': valid_tokens,
		'embedding_metadata': embedding_metadata,
	}
	for key, selected_path in selected.items():
		identity = _mapping(identities.get(key), f'checkpoint identity {key}')
		declared = identity.get('path')
		if not isinstance(declared, str) or Path(declared).resolve(
			strict=False
		) != selected_path.resolve(strict=False):
			raise ValueError(f'checkpoint/source identity mismatch: {key} path')
	for key, value in identities.items():
		if key == 'name':
			continue
		identity = _mapping(value, f'checkpoint identity {key}')
		path_value = identity.get('path')
		hash_value = identity.get('sha256')
		if not isinstance(path_value, str) or not isinstance(hash_value, str):
			raise TypeError(f'checkpoint identity {key} requires path and sha256')
		path = Path(path_value)
		if not path.is_file():
			raise FileNotFoundError(f'checkpoint-bound source is missing: {path}')
		if (verify_array_hashes or path.suffix != '.npy') and (
			file_sha256(path) != hash_value
		):
			raise ValueError(f'checkpoint/source identity mismatch: {key} hash')


def _validate_manifest_identities(
	checkpoint: Mapping[str, object],
	*,
	train_path: Path,
	validation_path: Path,
) -> None:
	hashes = _mapping(
		checkpoint.get('tile_manifest_hashes'), 'checkpoint tile_manifest_hashes'
	)
	for split, path in (('train', train_path), ('validation', validation_path)):
		manifest = read_voxel_tile_manifest(path)
		if hashes.get(split) != manifest.identity_sha256:
			raise ValueError(
				f'checkpoint/source identity mismatch: {split} tile manifest'
			)


def _validate_checkpoint_classes(
	identities: Mapping[str, object], *, classes: Sequence[F3ClassInfo]
) -> None:
	identity = _mapping(
		identities.get('voxel_dataset_metadata'),
		'checkpoint identity voxel_dataset_metadata',
	)
	metadata = _read_json_object(Path(cast('str', identity['path'])))
	values = metadata.get('classes')
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		raise TypeError('voxel dataset metadata classes must be a sequence')
	expected = tuple(item.to_dict() for item in classes)
	actual: list[dict[str, object]] = []
	for value in values:
		entry = _mapping(value, 'voxel dataset class')
		actual.append(dict(entry))
	if tuple(actual) != expected:
		raise ValueError('class_info does not match checkpoint supervision classes')


def _prediction_metadata(
	config: F3LithologyVoxelInferenceConfig,
	*,
	plan: VoxelDecoderInferencePlan,
	write_probabilities: bool,
	summary: Mapping[str, object],
	coverage: Mapping[str, object],
) -> dict[str, object]:
	root = config.output_dir.resolve(strict=False)
	outputs = {
		'predictions': str(root / 'f3_voxel_predictions.npy'),
		'confidence': str(root / 'f3_voxel_confidence.npy'),
		'valid_mask': str(root / 'f3_valid_voxel_mask.npy'),
	}
	if write_probabilities:
		outputs['probabilities'] = str(root / 'f3_voxel_probabilities.npy')
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'prediction_kind': 'frozen_embedding_decoder',
		'decoder_architecture': dict(plan.decoder_spec),
		'model_tag': config.model['tag'],
		'class_probability_order': list(plan.class_ids),
		'classes': [item.to_dict() for item in plan.classes],
		'volume_shape_xyz': list(plan.volume_shape_xyz),
		'patch_size_xyz': list(plan.patch_size_xyz),
		'invalid_prediction_class_id': INVALID_PREDICTION_CLASS_ID,
		'invalid_confidence_value': INVALID_CONFIDENCE_VALUE,
		'inputs': {
			'embeddings': str(plan.embeddings),
			'embedding_metadata': str(plan.embedding_metadata),
			'valid_tokens': str(plan.valid_tokens),
			'class_info': str(config.class_info),
			'decoder_checkpoint': str(plan.checkpoint),
		},
		'source_identity': {
			'decoder_checkpoint': {
				'path': str(plan.checkpoint),
				'sha256': file_sha256(plan.checkpoint),
			},
			'resolved_decoder_config': {
				'path': str(plan.resolved_decoder_config),
				'sha256': file_sha256(plan.resolved_decoder_config),
			},
			'class_info': {
				'path': str(config.class_info),
				'sha256': file_sha256(config.class_info),
			},
			'artifact_identities': dict(plan.artifact_identities),
			'tile_manifests': {
				'train': {
					'path': str(plan.train_tile_manifest),
					'sha256': file_sha256(plan.train_tile_manifest),
				},
				'validation': {
					'path': str(plan.validation_tile_manifest),
					'sha256': file_sha256(plan.validation_tile_manifest),
				},
			},
		},
		'tile_geometry': {
			'core_size_tokens': list(config.tiles.core_size_tokens),
			'context_halo_tokens': list(config.tiles.context_halo_tokens),
			'token_grid_shape_xyz': list(plan.token_grid_shape_xyz),
		},
		'write_probabilities': write_probabilities,
		'coverage': dict(coverage),
		'outputs': outputs,
		'summary': dict(summary),
	}


def _flush_arrays(arrays: F3VoxelPredictionArrays) -> None:
	for value in (
		arrays.predictions,
		arrays.confidence,
		arrays.valid_mask,
		arrays.probabilities,
	):
		if isinstance(value, np.memmap):
			value.flush()


def _validate_written_coverage(
	summary: Mapping[str, object], coverage: Mapping[str, object]
) -> None:
	if summary['valid_voxel_count'] != coverage['valid_voxel_count']:
		raise AssertionError('valid voxel coverage changed while writing outputs')


def _resolve_device(value: str | torch.device) -> torch.device:
	if isinstance(value, torch.device):
		return value
	if value not in {'auto', 'cpu', 'cuda'}:
		raise ValueError('device must be auto, cpu, or cuda')
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if value == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available')
	return torch.device(value)


def _read_json_object(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as file_obj:
		value = json.load(file_obj)
	if not isinstance(value, Mapping):
		raise TypeError(f'JSON artifact must contain an object: {path}')
	return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _positive_triplet(value: object, label: str) -> tuple[int, int, int]:
	items = _positive_sequence(value, label)
	if len(items) != 3:
		raise ValueError(f'{label} must contain three positive integers')
	return (items[0], items[1], items[2])


def _positive_sequence(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError(f'{label} must be a non-empty integer sequence')
	return tuple(_positive_int(item, label) for item in value)


def _factor_sequence(value: object) -> tuple[tuple[int, int, int], ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
		raise TypeError('decoder.upsample_factors must be a non-empty sequence')
	return tuple(_positive_triplet(item, 'decoder.upsample_factors') for item in value)


def _positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


def _embedding_dim(metadata: Mapping[str, object]) -> int:
	value = metadata.get('embedding_dim')
	geometry = metadata.get('model_geometry')
	if value is None and isinstance(geometry, Mapping):
		value = geometry.get('encoder_dim')
	return _positive_int(value, 'embedding dimension')


def _bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		raise TypeError(f'{label} must be boolean')
	return value


def _nonempty_str(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _slices(
	start: Sequence[int], stop: Sequence[int]
) -> tuple[slice, slice, slice]:
	return cast(
		'tuple[slice, slice, slice]',
		tuple(slice(begin, end) for begin, end in zip(start, stop, strict=True)),
	)


__all__ = [
	'VoxelDecoderInferencePlan',
	'VoxelDecoderInferenceResult',
	'inspect_f3_lithology_voxel_inference',
	'predict_f3_lithology_voxels',
]
