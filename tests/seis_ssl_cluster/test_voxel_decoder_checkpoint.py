from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from seis_ssl_cluster.models.voxel_decoder import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	VoxelDecoder3D,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	BEST_SELECTION_EPSILON,
	CHECKPOINT_SCHEMA_VERSION,
	load_voxel_decoder_checkpoint,
	save_voxel_decoder_checkpoint,
	stable_model_state_sha256,
	validate_resume_identity,
	validation_is_better,
)


def _architecture() -> dict[str, object]:
	return {
		'spec': VOXEL_DECODER_SPEC,
		'embedding_dim': 2,
		'class_count': 2,
		'hidden_channels': [2],
		'upsample_factors': [[1, 1, 1]],
		'upsample_mode': VOXEL_DECODER_UPSAMPLE_MODE,
		'normalization': VOXEL_DECODER_NORMALIZATION,
	}


def _save_checkpoint(tmp_path, *, architecture=None):
	model = VoxelDecoder3D(
		embedding_dim=2,
		class_count=2,
		hidden_channels=(2,),
		upsample_factors=((1, 1, 1),),
		patch_size_xyz=(1, 1, 1),
	)
	optimizer = torch.optim.AdamW(model.parameters())
	resolved_config = {
		'decoder': _architecture() if architecture is None else architecture
	}
	path = tmp_path / 'checkpoint.pt'
	save_voxel_decoder_checkpoint(
		path,
		model=model,
		optimizer=optimizer,
		epoch=0,
		global_step=1,
		resolved_config=resolved_config,
		class_weights=(1.0, 1.0),
		artifact_identities={},
		tile_manifest_hashes={},
		best_selection_state=None,
		training_history=[],
		current_metrics={},
		checkpoint_kind='epoch',
	)
	return path, resolved_config


def test_best_selection_uses_fixed_lexicographic_rule() -> None:
	best = {'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 1.0}
	assert validation_is_better({'macro_f1': 0.6, 'mean_iou': 0.0, 'loss': 9.0}, best)
	assert validation_is_better({'macro_f1': 0.5, 'mean_iou': 0.5, 'loss': 9.0}, best)
	assert validation_is_better({'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 0.9}, best)
	assert not validation_is_better(dict(best), best)


def test_best_selection_ties_within_explicit_epsilon() -> None:
	best = {'macro_f1': 0.5, 'mean_iou': 0.4, 'loss': 1.0}
	candidate = dict(best)
	candidate['macro_f1'] = 0.5 + BEST_SELECTION_EPSILON / 2
	assert not validation_is_better(candidate, best)


def test_best_selection_rejects_invalid_metrics() -> None:
	with pytest.raises(TypeError, match='macro_f1'):
		validation_is_better({'macro_f1': None}, {'macro_f1': 1.0})


def test_loader_rejects_previous_model_semantics_schema(tmp_path) -> None:
	path = tmp_path / 'old.pt'
	payload = {
		'schema_version': CHECKPOINT_SCHEMA_VERSION - 1,
		'epoch': 0,
		'global_step': 0,
		'model_state_dict': {},
		'optimizer_state_dict': {},
		'best_selection_state': None,
		'training_history': [],
		'current_metrics': {},
		'resolved_config': {},
		'class_weights': [],
		'artifact_identities': {},
		'tile_manifest_hashes': {},
		'rng_states': {},
		'checkpoint_kind': 'completed',
	}
	torch.save(payload, path)

	with pytest.raises(ValueError, match=r'unsupported.*schema_version'):
		load_voxel_decoder_checkpoint(path)


def test_schema_five_round_trip_binds_decoder_architecture(tmp_path) -> None:
	path, resolved_config = _save_checkpoint(tmp_path)

	payload = load_voxel_decoder_checkpoint(path)

	assert payload['schema_version'] == 5 == CHECKPOINT_SCHEMA_VERSION
	assert payload['decoder_architecture'] == resolved_config['decoder']


def test_stable_model_state_hash_tracks_tensor_content() -> None:
	torch.manual_seed(17)
	first = VoxelDecoder3D(
		embedding_dim=2,
		class_count=2,
		hidden_channels=(2,),
		upsample_factors=((1, 1, 1),),
		patch_size_xyz=(1, 1, 1),
	)
	torch.manual_seed(17)
	second = VoxelDecoder3D(
		embedding_dim=2,
		class_count=2,
		hidden_channels=(2,),
		upsample_factors=((1, 1, 1),),
		patch_size_xyz=(1, 1, 1),
	)

	first_hash = stable_model_state_sha256(first)
	assert len(first_hash) == 64
	assert stable_model_state_sha256(second) == first_hash
	with torch.no_grad():
		next(second.parameters()).view(-1)[0] += 1.0
	assert stable_model_state_sha256(second) != first_hash


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('spec', 'frozen_embedding_decoder_v1'),
		('upsample_mode', 'trilinear'),
		('normalization', 'batch_norm'),
	],
)
def test_save_rejects_noncanonical_decoder_identity(
	tmp_path, field: str, value: str
) -> None:
	architecture = _architecture()
	architecture[field] = value

	with pytest.raises(ValueError, match=field):
		_save_checkpoint(tmp_path, architecture=architecture)


def test_loader_rejects_architecture_not_matching_resolved_config(tmp_path) -> None:
	path, _ = _save_checkpoint(tmp_path)
	payload = torch.load(path, map_location='cpu', weights_only=False)
	payload['decoder_architecture']['hidden_channels'] = [3]
	torch.save(payload, path)

	with pytest.raises(ValueError, match='does not match'):
		load_voxel_decoder_checkpoint(path)


def test_resume_identity_rejects_decoder_config_change(tmp_path) -> None:
	path, resolved_config = _save_checkpoint(tmp_path)
	payload = load_voxel_decoder_checkpoint(path)
	changed = deepcopy(resolved_config)
	changed['decoder']['hidden_channels'] = [3]

	with pytest.raises(ValueError, match='decoder architecture'):
		validate_resume_identity(
			payload,
			resolved_config=changed,
			class_weights=(1.0, 1.0),
			artifact_identities={},
			tile_manifest_hashes={},
		)
