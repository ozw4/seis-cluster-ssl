from __future__ import annotations

import pytest

from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner import (
	_shared_condition_contract,
)


def _triplet() -> list[dict[str, object]]:
	shared = {
		'split_id': 'split_000',
		'budget_id': 'cap25',
		'voxel_supervision_grid_sha256': 'grid',
		'selected_token_identity_sha256': 'tokens',
		'unique_token_xyz_sha256': 'xyz',
		'train_voxel_count': 100,
		'validation_voxel_count': 200,
		'validation_mask_sha256': 'validation',
		'canonical_valid_token_sha256': 'valid',
		'class_order': [0, 1, 2, 3, 4, 5],
		'class_weights': [1.0] * 6,
		'initial_model_state_sha256': 'initial',
		'decoder_architecture': {'spec': 'fixed'},
		'decoder_seed': 42000,
		'train_tile_manifest_sha256': 'train-manifest',
		'validation_tile_manifest_sha256': 'validation-manifest',
		'train_tile_identity_sha256': 'train-tiles',
		'validation_tile_identity_sha256': 'validation-tiles',
		'sampling_mode': 'uniform_tiles_with_replacement',
		'steps_per_epoch': 440,
		'sampling_sequence_sha256': 'sampling',
		'global_step': 22000,
		'metric_schema_sha256': 'metrics',
	}
	return [
		{**shared, 'model_role': role}
		for role in ('mae', 'm1_current_k6', 'mh_nocons')
	]


def test_full_condition_requires_three_model_paired_decoder_contract() -> None:
	_shared_condition_contract(
		_triplet(),
		models=('mae', 'm1_current_k6', 'mh_nocons'),
		context='full decoder run',
	)


@pytest.mark.parametrize(
	'key',
	[
		'initial_model_state_sha256',
		'class_weights',
		'train_tile_identity_sha256',
		'sampling_sequence_sha256',
	],
)
def test_full_condition_rejects_paired_decoder_contract_mismatch(key: str) -> None:
	rows = _triplet()
	rows[-1][key] = 'mismatch'
	with pytest.raises(ValueError, match=key):
		_shared_condition_contract(
			rows,
			models=('mae', 'm1_current_k6', 'mh_nocons'),
			context='full decoder run',
		)
