from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_runner import (
	LowLabelSplitJob,
	_selected_complete_triplets,
	_shared_condition_contract,
	_split_labels,
)
from seis_ssl_cluster.f3.splits import read_f3_line_geometry


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


def test_full_condition_skips_incomplete_selected_triplet() -> None:
	rows = _triplet()[:2]
	jobs = [
		LowLabelSplitJob('split_000', 'cap25', role, output_root=Path('unused'))
		for role in ('mae', 'm1_current_k6', 'mh_nocons')
	]
	assert _selected_complete_triplets(
		rows,
		jobs,
		models=('mae', 'm1_current_k6', 'mh_nocons'),
	) == []


def test_split_labels_rejects_drift_in_every_committed_source_identity(
	tmp_path: Path,
) -> None:
	paths = {
		'inventory': tmp_path / 'inventory.csv',
		'label_volume': tmp_path / 'labels.npy',
		'valid_tokens': tmp_path / 'valid_tokens.npy',
		'class_info': tmp_path / 'class_info.json',
		'source_label_segy': tmp_path / 'labels.sgy',
		'seismic_volume': tmp_path / 'seismic.npy',
	}
	for path in paths.values():
		path.write_bytes(path.name.encode())
	geometry = tmp_path / 'segy_geometry.json'
	geometry.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [2, 2, 2],
						'iline_min': 100,
						'iline_max': 101,
						'xline_min': 200,
						'xline_max': 201,
					}
				}
			}
		),
		encoding='utf-8',
	)
	paths['segy_geometry_json'] = geometry
	metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'inventory': _identity(paths['inventory']),
		'label_volume': _identity(paths['label_volume']),
		'reference_valid_tokens': _identity(paths['valid_tokens']),
		'labels': {
			'class_info': str(paths['class_info']),
			'source_label_segy': str(paths['source_label_segy']),
		},
		'source_identities': {
			key: _identity(paths[key])
			for key in (
				'class_info',
				'source_label_segy',
				'segy_geometry_json',
				'seismic_volume',
			)
		},
		'reference_embedding': {
			'metadata': {'source_amplitude_path': str(paths['seismic_volume'])}
		},
		'geometry': read_f3_line_geometry(geometry).to_dict(),
	}
	row = {'canonical_valid_tokens_sha256': file_sha256(paths['valid_tokens'])}
	assert _split_labels(row, metadata)['seismic_volume'] == paths['seismic_volume']
	for key in (
		'class_info',
		'source_label_segy',
		'segy_geometry_json',
		'seismic_volume',
	):
		paths[key].write_bytes(b'drift')
		with pytest.raises(ValueError, match='path/hash mismatch'):
			_split_labels(row, metadata)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}
