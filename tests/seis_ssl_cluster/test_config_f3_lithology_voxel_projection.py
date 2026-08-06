from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_voxel_projection_config_from_mapping,
)


def projection_config(tmp_path: Path) -> dict[str, object]:
	"""Write a valid synthetic source artifact and return its raw config."""
	artifact_root = tmp_path / 'artifacts'
	source = artifact_root / 'token_predictions'
	source.mkdir(parents=True)
	class_info = artifact_root / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				'2': {'name': 'class-two', 'color': [1, 2, 3]},
				'5': {'name': 'class-five', 'color': [4, 5, 6]},
			}
		),
		encoding='utf-8',
	)
	predictions = np.asarray([[[2]], [[5]]], dtype=np.int16)
	probabilities = np.asarray(
		[[[[0.75, 0.25]]], [[[0.25, 0.75]]]], dtype=np.float32
	)
	valid_tokens = np.ones((2, 1, 1), dtype=np.bool_)
	predictions_path = source / 'f3_token_predictions.npy'
	probabilities_path = source / 'f3_token_probabilities.npy'
	valid_tokens_path = source / 'f3_valid_token_grid.npy'
	metadata_path = source / 'prediction_metadata.json'
	np.save(predictions_path, predictions)
	np.save(probabilities_path, probabilities)
	np.save(valid_tokens_path, valid_tokens)
	metadata = {
		'artifact_type': 'f3_lithology_token_predictions',
		'dataset': {'name': 'f3_facies_benchmark', 'version': 'test-v1'},
		'model': {'tag': 'test-model', 'freeze_encoder': True},
		'embeddings': {'spec': 'test-embeddings'},
		'probe': {'spec': 'test-probe'},
		'classes': [
			{'class_id': 2, 'class_name': 'class-two'},
			{'class_id': 5, 'class_name': 'class-five'},
		],
		'class_probability_order': [2, 5],
		'invalid_prediction_class_id': -1,
		'invalid_probability_value': 'nan',
		'embedding': {
			'patch_size_xyz': [2, 2, 2],
			'token_grid_shape_xyz': [2, 1, 1],
		},
		'geometry': {'shape_xyz': [3, 2, 2]},
		'outputs': {
			'token_predictions': str(predictions_path),
			'probability_volume': str(probabilities_path),
			'valid_token_grid': str(valid_tokens_path),
			'metadata_json': str(metadata_path),
		},
		'summary': {
			'token_grid_shape_xyz': [2, 1, 1],
			'probability_grid_shape': [2, 1, 1, 2],
			'valid_token_count': 2,
			'invalid_token_count': 0,
		},
	}
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'f3_root': str(tmp_path / 'f3'),
		},
		'dataset': {'name': 'f3_facies_benchmark', 'version': 'test-v1'},
		'model': {'tag': 'test-model', 'freeze_encoder': True},
		'labels': {'class_info': str(class_info)},
		'token_predictions': {
			'input_dir': str(source),
			'predictions': str(predictions_path),
			'probabilities': str(probabilities_path),
			'valid_tokens': str(valid_tokens_path),
			'metadata_json': str(metadata_path),
		},
		'voxel_projection': {
			'output_dir': str(artifact_root / 'voxel_predictions')
		},
	}


def test_projection_config_resolves_geometry_and_defaults(tmp_path: Path) -> None:
	resolved = f3_lithology_voxel_projection_config_from_mapping(
		projection_config(tmp_path)
	)

	assert resolved.mode == 'nearest'
	assert resolved.write_probabilities is False
	assert resolved.overwrite is False
	assert resolved.source.token_grid_shape_xyz == (2, 1, 1)
	assert resolved.source.patch_size_xyz == (2, 2, 2)
	assert resolved.source.volume_shape_xyz == (3, 2, 2)
	assert resolved.output_dir == tmp_path / 'artifacts' / 'voxel_predictions'


def test_projection_config_accepts_inventory_class_info_payload(
	tmp_path: Path,
) -> None:
	config = projection_config(tmp_path)
	class_info = Path(config['labels']['class_info'])  # type: ignore[index]
	class_info.write_text(
		json.dumps(
			{
				'class_count': 2,
				'classes': [
					{
						'class_id': 2,
						'class_name': 'class-two',
						'hex_color': '#010203',
						'rgb': [1, 2, 3],
					},
					{
						'class_id': 5,
						'class_name': 'class-five',
						'hex_color': '#040506',
						'rgb': [4, 5, 6],
					},
				],
				'source_path': '/source/F3/interpretation/class_info.json',
			}
		),
		encoding='utf-8',
	)

	resolved = f3_lithology_voxel_projection_config_from_mapping(config)

	assert resolved.source.class_probability_order == (2, 5)


@pytest.mark.parametrize(
	('section', 'key', 'value', 'error'),
	[
		('dataset', 'unknown', True, ValueError),
		('voxel_projection', 'mode', 'trilinear', ValueError),
		('voxel_projection', 'write_probabilities', 1, TypeError),
		('voxel_projection', 'output_dir', 'relative', ValueError),
	],
)
def test_projection_config_rejects_unknown_mode_type_and_path(
	tmp_path: Path,
	section: str,
	key: str,
	value: object,
	error: type[Exception],
) -> None:
	config = projection_config(tmp_path)
	config[section][key] = value  # type: ignore[index]
	with pytest.raises(error):
		f3_lithology_voxel_projection_config_from_mapping(config)


def test_projection_config_rejects_unbound_source_path(tmp_path: Path) -> None:
	config = projection_config(tmp_path)
	config['token_predictions']['predictions'] = str(  # type: ignore[index]
		tmp_path / 'artifacts' / 'other.npy'
	)
	with pytest.raises(ValueError, match='must identify'):
		f3_lithology_voxel_projection_config_from_mapping(config)


def test_projection_config_rejects_class_order_mismatch(tmp_path: Path) -> None:
	config = projection_config(tmp_path)
	class_info = config['labels']['class_info']  # type: ignore[index]
	Path(class_info).write_text(
		json.dumps({'5': {'name': 'five', 'color': [1, 2, 3]}}),
		encoding='utf-8',
	)
	with pytest.raises(ValueError, match='class order must match'):
		f3_lithology_voxel_projection_config_from_mapping(config)


def test_projection_config_rejects_existing_output_without_overwrite(
	tmp_path: Path,
) -> None:
	config = projection_config(tmp_path)
	output_dir = Path(config['voxel_projection']['output_dir'])  # type: ignore[index]
	output_dir.mkdir()
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		f3_lithology_voxel_projection_config_from_mapping(config)


def test_projection_config_rejects_output_containing_class_info_with_overwrite(
	tmp_path: Path,
) -> None:
	config = projection_config(tmp_path)
	class_info = Path(config['labels']['class_info'])  # type: ignore[index]
	labels_dir = class_info.parent / 'labels'
	labels_dir.mkdir()
	moved_class_info = labels_dir / class_info.name
	class_info.replace(moved_class_info)
	config['labels']['class_info'] = str(moved_class_info)  # type: ignore[index]
	config['voxel_projection']['output_dir'] = str(labels_dir)  # type: ignore[index]
	config['voxel_projection']['overwrite'] = True  # type: ignore[index]

	with pytest.raises(ValueError, match=r'must not overlap labels\.class_info'):
		f3_lithology_voxel_projection_config_from_mapping(config)

	assert moved_class_info.is_file()
