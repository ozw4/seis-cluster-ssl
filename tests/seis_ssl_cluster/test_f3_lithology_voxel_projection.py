from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology.voxel_geometry import (
	project_token_grid_nearest,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	discover_f3_voxel_probability_path,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_projection import (
	project_f3_lithology_tokens_to_voxels,
)

if TYPE_CHECKING:
	from pathlib import Path

VOLUME_SHAPE = (3, 3, 5)
PATCH_SIZE = (2, 2, 2)
TOKEN_SHAPE = (2, 2, 3)
CLASS_ORDER = (5, 2, 9)


@pytest.mark.parametrize('write_probabilities', [False, True])
def test_projects_exact_repeat_crop_invalid_tokens_and_provenance(
	tmp_path: Path, write_probabilities: bool  # noqa: FBT001
) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	before = {path.name: path.read_bytes() for path in source.iterdir()}

	result = project_f3_lithology_tokens_to_voxels(
		source,
		tmp_path / 'voxels',
		write_probabilities=write_probabilities,
		token_axis_chunk_size=1,
	)
	artifact = validate_f3_voxel_prediction_artifact(result.output_dir, mmap_mode='r')
	expected_valid = project_token_grid_nearest(
		np.load(source / 'f3_valid_token_grid.npy'),
		patch_size_xyz=PATCH_SIZE,
		volume_shape_xyz=VOLUME_SHAPE,
	)
	expected_hard = project_token_grid_nearest(
		np.load(source / 'f3_token_predictions.npy'),
		patch_size_xyz=PATCH_SIZE,
		volume_shape_xyz=VOLUME_SHAPE,
	)
	token_probabilities = np.load(source / 'f3_token_probabilities.npy')
	expected_confidence = project_token_grid_nearest(
		np.max(token_probabilities, axis=-1),
		patch_size_xyz=PATCH_SIZE,
		volume_shape_xyz=VOLUME_SHAPE,
	)

	assert artifact.arrays.predictions.shape == VOLUME_SHAPE
	assert np.array_equal(artifact.arrays.valid_mask, expected_valid)
	assert np.array_equal(
		artifact.arrays.predictions[expected_valid], expected_hard[expected_valid]
	)
	assert np.all(artifact.arrays.predictions[~expected_valid] == -1)
	assert np.allclose(
		artifact.arrays.confidence[expected_valid],
		expected_confidence[expected_valid],
		rtol=1e-3,
	)
	assert np.all(np.isnan(artifact.arrays.confidence[~expected_valid]))
	assert (artifact.arrays.probabilities is not None) is write_probabilities
	assert (
		discover_f3_voxel_probability_path(result.output_dir) is not None
	) is write_probabilities
	if artifact.arrays.probabilities is not None:
		expected_probabilities = project_token_grid_nearest(
			token_probabilities,
			patch_size_xyz=PATCH_SIZE,
			volume_shape_xyz=VOLUME_SHAPE,
		)
		assert artifact.arrays.probabilities.shape == (*VOLUME_SHAPE, 3)
		assert np.allclose(
			artifact.arrays.probabilities[expected_valid],
			expected_probabilities[expected_valid],
			rtol=1e-3,
		)
		assert np.all(np.isnan(artifact.arrays.probabilities[~expected_valid]))
	metadata = artifact.metadata
	assert metadata['prediction_kind'] == 'token_projection_nearest'
	assert 'not a learned voxel prediction' in metadata['prediction_semantics']
	assert metadata['class_probability_order'] == list(CLASS_ORDER)
	assert metadata['source_identity']['probe_spec'] == {'spec': 'test_probe'}
	assert set(metadata['source_identity']['token_artifact_files']) == {
		'prediction_metadata',
		'token_predictions',
		'token_probabilities',
		'valid_token_grid',
	}
	assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_chunked_projection_matches_larger_axis_chunk(tmp_path: Path) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	first = project_f3_lithology_tokens_to_voxels(
		source,
		tmp_path / 'chunk_one',
		write_probabilities=True,
		token_axis_chunk_size=1,
	)
	second = project_f3_lithology_tokens_to_voxels(
		source,
		tmp_path / 'chunk_all',
		write_probabilities=True,
		token_axis_chunk_size=TOKEN_SHAPE[0],
	)
	first_artifact = validate_f3_voxel_prediction_artifact(first.output_dir)
	second_artifact = validate_f3_voxel_prediction_artifact(second.output_dir)
	for first_array, second_array in (
		(first_artifact.arrays.predictions, second_artifact.arrays.predictions),
		(first_artifact.arrays.confidence, second_artifact.arrays.confidence),
		(first_artifact.arrays.valid_mask, second_artifact.arrays.valid_mask),
		(first_artifact.arrays.probabilities, second_artifact.arrays.probabilities),
	):
		assert np.array_equal(first_array, second_array, equal_nan=True)


def test_rejects_hard_probability_argmax_mismatch_before_writing(
	tmp_path: Path,
) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	predictions_path = source / 'f3_token_predictions.npy'
	predictions = np.load(predictions_path)
	predictions[0, 0, 0] = 2
	np.save(predictions_path, predictions)
	output = tmp_path / 'voxels'

	with pytest.raises(ValueError, match='must match probability argmax'):
		project_f3_lithology_tokens_to_voxels(source, output)

	assert not output.exists()


def test_rejects_metadata_geometry_mismatch(tmp_path: Path) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	metadata_path = source / 'prediction_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['geometry']['shape_xyz'] = [5, 3, 5]
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='geometry is inconsistent'):
		project_f3_lithology_tokens_to_voxels(source, tmp_path / 'voxels')


def test_rejects_token_metadata_output_mismatch_before_writing(
	tmp_path: Path,
) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	metadata_path = source / 'prediction_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['outputs']['token_predictions'] = str(source / 'other.npy')
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	output = tmp_path / 'voxels'

	with pytest.raises(
		ValueError, match=r'outputs\.token_predictions does not identify'
	):
		project_f3_lithology_tokens_to_voxels(source, output)

	assert not output.exists()


def test_rejects_output_dir_equal_to_source_without_modifying_source(
	tmp_path: Path,
) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	before = {path.name: path.read_bytes() for path in source.iterdir()}

	with pytest.raises(ValueError, match='must differ from token_prediction_dir'):
		project_f3_lithology_tokens_to_voxels(
			source,
			source / '..' / source.name,
			overwrite=True,
		)

	assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_rejects_output_dir_ancestor_of_source_without_modifying_source(
	tmp_path: Path,
) -> None:
	output = tmp_path / 'output'
	output.mkdir()
	marker = output / 'keep.txt'
	marker.write_text('preserve me', encoding='utf-8')
	source = _write_token_artifact(output / 'tokens')
	before = {path.name: path.read_bytes() for path in source.iterdir()}

	with pytest.raises(ValueError, match='must not contain token_prediction_dir'):
		project_f3_lithology_tokens_to_voxels(
			source,
			output,
			overwrite=True,
		)

	assert marker.read_text(encoding='utf-8') == 'preserve me'
	assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_rejects_output_dir_inside_source_without_modifying_source(
	tmp_path: Path,
) -> None:
	source = _write_token_artifact(tmp_path / 'tokens')
	before = {path.name: path.read_bytes() for path in source.iterdir()}

	with pytest.raises(ValueError, match='must not be inside token_prediction_dir'):
		project_f3_lithology_tokens_to_voxels(
			source,
			source / 'f3_token_predictions.npy',
			overwrite=True,
		)

	assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def _write_token_artifact(root: Path) -> Path:
	root.mkdir()
	class_indices = np.arange(np.prod(TOKEN_SHAPE)).reshape(TOKEN_SHAPE) % 3
	valid = np.ones(TOKEN_SHAPE, dtype=np.bool_)
	valid[1, 1, 2] = False
	probabilities = np.full((*TOKEN_SHAPE, 3), 0.05, dtype=np.float32)
	for class_index in range(3):
		probabilities[..., class_index][class_indices == class_index] = 0.9
	probabilities[~valid] = np.nan
	predictions = np.asarray(CLASS_ORDER, dtype=np.int16)[class_indices]
	predictions[~valid] = -1
	np.save(root / 'f3_token_predictions.npy', predictions)
	np.save(root / 'f3_token_probabilities.npy', probabilities)
	np.save(root / 'f3_valid_token_grid.npy', valid)
	metadata = {
		'artifact_type': 'f3_lithology_token_predictions',
		'model': {'tag': 'test_encoder', 'checkpoint_sha256': 'model-hash'},
		'embeddings': {'spec': 'test_embeddings'},
		'probe': {'spec': 'test_probe'},
		'classes': [
			{'class_id': class_id, 'class_name': f'class-{class_id}'}
			for class_id in CLASS_ORDER
		],
		'class_probability_order': list(CLASS_ORDER),
		'invalid_prediction_class_id': -1,
		'invalid_probability_value': 'nan',
		'embedding': {
			'survey_id': 'test',
			'patch_size_xyz': list(PATCH_SIZE),
			'token_grid_shape_xyz': list(TOKEN_SHAPE),
			'embedding_dim': 4,
		},
		'geometry': {'shape_xyz': list(VOLUME_SHAPE)},
		'outputs': {
			'token_predictions': str(root / 'f3_token_predictions.npy'),
			'probability_volume': str(root / 'f3_token_probabilities.npy'),
			'valid_token_grid': str(root / 'f3_valid_token_grid.npy'),
			'metadata_json': str(root / 'prediction_metadata.json'),
		},
		'summary': {
			'token_grid_shape_xyz': list(TOKEN_SHAPE),
			'probability_grid_shape': [*TOKEN_SHAPE, 3],
			'valid_token_count': int(np.count_nonzero(valid)),
			'invalid_token_count': int(valid.size - np.count_nonzero(valid)),
		},
	}
	(root / 'prediction_metadata.json').write_text(
		json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
	return root
