from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_evaluation import (
	f3_lithology_voxel_evaluation_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.io.labels import read_class_info
from seis_ssl_cluster.f3.lithology.voxel_evaluation import (
	evaluate_f3_lithology_voxels,
	inspect_f3_lithology_voxel_evaluation,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	open_f3_voxel_prediction_memmaps,
	validate_f3_voxel_prediction_arrays,
	write_f3_voxel_prediction_metadata,
)


def test_perfect_evaluation_uses_unique_aggregate_and_duplicate_slices(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	result = evaluate_f3_lithology_voxels(config)

	metrics = json.loads(result.metrics_json.read_text(encoding='utf-8'))
	assert metrics['accuracy'] == 1.0
	assert metrics['evaluation_voxel_count'] == 12
	with result.validation_slice_metrics_csv.open(
		newline='', encoding='utf-8'
	) as file_obj:
		slices = list(csv.DictReader(file_obj))
	assert len(slices) == 2
	assert sum(int(row['voxel_count']) for row in slices) == 16
	boundary = json.loads(result.boundary_metrics_json.read_text(encoding='utf-8'))
	assert boundary['vertical_boundary_recall_at_1'] == 1.0
	assert boundary['vertical_boundary_class_3_recall_at_1'] == 1.0
	assert boundary['vertical_boundary_class_5_recall_at_1'] == 1.0
	assert json.dumps(boundary, allow_nan=False)


def test_chunk_size_is_metric_invariant(tmp_path: Path) -> None:
	first = _fixture(tmp_path / 'first', chunk_size_x=1)
	second = _fixture(tmp_path / 'second', chunk_size_x=8)
	first_result = evaluate_f3_lithology_voxels(first)
	second_result = evaluate_f3_lithology_voxels(second)

	first_metrics = json.loads(first_result.metrics_json.read_text(encoding='utf-8'))
	second_metrics = json.loads(second_result.metrics_json.read_text(encoding='utf-8'))
	assert first_metrics == second_metrics


def test_error_prediction_matches_direct_unique_voxel_accuracy(tmp_path: Path) -> None:
	config = _fixture(tmp_path, error_coordinate=(0, 0, 0))
	result = evaluate_f3_lithology_voxels(config)
	metrics = json.loads(result.metrics_json.read_text(encoding='utf-8'))
	assert metrics['accuracy'] == pytest.approx(11 / 12)


def test_no_boundary_and_zero_support_use_standard_json_nulls(tmp_path: Path) -> None:
	config = _fixture(tmp_path, constant_labels=True)
	result = evaluate_f3_lithology_voxels(config)
	boundary = json.loads(result.boundary_metrics_json.read_text(encoding='utf-8'))
	assert boundary['vertical_boundary_recall_at_1'] is None
	assert 'vertical_boundary_recall_at_1' in boundary['undefined_reasons']
	assert json.dumps(boundary, allow_nan=False)
	metrics = json.loads(result.metrics_json.read_text(encoding='utf-8'))
	assert metrics['per_class_support']['3'] == 0
	assert metrics['per_class_support']['5'] == 0


def test_v0_token_projection_uses_the_same_evaluator(tmp_path: Path) -> None:
	config = _fixture(tmp_path, prediction_kind='token_projection_nearest')
	result = evaluate_f3_lithology_voxels(config)
	metrics = json.loads(result.metrics_json.read_text(encoding='utf-8'))
	assert metrics['accuracy'] == 1.0


def test_v0_token_projection_allows_model_specific_embedding_sources(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path, prediction_kind='token_projection_nearest')
	metadata_path = _token_metadata_path(config.prediction_input_dir)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	model_metadata = tmp_path / 'model.embedding_metadata.json'
	model_metadata.write_text('{}\n', encoding='utf-8')
	model_valid_tokens = tmp_path / 'model.valid_tokens.npy'
	np.save(model_valid_tokens, np.ones((1, 1, 2), dtype=np.bool_))
	metadata['inputs']['embedding_metadata_json'] = str(model_metadata)
	metadata['inputs']['valid_tokens_path'] = str(model_valid_tokens)
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	_refresh_token_metadata_identity(config.prediction_input_dir, metadata_path)

	inspection = inspect_f3_lithology_voxel_evaluation(config)

	assert inspection.validation_voxel_count == 12


def test_v0_token_projection_requires_source_dataset_identity(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path, prediction_kind='token_projection_nearest')
	metadata_path = _token_metadata_path(config.prediction_input_dir)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata.pop('dataset')
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	_refresh_token_metadata_identity(config.prediction_input_dir, metadata_path)

	with pytest.raises(ValueError, match='source dataset identity mismatch'):
		inspect_f3_lithology_voxel_evaluation(config)


@pytest.mark.parametrize('inputs', [None, [], {}])
def test_v0_token_projection_requires_complete_source_inputs(
	tmp_path: Path, inputs: object
) -> None:
	config = _fixture(tmp_path, prediction_kind='token_projection_nearest')
	metadata_path = _token_metadata_path(config.prediction_input_dir)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['inputs'] = inputs
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	_refresh_token_metadata_identity(config.prediction_input_dir, metadata_path)

	with pytest.raises(
		(TypeError, ValueError),
		match=r'token prediction inputs|source identity mismatch',
	):
		inspect_f3_lithology_voxel_evaluation(config)


def test_validation_voxel_outside_prediction_mask_is_rejected(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path, invalid_coordinate=(0, 0, 0))
	with pytest.raises(ValueError, match='outside the prediction valid mask'):
		inspect_f3_lithology_voxel_evaluation(config)


def test_prediction_supervision_identity_mismatch_is_rejected(
	tmp_path: Path,
) -> None:
	config = _fixture(tmp_path)
	metadata_path = config.prediction_input_dir / 'prediction_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['source_identity']['artifact_identities']['voxel_split_grid'][
		'sha256'
	] = '0' * 64
	write_f3_voxel_prediction_metadata(metadata_path, metadata)
	with pytest.raises(ValueError, match='source identity mismatch'):
		inspect_f3_lithology_voxel_evaluation(config)


def _fixture(  # noqa: PLR0913, PLR0915
	root: Path,
	*,
	chunk_size_x: int = 1,
	invalid_coordinate: tuple[int, int, int] | None = None,
	error_coordinate: tuple[int, int, int] | None = None,
	constant_labels: bool = False,
	prediction_kind: str = 'frozen_embedding_decoder',
):
	artifact_root = root / 'artifacts'
	f3_root = root / 'f3'
	label_artifacts = artifact_root / 'labels'
	supervision = artifact_root / 'supervision'
	predictions_dir = artifact_root / 'predictions'
	for path in (f3_root, label_artifacts, supervision, predictions_dir):
		path.mkdir(parents=True, exist_ok=True)
	class_info = label_artifacts / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				'0': {'name': 'zero', 'color': [0, 0, 0]},
				'3': {'name': 'three', 'color': [3, 3, 3]},
				'5': {'name': 'five', 'color': [5, 5, 5]},
			}
		),
		encoding='utf-8',
	)
	labels = np.asarray(
		[
			[[0, 0, 3, 5], [0, 0, 3, 5]],
			[[0, 0, 3, 5], [0, 0, 3, 5]],
		],
		dtype=np.int16,
	)
	if constant_labels:
		labels[...] = 0
	label_path = label_artifacts / 'labels.npy'
	np.save(label_path, labels)
	label_segy = f3_root / 'labels.segy'
	label_segy.write_bytes(b'synthetic-segy')
	geometry_path = label_artifacts / 'geometry.json'
	geometry = {
		'cube_shape': [2, 2, 4],
		'iline_min': 100,
		'iline_max': 101,
		'xline_min': 200,
		'xline_max': 201,
	}
	geometry_path.write_text(json.dumps(geometry), encoding='utf-8')
	inventory = label_artifacts / 'inventory.csv'
	inventory.write_text(
		'relative_path,split,slice_type,slice_index\n'
		'a.png,validation,inline,100\n'
		'b.png,validation,crossline,200\n',
		encoding='utf-8',
	)
	reference_metadata = artifact_root / 'embedding_metadata.json'
	reference_metadata.write_text('{}\n', encoding='utf-8')
	valid_tokens = artifact_root / 'valid_tokens.npy'
	np.save(valid_tokens, np.ones((1, 1, 2), dtype=np.bool_))
	grid = np.zeros(labels.shape, dtype=np.uint8)
	grid[0, :, :] = 2
	grid[:, 0, :] = 2
	grid_path = supervision / 'supervision_split_grid.npy'
	np.save(grid_path, grid)
	classes = read_class_info(class_info)
	supervision_metadata = {
		'artifact_type': 'f3_lithology_voxel_supervision',
		'schema_version': 1,
		'dataset': {'name': 'f3', 'version': 'test'},
		'labels': {
			'source_label_segy': str(label_segy),
			'class_info': str(class_info),
		},
		'classes': [item.to_dict() for item in classes],
		'geometry': {
			'shape_xyz': [2, 2, 4],
			'inline_min': 100,
			'inline_max': 101,
			'crossline_min': 200,
			'crossline_max': 201,
		},
		'split_codes': {'unsupervised': 0, 'train': 1, 'validation': 2},
		'reference_embedding': {
			'path': str(reference_metadata),
			'sha256': file_sha256(reference_metadata),
			'patch_size': [1, 1, 2],
			'volume_shape_xyz': [2, 2, 4],
		},
		'reference_valid_tokens': {
			'path': str(valid_tokens),
			'sha256': file_sha256(valid_tokens),
		},
		'label_volume': {'path': str(label_path), 'sha256': file_sha256(label_path)},
		'inventory': {'path': str(inventory), 'sha256': file_sha256(inventory)},
	}
	supervision_metadata_path = supervision / 'voxel_dataset_metadata.json'
	supervision_metadata_path.write_text(
		json.dumps(supervision_metadata), encoding='utf-8'
	)
	arrays = open_f3_voxel_prediction_memmaps(
		predictions_dir, volume_shape_xyz=labels.shape, class_count=len(classes)
	)
	arrays.predictions[...] = labels
	arrays.confidence[...] = 1.0
	arrays.valid_mask[...] = True
	if error_coordinate is not None:
		arrays.predictions[error_coordinate] = 3
	if invalid_coordinate is not None:
		arrays.predictions[invalid_coordinate] = -1
		arrays.confidence[invalid_coordinate] = np.nan
		arrays.valid_mask[invalid_coordinate] = False
	for array in (arrays.predictions, arrays.confidence, arrays.valid_mask):
		array.flush()
	summary = validate_f3_voxel_prediction_arrays(
		arrays,
		volume_shape_xyz=labels.shape,
		class_probability_order=[item.class_id for item in classes],
	)
	if prediction_kind == 'token_projection_nearest':
		source_identity = _token_source_identity(
			artifact_root / 'tokens',
			dataset={'name': 'f3', 'version': 'test'},
			label_path=label_path,
			class_info=class_info,
			inventory=inventory,
			geometry_path=geometry_path,
			label_segy=label_segy,
			reference_metadata=reference_metadata,
			valid_tokens=valid_tokens,
		)
	else:
		source_identity = {
			'artifact_identities': {
				'voxel_dataset_metadata': {
					'path': str(supervision_metadata_path),
					'sha256': file_sha256(supervision_metadata_path),
				},
				'voxel_split_grid': {
					'path': str(grid_path),
					'sha256': file_sha256(grid_path),
				},
				'label_volume': {
					'path': str(label_path),
					'sha256': file_sha256(label_path),
				},
			}
		}
	prediction_metadata = {
		'artifact_type': 'f3_lithology_voxel_predictions',
		'schema_version': 1,
		'prediction_kind': prediction_kind,
		'model_tag': 'test-model',
		'class_probability_order': [item.class_id for item in classes],
		'classes': [item.to_dict() for item in classes],
		'volume_shape_xyz': list(labels.shape),
		'patch_size_xyz': [1, 1, 2],
		'invalid_prediction_class_id': -1,
		'invalid_confidence_value': 'nan',
		'inputs': {},
		'source_identity': source_identity,
		'outputs': {
			'predictions': str(predictions_dir / 'f3_voxel_predictions.npy'),
			'confidence': str(predictions_dir / 'f3_voxel_confidence.npy'),
			'valid_mask': str(predictions_dir / 'f3_valid_voxel_mask.npy'),
		},
		'summary': summary,
	}
	write_f3_voxel_prediction_metadata(
		predictions_dir / 'prediction_metadata.json', prediction_metadata
	)
	raw = {
		'paths': {'artifact_root': str(artifact_root), 'f3_root': str(f3_root)},
		'dataset': {'name': 'f3', 'version': 'test'},
		'labels': {
			'source_label_volume': str(label_path),
			'source_label_segy': str(label_segy),
			'png_label_inventory': str(inventory),
			'segy_geometry_json': str(geometry_path),
			'class_info': str(class_info),
		},
		'voxel_predictions': {'input_dir': str(predictions_dir)},
		'voxel_dataset': {'input_dir': str(supervision)},
		'evaluation': {
			'monitored_class_ids': [3, 5],
			'boundary_tolerances': [1, 2],
			'boundary_region_radii': [1, 2],
			'chunk_size_x': chunk_size_x,
		},
		'outputs': {'output_dir': str(artifact_root / 'evaluation')},
	}
	return f3_lithology_voxel_evaluation_config_from_mapping(raw)


def _token_source_identity(  # noqa: PLR0913
	token_dir: Path,
	*,
	dataset: dict[str, str],
	label_path: Path,
	class_info: Path,
	inventory: Path,
	geometry_path: Path,
	label_segy: Path,
	reference_metadata: Path,
	valid_tokens: Path,
) -> dict[str, object]:
	token_dir.mkdir()
	paths = {
		'token_predictions': token_dir / 'f3_token_predictions.npy',
		'token_probabilities': token_dir / 'f3_token_probabilities.npy',
		'valid_token_grid': token_dir / 'f3_valid_token_grid.npy',
		'prediction_metadata': token_dir / 'prediction_metadata.json',
	}
	np.save(paths['token_predictions'], np.zeros((1, 1, 2), dtype=np.int16))
	np.save(
		paths['token_probabilities'], np.ones((1, 1, 2, 3), dtype=np.float32) / 3
	)
	np.save(paths['valid_token_grid'], np.ones((1, 1, 2), dtype=np.bool_))
	paths['prediction_metadata'].write_text(
		json.dumps(
			{
				'dataset': dataset,
				'inputs': {
					'label_volume': str(label_path),
					'class_info': str(class_info),
					'png_label_inventory': str(inventory),
					'segy_geometry_json': str(geometry_path),
					'source_label_segy': str(label_segy),
					'embedding_metadata_json': str(reference_metadata),
					'valid_tokens_path': str(valid_tokens),
				},
			}
		),
		encoding='utf-8',
	)
	return {
		'token_artifact_files': {
			name: {'path': str(path), 'sha256': file_sha256(path)}
			for name, path in paths.items()
		}
	}


def _token_metadata_path(prediction_dir: Path) -> Path:
	metadata = json.loads(
		(prediction_dir / 'prediction_metadata.json').read_text(encoding='utf-8')
	)
	return Path(
		metadata['source_identity']['token_artifact_files']['prediction_metadata'][
			'path'
		]
	)


def _refresh_token_metadata_identity(
	prediction_dir: Path, token_metadata_path: Path
) -> None:
	metadata_path = prediction_dir / 'prediction_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['source_identity']['token_artifact_files']['prediction_metadata'][
		'sha256'
	] = file_sha256(token_metadata_path)
	write_f3_voxel_prediction_metadata(metadata_path, metadata)
