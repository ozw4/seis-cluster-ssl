from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.f3.lithology import voxel_prediction_artifact as artifact_module
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArrays,
	commit_f3_voxel_prediction_artifact,
	create_f3_voxel_prediction_staging_paths,
	discover_f3_voxel_probability_path,
	f3_voxel_prediction_artifact_paths,
	open_f3_voxel_prediction_memmaps,
	read_f3_voxel_prediction_metadata,
	validate_f3_voxel_prediction_arrays,
	validate_f3_voxel_prediction_artifact,
	write_f3_voxel_prediction_metadata,
)

SHAPE = (2, 2, 2)
CLASS_IDS = (2, 5)


def _metadata(
	summary: dict[str, object], *, include_probabilities: bool = False
) -> dict[str, object]:
	metadata = {
		'artifact_type': 'f3_lithology_voxel_predictions',
		'schema_version': 1,
		'prediction_kind': 'token_projection_nearest',
		'model_tag': 'test-model',
		'class_probability_order': list(CLASS_IDS),
		'classes': [
			{'class_id': 2, 'class_name': 'two'},
			{'class_id': 5, 'class_name': 'five'},
		],
		'volume_shape_xyz': list(SHAPE),
		'patch_size_xyz': [2, 2, 2],
		'invalid_prediction_class_id': -1,
		'invalid_confidence_value': 'nan',
		'inputs': {'token_predictions': 'tokens.npy'},
		'source_identity': {'token_prediction_sha256': 'abc'},
		'outputs': {
			'predictions': 'f3_voxel_predictions.npy',
			'confidence': 'f3_voxel_confidence.npy',
			'valid_mask': 'f3_valid_voxel_mask.npy',
		},
		'summary': summary,
	}
	if include_probabilities:
		metadata['outputs']['probabilities'] = 'f3_voxel_probabilities.npy'
	return metadata


def _write_artifact(
	output_dir: Path, *, include_probabilities: bool = False, first_class: int = 2
) -> None:
	paths = f3_voxel_prediction_artifact_paths(output_dir)
	arrays = open_f3_voxel_prediction_memmaps(
		paths,
		volume_shape_xyz=SHAPE,
		class_count=len(CLASS_IDS),
		include_probabilities=include_probabilities,
	)
	arrays.valid_mask[0, :, :] = True
	arrays.predictions[0, :, :] = first_class
	arrays.confidence[0, :, :] = np.float16(0.75)
	if arrays.probabilities is not None:
		arrays.probabilities[0, :, :, :] = np.asarray(
			[0.75, 0.25] if first_class == 2 else [0.25, 0.75],
			dtype=np.float16,
		)
	for array in (
		arrays.predictions,
		arrays.confidence,
		arrays.valid_mask,
		arrays.probabilities,
	):
		if isinstance(array, np.memmap):
			array.flush()
	summary = validate_f3_voxel_prediction_arrays(
		arrays,
		volume_shape_xyz=SHAPE,
		class_probability_order=CLASS_IDS,
		chunk_voxels=3,
	)
	write_f3_voxel_prediction_metadata(
		paths.metadata,
		_metadata(summary, include_probabilities=include_probabilities),
	)


def _valid_arrays(*, probabilities: bool = False) -> F3VoxelPredictionArrays:
	mask = np.zeros(SHAPE, dtype=np.bool_)
	mask[0] = True
	predictions = np.full(SHAPE, -1, dtype=np.int16)
	predictions[0] = 2
	confidence = np.full(SHAPE, np.nan, dtype=np.float16)
	confidence[0] = np.float16(0.75)
	probability_array = None
	if probabilities:
		probability_array = np.full((*SHAPE, 2), np.nan, dtype=np.float16)
		probability_array[0] = [0.75, 0.25]
	return F3VoxelPredictionArrays(
		predictions=predictions,
		confidence=confidence,
		valid_mask=mask,
		probabilities=probability_array,
	)


@pytest.mark.parametrize('include_probabilities', [False, True])
def test_round_trip_and_optional_probability_discovery(
	tmp_path: Path, include_probabilities
) -> None:
	output_dir = tmp_path / 'predictions'
	_write_artifact(output_dir, include_probabilities=include_probabilities)

	artifact = validate_f3_voxel_prediction_artifact(output_dir, mmap_mode='r')

	assert isinstance(artifact.arrays.predictions, np.memmap)
	assert artifact.arrays.predictions.dtype == np.int16
	assert artifact.arrays.confidence.dtype == np.float16
	assert artifact.arrays.valid_mask.dtype == np.bool_
	assert (artifact.arrays.probabilities is not None) is include_probabilities
	assert (discover_f3_voxel_probability_path(output_dir) is not None) is (
		include_probabilities
	)
	assert artifact.metadata['summary'] == {
		'valid_voxel_count': 4,
		'invalid_voxel_count': 4,
		'class_prediction_counts': {'2': 4, '5': 0},
	}


@pytest.mark.parametrize(
	('mutation', 'match'),
	[
		(
			lambda arrays: arrays.predictions.__setitem__((1, 0, 0), 2),
			'invalid voxels must use prediction class ID -1',
		),
		(
			lambda arrays: arrays.confidence.__setitem__((1, 0, 0), 0.0),
			'invalid voxels must use NaN confidence',
		),
		(
			lambda arrays: arrays.predictions.__setitem__((0, 0, 0), 9),
			'unknown prediction class ID',
		),
	],
)
def test_invalid_sentinel_and_unknown_predictions_are_rejected(
	mutation: object, match: str
) -> None:
	arrays = _valid_arrays()
	mutation(arrays)  # type: ignore[operator]
	with pytest.raises(ValueError, match=match):
		validate_f3_voxel_prediction_arrays(
			arrays,
			volume_shape_xyz=SHAPE,
			class_probability_order=CLASS_IDS,
		)


def test_probability_sum_and_confidence_are_validated() -> None:
	arrays = _valid_arrays(probabilities=True)
	assert arrays.probabilities is not None
	arrays.probabilities[0, 0, 0] = [0.6, 0.3]
	with pytest.raises(ValueError, match='sum to 1'):
		validate_f3_voxel_prediction_arrays(
			arrays,
			volume_shape_xyz=SHAPE,
			class_probability_order=CLASS_IDS,
		)

	arrays.probabilities[0, 0, 0] = [0.6, 0.4]
	with pytest.raises(ValueError, match='confidence must equal'):
		validate_f3_voxel_prediction_arrays(
			arrays,
			volume_shape_xyz=SHAPE,
			class_probability_order=CLASS_IDS,
		)


def test_class_order_mismatch_is_rejected(tmp_path: Path) -> None:
	output_dir = tmp_path / 'predictions'
	_write_artifact(output_dir)
	paths = f3_voxel_prediction_artifact_paths(output_dir)
	metadata = dict(read_f3_voxel_prediction_metadata(paths.metadata))
	metadata['class_probability_order'] = [5, 2]
	write_f3_voxel_prediction_metadata(paths.metadata, metadata)

	with pytest.raises(ValueError, match='classes order must match'):
		validate_f3_voxel_prediction_artifact(output_dir)


@pytest.mark.parametrize(
	('outputs', 'error', 'match'),
	[
		({}, TypeError, 'outputs.predictions'),
		(
			{
				'predictions': 'other.npy',
				'confidence': 'f3_voxel_confidence.npy',
				'valid_mask': 'f3_valid_voxel_mask.npy',
			},
			ValueError,
			'outputs.predictions does not identify',
		),
	],
)
def test_metadata_outputs_are_required_and_bound_to_artifact_paths(
	tmp_path: Path,
	outputs: dict[str, str],
	error: type[Exception],
	match: str,
) -> None:
	output_dir = tmp_path / 'predictions'
	_write_artifact(output_dir)
	paths = f3_voxel_prediction_artifact_paths(output_dir)
	metadata = dict(read_f3_voxel_prediction_metadata(paths.metadata))
	metadata['outputs'] = outputs
	write_f3_voxel_prediction_metadata(paths.metadata, metadata)

	with pytest.raises(error, match=match):
		validate_f3_voxel_prediction_artifact(output_dir)


def test_metadata_declared_probability_file_must_exist(tmp_path: Path) -> None:
	output_dir = tmp_path / 'predictions'
	_write_artifact(output_dir)
	paths = f3_voxel_prediction_artifact_paths(output_dir)
	metadata = dict(read_f3_voxel_prediction_metadata(paths.metadata))
	outputs = dict(metadata['outputs'])  # type: ignore[arg-type]
	outputs['probabilities'] = 'f3_voxel_probabilities.npy'
	metadata['outputs'] = outputs
	write_f3_voxel_prediction_metadata(paths.metadata, metadata)

	with pytest.raises(FileNotFoundError, match='metadata declares'):
		validate_f3_voxel_prediction_artifact(output_dir)


@pytest.mark.parametrize('metadata_only', [False, True])
def test_partial_artifact_is_rejected(
	tmp_path: Path, metadata_only
) -> None:
	paths = f3_voxel_prediction_artifact_paths(tmp_path / 'partial')
	paths.output_dir.mkdir()
	if metadata_only:
		write_f3_voxel_prediction_metadata(paths.metadata, _metadata({}))
	else:
		np.save(paths.predictions, np.full(SHAPE, -1, dtype=np.int16))

	with pytest.raises(FileNotFoundError, match='incomplete'):
		validate_f3_voxel_prediction_artifact(paths)


def test_staged_commit_and_overwrite_safety(tmp_path: Path) -> None:
	target = tmp_path / 'run'
	staging = create_f3_voxel_prediction_staging_paths(target)
	_write_artifact(staging.output_dir)
	committed = commit_f3_voxel_prediction_artifact(staging, target)
	assert committed.output_dir == target
	assert not staging.output_dir.exists()

	replacement = create_f3_voxel_prediction_staging_paths(target, overwrite=True)
	_write_artifact(replacement.output_dir, first_class=5)
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		commit_f3_voxel_prediction_artifact(replacement, target)
	assert validate_f3_voxel_prediction_artifact(target).metadata['summary'] == {
		'valid_voxel_count': 4,
		'invalid_voxel_count': 4,
		'class_prediction_counts': {'2': 4, '5': 0},
	}

	commit_f3_voxel_prediction_artifact(replacement, target, overwrite=True)
	assert validate_f3_voxel_prediction_artifact(target).metadata['summary'] == {
		'valid_voxel_count': 4,
		'invalid_voxel_count': 4,
		'class_prediction_counts': {'2': 0, '5': 4},
	}


def test_atomic_overwrite_rejects_unsupported_platform_without_data_loss(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	target = tmp_path / 'run'
	staging = create_f3_voxel_prediction_staging_paths(target)
	_write_artifact(staging.output_dir)
	commit_f3_voxel_prediction_artifact(staging, target)
	replacement = create_f3_voxel_prediction_staging_paths(target, overwrite=True)
	_write_artifact(replacement.output_dir, first_class=5)

	def unsupported_libc(*_args: object, **_kwargs: object) -> object:
		return object()

	monkeypatch.setattr(artifact_module.ctypes, 'CDLL', unsupported_libc)
	with pytest.raises(NotImplementedError, match=r'requires renameat2'):
		commit_f3_voxel_prediction_artifact(replacement, target, overwrite=True)

	assert replacement.output_dir.is_dir()
	assert validate_f3_voxel_prediction_artifact(target).metadata['summary'] == {
		'valid_voxel_count': 4,
		'invalid_voxel_count': 4,
		'class_prediction_counts': {'2': 4, '5': 0},
	}


def test_non_overwrite_commit_does_not_replace_racing_target(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	target = tmp_path / 'run'
	staging = create_f3_voxel_prediction_staging_paths(target)
	_write_artifact(staging.output_dir)
	original_exists = Path.exists
	target_checks = 0

	def create_target_after_checks(path: Path) -> bool:
		nonlocal target_checks
		if path == target:
			target_checks += 1
			if target_checks == 2:
				target.mkdir()
				(target / 'marker').write_text('existing', encoding='utf-8')
				return False
		return original_exists(path)

	monkeypatch.setattr(Path, 'exists', create_target_after_checks)
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		commit_f3_voxel_prediction_artifact(staging, target)

	assert target_checks == 2
	assert staging.output_dir.is_dir()
	assert (target / 'marker').read_text(encoding='utf-8') == 'existing'


def test_metadata_is_standard_json_and_rejects_nan(tmp_path: Path) -> None:
	path = tmp_path / 'metadata.json'
	write_f3_voxel_prediction_metadata(path, _metadata({}))
	json.loads(
		path.read_text(encoding='utf-8'),
		parse_constant=lambda value: pytest.fail(f'non-standard constant: {value}'),
	)

	with pytest.raises(ValueError, match='Out of range float values'):
		write_f3_voxel_prediction_metadata(path, {'invalid': float('nan')})


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_metadata_reader_rejects_non_standard_json_constants(
	tmp_path: Path, constant: str
) -> None:
	path = tmp_path / 'metadata.json'
	path.write_text(f'{{"summary": {constant}}}', encoding='utf-8')

	with pytest.raises(ValueError, match='non-standard JSON constant'):
		read_f3_voxel_prediction_metadata(path)


def test_extra_summary_fields_are_rejected(tmp_path: Path) -> None:
	output_dir = tmp_path / 'predictions'
	_write_artifact(output_dir)
	paths = f3_voxel_prediction_artifact_paths(output_dir)
	metadata = dict(read_f3_voxel_prediction_metadata(paths.metadata))
	summary = dict(metadata['summary'])  # type: ignore[arg-type]
	summary['extra'] = 'unexpected'
	metadata['summary'] = summary
	write_f3_voxel_prediction_metadata(paths.metadata, metadata)

	with pytest.raises(ValueError, match='summary does not match'):
		validate_f3_voxel_prediction_artifact(output_dir)
