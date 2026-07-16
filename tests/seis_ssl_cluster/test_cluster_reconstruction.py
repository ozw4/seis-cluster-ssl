from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.clustering.reconstruct as reconstruct_module
import seis_ssl_cluster.clustering.summaries as summaries_module
from seis_ssl_cluster.clustering.reconstruct import (
	reconstruct_labels_for_survey,
	reconstruct_voxel_labels,
)
from seis_ssl_cluster.clustering.summaries import (
	ClusterSummaryInput,
	write_cluster_summaries,
)
from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	normalize_amplitude,
	write_normalization_stats,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_token_labels_upsample_to_clipped_voxel_shape_and_keep_invalid(
	tmp_path: Path,
) -> None:
	token_labels = np.array(
		[
			[[0, -1], [1, 2]],
			[[1, 2], [-1, 0]],
		],
		dtype=np.int32,
	)
	voxel_path = tmp_path / 'survey.cluster_labels_voxel.npy'

	voxels = reconstruct_voxel_labels(
		token_labels,
		patch_size_xyz=(2, 2, 2),
		volume_shape_xyz=(3, 4, 3),
		output_path=voxel_path,
	)

	assert voxel_path.is_file()
	assert voxels.shape == (3, 4, 3)
	assert voxels[0, 0, 0] == 0
	assert voxels[0, 0, 2] == -1
	assert voxels[0, 3, 0] == 1
	assert voxels[2, 3, 0] == -1
	np.testing.assert_array_equal(np.load(voxel_path), voxels)


@pytest.mark.parametrize(
	('patch_size', 'volume_shape'),
	[
		((1, 1, 1), (3, 4, 5)),
		((2, 3, 1), (5, 10, 5)),
		((3, 1, 4), (8, 4, 17)),
	],
)
@pytest.mark.parametrize('label_dtype', [np.int8, np.uint8, np.int32, np.int64])
@pytest.mark.parametrize('chunk_size', [1, 2, 8])
def test_vectorized_voxel_reconstruction_matches_reference_loop(
	tmp_path: Path,
	patch_size: tuple[int, int, int],
	volume_shape: tuple[int, int, int],
	label_dtype: np.dtype,
	chunk_size: int,
) -> None:
	token_shape = tuple(
		(volume_axis + patch_axis - 1) // patch_axis
		for volume_axis, patch_axis in zip(
			volume_shape,
			patch_size,
			strict=True,
		)
	)
	labels = (
		np.arange(np.prod(token_shape), dtype=np.int64).reshape(token_shape) % 7 - 1
	).astype(label_dtype)
	reference = _reference_voxel_reconstruction(
		labels,
		patch_size=patch_size,
		volume_shape=volume_shape,
	)

	in_memory = reconstruct_voxel_labels(
		labels,
		patch_size_xyz=patch_size,
		volume_shape_xyz=volume_shape,
		token_axis_chunk_size=chunk_size,
	)
	output_path = tmp_path / (
		f'voxels-{patch_size}-{label_dtype}-{chunk_size}.npy'
	)
	memmap = reconstruct_voxel_labels(
		labels,
		patch_size_xyz=patch_size,
		volume_shape_xyz=volume_shape,
		output_path=output_path,
		token_axis_chunk_size=chunk_size,
	)

	assert in_memory.dtype == np.dtype(np.int32)
	assert memmap.dtype == np.dtype(np.int32)
	np.testing.assert_array_equal(in_memory, reference)
	np.testing.assert_array_equal(memmap, reference)
	np.testing.assert_array_equal(np.load(output_path), reference)


def test_voxel_reconstruction_rejects_invalid_chunk_size() -> None:
	with pytest.raises(ValueError, match='token_axis_chunk_size must be positive'):
		reconstruct_voxel_labels(
			np.zeros((1, 1, 1), dtype=np.int32),
			patch_size_xyz=(1, 1, 1),
			volume_shape_xyz=(1, 1, 1),
			token_axis_chunk_size=0,
		)


def test_voxel_reconstruction_temporary_is_token_chunk_bounded(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	labels = np.arange(7 * 2 * 2, dtype=np.int32).reshape(7, 2, 2)
	original_broadcast_to = np.broadcast_to
	broadcast_shapes: list[tuple[int, ...]] = []

	def tracked_broadcast_to(
		array: np.ndarray,
		shape: tuple[int, ...],
	) -> np.ndarray:
		broadcast_shapes.append(shape)
		return original_broadcast_to(array, shape)

	monkeypatch.setattr(reconstruct_module.np, 'broadcast_to', tracked_broadcast_to)

	reconstruct_voxel_labels(
		labels,
		patch_size_xyz=(3, 2, 4),
		volume_shape_xyz=(20, 4, 8),
		token_axis_chunk_size=2,
	)

	assert broadcast_shapes
	assert all(shape[0] <= 2 for shape in broadcast_shapes)


def _reference_voxel_reconstruction(
	labels: np.ndarray,
	*,
	patch_size: tuple[int, int, int],
	volume_shape: tuple[int, int, int],
) -> np.ndarray:
	output = np.empty(volume_shape, dtype=np.int32)
	for x_index in range(labels.shape[0]):
		for y_index in range(labels.shape[1]):
			for z_index in range(labels.shape[2]):
				x_start = x_index * patch_size[0]
				y_start = y_index * patch_size[1]
				z_start = z_index * patch_size[2]
				output[
					x_start : min(x_start + patch_size[0], volume_shape[0]),
					y_start : min(y_start + patch_size[1], volume_shape[1]),
					z_start : min(z_start + patch_size[2], volume_shape[2]),
				] = labels[x_index, y_index, z_index]
	return output


def test_reconstruct_labels_for_survey_uses_embedding_metadata_shape(
	tmp_path: Path,
) -> None:
	labels_dir = tmp_path / 'labels' / 'k3'
	embedding_dir = tmp_path / 'embeddings'
	labels_dir.mkdir(parents=True)
	embedding_dir.mkdir()
	token_path = labels_dir / 'survey.cluster_labels_token.npy'
	embedding_metadata_path = embedding_dir / 'survey.embedding_metadata.json'
	label_metadata_path = labels_dir / 'survey.cluster_label_metadata.json'
	np.save(token_path, np.zeros((2, 2, 1), dtype=np.int32))
	embedding_metadata_path.write_text(
		json.dumps(
			{
				'patch_size': [2, 3, 4],
				'volume_shape_xyz': [3, 5, 4],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	label_metadata_path.write_text(
		json.dumps(
			{
				'embedding_input': {
					'metadata_path': str(embedding_metadata_path),
				},
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = reconstruct_labels_for_survey(
		token_path,
		metadata_path=label_metadata_path,
	)

	assert result.voxel_labels_path == labels_dir / 'survey.cluster_labels_voxel.npy'
	assert np.load(result.voxel_labels_path).shape == (3, 5, 4)


def test_reconstruct_labels_for_survey_skips_compatible_existing_voxels(
	tmp_path: Path,
) -> None:
	labels_dir = tmp_path / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	token_path = labels_dir / 'survey.cluster_labels_token.npy'
	metadata_path = labels_dir / 'survey.cluster_label_metadata.json'
	voxel_path = labels_dir / 'survey.cluster_labels_voxel.npy'
	np.save(token_path, np.ones((1, 1, 1), dtype=np.int32))
	np.save(voxel_path, np.full((2, 2, 2), 7, dtype=np.int32))
	metadata_path.write_text(
		json.dumps({'patch_size': [2, 2, 2], 'volume_shape_xyz': [2, 2, 2]})
		+ '\n',
		encoding='utf-8',
	)

	result = reconstruct_labels_for_survey(
		token_path,
		metadata_path=metadata_path,
		skip_existing_voxel_labels=True,
	)

	assert result.skipped_existing_voxel_labels is True
	np.testing.assert_array_equal(
		np.load(voxel_path),
		np.full((2, 2, 2), 7, dtype=np.int32),
	)


def test_reconstruct_labels_for_survey_rejects_incompatible_existing_voxels(
	tmp_path: Path,
) -> None:
	labels_dir = tmp_path / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	token_path = labels_dir / 'survey.cluster_labels_token.npy'
	metadata_path = labels_dir / 'survey.cluster_label_metadata.json'
	voxel_path = labels_dir / 'survey.cluster_labels_voxel.npy'
	np.save(token_path, np.ones((1, 1, 1), dtype=np.int32))
	np.save(voxel_path, np.ones((2, 2, 2), dtype=np.float32))
	metadata_path.write_text(
		json.dumps({'patch_size': [2, 2, 2], 'volume_shape_xyz': [2, 2, 2]})
		+ '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='incompatible existing voxel labels'):
		reconstruct_labels_for_survey(
			token_path,
			metadata_path=metadata_path,
			skip_existing_voxel_labels=True,
		)


def test_cluster_summary_counts_equal_valid_assigned_tokens(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(summaries_module, '_TOKEN_CHUNK_SIZE', 3)
	labels_path = tmp_path / 'survey.cluster_labels_token.npy'
	embeddings_path = tmp_path / 'survey.embeddings.npy'
	np.save(
		labels_path,
		np.array([[[0, 1], [-1, 1]], [[2, -1], [2, 2]]], dtype=np.int32),
	)
	np.save(embeddings_path, np.ones((2, 2, 2, 3), dtype=np.float32))

	artifacts = write_cluster_summaries(
		[
			ClusterSummaryInput(
				survey_id='survey',
				labels_path=labels_path,
				embeddings_path=embeddings_path,
			),
		],
		k=3,
		output_dir=tmp_path / 'summary',
	)

	payload = json.loads(artifacts.json_path.read_text(encoding='utf-8'))
	assert payload['total_valid_token_count'] == 6
	assert payload['total_invalid_token_count'] == 2
	assert [row['token_count'] for row in payload['clusters']] == [1, 2, 3]
	with artifacts.csv_path.open(encoding='utf-8', newline='') as file_obj:
		csv_rows = list(csv.DictReader(file_obj))
	assert csv_rows[0].keys() == {
		'cluster',
		'token_count',
		'valid_fraction',
		'mean_amplitude_norm',
		'std_amplitude_norm',
		'mean_embedding_norm',
		'survey_coverage_count',
		'survey_coverage_fraction',
	}
	assert [int(row['token_count']) for row in csv_rows] == [1, 2, 3]
	assert [float(row['mean_embedding_norm']) for row in csv_rows] == pytest.approx(
		[float(np.sqrt(3.0))] * 3,
	)
	assert [int(row['survey_coverage_count']) for row in csv_rows] == [1, 1, 1]
	assert artifacts.png_path.is_file()


def test_cluster_summary_amplitude_std_uses_patch_voxels(tmp_path: Path) -> None:
	labels_path = tmp_path / 'survey.cluster_labels_token.npy'
	metadata_path = tmp_path / 'survey.cluster_label_metadata.json'
	amplitude_path = tmp_path / 'survey.npy'
	stats_path = tmp_path / 'survey.normalization_stats.json'
	volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
	stats = SurveyNormalizationStats(
		survey_id='survey',
		source_path=amplitude_path,
		grid_order=('x', 'y', 'z'),
		clip_low_percentile=0.0,
		clip_high_percentile=100.0,
		clip_low=0.0,
		clip_high=7.0,
		median=0.0,
		iqr=1.0,
		eps=1.0e-6,
	)
	np.save(labels_path, np.array([[[0]]], dtype=np.int32))
	np.save(amplitude_path, volume)
	write_normalization_stats(stats, stats_path)
	metadata_path.write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'normalization_stats_path': str(stats_path),
				'patch_size': [2, 2, 2],
				'volume_shape_xyz': [2, 2, 2],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	artifacts = write_cluster_summaries(
		[
			ClusterSummaryInput(
				survey_id='survey',
				labels_path=labels_path,
				metadata_path=metadata_path,
			),
		],
		k=1,
		output_dir=tmp_path / 'summary',
		include_amplitude_norm=True,
	)

	payload = json.loads(artifacts.json_path.read_text(encoding='utf-8'))
	cluster = payload['clusters'][0]
	expected = normalize_amplitude(volume, stats)
	assert cluster['token_count'] == 1
	assert cluster['mean_amplitude_norm'] == pytest.approx(float(np.mean(expected)))
	assert cluster['std_amplitude_norm'] == pytest.approx(float(np.std(expected)))
