from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.embedding.merge as merge_module
from seis_ssl_cluster.embedding import EmbeddingMerger, SlidingWindow

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize('input_dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('chunk_size_x', [1, 2, 8])
def test_embedding_merge_matches_reference_loop(
	input_dtype: np.dtype,
	chunk_size_x: int,
) -> None:
	rng = np.random.default_rng(259)
	shape = (4, 3, 2)
	dim = 3
	merger = EmbeddingMerger(token_grid_shape_xyz=shape, embedding_dim=dim)
	reference_sums = np.zeros((*shape, dim), dtype=np.float32)
	reference_counts = np.zeros(shape, dtype=np.uint32)

	for x_start, mask_kind in ((0, 'mixed'), (1, 'all'), (1, 'none')):
		window_shape = (3, 3, 2)
		embeddings = rng.normal(size=(*window_shape, dim)).astype(input_dtype)
		if mask_kind == 'mixed':
			valid = rng.random(window_shape) > 0.4
		elif mask_kind == 'all':
			valid = np.ones(window_shape, dtype=np.bool_)
		else:
			valid = np.zeros(window_shape, dtype=np.bool_)
		merger.add_window(
			SlidingWindow(start_xyz=(x_start, 0, 0), size_xyz=window_shape),
			patch_size_xyz=(1, 1, 1),
			token_embeddings=embeddings,
			token_valid_mask=valid,
		)
		for x_index in range(window_shape[0]):
			for y_index in range(window_shape[1]):
				for z_index in range(window_shape[2]):
					if valid[x_index, y_index, z_index]:
						target = (x_start + x_index, y_index, z_index)
						reference_sums[target] += embeddings[
							x_index, y_index, z_index
						].astype(np.float32)
						reference_counts[target] += 1

	actual, actual_valid = merger.finalize(
		output_dtype=np.float32,
		chunk_size_x=chunk_size_x,
	)
	reference = np.zeros_like(reference_sums)
	reference_valid = reference_counts > 0
	np.divide(
		reference_sums,
		reference_counts[..., np.newaxis],
		out=reference,
		where=reference_valid[..., np.newaxis],
	)

	np.testing.assert_array_equal(merger.counts, reference_counts)
	np.testing.assert_array_equal(actual_valid, reference_valid)
	np.testing.assert_allclose(actual, reference, rtol=1.0e-6, atol=0.0)
	assert set(np.unique(reference_counts)) >= {0, 1, 2}


@pytest.mark.parametrize('output_dtype', [np.float16, np.float32])
@pytest.mark.parametrize('chunk_size_x', [1, 3, 10])
def test_embedding_average_memmap_matches_in_memory(
	tmp_path: Path,
	output_dtype: np.dtype,
	chunk_size_x: int,
) -> None:
	merger = EmbeddingMerger(token_grid_shape_xyz=(5, 2, 2), embedding_dim=2)
	merger.sums[...] = np.arange(40, dtype=np.float32).reshape(5, 2, 2, 2)
	merger.counts[...] = np.array(
		[
			[[0, 1], [2, 3]],
			[[1, 2], [3, 0]],
			[[2, 3], [0, 1]],
			[[3, 0], [1, 2]],
			[[0, 1], [2, 3]],
		],
		dtype=np.uint32,
	)
	expected_embeddings, expected_valid = merger.finalize(
		output_dtype=output_dtype,
		chunk_size_x=2,
	)
	embedding_path = tmp_path / 'embeddings.npy'
	valid_path = tmp_path / 'valid.npy'

	merger.write_average(
		embedding_path=embedding_path,
		valid_tokens_path=valid_path,
		output_dtype=output_dtype,
		chunk_size_x=chunk_size_x,
	)

	actual_embeddings = np.load(embedding_path, mmap_mode='r')
	actual_valid = np.load(valid_path, mmap_mode='r')
	assert actual_embeddings.dtype == np.dtype(output_dtype)
	assert actual_valid.dtype == np.dtype(np.bool_)
	np.testing.assert_array_equal(actual_embeddings, expected_embeddings)
	np.testing.assert_array_equal(actual_valid, expected_valid)
	np.testing.assert_array_equal(actual_embeddings[~actual_valid], 0)


def test_embedding_finalize_rejects_invalid_chunk_size() -> None:
	merger = EmbeddingMerger(token_grid_shape_xyz=(1, 1, 1), embedding_dim=1)
	with pytest.raises(ValueError, match='chunk_size_x must be positive'):
		merger.finalize(chunk_size_x=0)


def test_embedding_write_average_rejects_invalid_chunk_size(tmp_path: Path) -> None:
	merger = EmbeddingMerger(token_grid_shape_xyz=(1, 1, 1), embedding_dim=1)
	with pytest.raises(ValueError, match='chunk_size_x must be positive'):
		merger.write_average(
			embedding_path=tmp_path / 'embeddings.npy',
			valid_tokens_path=tmp_path / 'valid.npy',
			output_dtype=np.float32,
			chunk_size_x=0,
		)


def test_embedding_average_temporary_is_chunk_bounded(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	merger = EmbeddingMerger(token_grid_shape_xyz=(7, 3, 2), embedding_dim=4)
	merger.sums[...] = 1
	merger.counts[...] = 1
	original_divide = np.divide
	input_shapes: list[tuple[int, ...]] = []

	def tracked_divide(*args: object, **kwargs: object) -> np.ndarray:
		input_shapes.append(np.shape(args[0]))
		return original_divide(*args, **kwargs)

	monkeypatch.setattr(merge_module.np, 'divide', tracked_divide)

	merger.finalize(output_dtype=np.float32, chunk_size_x=2)

	assert input_shapes
	assert all(shape[0] <= 2 for shape in input_shapes)
