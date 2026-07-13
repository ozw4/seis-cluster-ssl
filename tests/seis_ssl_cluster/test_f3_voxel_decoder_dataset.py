from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data.f3_voxel_decoder_dataset import (
	F3VoxelDecoderDataset,
	build_f3_voxel_decoder_dataloader,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_tiles import build_voxel_tile_manifest


def _dataset(tmp_path) -> F3VoxelDecoderDataset:
	embeddings = np.arange(4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3)
	valid = np.ones((4, 2, 2), dtype=np.bool_)
	valid[0, 0, 0] = False
	labels = np.zeros((8, 4, 4), dtype=np.int16)
	labels[3, 0, 0] = -1
	split = np.ones(labels.shape, dtype=np.uint8)
	split[:2, :2, :2] = 0
	paths = {
		'embeddings': tmp_path / 'embeddings.npy',
		'valid': tmp_path / 'valid.npy',
		'labels': tmp_path / 'labels.npy',
		'split': tmp_path / 'split.npy',
		'metadata': tmp_path / 'embedding.json',
		'supervision': tmp_path / 'supervision.json',
	}
	for key, value in (
		('embeddings', embeddings),
		('valid', valid),
		('labels', labels),
		('split', split),
	):
		np.save(paths[key], value, allow_pickle=False)
	metadata = {
		'volume_shape_xyz': [8, 4, 4],
		'patch_size': [2, 2, 2],
		'token_grid_shape': [4, 2, 2],
		'model_geometry': {'encoder_dim': 3},
		'preprocessing': {'normalized_clip_abs': None},
		'zero_mask': {'enabled': True},
	}
	paths['metadata'].write_text(json.dumps(metadata), encoding='utf-8')
	paths['supervision'].write_text(
		json.dumps(
			{
				'reference_embedding': {'metadata': metadata},
				'reference_valid_tokens': {'sha256': file_sha256(paths['valid'])},
			}
		),
		encoding='utf-8',
	)
	manifest = build_voxel_tile_manifest(
		split,
		labels,
		split='train',
		patch_size_xyz=(2, 2, 2),
		core_size_tokens=(2, 2, 2),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0,),
		canonical_valid_tokens=valid,
	)
	return F3VoxelDecoderDataset(
		paths['embeddings'],
		paths['valid'],
		paths['metadata'],
		paths['labels'],
		paths['split'],
		manifest,
		supervision_metadata_path=paths['supervision'],
	)


def test_mmap_dataset_masks_context_invalid_and_unknown_voxels(tmp_path) -> None:
	dataset = _dataset(tmp_path)
	item = dataset[0]

	assert isinstance(dataset._embeddings, np.memmap)  # noqa: SLF001
	assert item['embeddings'].shape == (3, 4, 4, 4)
	assert item['embeddings'].dtype == torch.float32
	assert item['token_valid_mask'].shape == (4, 4, 4)
	assert item['labels'].shape == (8, 8, 8)
	assert not item['supervision_mask'][0].any()
	assert not item['supervision_mask'][~item['core_mask']].any()
	assert not item['supervision_mask'][2:4, 2:4, 2:4].any()
	assert (item['labels'] == -1).any()


def test_dataset_collates_and_shuffle_is_seeded(tmp_path) -> None:
	dataset = _dataset(tmp_path)
	batch = next(
		iter(
			build_f3_voxel_decoder_dataloader(
				dataset, batch_size=2, shuffle=False, seed=4
			)
		)
	)
	assert batch['embeddings'].shape[0] == 2
	orders = []
	for _ in range(2):
		loader = build_f3_voxel_decoder_dataloader(
			dataset, batch_size=1, shuffle=True, seed=19
		)
		orders.append([item['tile_id'][0] for item in loader])
	assert orders[0] == orders[1]


def test_dataset_rejects_valid_token_hash_mismatch(tmp_path) -> None:
	dataset = _dataset(tmp_path)
	valid = np.load(dataset.valid_tokens_path)
	valid[0, 0, 0] = True
	np.save(dataset.valid_tokens_path, valid, allow_pickle=False)

	with pytest.raises(ValueError, match='valid-token hash'):
		F3VoxelDecoderDataset(
			dataset.embedding_path,
			dataset.valid_tokens_path,
			dataset.embedding_metadata_path,
			dataset.label_volume_path,
			dataset.supervision_split_grid_path,
			dataset.manifest,
			supervision_metadata_path=dataset.supervision_metadata_path,
		)


def test_dataset_rejects_missing_canonical_valid_token_hash(tmp_path) -> None:
	dataset = _dataset(tmp_path)
	dataset.supervision_metadata_path.write_text(
		json.dumps({'reference_embedding': {'metadata': dataset.embedding_metadata}}),
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match=r'reference_valid_tokens\.sha256'):
		F3VoxelDecoderDataset(
			dataset.embedding_path,
			dataset.valid_tokens_path,
			dataset.embedding_metadata_path,
			dataset.label_volume_path,
			dataset.supervision_split_grid_path,
			dataset.manifest,
			supervision_metadata_path=dataset.supervision_metadata_path,
		)


@pytest.mark.parametrize('key', ['preprocessing', 'zero_mask'])
def test_dataset_rejects_pairing_geometry_missing_from_both_metadata(
	tmp_path, key
) -> None:
	dataset = _dataset(tmp_path)
	metadata = dict(dataset.embedding_metadata)
	metadata.pop(key)
	dataset.embedding_metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	dataset.supervision_metadata_path.write_text(
		json.dumps(
			{
				'reference_embedding': {'metadata': metadata},
				'reference_valid_tokens': {
					'sha256': file_sha256(dataset.valid_tokens_path)
				},
			}
		),
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match=rf'pairing mismatch for {key}'):
		F3VoxelDecoderDataset(
			dataset.embedding_path,
			dataset.valid_tokens_path,
			dataset.embedding_metadata_path,
			dataset.label_volume_path,
			dataset.supervision_split_grid_path,
			dataset.manifest,
			supervision_metadata_path=dataset.supervision_metadata_path,
		)


def test_dataset_rejects_reference_metadata_without_embedding_dimension(
	tmp_path,
) -> None:
	dataset = _dataset(tmp_path)
	reference_metadata = dict(dataset.embedding_metadata)
	reference_metadata.pop('model_geometry')
	dataset.supervision_metadata_path.write_text(
		json.dumps(
			{
				'reference_embedding': {'metadata': reference_metadata},
				'reference_valid_tokens': {
					'sha256': file_sha256(dataset.valid_tokens_path)
				},
			}
		),
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='pairing mismatch for embedding dim'):
		F3VoxelDecoderDataset(
			dataset.embedding_path,
			dataset.valid_tokens_path,
			dataset.embedding_metadata_path,
			dataset.label_volume_path,
			dataset.supervision_split_grid_path,
			dataset.manifest,
			supervision_metadata_path=dataset.supervision_metadata_path,
		)
