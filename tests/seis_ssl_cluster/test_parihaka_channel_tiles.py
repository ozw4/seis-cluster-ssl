# ruff: noqa: TC003

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.embedding.writer import output_paths
from seis_ssl_cluster.parihaka.channel_data import SectionLines
from seis_ssl_cluster.parihaka.channel_decoder import (
	ChannelTileDataset,
	DecoderTiles,
	EmbeddingGeometry,
)
from seis_ssl_cluster.parihaka.channel_tiles import (
	ChannelTileSettings,
	build_channel_tile_targets,
	enumerate_channel_tile_records,
)


def _fixture(tmp_path: Path) -> tuple[
	ChannelTileSettings,
	np.ndarray,
	np.ndarray,
	Path,
]:
	settings = ChannelTileSettings(
		volume_shape_xyz=(12, 13, 14),
		token_grid_shape_xyz=(2, 2, 2),
		patch_size_xyz=(8, 8, 8),
		core_size_tokens=(1, 1, 1),
		context_halo_tokens=(1, 1, 1),
	)
	valid = np.ones(settings.token_grid_shape_xyz, dtype=np.bool_)
	valid[1, 1, 1] = False
	labels = np.ones(settings.volume_shape_xyz, dtype=np.int8)
	labels[:, :, ::2] = 5
	labels_path = tmp_path / 'labels.npy'
	np.save(labels_path, labels)
	return settings, valid, labels, labels_path


def test_shared_helper_matches_frozen_dataset(tmp_path: Path) -> None:
	settings, valid, labels, labels_path = _fixture(tmp_path)
	paths = output_paths(tmp_path / 'embeddings', 'parihaka')
	paths.embeddings.parent.mkdir(parents=True)
	np.save(paths.embeddings, np.zeros((*valid.shape, 3), dtype=np.float16))
	np.save(paths.valid_tokens, valid)
	lines = SectionLines((0,), (0,))
	validation = SectionLines((8,), (8,))
	reserved = SectionLines((0, 9), (0, 9))
	records, counts = enumerate_channel_tile_records(
		valid_tokens=valid,
		training_selection_mask=valid.copy(),
		labels=labels,
		settings=settings,
		train=lines,
		validation=validation,
		reserved_training=reserved,
		split='train',
	)
	geometry = EmbeddingGeometry(
		pretrained=paths,
		random=paths,
		volume_shape_xyz=settings.volume_shape_xyz,
		token_grid_shape_xyz=settings.token_grid_shape_xyz,
		patch_size_xyz=settings.patch_size_xyz,
		embedding_shape=(*valid.shape, 3),
		embedding_dim=3,
		pretrained_metadata={},
		random_metadata={},
		pretrained_model_source={},
		random_model_source={},
	)
	dataset = ChannelTileDataset(
		embedding_path=paths.embeddings,
		valid_tokens_path=paths.valid_tokens,
		labels_path=labels_path,
		geometry=geometry,
		lines=lines,
		validation=validation,
		reserved_training=reserved,
		split='train',
		tiles=DecoderTiles((1, 1, 1), (1, 1, 1)),
		training_selection_mask=valid.copy(),
	)
	assert dataset.records == records
	assert dataset.class_counts == counts
	for index, record in enumerate(records):
		targets = build_channel_tile_targets(
			record=record,
			valid_tokens=valid,
			training_selection_mask=valid.copy(),
			labels=labels,
			settings=settings,
			train=lines,
			validation=validation,
			reserved_training=reserved,
			split='train',
		)
		item = dataset[index]
		assert item['tile_id'] == record.tile_id
		assert torch.equal(
			item['supervision_mask'], torch.from_numpy(targets.supervision_mask)
		)


def test_edge_tile_padding_core_halo_and_split_priority(tmp_path: Path) -> None:
	settings, valid, labels, _ = _fixture(tmp_path)
	lines = SectionLines((0, 8), (0, 8))
	validation = SectionLines((8,), (8,))
	reserved = SectionLines((0, 8, 9), (0, 8, 9))
	records, _ = enumerate_channel_tile_records(
		valid_tokens=valid,
		training_selection_mask=None,
		labels=labels,
		settings=settings,
		train=lines,
		validation=validation,
		reserved_training=reserved,
		split='validation',
	)
	record = next(item for item in records if item.core_start_token == (1, 1, 0))
	targets = build_channel_tile_targets(
		record=record,
		valid_tokens=valid,
		training_selection_mask=None,
		labels=labels,
		settings=settings,
		train=lines,
		validation=validation,
		reserved_training=reserved,
		split='validation',
	)
	assert targets.token_valid_mask.shape == (3, 3, 3)
	assert targets.labels.shape == (24, 24, 24)
	assert targets.core_mask[8:16, 8:16, 8:16].all()
	assert not targets.core_mask[:8].any()
	assert targets.supervision_mask[9, 8, 8]
	assert not targets.supervision_mask[9, 9, 8]
	assert targets.supervision_mask[8, 8, 8]
	section_voxels = int(np.count_nonzero(targets.section_mask & targets.core_mask))
	assert section_voxels == 64
