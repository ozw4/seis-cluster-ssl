"""Tests for voxel-decoder train and validation epochs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	VoxelTileManifest,
	VoxelTileRecord,
)
from seis_ssl_cluster.training.voxel_decoder.epoch import (
	train_voxel_decoder_one_epoch,
	validate_voxel_decoder_one_epoch,
)


class _PointDecoder(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.projection = torch.nn.Conv3d(1, 2, kernel_size=1, bias=False)
		with torch.no_grad():
			self.projection.weight[:, 0, 0, 0, 0] = torch.tensor([-1.0, 1.0])

	def forward(
		self, embeddings: torch.Tensor, token_valid_mask: torch.Tensor
	) -> torch.Tensor:
		return self.projection(embeddings.masked_fill(~token_valid_mask[:, None], 0))


class _ValidationTiles(Dataset[dict[str, object]]):
	def __init__(self) -> None:
		self.samples = (
			(self._sample([-2.0, 1.0], [0, 1], 'validation_000000')),
			(self._sample([2.0, -1.0], [1, 1], 'validation_000001')),
		)
		self.manifest = _manifest()

	@staticmethod
	def _sample(
		values: list[float], labels: list[int], tile_id: str
	) -> dict[str, object]:
		return {
			'embeddings': torch.tensor(values).reshape(1, 2, 1, 1),
			'token_valid_mask': torch.ones(2, 1, 1, dtype=torch.bool),
			'labels': torch.tensor(labels).reshape(2, 1, 1),
			'supervision_mask': torch.ones(2, 1, 1, dtype=torch.bool),
			'core_mask': torch.ones(2, 1, 1, dtype=torch.bool),
			'tile_id': tile_id,
		}

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> dict[str, object]:
		return self.samples[index]


def _manifest() -> VoxelTileManifest:
	tiles = tuple(
		VoxelTileRecord(
			tile_id=f'validation_{index:06d}',
			split='validation',
			core_start_token_xyz=(index, 0, 0),
			core_stop_token_xyz=(index + 1, 1, 1),
			input_start_token_xyz=(index, 0, 0),
			input_stop_token_xyz=(index + 1, 1, 1),
			input_padding_before_xyz=(0, 0, 0),
			input_padding_after_xyz=(0, 0, 0),
			core_voxel_start_xyz=(2 * index, 0, 0),
			core_voxel_stop_xyz=(2 * index + 2, 1, 1),
			supervised_voxel_count=2,
			per_class_supervised_counts={'0': 1 if index == 0 else 0, '1': index + 1},
		)
		for index in range(2)
	)
	return VoxelTileManifest(
		split='validation',
		volume_shape_xyz=(4, 1, 1),
		token_grid_shape_xyz=(2, 1, 1),
		patch_size_xyz=(2, 1, 1),
		core_size_tokens=(1, 1, 1),
		context_halo_tokens=(0, 0, 0),
		class_ids=(0, 1),
		tiles=tiles,
	)


def test_train_step_updates_decoder_but_not_embedding_input() -> None:
	decoder = _PointDecoder()
	embeddings = torch.tensor([[[[[-1.0]], [[1.0]]]]], requires_grad=True)
	batch = {
		'embeddings': embeddings,
		'token_valid_mask': torch.ones(1, 2, 1, 1, dtype=torch.bool),
		'labels': torch.tensor([[[[0]], [[1]]]]),
		'supervision_mask': torch.ones(1, 2, 1, 1, dtype=torch.bool),
		'core_mask': torch.ones(1, 2, 1, 1, dtype=torch.bool),
	}
	before = decoder.projection.weight.detach().clone()
	optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

	metrics = train_voxel_decoder_one_epoch(
		decoder=decoder,
		dataloader=[batch],  # type: ignore[arg-type]
		optimizer=optimizer,
		class_weights=torch.ones(2),
		grad_clip_norm=1.0,
	)

	assert not torch.equal(decoder.projection.weight, before)
	assert embeddings.grad is None
	assert metrics['supervised_voxel_count'] == 2


def test_validation_confusion_and_metrics_are_batch_size_invariant() -> None:
	decoder = _PointDecoder()
	dataset = _ValidationTiles()

	one_batch = validate_voxel_decoder_one_epoch(
		decoder=decoder,
		dataloader=DataLoader(dataset, batch_size=2),
		class_weights=torch.tensor([1.0, 2.0]),
	)
	two_batches = validate_voxel_decoder_one_epoch(
		decoder=decoder,
		dataloader=DataLoader(dataset, batch_size=1),
		class_weights=torch.tensor([1.0, 2.0]),
	)

	assert np.array_equal(one_batch['confusion_matrix'], [[1, 0], [1, 2]])
	for key in ('loss', 'accuracy', 'balanced_accuracy', 'macro_f1', 'mean_iou'):
		assert one_batch[key] == pytest.approx(two_batches[key])
	assert one_batch['per_class_support'] == {'0': 1, '1': 3}


def test_validation_rejects_incomplete_manifest_coverage() -> None:
	dataset = _ValidationTiles()
	with pytest.raises(ValueError, match='exactly once'):
		validate_voxel_decoder_one_epoch(
			decoder=_PointDecoder(),
			dataloader=DataLoader(
				dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler([0])
			),
			class_weights=torch.ones(2),
		)


def test_cpu_amp_smoke() -> None:
	dataset = _ValidationTiles()
	metrics = validate_voxel_decoder_one_epoch(
		decoder=_PointDecoder(),
		dataloader=DataLoader(dataset, batch_size=2),
		class_weights=torch.ones(2),
		amp_enabled=True,
	)
	assert np.isfinite(metrics['loss'])
