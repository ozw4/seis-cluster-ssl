"""Batch-size-invariant epoch aggregation for the voxel decoder."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
)
from seis_ssl_cluster.f3.lithology.voxel_metrics import update_confusion_matrix
from seis_ssl_cluster.training.voxel_decoder.losses import (
	masked_weighted_voxel_cross_entropy,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.f3.lithology.voxel_tiles import VoxelTileManifest


def train_voxel_decoder_one_epoch(  # noqa: PLR0913
	*,
	decoder: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader[Any],
	optimizer: torch.optim.Optimizer,
	class_weights: torch.Tensor,
	class_ids: Sequence[int] | None = None,
	device: torch.device | str | None = None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	grad_clip_norm: float | None = None,
) -> dict[str, object]:
	"""Train only the decoder for one complete dataloader pass."""
	decoder.train()
	return _run_epoch(
		decoder=decoder,
		dataloader=dataloader,
		class_weights=class_weights,
		class_ids=class_ids,
		device=device,
		optimizer=optimizer,
		amp_enabled=amp_enabled,
		scaler=scaler,
		grad_clip_norm=grad_clip_norm,
		tile_manifest=None,
	)


def validate_voxel_decoder_one_epoch(  # noqa: PLR0913
	*,
	decoder: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader[Any],
	class_weights: torch.Tensor,
	class_ids: Sequence[int] | None = None,
	device: torch.device | str | None = None,
	amp_enabled: bool = False,
	tile_manifest: VoxelTileManifest | None = None,
) -> dict[str, object]:
	"""Evaluate every validation tile once and verify manifest coverage."""
	decoder.eval()
	manifest = tile_manifest or getattr(dataloader.dataset, 'manifest', None)
	if manifest is None:
		raise ValueError('validation requires a voxel tile manifest')
	if manifest.split != 'validation':
		raise ValueError('validation requires a validation tile manifest')
	return _run_epoch(
		decoder=decoder,
		dataloader=dataloader,
		class_weights=class_weights,
		class_ids=class_ids,
		device=device,
		optimizer=None,
		amp_enabled=amp_enabled,
		scaler=None,
		grad_clip_norm=None,
		tile_manifest=manifest,
	)


def _run_epoch(  # noqa: C901, PLR0913
	*,
	decoder: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader[Any],
	class_weights: torch.Tensor,
	class_ids: Sequence[int] | None,
	device: torch.device | str | None,
	optimizer: torch.optim.Optimizer | None,
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	grad_clip_norm: float | None,
	tile_manifest: VoxelTileManifest | None,
) -> dict[str, object]:
	run_device = _decoder_device(decoder, device)
	weights = class_weights.detach().to(device=run_device, dtype=torch.float32)
	ids = _validated_class_ids(class_ids, weights.shape[0], tile_manifest)
	if grad_clip_norm is not None and (
		not isinstance(grad_clip_norm, Real)
		or isinstance(grad_clip_norm, bool)
		or not np.isfinite(grad_clip_norm)
		or grad_clip_norm <= 0
	):
		raise ValueError('grad_clip_norm must be finite and positive')

	confusion = np.zeros((len(ids), len(ids)), dtype=np.int64)
	total_weighted_ce_sum = 0.0
	total_unweighted_ce_sum = 0.0
	total_weight_sum = 0.0
	total_voxels = 0
	class_counts = np.zeros(len(ids), dtype=np.int64)
	observed_tiles: Counter[str] = Counter()
	batches = 0

	grad_context = torch.enable_grad() if optimizer is not None else torch.no_grad()
	with grad_context:
		for raw_batch in dataloader:
			batch = _validated_batch(raw_batch, run_device)
			embeddings = batch['embeddings'].detach()
			labels = batch['labels']
			mask = batch['supervision_mask'] & batch['core_mask']
			encoded_labels = _encode_labels(labels, mask, ids)
			if optimizer is not None:
				optimizer.zero_grad(set_to_none=True)
			with _autocast(run_device, enabled=amp_enabled):
				logits = decoder(embeddings, batch['token_valid_mask'])
				loss, summary = masked_weighted_voxel_cross_entropy(
					logits,
					encoded_labels,
					mask,
					weights,
				)
			if not torch.isfinite(loss):
				raise FloatingPointError('non-finite voxel decoder loss')
			if optimizer is not None:
				_backward_and_step(
					loss,
					decoder=decoder,
					optimizer=optimizer,
					scaler=scaler,
					grad_clip_norm=grad_clip_norm,
				)

			count = int(summary['supervised_voxel_count'])
			weight_sum = summary['class_weight_sum']
			total_voxels += count
			total_weight_sum += weight_sum
			total_weighted_ce_sum += summary['weighted_cross_entropy'] * weight_sum
			total_unweighted_ce_sum += summary['unweighted_cross_entropy'] * count
			for index in range(len(ids)):
				class_counts[index] += int(summary[f'class_{index}_count'])
			predicted = logits.detach().argmax(dim=1)
			update_confusion_matrix(
				confusion,
				encoded_labels.detach().cpu().numpy(),
				predicted.cpu().numpy(),
				valid_mask=mask.detach().cpu().numpy(),
				class_ids=tuple(range(len(ids))),
			)
			if tile_manifest is not None:
				_validate_and_record_tiles(batch, mask, tile_manifest, observed_tiles)
			batches += 1

	if batches == 0:
		raise ValueError('dataloader produced no batches')
	if total_voxels <= 0 or total_weight_sum <= 0:
		raise ValueError('epoch requires supervised voxels with positive weight')
	if tile_manifest is not None:
		_validate_complete_coverage(tile_manifest, observed_tiles, total_voxels)

	classes = tuple(F3ClassInfo(class_id, str(class_id), (0, 0, 0)) for class_id in ids)
	metrics = lithology_metrics_from_confusion_matrix(confusion, classes)
	metrics.update(
		{
			'loss': total_weighted_ce_sum / total_weight_sum,
			'weighted_cross_entropy': total_weighted_ce_sum / total_weight_sum,
			'unweighted_cross_entropy': total_unweighted_ce_sum / total_voxels,
			'class_weight_sum': total_weight_sum,
			'supervised_voxel_count': total_voxels,
			'class_counts': {
				str(class_id): int(count)
				for class_id, count in zip(ids, class_counts, strict=True)
			},
		}
	)
	return metrics


def _validated_batch(
	raw_batch: object, device: torch.device
) -> dict[str, Any]:
	if not isinstance(raw_batch, Mapping):
		raise TypeError('voxel decoder batch must be a mapping')
	batch = dict(raw_batch)
	for key in (
		'embeddings',
		'token_valid_mask',
		'labels',
		'supervision_mask',
		'core_mask',
	):
		value = batch.get(key)
		if not isinstance(value, torch.Tensor):
			raise TypeError(f'batch {key!r} must be a tensor')
		batch[key] = value.to(device=device, non_blocking=True)
	return batch


def _encode_labels(
	labels: torch.Tensor, mask: torch.Tensor, class_ids: tuple[int, ...]
) -> torch.Tensor:
	encoded = torch.full_like(labels, -1)
	matched = torch.zeros_like(mask)
	for index, class_id in enumerate(class_ids):
		selected = labels == class_id
		encoded[selected] = index
		matched |= selected
	if (mask & ~matched).any():
		unknown = torch.unique(labels[mask & ~matched]).detach().cpu().tolist()
		raise ValueError(f'masked labels contain unknown classes: {unknown!r}')
	return encoded


def _backward_and_step(
	loss: torch.Tensor,
	*,
	decoder: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	grad_clip_norm: float | None,
) -> None:
	if scaler is None:
		loss.backward()
	else:
		scaler.scale(loss).backward()
	if grad_clip_norm is not None:
		if scaler is not None:
			scaler.unscale_(optimizer)
		grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip_norm)
		if not torch.isfinite(grad_norm):
			raise FloatingPointError('non-finite voxel decoder gradient norm')
	if scaler is None:
		optimizer.step()
	else:
		scaler.step(optimizer)
		scaler.update()


def _autocast(
	device: torch.device, *, enabled: bool
) -> AbstractContextManager[None]:
	if not enabled:
		return nullcontext()
	return torch.autocast(device_type=device.type)


def _decoder_device(
	decoder: torch.nn.Module, requested: torch.device | str | None
) -> torch.device:
	if requested is not None:
		return torch.device(requested)
	parameter = next(decoder.parameters(), None)
	return parameter.device if parameter is not None else torch.device('cpu')


def _validated_class_ids(
	class_ids: Sequence[int] | None,
	class_count: int,
	manifest: VoxelTileManifest | None,
) -> tuple[int, ...]:
	values = tuple(
		manifest.class_ids
		if class_ids is None and manifest is not None
		else range(class_count)
		if class_ids is None
		else class_ids
	)
	if len(values) != class_count or len(set(values)) != len(values):
		raise ValueError('class_ids must be unique and match class_weights length')
	if any(
		isinstance(value, bool) or not isinstance(value, Integral) for value in values
	):
		raise TypeError('class_ids must contain integers')
	values = tuple(int(value) for value in values)
	if manifest is not None and values != tuple(manifest.class_ids):
		raise ValueError('class_ids must match the validation tile manifest')
	return values


def _validate_and_record_tiles(
	batch: Mapping[str, Any],
	mask: torch.Tensor,
	manifest: VoxelTileManifest,
	observed: Counter[str],
) -> None:
	raw_ids = batch.get('tile_id')
	tile_ids = [raw_ids] if isinstance(raw_ids, str) else list(raw_ids or ())
	if len(tile_ids) != mask.shape[0] or not all(
		isinstance(item, str) for item in tile_ids
	):
		raise ValueError('validation batch must provide one tile_id per sample')
	expected = {tile.tile_id: tile for tile in manifest.tiles}
	per_sample_counts = mask.reshape(mask.shape[0], -1).sum(dim=1).cpu().tolist()
	for tile_id, count in zip(tile_ids, per_sample_counts, strict=True):
		if tile_id not in expected:
			raise ValueError(f'validation produced unknown tile_id {tile_id!r}')
		if int(count) != expected[tile_id].supervised_voxel_count:
			raise ValueError(
				f'validation supervision count mismatch for tile {tile_id!r}'
			)
		observed[tile_id] += 1


def _validate_complete_coverage(
	manifest: VoxelTileManifest,
	observed: Counter[str],
	total_voxels: int,
) -> None:
	expected_ids = {tile.tile_id for tile in manifest.tiles}
	if set(observed) != expected_ids or any(count != 1 for count in observed.values()):
		raise ValueError('validation tiles must cover the manifest exactly once')
	expected_count = sum(tile.supervised_voxel_count for tile in manifest.tiles)
	if total_voxels != expected_count:
		raise ValueError('validation supervised voxel count does not match manifest')


__all__ = [
	'train_voxel_decoder_one_epoch',
	'validate_voxel_decoder_one_epoch',
]
