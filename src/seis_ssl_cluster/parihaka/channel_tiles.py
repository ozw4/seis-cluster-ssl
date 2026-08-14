"""Shared deterministic tile geometry for Parihaka Channel supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.parihaka.channel_data import (
	CHANNEL_CLASS_ID,
	SectionLines,
	split_mask_for_crop,
)

if TYPE_CHECKING:
	from collections.abc import Sequence

CHANNEL_PATCH_SIZE_VOXELS = (8, 8, 8)
CHANNEL_CORE_SIZE_TOKENS = (8, 8, 8)
CHANNEL_CONTEXT_HALO_TOKENS = (1, 1, 1)


@dataclass(frozen=True)
class ChannelTileSettings:
	"""Volume, token, core, and halo geometry for Channel tiles."""

	volume_shape_xyz: tuple[int, int, int]
	token_grid_shape_xyz: tuple[int, int, int]
	patch_size_xyz: tuple[int, int, int]
	core_size_tokens: tuple[int, int, int]
	context_halo_tokens: tuple[int, int, int]

	@property
	def input_size_tokens(self) -> tuple[int, int, int]:
		"""Return the fixed halo-padded encoder input shape in tokens."""
		return tuple(
			self.core_size_tokens[axis] + 2 * self.context_halo_tokens[axis]
			for axis in range(3)
		)  # type: ignore[return-value]

	@property
	def input_size_voxels(self) -> tuple[int, int, int]:
		"""Return the fixed halo-padded encoder input shape in voxels."""
		return tuple(
			self.input_size_tokens[axis] * self.patch_size_xyz[axis]
			for axis in range(3)
		)  # type: ignore[return-value]


@dataclass(frozen=True)
class ChannelTileRecord:
	"""One non-empty core tile in deterministic grid order."""

	tile_id: int
	core_start_token: tuple[int, int, int]
	core_stop_token: tuple[int, int, int]
	supervised_voxels: int

	@property
	def index(self) -> int:
		"""Retain the frozen dataset's historical record identifier spelling."""
		return self.tile_id


@dataclass(frozen=True)
class ChannelTileTargets:
	"""Halo-padded masks, labels, and source coordinates for one tile."""

	token_source_start: tuple[int, int, int]
	token_source_stop: tuple[int, int, int]
	token_destination_start: tuple[int, int, int]
	token_destination_stop: tuple[int, int, int]
	input_start_token: tuple[int, int, int]
	labels: np.ndarray
	token_valid_mask: np.ndarray
	section_mask: np.ndarray
	core_mask: np.ndarray
	supervision_mask: np.ndarray


def enumerate_channel_tile_records(  # noqa: PLR0913
	*,
	valid_tokens: np.ndarray,
	labels: np.ndarray,
	settings: ChannelTileSettings,
	train: SectionLines,
	validation: SectionLines,
	reserved_training: SectionLines,
	split: str,
) -> tuple[tuple[ChannelTileRecord, ...], tuple[int, int]]:
	"""Enumerate supervised core tiles and count binary classes once."""
	_validate_inputs(valid_tokens, labels, settings)
	records: list[ChannelTileRecord] = []
	counts = np.zeros(2, dtype=np.int64)
	tile_id = 0
	core = settings.core_size_tokens
	grid = settings.token_grid_shape_xyz
	patch = settings.patch_size_xyz
	for tx in range(0, grid[0], core[0]):
		for ty in range(0, grid[1], core[1]):
			for tz in range(0, grid[2], core[2]):
				start_token = (tx, ty, tz)
				stop_token = tuple(
					min(start_token[axis] + core[axis], grid[axis])
					for axis in range(3)
				)
				voxel_start = tuple(
					start_token[axis] * patch[axis] for axis in range(3)
				)
				voxel_stop = tuple(
					min(stop_token[axis] * patch[axis], settings.volume_shape_xyz[axis])
					for axis in range(3)
				)
				shape = _difference(voxel_stop, voxel_start)
				section = split_mask_for_crop(
					shape=shape,
					start_xyz=voxel_start,
					train=train,
					validation=validation,
					reserved_training=reserved_training,
					split=split,
				)
				token_valid = valid_tokens[_slices(start_token, stop_token)]
				valid_voxels = _expand_token_mask(token_valid, patch, shape)
				supervision = _supervision_mask(
					core_mask=np.ones(shape, dtype=np.bool_),
					valid_voxels=valid_voxels,
					section_mask=section,
					valid_labels=np.ones(shape, dtype=np.bool_),
				)
				total = int(np.count_nonzero(supervision))
				if total:
					binary = (
						labels[_slices(voxel_start, voxel_stop)] == CHANNEL_CLASS_ID
					)
					positive = int(np.count_nonzero(binary & supervision))
					counts += (total - positive, positive)
					records.append(
						ChannelTileRecord(tile_id, start_token, stop_token, total)
					)
				tile_id += 1
	return tuple(records), (int(counts[0]), int(counts[1]))


def build_channel_tile_targets(  # noqa: PLR0913
	*,
	record: ChannelTileRecord,
	valid_tokens: np.ndarray,
	labels: np.ndarray,
	settings: ChannelTileSettings,
	train: SectionLines,
	validation: SectionLines,
	reserved_training: SectionLines,
	split: str,
) -> ChannelTileTargets:
	"""Build the shared halo-padded target and supervision contract."""
	_validate_inputs(valid_tokens, labels, settings)
	patch = settings.patch_size_xyz
	halo = settings.context_halo_tokens
	input_start = tuple(
		record.core_start_token[axis] - halo[axis] for axis in range(3)
	)
	token_source_start = tuple(max(0, value) for value in input_start)
	token_source_stop = tuple(
		min(
			record.core_stop_token[axis] + halo[axis],
			settings.token_grid_shape_xyz[axis],
		)
		for axis in range(3)
	)
	token_destination_start = _difference(token_source_start, input_start)
	token_destination_stop = tuple(
		token_destination_start[axis]
		+ token_source_stop[axis]
		- token_source_start[axis]
		for axis in range(3)
	)
	token_source = _slices(token_source_start, token_source_stop)
	token_destination = _slices(token_destination_start, token_destination_stop)
	token_mask = np.zeros(settings.input_size_tokens, dtype=np.bool_)
	token_mask[token_destination] = valid_tokens[token_source]

	voxel_global_start = tuple(input_start[axis] * patch[axis] for axis in range(3))
	voxel_source_start = tuple(max(0, value) for value in voxel_global_start)
	voxel_source_stop = tuple(
		min(
			(record.core_stop_token[axis] + halo[axis]) * patch[axis],
			settings.volume_shape_xyz[axis],
		)
		for axis in range(3)
	)
	voxel_destination_start = _difference(voxel_source_start, voxel_global_start)
	voxel_destination_stop = tuple(
		voxel_destination_start[axis]
		+ voxel_source_stop[axis]
		- voxel_source_start[axis]
		for axis in range(3)
	)
	voxel_source = _slices(voxel_source_start, voxel_source_stop)
	voxel_destination = _slices(voxel_destination_start, voxel_destination_stop)

	label_crop = np.full(settings.input_size_voxels, -1, dtype=np.int64)
	label_crop[voxel_destination] = (
		np.asarray(labels[voxel_source]) == CHANNEL_CLASS_ID
	).astype(np.int64)
	section_mask = np.zeros(settings.input_size_voxels, dtype=np.bool_)
	section_mask[voxel_destination] = split_mask_for_crop(
		shape=_difference(voxel_source_stop, voxel_source_start),
		start_xyz=voxel_source_start,
		train=train,
		validation=validation,
		reserved_training=reserved_training,
		split=split,
	)
	core_mask = np.zeros(settings.input_size_voxels, dtype=np.bool_)
	core_start = tuple(halo[axis] * patch[axis] for axis in range(3))
	core_stop = tuple(
		core_start[axis]
		+ (record.core_stop_token[axis] - record.core_start_token[axis]) * patch[axis]
		for axis in range(3)
	)
	core_mask[_slices(core_start, core_stop)] = True
	valid_voxels = _expand_token_mask(
		token_mask,
		patch,
		settings.input_size_voxels,
	)
	supervision = _supervision_mask(
		core_mask=core_mask,
		valid_voxels=valid_voxels,
		section_mask=section_mask,
		valid_labels=label_crop >= 0,
	)
	if int(np.count_nonzero(supervision)) != record.supervised_voxels:
		raise RuntimeError('runtime section mask no longer matches tile inspection')
	return ChannelTileTargets(
		token_source_start=token_source_start,
		token_source_stop=token_source_stop,
		token_destination_start=token_destination_start,
		token_destination_stop=token_destination_stop,
		input_start_token=input_start,
		labels=label_crop,
		token_valid_mask=token_mask,
		section_mask=section_mask,
		core_mask=core_mask,
		supervision_mask=supervision,
	)


def _validate_inputs(
	valid_tokens: np.ndarray,
	labels: np.ndarray,
	settings: ChannelTileSettings,
) -> None:
	if tuple(valid_tokens.shape) != settings.token_grid_shape_xyz:
		raise ValueError('valid-token shape does not match channel tile settings')
	if valid_tokens.dtype != np.bool_:
		raise TypeError('valid-token mask must have dtype bool')
	if tuple(labels.shape) != settings.volume_shape_xyz:
		raise ValueError('label shape does not match channel tile settings')
	for name, value in (
		('volume_shape_xyz', settings.volume_shape_xyz),
		('token_grid_shape_xyz', settings.token_grid_shape_xyz),
		('patch_size_xyz', settings.patch_size_xyz),
		('core_size_tokens', settings.core_size_tokens),
	):
		if len(value) != 3 or any(axis <= 0 for axis in value):
			raise ValueError(f'{name} must be a positive integer triple')
	if len(settings.context_halo_tokens) != 3 or any(
		axis < 0 for axis in settings.context_halo_tokens
	):
		raise ValueError('context_halo_tokens must be a nonnegative integer triple')


def _expand_token_mask(
	mask: np.ndarray,
	patch: Sequence[int],
	shape: Sequence[int],
) -> np.ndarray:
	result = np.asarray(mask, dtype=np.bool_)
	for axis, repeats in enumerate(patch):
		result = np.repeat(result, repeats, axis=axis)
	return result[tuple(slice(0, size) for size in shape)]


def _supervision_mask(
	*,
	core_mask: np.ndarray,
	valid_voxels: np.ndarray,
	section_mask: np.ndarray,
	valid_labels: np.ndarray,
) -> np.ndarray:
	return core_mask & valid_voxels & section_mask & valid_labels


def _difference(
	stop: Sequence[int], start: Sequence[int]
) -> tuple[int, int, int]:
	return tuple(
		stop_axis - start_axis
		for start_axis, stop_axis in zip(start, stop, strict=True)
	)  # type: ignore[return-value]


def _slices(start: Sequence[int], stop: Sequence[int]) -> tuple[slice, slice, slice]:
	return tuple(slice(a, b) for a, b in zip(start, stop, strict=True))  # type: ignore[return-value]


__all__ = [
	'CHANNEL_CONTEXT_HALO_TOKENS',
	'CHANNEL_CORE_SIZE_TOKENS',
	'CHANNEL_PATCH_SIZE_VOXELS',
	'ChannelTileRecord',
	'ChannelTileSettings',
	'ChannelTileTargets',
	'build_channel_tile_targets',
	'enumerate_channel_tile_records',
]
