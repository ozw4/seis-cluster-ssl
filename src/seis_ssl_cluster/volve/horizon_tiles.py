'''Deterministic lateral tiles shared by Volve horizon regimes.'''

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import cast

import numpy as np

from seis_ssl_cluster.data.window_preprocessing import reduce_valid_mask_to_tokens
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_model import (
	HORIZON_CONTEXT_HALO_TOKENS,
	HORIZON_CORE_SIZE_TOKENS,
	HORIZON_INPUT_SIZE_TOKENS,
	HORIZON_INPUT_SIZE_VOXELS,
	HORIZON_PATCH_SIZE,
)

HORIZON_WINDOW_START = 552
HORIZON_WINDOW_STOP = 768
HORIZON_WINDOW_LENGTH = HORIZON_WINDOW_STOP - HORIZON_WINDOW_START
HORIZON_MIN_TOKEN_VALID_FRACTION = 0.1


@dataclass(frozen=True)
class HorizonTileSettings:
	'''Fixed production geometry plus the canonical lateral survey shape.'''

	lateral_shape_xy: tuple[int, int]
	patch_size_xyz: tuple[int, int, int] = HORIZON_PATCH_SIZE
	core_size_tokens: tuple[int, int, int] = HORIZON_CORE_SIZE_TOKENS
	context_halo_tokens: tuple[int, int, int] = HORIZON_CONTEXT_HALO_TOKENS
	window_start: int = HORIZON_WINDOW_START
	window_stop: int = HORIZON_WINDOW_STOP
	min_token_valid_fraction: float = HORIZON_MIN_TOKEN_VALID_FRACTION

	@property
	def token_grid_shape(self) -> tuple[int, int, int]:
		'''Return the survey token grid, with the full vertical window.'''
		return (
			_ceil_div(self.lateral_shape_xy[0], self.patch_size_xyz[0]),
			_ceil_div(self.lateral_shape_xy[1], self.patch_size_xyz[1]),
			self.core_size_tokens[2],
		)

	@property
	def input_size_tokens(self) -> tuple[int, int, int]:
		'''Return the fixed halo-padded input token shape.'''
		return cast(
			'tuple[int, int, int]',
			tuple(
				self.core_size_tokens[axis]
				+ 2 * self.context_halo_tokens[axis]
				for axis in range(3)
			),
		)

	@property
	def input_size_voxels(self) -> tuple[int, int, int]:
		'''Return the fixed end-to-end raw crop shape.'''
		return cast(
			'tuple[int, int, int]',
			tuple(
				self.input_size_tokens[axis] * self.patch_size_xyz[axis]
				for axis in range(3)
			),
		)

	@property
	def core_size_voxels(self) -> tuple[int, int, int]:
		'''Return the supervised core shape.'''
		return cast(
			'tuple[int, int, int]',
			tuple(
				self.core_size_tokens[axis] * self.patch_size_xyz[axis]
				for axis in range(3)
			),
		)

	def validate(self) -> None:
		'''Require the fixed 005 tile contract.'''
		if len(self.lateral_shape_xy) != 2 or any(
			not isinstance(axis, int) or isinstance(axis, bool) or axis <= 0
			for axis in self.lateral_shape_xy
		):
			raise ValueError('lateral_shape_xy must be a positive integer pair')
		if self.patch_size_xyz != HORIZON_PATCH_SIZE:
			raise ValueError(f'patch_size_xyz must be {HORIZON_PATCH_SIZE!r}')
		if self.core_size_tokens != HORIZON_CORE_SIZE_TOKENS:
			raise ValueError(
				f'core_size_tokens must be {HORIZON_CORE_SIZE_TOKENS!r}'
			)
		if self.context_halo_tokens != HORIZON_CONTEXT_HALO_TOKENS:
			raise ValueError(
				'context_halo_tokens must be '
				f'{HORIZON_CONTEXT_HALO_TOKENS!r}'
			)
		if (self.window_start, self.window_stop) != (
			HORIZON_WINDOW_START,
			HORIZON_WINDOW_STOP,
		):
			raise ValueError(
				'Volve horizon sample window must be exactly [552, 768)'
			)
		if self.input_size_tokens != HORIZON_INPUT_SIZE_TOKENS:
			raise ValueError('horizon input token geometry is inconsistent')
		if self.input_size_voxels != HORIZON_INPUT_SIZE_VOXELS:
			raise ValueError('horizon input voxel geometry is inconsistent')
		if (
			not isinstance(self.min_token_valid_fraction, Real)
			or isinstance(self.min_token_valid_fraction, bool)
			or not 0.0 <= float(self.min_token_valid_fraction) <= 1.0
		):
			raise ValueError('min_token_valid_fraction must be in [0, 1]')


@dataclass(frozen=True)
class HorizonTileRecord:
	'''One non-empty 64 by 64 lateral core in deterministic grid order.'''

	tile_id: int
	core_start_token_xy: tuple[int, int]
	core_stop_token_xy: tuple[int, int]
	per_horizon_observation_counts: tuple[int, ...]

	@property
	def supervised_observation_count(self) -> int:
		'''Return the total native supervision count in this tile.'''
		return sum(self.per_horizon_observation_counts)


@dataclass(frozen=True)
class HorizonTileTargets:
	'''Fixed-core fractional targets and masks for one lateral tile.'''

	sample_float: np.ndarray
	native_valid_mask: np.ndarray
	split_mask: np.ndarray
	trace_valid_mask: np.ndarray
	supervision_mask: np.ndarray
	input_start_token: tuple[int, int, int]
	token_source_start: tuple[int, int, int]
	token_source_stop: tuple[int, int, int]
	token_destination_start: tuple[int, int, int]
	token_destination_stop: tuple[int, int, int]


@dataclass(frozen=True)
class FrozenHorizonTile:
	'''Halo-padded frozen embeddings in channel-first decoder format.'''

	embeddings: np.ndarray
	token_valid_mask: np.ndarray


@dataclass(frozen=True)
class RawHorizonTile:
	'''Halo-padded preprocessed amplitude and explicit validity for end-to-end use.'''

	amplitude: np.ndarray
	local_valid_mask: np.ndarray
	token_valid_mask: np.ndarray


def horizon_supervision_mask(  # noqa: PLR0913
	*,
	sample_float: np.ndarray,
	native_valid_mask: np.ndarray,
	split_mask: np.ndarray,
	trace_valid_mask: np.ndarray,
	window_start: int = HORIZON_WINDOW_START,
	window_stop: int = HORIZON_WINDOW_STOP,
) -> np.ndarray:
	'''Combine native, split, trace, finite, and sample-window validity.'''
	samples = np.asarray(sample_float)
	native = np.asarray(native_valid_mask)
	split = np.asarray(split_mask)
	traces = np.asarray(trace_valid_mask)
	_validate_horizon_arrays(samples, native, split, traces)
	return (
		native
		& split
		& traces[np.newaxis, :, :]
		& np.isfinite(samples)
		& (samples >= window_start)
		& (samples < window_stop)
	)


def enumerate_horizon_tile_records(
	*,
	sample_float: np.ndarray,
	native_valid_mask: np.ndarray,
	split_mask: np.ndarray,
	trace_valid_mask: np.ndarray,
	settings: HorizonTileSettings,
) -> tuple[HorizonTileRecord, ...]:
	'''Enumerate non-empty lateral cores without vertical tiling.'''
	settings.validate()
	_validate_source_shapes(
		sample_float, native_valid_mask, split_mask, trace_valid_mask, settings
	)
	mask = horizon_supervision_mask(
		sample_float=sample_float,
		native_valid_mask=native_valid_mask,
		split_mask=split_mask,
		trace_valid_mask=trace_valid_mask,
		window_start=settings.window_start,
		window_stop=settings.window_stop,
	)
	records: list[HorizonTileRecord] = []
	tile_id = 0
	core_x, core_y = settings.core_size_tokens[:2]
	grid_x, grid_y = settings.token_grid_shape[:2]
	patch_x, patch_y = settings.patch_size_xyz[:2]
	for token_x in range(0, grid_x, core_x):
		for token_y in range(0, grid_y, core_y):
			stop_x = min(token_x + core_x, grid_x)
			stop_y = min(token_y + core_y, grid_y)
			voxel_slice = (
				slice(
					token_x * patch_x,
					min(stop_x * patch_x, settings.lateral_shape_xy[0]),
				),
				slice(
					token_y * patch_y,
					min(stop_y * patch_y, settings.lateral_shape_xy[1]),
				),
			)
			counts = tuple(
				int(np.count_nonzero(mask[horizon][voxel_slice]))
				for horizon in range(len(HORIZON_NAMES))
			)
			if sum(counts):
				records.append(
					HorizonTileRecord(
						tile_id=tile_id,
						core_start_token_xy=(token_x, token_y),
						core_stop_token_xy=(stop_x, stop_y),
						per_horizon_observation_counts=counts,
					)
				)
			tile_id += 1
	return tuple(records)


def build_horizon_tile_targets(  # noqa: PLR0913
	*,
	record: HorizonTileRecord,
	sample_float: np.ndarray,
	native_valid_mask: np.ndarray,
	split_mask: np.ndarray,
	trace_valid_mask: np.ndarray,
	settings: HorizonTileSettings,
) -> HorizonTileTargets:
	'''Build fixed-size central-core targets for one record.'''
	settings.validate()
	_validate_source_shapes(
		sample_float, native_valid_mask, split_mask, trace_valid_mask, settings
	)
	core_x, core_y = settings.core_size_voxels[:2]
	start_x = record.core_start_token_xy[0] * settings.patch_size_xyz[0]
	start_y = record.core_start_token_xy[1] * settings.patch_size_xyz[1]
	stop_x = min(start_x + core_x, settings.lateral_shape_xy[0])
	stop_y = min(start_y + core_y, settings.lateral_shape_xy[1])
	source = (slice(start_x, stop_x), slice(start_y, stop_y))
	destination = (slice(0, stop_x - start_x), slice(0, stop_y - start_y))

	targets = np.full((len(HORIZON_NAMES), core_x, core_y), np.nan, dtype=np.float32)
	native = np.zeros(targets.shape, dtype=np.bool_)
	split = np.zeros(targets.shape, dtype=np.bool_)
	traces = np.zeros((core_x, core_y), dtype=np.bool_)
	targets[(slice(None), *destination)] = np.asarray(sample_float)[
		(slice(None), *source)
	]
	native[(slice(None), *destination)] = np.asarray(native_valid_mask)[
		(slice(None), *source)
	]
	split[(slice(None), *destination)] = np.asarray(split_mask)[
		(slice(None), *source)
	]
	traces[destination] = np.asarray(trace_valid_mask)[source]
	supervision = horizon_supervision_mask(
		sample_float=targets,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
		window_start=settings.window_start,
		window_stop=settings.window_stop,
	)
	counts = tuple(
		int(np.count_nonzero(supervision[index]))
		for index in range(len(HORIZON_NAMES))
	)
	if counts != record.per_horizon_observation_counts:
		raise RuntimeError('runtime horizon masks no longer match tile enumeration')
	window_input_start = (
		record.core_start_token_xy[0] - settings.context_halo_tokens[0],
		record.core_start_token_xy[1] - settings.context_halo_tokens[1],
		0,
	)
	source_start, source_stop, destination_start, destination_stop = (
		_context_token_bounds(window_input_start, settings)
	)
	input_start = (
		window_input_start[0],
		window_input_start[1],
		settings.window_start // settings.patch_size_xyz[2],
	)
	return HorizonTileTargets(
		sample_float=targets,
		native_valid_mask=native,
		split_mask=split,
		trace_valid_mask=traces,
		supervision_mask=supervision,
		input_start_token=input_start,
		token_source_start=source_start,
		token_source_stop=source_stop,
		token_destination_start=destination_start,
		token_destination_stop=destination_stop,
	)


def build_frozen_horizon_tile(
	*,
	record: HorizonTileRecord,
	embeddings: np.ndarray,
	valid_tokens: np.ndarray,
	settings: HorizonTileSettings,
) -> FrozenHorizonTile:
	'''Crop and edge-pad frozen `[TX,TY,27,D]` embedding inputs.'''
	settings.validate()
	values = np.asarray(embeddings)
	valid = np.asarray(valid_tokens)
	if values.ndim != 4 or tuple(values.shape[:3]) != settings.token_grid_shape:
		raise ValueError('embeddings must have shape [TX,TY,27,D]')
	if not np.issubdtype(values.dtype, np.floating):
		raise TypeError('embeddings must have a floating dtype')
	if valid.shape != values.shape[:3] or valid.dtype != np.bool_:
		raise ValueError('valid_tokens must be a bool mask matching the token grid')
	input_start = (
		record.core_start_token_xy[0] - settings.context_halo_tokens[0],
		record.core_start_token_xy[1] - settings.context_halo_tokens[1],
		0,
	)
	source_start, source_stop, destination_start, destination_stop = (
		_context_token_bounds(input_start, settings)
	)
	source = _slices(source_start, source_stop)
	destination = _slices(destination_start, destination_stop)
	crop = np.zeros((*settings.input_size_tokens, values.shape[-1]), dtype=np.float32)
	mask = np.zeros(settings.input_size_tokens, dtype=np.bool_)
	source_values = np.asarray(values[source], dtype=np.float32)
	source_valid = valid[source]
	if not np.isfinite(source_values[source_valid]).all():
		raise ValueError('valid frozen embeddings must be finite')
	crop[destination] = np.where(
		source_valid[..., np.newaxis], source_values, 0.0
	)
	mask[destination] = source_valid
	return FrozenHorizonTile(
		embeddings=np.ascontiguousarray(np.moveaxis(crop, -1, 0)),
		token_valid_mask=mask,
	)


def build_raw_horizon_tile(
	*,
	record: HorizonTileRecord,
	preprocessed_amplitude: np.ndarray,
	trace_valid_mask: np.ndarray,
	settings: HorizonTileSettings,
) -> RawHorizonTile:
	'''Crop a preprocessed survey window, padding edges and missing traces with zero.'''
	settings.validate()
	values = np.asarray(preprocessed_amplitude)
	traces = np.asarray(trace_valid_mask)
	if values.ndim != 3 or tuple(values.shape[:2]) != settings.lateral_shape_xy:
		raise ValueError('preprocessed_amplitude must match the survey lateral shape')
	if values.shape[2] < settings.window_stop:
		raise ValueError('preprocessed_amplitude does not contain the horizon window')
	if not np.issubdtype(values.dtype, np.floating):
		raise TypeError('preprocessed_amplitude must have a floating dtype')
	if traces.shape != settings.lateral_shape_xy or traces.dtype != np.bool_:
		raise ValueError('trace_valid_mask must be a bool lateral survey mask')
	start_x = (
		record.core_start_token_xy[0] - settings.context_halo_tokens[0]
	) * settings.patch_size_xyz[0]
	start_y = (
		record.core_start_token_xy[1] - settings.context_halo_tokens[1]
	) * settings.patch_size_xyz[1]
	input_x, input_y, input_z = settings.input_size_voxels
	stop_x = min(start_x + input_x, settings.lateral_shape_xy[0])
	stop_y = min(start_y + input_y, settings.lateral_shape_xy[1])
	source_x = slice(max(0, start_x), stop_x)
	source_y = slice(max(0, start_y), stop_y)
	destination_x = slice(max(0, -start_x), max(0, -start_x) + stop_x - max(0, start_x))
	destination_y = slice(max(0, -start_y), max(0, -start_y) + stop_y - max(0, start_y))
	window = np.asarray(
		values[source_x, source_y, settings.window_start : settings.window_stop],
		dtype=np.float32,
	)
	valid_lateral = traces[source_x, source_y]
	if not np.isfinite(window[valid_lateral]).all():
		raise ValueError('valid preprocessed amplitude values must be finite')
	amplitude = np.zeros((input_x, input_y, input_z), dtype=np.float32)
	local_valid = np.zeros((input_x, input_y, input_z), dtype=np.bool_)
	destination = (destination_x, destination_y, slice(None))
	amplitude[destination] = np.where(valid_lateral[..., np.newaxis], window, 0.0)
	local_valid[destination] = valid_lateral[..., np.newaxis]
	token_valid = reduce_valid_mask_to_tokens(
		local_valid,
		patch_size_xyz=settings.patch_size_xyz,
		min_valid_fraction=settings.min_token_valid_fraction,
	)
	return RawHorizonTile(
		amplitude=amplitude[np.newaxis, ...],
		local_valid_mask=local_valid,
		token_valid_mask=token_valid,
	)


def _validate_horizon_arrays(
	samples: np.ndarray,
	native: np.ndarray,
	split: np.ndarray,
	traces: np.ndarray,
) -> None:
	if samples.ndim != 3 or samples.shape[0] != len(HORIZON_NAMES):
		raise ValueError('sample_float must have shape [5,X,Y]')
	if native.shape != samples.shape or split.shape != samples.shape:
		raise ValueError('native and split masks must match sample_float')
	if native.dtype != np.bool_ or split.dtype != np.bool_:
		raise TypeError('native and split masks must have dtype bool')
	if traces.shape != samples.shape[1:] or traces.dtype != np.bool_:
		raise TypeError('trace_valid_mask must be a matching bool lateral mask')


def _validate_source_shapes(
	sample_float: np.ndarray,
	native_valid_mask: np.ndarray,
	split_mask: np.ndarray,
	trace_valid_mask: np.ndarray,
	settings: HorizonTileSettings,
) -> None:
	_validate_horizon_arrays(
		np.asarray(sample_float),
		np.asarray(native_valid_mask),
		np.asarray(split_mask),
		np.asarray(trace_valid_mask),
	)
	if tuple(np.asarray(sample_float).shape[1:]) != settings.lateral_shape_xy:
		raise ValueError('horizon arrays do not match settings.lateral_shape_xy')


def _context_token_bounds(
	input_start: tuple[int, int, int], settings: HorizonTileSettings
) -> tuple[
	tuple[int, int, int],
	tuple[int, int, int],
	tuple[int, int, int],
	tuple[int, int, int],
]:
	source_start = cast(
		'tuple[int, int, int]', tuple(max(0, value) for value in input_start)
	)
	source_stop = cast(
		'tuple[int, int, int]',
		tuple(
			min(
				input_start[axis] + settings.input_size_tokens[axis],
				settings.token_grid_shape[axis],
			)
			for axis in range(3)
		),
	)
	destination_start = cast(
		'tuple[int, int, int]',
		tuple(source_start[axis] - input_start[axis] for axis in range(3)),
	)
	destination_stop = cast(
		'tuple[int, int, int]',
		tuple(
			destination_start[axis] + source_stop[axis] - source_start[axis]
			for axis in range(3)
		),
	)
	return source_start, source_stop, destination_start, destination_stop


def _slices(
	start: tuple[int, int, int], stop: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
	return tuple(slice(start[axis], stop[axis]) for axis in range(3))  # type: ignore[return-value]


def _ceil_div(value: int, divisor: int) -> int:
	return (value + divisor - 1) // divisor


__all__ = [
	'HORIZON_MIN_TOKEN_VALID_FRACTION',
	'HORIZON_WINDOW_LENGTH',
	'HORIZON_WINDOW_START',
	'HORIZON_WINDOW_STOP',
	'FrozenHorizonTile',
	'HorizonTileRecord',
	'HorizonTileSettings',
	'HorizonTileTargets',
	'RawHorizonTile',
	'build_frozen_horizon_tile',
	'build_horizon_tile_targets',
	'build_raw_horizon_tile',
	'enumerate_horizon_tile_records',
	'horizon_supervision_mask',
]
