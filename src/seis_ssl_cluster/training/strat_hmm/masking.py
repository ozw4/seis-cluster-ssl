"""Deterministic XY token-column masking for center-trace pretraining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

COMMON_HARD_TARGET_HEAD_KS = (6, 8, 10)
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_COLUMN_SALT = 0xD1B54A32D192ED03
_REPLACEMENT_TOKEN_SALT = 0xA0761D6478BD642F

CommonHardTargetValidMaskInput: TypeAlias = (
	torch.Tensor | Mapping[int, torch.Tensor] | Sequence[torch.Tensor]
)


@dataclass(frozen=True)
class XYTokenColumnMaskPlan:
	"""Auditable full-Z XY-column mask decisions for one batch."""

	mask: torch.Tensor
	eligible_counts: torch.Tensor
	selected_counts: torch.Tensor
	selected_xy_coordinates: torch.Tensor

	@property
	def selected_coordinates(self) -> torch.Tensor:
		"""Return the selected ``(tx, ty)`` coordinates, padded with ``-1``."""
		return self.selected_xy_coordinates


def plan_xy_token_column_mask(  # noqa: PLR0913
	common_hard_target_valid_mask: CommonHardTargetValidMaskInput,
	student_token_valid_mask: torch.Tensor,
	*,
	column_fraction: float = 0.10,
	training_seed: int,
	epoch: int,
	global_step: int,
	batch_index: int,
	sample_indices: torch.Tensor | Sequence[int] | None = None,
) -> XYTokenColumnMaskPlan:
	"""Plan a deterministic full-Z mask over eligible XY token columns.

	The planner uses integer mixing and ordered scores rather than a global or
	device RNG.  A sequence of three hard-target masks is interpreted in the
	fixed ``K=(6, 8, 10)`` order; a mapping must use those keys exactly.
	"""
	common_mask = _resolve_common_hard_target_valid_mask(common_hard_target_valid_mask)
	_validate_fraction(column_fraction)
	for value, name in (
		(training_seed, 'training_seed'),
		(epoch, 'epoch'),
		(global_step, 'global_step'),
		(batch_index, 'batch_index'),
	):
		_validate_identity_int(value, name)

	batch_size = int(common_mask.shape[0])
	ty_size = int(common_mask.shape[2])
	if batch_size == 0:
		raise ValueError(
			'common hard-target valid mask must contain at least one sample'
		)
	student_mask = _normalize_student_token_valid_mask(
		student_token_valid_mask,
		common_mask,
	)
	sample_ids = _normalize_sample_indices(sample_indices, batch_size)
	eligible_columns = (common_mask & student_mask).any(dim=3)
	eligible_counts_cpu = (
		eligible_columns.reshape(batch_size, -1)
		.sum(dim=1, dtype=torch.int64)
		.detach()
		.cpu()
		.tolist()
	)
	if any(int(count) < 2 for count in eligible_counts_cpu):
		bad_sample = next(
			index for index, count in enumerate(eligible_counts_cpu) if int(count) < 2
		)
		raise ValueError(
			'each sample must have at least two eligible XY columns; '
			f'sample_index={bad_sample}, '
			f'eligible_count={int(eligible_counts_cpu[bad_sample])}',
		)

	selected_counts_cpu = [
		_selected_column_count(int(count)) for count in eligible_counts_cpu
	]
	selected_counts = torch.tensor(
		selected_counts_cpu,
		dtype=torch.int64,
		device=common_mask.device,
	)
	eligible_counts = torch.tensor(
		[int(count) for count in eligible_counts_cpu],
		dtype=torch.int64,
		device=common_mask.device,
	)
	mask = torch.zeros_like(common_mask)
	max_selected_count = max(selected_counts_cpu)
	selected_coordinates = torch.full(
		(batch_size, max_selected_count, 2),
		-1,
		dtype=torch.int64,
		device=common_mask.device,
	)

	for sample_position, selected_count in enumerate(selected_counts_cpu):
		eligible_coordinates = [
			(int(tx), int(ty))
			for tx, ty in torch.nonzero(
				eligible_columns[sample_position],
				as_tuple=False,
			)
			.detach()
			.cpu()
			.tolist()
		]
		identity = _mixed_identity(
			training_seed=training_seed,
			epoch=epoch,
			global_step=global_step,
			batch_index=batch_index,
			sample_index=sample_ids[sample_position],
		)
		ranked_coordinates = sorted(
			eligible_coordinates,
			key=lambda coordinate: (
				_selection_score(identity, coordinate[0] * ty_size + coordinate[1]),
				coordinate[0],
				coordinate[1],
			),
		)
		selected = ranked_coordinates[:selected_count]
		for selection_position, (tx, ty) in enumerate(selected):
			mask[sample_position, tx, ty, :] = True
			selected_coordinates[sample_position, selection_position, 0] = tx
			selected_coordinates[sample_position, selection_position, 1] = ty

	return XYTokenColumnMaskPlan(
		mask=mask,
		eligible_counts=eligible_counts,
		selected_counts=selected_counts,
		selected_xy_coordinates=selected_coordinates,
	)


def validate_common_hard_target_valid_masks(
	valid_masks: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
) -> torch.Tensor:
	"""Validate and return the shared K=6/8/10 hard-target valid mask."""
	if isinstance(valid_masks, Mapping):
		if set(valid_masks) != set(COMMON_HARD_TARGET_HEAD_KS):
			raise ValueError(
				'common hard-target valid masks must contain exactly K=6, K=8, and K=10'
			)
		masks = [valid_masks[k] for k in COMMON_HARD_TARGET_HEAD_KS]
	elif isinstance(valid_masks, Sequence) and not isinstance(
		valid_masks,
		(str, bytes),
	):
		if len(valid_masks) != len(COMMON_HARD_TARGET_HEAD_KS):
			raise ValueError(
				'common hard-target valid masks must contain exactly three masks '
				'for K=6/8/10'
			)
		masks = list(valid_masks)
	else:
		raise TypeError(
			'common hard-target valid masks must be a K mapping or a '
			'three-mask sequence'
		)

	first = _validate_hard_target_valid_mask(masks[0], name='K=6')
	for head_k, mask in zip(COMMON_HARD_TARGET_HEAD_KS[1:], masks[1:], strict=True):
		validated = _validate_hard_target_valid_mask(mask, name=f'K={head_k}')
		if validated.device != first.device:
			raise ValueError(
				'common hard-target valid masks must be on the same device'
			)
		if tuple(validated.shape) != tuple(first.shape):
			raise ValueError(
				'common hard-target valid masks must have identical shapes'
			)
		if not torch.equal(validated, first):
			raise ValueError('K=6/8/10 hard-target valid masks must be identical')
	return first


def center_trace_replacement_token_seed(training_seed: int) -> int:
	"""Return the fixed local-generator seed for a training seed."""
	_validate_identity_int(training_seed, 'training_seed')
	return _splitmix64(training_seed ^ _REPLACEMENT_TOKEN_SALT) & ((1 << 63) - 1)


def _resolve_common_hard_target_valid_mask(
	value: CommonHardTargetValidMaskInput,
) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return _validate_hard_target_valid_mask(value, name='common')
	return validate_common_hard_target_valid_masks(value)


def _validate_hard_target_valid_mask(
	mask: object,
	*,
	name: str,
) -> torch.Tensor:
	if not isinstance(mask, torch.Tensor):
		raise TypeError(
			f'{name} hard-target valid mask must be a tensor; got {type(mask).__name__}'
		)
	if mask.ndim != 4:
		raise ValueError(
			f'{name} hard-target valid mask must have shape [B, TX, TY, TZ]; '
			f'got shape={tuple(mask.shape)!r}'
		)
	if any(int(size) <= 0 for size in mask.shape):
		raise ValueError(
			f'{name} hard-target valid mask dimensions must be positive; '
			f'got shape={tuple(mask.shape)!r}'
		)
	if mask.dtype != torch.bool:
		raise TypeError(
			f'{name} hard-target valid mask must have dtype torch.bool; '
			f'got {mask.dtype}'
		)
	return mask


def _normalize_student_token_valid_mask(
	mask: torch.Tensor,
	common_mask: torch.Tensor,
) -> torch.Tensor:
	if not isinstance(mask, torch.Tensor):
		raise TypeError(
			f'student token-valid mask must be a tensor; got {type(mask).__name__}'
		)
	if mask.dtype != torch.bool:
		raise TypeError(
			f'student token-valid mask must have dtype torch.bool; got {mask.dtype}'
		)
	if mask.device != common_mask.device:
		raise ValueError(
			'student token-valid mask must be on the same device as the '
			'common hard-target valid mask; '
			f'got mask_device={mask.device}, common_device={common_mask.device}'
		)
	if mask.ndim == 4:
		if tuple(mask.shape) != tuple(common_mask.shape):
			raise ValueError(
				'student token-valid mask grid shape must match the common '
				f'hard-target mask; got shape={tuple(mask.shape)!r}, '
				f'expected={tuple(common_mask.shape)!r}'
			)
		return mask
	if mask.ndim == 2:
		expected_tokens = int(
			common_mask.shape[1] * common_mask.shape[2] * common_mask.shape[3]
		)
		if tuple(mask.shape) != (int(common_mask.shape[0]), expected_tokens):
			raise ValueError(
				'student token-valid mask must have shape [B, T] with '
				f'T={expected_tokens}; got shape={tuple(mask.shape)!r}'
			)
		return mask.reshape(common_mask.shape)
	raise ValueError(
		'student token-valid mask must have shape [B, TX, TY, TZ] or [B, T]; '
		f'got shape={tuple(mask.shape)!r}'
	)


def _normalize_sample_indices(
	value: torch.Tensor | Sequence[int] | None,
	batch_size: int,
) -> tuple[int, ...]:
	if value is None:
		return tuple(range(batch_size))
	if isinstance(value, torch.Tensor):
		if value.ndim != 1 or int(value.shape[0]) != batch_size:
			raise ValueError(
				'sample_indices must have shape [B]; '
				f'got shape={tuple(value.shape)!r}, expected=({batch_size},)'
			)
		if (
			value.dtype == torch.bool
			or value.dtype.is_floating_point
			or value.dtype.is_complex
		):
			raise TypeError(
				f'sample_indices must contain integer values; got dtype={value.dtype}'
			)
		values = value.detach().cpu().tolist()
	else:
		if isinstance(value, (str, bytes)) or len(value) != batch_size:
			raise ValueError(
				'sample_indices must contain one integer identity per sample; '
				f'got length='
				f'{len(value) if not isinstance(value, (str, bytes)) else None!r}, '
				f'expected={batch_size}'
			)
		values = list(value)
	result: list[int] = []
	for index, sample_index in enumerate(values):
		_validate_identity_int(sample_index, f'sample_indices[{index}]')
		result.append(sample_index)
	return tuple(result)


def _validate_fraction(value: float) -> None:
	if isinstance(value, bool) or not isinstance(value, (float, int)):
		raise TypeError(f'column_fraction must be a float equal to 0.10; got {value!r}')
	if value != 0.10:
		raise ValueError(f'column_fraction must equal 0.10; got {value!r}')


def _validate_identity_int(value: object, name: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{name} must be a nonnegative integer; got {value!r}')
	if value < 0:
		raise ValueError(f'{name} must be nonnegative; got {value!r}')


def _selected_column_count(eligible_count: int) -> int:
	"""Return ``min(N - 1, max(1, floor(0.10 * N + 0.5)))`` exactly."""
	rounded_fraction = (eligible_count + 5) // 10
	return min(eligible_count - 1, max(1, rounded_fraction))


def _mixed_identity(
	*,
	training_seed: int,
	epoch: int,
	global_step: int,
	batch_index: int,
	sample_index: int,
) -> int:
	value = 0x243F6A8885A308D3
	for field in (training_seed, epoch, global_step, batch_index, sample_index):
		value = _splitmix64(value ^ (field & _UINT64_MASK))
	return value


def _selection_score(identity: int, column_index: int) -> int:
	return _splitmix64(
		identity ^ _splitmix64((column_index + _COLUMN_SALT) & _UINT64_MASK)
	)


def _splitmix64(value: int) -> int:
	z = (value + _SPLITMIX_INCREMENT) & _UINT64_MASK
	z = ((z ^ (z >> 30)) * _SPLITMIX_MULTIPLIER_1) & _UINT64_MASK
	z = ((z ^ (z >> 27)) * _SPLITMIX_MULTIPLIER_2) & _UINT64_MASK
	return (z ^ (z >> 31)) & _UINT64_MASK


__all__ = [
	'COMMON_HARD_TARGET_HEAD_KS',
	'CommonHardTargetValidMaskInput',
	'XYTokenColumnMaskPlan',
	'center_trace_replacement_token_seed',
	'plan_xy_token_column_mask',
	'validate_common_hard_target_valid_masks',
]
