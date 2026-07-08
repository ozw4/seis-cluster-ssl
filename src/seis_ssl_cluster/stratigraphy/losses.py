"""Masked losses for stratigraphic prototype pretraining."""

from __future__ import annotations

import math

import torch
import torch.nn.functional


def ordered_soft_coordinate(probs: torch.Tensor) -> torch.Tensor:
	"""Map ordered prototype probabilities to a normalized coordinate in [0, 1]."""
	_validate_probs_prefix(probs, name='probs')
	num_prototypes = probs.shape[-1]
	if num_prototypes == 1:
		return probs.new_zeros(probs.shape[:-1])
	coordinates = torch.linspace(
		0.0,
		1.0,
		steps=num_prototypes,
		device=probs.device,
		dtype=probs.dtype,
	)
	return (probs * coordinates).sum(dim=-1)


def structured_hmm_prototype_loss(  # noqa: PLR0913
	logits: torch.Tensor,
	labels: torch.Tensor,
	*,
	valid_mask: torch.Tensor,
	confidence: torch.Tensor | None = None,
	ignore_index: int = -1,
	eps: float = 1.0e-8,
) -> torch.Tensor:
	"""Return weighted CE against valid HMM pseudo-label tokens."""
	_validate_eps(eps)
	_validate_logits(logits)
	prefix_shape = logits.shape[:-1]
	num_prototypes = logits.shape[-1]
	_validate_labels(labels, prefix_shape)
	_validate_valid_mask(valid_mask, prefix_shape, logits.device)
	_validate_same_device(logits, labels, valid_mask)

	selection = valid_mask & labels.ne(ignore_index)
	flat_selection = selection.reshape(-1)
	if not bool(flat_selection.any().item()):
		msg = 'structured HMM prototype loss has no target tokens after masking'
		raise ValueError(msg)

	flat_labels = labels.reshape(-1)
	selected_labels = flat_labels[flat_selection]
	_validate_selected_labels(selected_labels, num_prototypes, ignore_index)

	if confidence is None:
		selected_weights = logits.new_ones(selected_labels.shape)
	else:
		_validate_confidence(confidence, prefix_shape, logits.device)
		selected_weights = confidence.reshape(-1)[flat_selection].to(dtype=logits.dtype)
		_validate_weights(selected_weights)

	positive_weight = selected_weights > 0
	total_weight = selected_weights[positive_weight].sum()
	if not bool(total_weight.detach().gt(eps).item()):
		msg = 'structured HMM prototype loss has no positive-weight target tokens'
		raise ValueError(msg)

	selected_logits = logits.reshape(-1, num_prototypes)[flat_selection]
	token_loss = torch.nn.functional.cross_entropy(
		selected_logits,
		selected_labels.long(),
		reduction='none',
	)
	return (token_loss * selected_weights).sum() / total_weight


def usage_entropy_floor_loss(
	probs: torch.Tensor,
	*,
	valid_mask: torch.Tensor,
	entropy_floor: float,
	eps: float = 1.0e-8,
) -> torch.Tensor:
	"""Penalize valid-token prototype usage entropy below ``entropy_floor``."""
	_validate_eps(eps)
	_validate_probability_tensor(probs)
	prefix_shape = probs.shape[:-1]
	_validate_valid_mask(valid_mask, prefix_shape, probs.device)
	_validate_entropy_floor(entropy_floor, probs.shape[-1])
	_validate_same_device(probs, valid_mask)

	flat_valid = valid_mask.reshape(-1)
	if not bool(flat_valid.any().item()):
		msg = 'usage entropy floor loss requires at least one valid token'
		raise ValueError(msg)
	q_bar = probs.reshape(-1, probs.shape[-1])[flat_valid].mean(dim=0)
	entropy = -(q_bar * (q_bar + eps).log()).sum()
	floor = probs.new_tensor(float(entropy_floor))
	return (floor - entropy).clamp_min(0.0).square()


def feature_distillation_loss(
	student_features: torch.Tensor,
	teacher_features: torch.Tensor,
	*,
	valid_mask: torch.Tensor,
	eps: float = 1.0e-8,
) -> torch.Tensor:
	"""Return masked mean cosine distance to stop-gradient teacher features."""
	_validate_eps(eps)
	_validate_matching_feature_tensors(student_features, teacher_features)
	prefix_shape = student_features.shape[:-1]
	_validate_valid_mask(valid_mask, prefix_shape, student_features.device)
	_validate_same_device(student_features, teacher_features, valid_mask)

	flat_valid = valid_mask.reshape(-1)
	if not bool(flat_valid.any().item()):
		msg = 'feature distillation loss requires at least one valid token'
		raise ValueError(msg)
	student_flat = student_features.reshape(-1, student_features.shape[-1])[flat_valid]
	teacher_flat = teacher_features.detach().reshape(
		-1,
		teacher_features.shape[-1],
	)[flat_valid]
	teacher_flat = teacher_flat.to(dtype=student_flat.dtype)
	cosine = torch.nn.functional.cosine_similarity(
		student_flat,
		teacher_flat,
		dim=-1,
		eps=eps,
	)
	return (1.0 - cosine).mean()


def _validate_eps(eps: float) -> None:
	if not isinstance(eps, (float, int)) or isinstance(eps, bool):
		msg = f'eps must be a float; got {eps!r}'
		raise TypeError(msg)
	if not math.isfinite(float(eps)) or float(eps) <= 0.0:
		msg = f'eps must be positive and finite; got {eps!r}'
		raise ValueError(msg)


def _validate_logits(logits: torch.Tensor) -> None:
	if not isinstance(logits, torch.Tensor):
		msg = f'logits must be a torch.Tensor; got {type(logits)!r}'
		raise TypeError(msg)
	if logits.ndim < 2:
		msg = (
			'logits must have at least two dimensions with prototypes last; '
			f'got shape={tuple(logits.shape)!r}'
		)
		raise ValueError(msg)
	if logits.shape[-1] <= 0:
		msg = 'logits must contain at least one prototype'
		raise ValueError(msg)
	if not torch.is_floating_point(logits):
		msg = f'logits must be floating point; got {logits.dtype}'
		raise TypeError(msg)


def _validate_labels(labels: torch.Tensor, prefix_shape: torch.Size) -> None:
	if not isinstance(labels, torch.Tensor):
		msg = f'labels must be a torch.Tensor; got {type(labels)!r}'
		raise TypeError(msg)
	if tuple(labels.shape) != tuple(prefix_shape):
		msg = (
			f'labels shape must match logits prefix {tuple(prefix_shape)!r}; '
			f'got {tuple(labels.shape)!r}'
		)
		raise ValueError(msg)
	if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
		msg = f'labels must have integer dtype; got {labels.dtype}'
		raise TypeError(msg)


def _validate_valid_mask(
	valid_mask: torch.Tensor,
	prefix_shape: torch.Size,
	device: torch.device,
) -> None:
	if not isinstance(valid_mask, torch.Tensor):
		msg = f'valid_mask must be a torch.Tensor; got {type(valid_mask)!r}'
		raise TypeError(msg)
	if tuple(valid_mask.shape) != tuple(prefix_shape):
		msg = (
			f'valid_mask shape must match tensor prefix {tuple(prefix_shape)!r}; '
			f'got {tuple(valid_mask.shape)!r}'
		)
		raise ValueError(msg)
	if valid_mask.dtype != torch.bool:
		msg = f'valid_mask must have dtype torch.bool; got {valid_mask.dtype}'
		raise TypeError(msg)
	if valid_mask.device != device:
		msg = (
			'valid_mask must be on the same device as the tensor; '
			f'got valid_mask_device={valid_mask.device}, tensor_device={device}'
		)
		raise ValueError(msg)


def _validate_selected_labels(
	labels: torch.Tensor,
	num_prototypes: int,
	ignore_index: int,
) -> None:
	if bool(labels.lt(0).any().item()) or bool(labels.ge(num_prototypes).any().item()):
		msg = (
			'selected labels must be in prototype range '
			f'[0, {num_prototypes}); got ignore_index={ignore_index!r}'
		)
		raise ValueError(msg)


def _validate_confidence(
	confidence: torch.Tensor,
	prefix_shape: torch.Size,
	device: torch.device,
) -> None:
	if not isinstance(confidence, torch.Tensor):
		msg = f'confidence must be a torch.Tensor; got {type(confidence)!r}'
		raise TypeError(msg)
	if tuple(confidence.shape) != tuple(prefix_shape):
		msg = (
			f'confidence shape must match logits prefix {tuple(prefix_shape)!r}; '
			f'got {tuple(confidence.shape)!r}'
		)
		raise ValueError(msg)
	if not torch.is_floating_point(confidence):
		msg = f'confidence must be floating point; got {confidence.dtype}'
		raise TypeError(msg)
	if confidence.device != device:
		msg = (
			'confidence must be on the same device as logits; '
			f'got confidence_device={confidence.device}, logits_device={device}'
		)
		raise ValueError(msg)


def _validate_weights(weights: torch.Tensor) -> None:
	if not bool(torch.isfinite(weights).all().item()):
		msg = 'confidence weights must be finite'
		raise ValueError(msg)
	if bool(weights.lt(0).any().item()):
		msg = 'confidence weights must be nonnegative'
		raise ValueError(msg)


def _validate_probability_tensor(probs: torch.Tensor) -> None:
	_validate_probs_prefix(probs, name='probs')
	if not bool(torch.isfinite(probs).all().item()):
		msg = 'probs must be finite'
		raise ValueError(msg)
	if bool(probs.lt(0).any().item()):
		msg = 'probs must be nonnegative'
		raise ValueError(msg)
	prob_sums = probs.sum(dim=-1)
	if not bool(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1.0e-4)):
		msg = 'probs must sum to 1 along the prototype dimension'
		raise ValueError(msg)


def _validate_probs_prefix(probs: torch.Tensor, *, name: str) -> None:
	if not isinstance(probs, torch.Tensor):
		msg = f'{name} must be a torch.Tensor; got {type(probs)!r}'
		raise TypeError(msg)
	if probs.ndim < 1:
		msg = f'{name} must have at least one prototype dimension'
		raise ValueError(msg)
	if probs.shape[-1] <= 0:
		msg = f'{name} must contain at least one prototype'
		raise ValueError(msg)
	if not torch.is_floating_point(probs):
		msg = f'{name} must be floating point; got {probs.dtype}'
		raise TypeError(msg)


def _validate_entropy_floor(entropy_floor: float, num_prototypes: int) -> None:
	if not isinstance(entropy_floor, (float, int)) or isinstance(entropy_floor, bool):
		msg = f'entropy_floor must be a float; got {entropy_floor!r}'
		raise TypeError(msg)
	entropy_floor = float(entropy_floor)
	max_entropy = math.log(num_prototypes)
	if (
		not math.isfinite(entropy_floor)
		or entropy_floor < 0.0
		or entropy_floor > max_entropy + 1.0e-8
	):
		msg = (
			'entropy_floor must be finite and in [0, log(num_prototypes)]; '
			f'got {entropy_floor!r}'
		)
		raise ValueError(msg)


def _validate_matching_feature_tensors(
	student_features: torch.Tensor,
	teacher_features: torch.Tensor,
) -> None:
	if not isinstance(student_features, torch.Tensor):
		msg = (
			'student_features must be a torch.Tensor; '
			f'got {type(student_features)!r}'
		)
		raise TypeError(msg)
	if not isinstance(teacher_features, torch.Tensor):
		msg = (
			'teacher_features must be a torch.Tensor; '
			f'got {type(teacher_features)!r}'
		)
		raise TypeError(msg)
	if tuple(student_features.shape) != tuple(teacher_features.shape):
		msg = (
			'student_features and teacher_features shapes must match; '
			f'got {tuple(student_features.shape)!r} and '
			f'{tuple(teacher_features.shape)!r}'
		)
		raise ValueError(msg)
	if student_features.ndim < 2:
		msg = (
			'features must have at least two dimensions with channels last; '
			f'got shape={tuple(student_features.shape)!r}'
		)
		raise ValueError(msg)
	if student_features.shape[-1] <= 0:
		msg = 'features must contain at least one channel'
		raise ValueError(msg)
	if not torch.is_floating_point(student_features):
		msg = f'student_features must be floating point; got {student_features.dtype}'
		raise TypeError(msg)
	if not torch.is_floating_point(teacher_features):
		msg = f'teacher_features must be floating point; got {teacher_features.dtype}'
		raise TypeError(msg)


def _validate_same_device(*tensors: torch.Tensor) -> None:
	devices = {tensor.device for tensor in tensors}
	if len(devices) != 1:
		device_names = sorted(map(str, devices))
		msg = f'all tensors must be on the same device; got {device_names!r}'
		raise ValueError(msg)


__all__ = [
	'feature_distillation_loss',
	'ordered_soft_coordinate',
	'structured_hmm_prototype_loss',
	'usage_entropy_floor_loss',
]
