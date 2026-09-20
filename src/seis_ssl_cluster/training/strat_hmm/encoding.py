"""Reuse an identical, frozen encoder prefix within an unmasked training batch."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn
from torch.nn.modules import module as module_hooks
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from seis_ssl_cluster.models.common.transformer import (
	TransformerBlock,
	TransformerStack,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D

TokenEncoding: TypeAlias = dict[str, torch.Tensor | tuple[int, int, int] | None]

_SUPPORTED_MODULES = (
	nn.Linear,
	NonDynamicallyQuantizableLinear,
	nn.LayerNorm,
	nn.Dropout,
	nn.GELU,
	nn.MultiheadAttention,
	nn.ModuleList,
	nn.Sequential,
	TransformerBlock,
	TransformerStack,
)
_ENCODING_METHODS = (
	'forward',
	'_forward_unchecked',
	'encode_tokens',
	'prepare_encoder_tokens',
	'_project_patches',
	'_position_embedding',
)


class UnmaskedEncoderPair:
	"""Encode a student/teacher pair, sharing only a verified frozen prefix.

	Create one instance per epoch, after loading checkpoints and setting student
	trainability. Parameter equality is checked once; version/identity and module
	configuration guards invalidate sharing after ordinary in-place updates,
	parameter replacement, or mode changes. No model parameters or cached batch
	features are retained beyond the lifetime of this epoch helper.
	"""

	def __init__(
		self,
		student: AmplitudeMAE3D,
		teacher: AmplitudeMAE3D | None,
		*,
		use_teacher: bool,
	) -> None:
		"""Retain the pair without changing its state or checking tensor values."""
		self.student = student
		self.teacher = teacher
		self.use_teacher = use_teacher
		self.prefix_layers = 0
		self._checked = False
		self._signature: tuple[object, ...] | None = None

	def encode(
		self,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor,
	) -> tuple[TokenEncoding, TokenEncoding | None]:
		"""Return normal encodings, falling back whenever sharing is ineligible."""
		if self.use_teacher and self.teacher is not None and not x.requires_grad:
			if not self._checked:
				self._check_prefix()
			if self.prefix_layers:
				if self._signature == self._current_signature():
					return self._encode_shared(x, valid_mask)
				# A callback or another caller changed the supposedly frozen state.
				self.prefix_layers = 0

		with torch.set_grad_enabled(_has_trainable_parameters(self.student)):
			encoded = self.student.encode_tokens(x, valid_mask=valid_mask)
		teacher_encoded = None
		if self.use_teacher:
			if self.teacher is None:
				raise ValueError(
					'teacher is required when loss.distillation_weight is positive'
				)
			self.teacher.eval()
			with torch.no_grad():
				teacher_encoded = self.teacher.encode_tokens(x, valid_mask=valid_mask)
		return encoded, teacher_encoded

	def _check_prefix(self) -> None:  # noqa: C901, PLR0911
		self._checked = True
		student, teacher = self.student, self.teacher
		if type(student) is not AmplitudeMAE3D or type(teacher) is not AmplitudeMAE3D:
			return
		if any(
			type(model.encoder) is not TransformerStack for model in (student, teacher)
		):
			return
		if _geometry(student) != _geometry(teacher) or not _same_position_cache(
			student, teacher
		):
			return
		if _structure(student.encoder) is None or _structure(teacher.encoder) is None:
			return
		if any(
			_has_trainable_parameters(model.patch_projection)
			for model in (student, teacher)
		):
			return
		depth = 0
		for left, right in zip(
			student.encoder.layers, teacher.encoder.layers, strict=True
		):
			if _has_trainable_parameters(left) or _has_trainable_parameters(right):
				break
			if any(
				module.training for block in (left, right) for module in block.modules()
			):
				break
			depth += 1
		if depth == 0:
			return
		left_modules = _prefix_modules(student, depth)
		right_modules = _prefix_modules(teacher, depth)
		if any(
			_structure(left) != _structure(right)
			for left, right in zip(left_modules, right_modules, strict=True)
		):
			return
		left_params = _prefix_parameters(student, depth)
		right_params = _prefix_parameters(teacher, depth)
		if not _identical_parameters(left_params, right_params):
			return
		self.prefix_layers = depth
		self._signature = self._current_signature()
		if self._signature is None:
			self.prefix_layers = 0

	def _current_signature(self) -> tuple[object, ...] | None:
		if _has_global_hooks():
			return None
		result: list[object] = []
		for model in (self.student, self.teacher):
			if type(model) is not AmplitudeMAE3D or _has_custom_calls(model):
				return None
			structures = (_structure(model.patch_projection), _structure(model.encoder))
			if None in structures or model.patch_projection.training:
				return None
			result.extend((_geometry(model), structures))
			result.append(tuple(id(module) for module in model.encoder.modules()))
			result.extend(
				(
					id(parameter),
					parameter._version,  # noqa: SLF001
					parameter.data_ptr(),
					parameter.requires_grad,
					parameter.dtype,
					parameter.device,
					tuple(parameter.shape),
					parameter.stride(),
				)
				for parameter in _prefix_parameters(model, self.prefix_layers)
			)
			result.append(
				tuple(
					(key, id(value), value._version)  # noqa: SLF001
					for key, value in model._position_embedding_cache.items()  # noqa: SLF001
				)
			)
		return tuple(result)

	def _encode_shared(
		self,
		x: torch.Tensor,
		valid_mask: torch.Tensor,
	) -> tuple[TokenEncoding, TokenEncoding]:
		teacher = self.teacher
		if teacher is None:
			raise RuntimeError('a shared prefix requires a teacher')
		with torch.no_grad():
			tokens, grid, token_valid_mask = self.student.prepare_encoder_tokens(
				x,
				valid_mask=valid_mask,
			)
			padding_mask = None if token_valid_mask is None else ~token_valid_mask
			prefix = self.student.encoder(
				tokens, padding_mask, stop_layer=self.prefix_layers
			)
		with torch.set_grad_enabled(_has_trainable_parameters(self.student)):
			student_tokens = self.student.encoder(
				prefix, padding_mask, start_layer=self.prefix_layers
			)
		teacher.eval()
		with torch.no_grad():
			# Keep each model's input checks and position-cache lifecycle unchanged.
			# Patch projection is cheap relative to the repeated transformer prefix.
			_, teacher_grid, teacher_valid_mask = teacher.prepare_encoder_tokens(
				x,
				valid_mask=valid_mask,
			)
			teacher_tokens = teacher.encoder(
				prefix, padding_mask, start_layer=self.prefix_layers
			)
		metadata = {'token_grid_shape': grid, 'token_valid_mask': token_valid_mask}
		self._signature = self._current_signature()
		return {'tokens': student_tokens, **metadata}, {
			'tokens': teacher_tokens,
			'token_grid_shape': teacher_grid,
			'token_valid_mask': teacher_valid_mask,
		}


def _has_trainable_parameters(module: nn.Module) -> bool:
	return any(parameter.requires_grad for parameter in module.parameters())


def _geometry(model: AmplitudeMAE3D) -> tuple[object, ...]:
	return (
		model.in_channels,
		model.patch_size_xyz,
		model.patch_volume,
		model.encoder_dim,
		len(model.encoder.layers),
	)


def _prefix_modules(model: AmplitudeMAE3D, depth: int) -> tuple[nn.Module, ...]:
	return (model.patch_projection, *list(model.encoder.layers)[:depth])


def _prefix_parameters(model: AmplitudeMAE3D, depth: int) -> tuple[nn.Parameter, ...]:
	return tuple(
		parameter
		for module in _prefix_modules(model, depth)
		for parameter in module.parameters()
	)


def _has_custom_calls(module: nn.Module) -> bool:
	return bool(
		module._forward_hooks  # noqa: SLF001
		or module._forward_pre_hooks  # noqa: SLF001
		or module._backward_hooks  # noqa: SLF001
		or module._backward_pre_hooks  # noqa: SLF001
		or module._compiled_call_impl is not None  # noqa: SLF001
		or any(name in vars(module) for name in _ENCODING_METHODS)
	)


def _structure(root: nn.Module) -> tuple[object, ...] | None:
	result: list[object] = []
	for name, module in root.named_modules():
		if type(module) not in _SUPPORTED_MODULES or _has_custom_calls(module):
			return None
		attention_config = None
		if isinstance(module, nn.MultiheadAttention):
			attention_config = (
				module.num_heads,
				module.head_dim,
				module.batch_first,
				module.dropout,
				module.add_zero_attn,
				module.kdim,
				module.vdim,
			)
		result.append(
			(name, type(module), module.training, module.extra_repr(), attention_config)
		)
	return tuple(result)


def _identical_parameters(
	left: tuple[torch.Tensor, ...],
	right: tuple[torch.Tensor, ...],
) -> bool:
	if len(left) != len(right):
		return False
	checks: list[torch.Tensor] = []
	for first, second in zip(left, right, strict=True):
		if (
			first.shape != second.shape
			or first.dtype != second.dtype
			or first.device != second.device
			or not first.is_contiguous()
			or not second.is_contiguous()
		):
			return False
		# Compare the representation too, including the sign of zero.
		checks.append(
			(
				first.detach().view(torch.uint8) == second.detach().view(torch.uint8)
			).all()
		)
	if not checks or len({check.device for check in checks}) != 1:
		return False
	return bool(torch.stack(checks).all())


def _same_position_cache(student: AmplitudeMAE3D, teacher: AmplitudeMAE3D) -> bool:
	left = student._position_embedding_cache  # noqa: SLF001
	right = teacher._position_embedding_cache  # noqa: SLF001
	if left.keys() != right.keys():
		return False
	return not left or _identical_parameters(
		tuple(left.values()),
		tuple(right[key] for key in left),
	)


def _has_global_hooks() -> bool:
	return any(
		getattr(module_hooks, name)
		for name in (
			'_global_forward_hooks',
			'_global_forward_pre_hooks',
			'_global_backward_hooks',
			'_global_backward_pre_hooks',
		)
	)
