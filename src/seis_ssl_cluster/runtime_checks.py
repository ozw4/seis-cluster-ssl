"""Runtime tensor-value check policy for performance-sensitive paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
	from collections.abc import Callable

RuntimeCheckMode = Literal['strict', 'once', 'minimal']
SUPPORTED_RUNTIME_CHECK_MODES = frozenset({'strict', 'once', 'minimal'})


@dataclass
class RuntimeChecks:
	"""Gate tensor-value checks while leaving structural checks unconditional."""

	mode: RuntimeCheckMode = 'once'
	_completed: set[str] = field(default_factory=set, init=False, repr=False)

	def __post_init__(self) -> None:
		"""Validate the configured mode."""
		if self.mode not in SUPPORTED_RUNTIME_CHECK_MODES:
			msg = (
				'runtime_check_mode must be one of '
				f'{sorted(SUPPORTED_RUNTIME_CHECK_MODES)!r}; got {self.mode!r}'
			)
			raise ValueError(msg)

	def check(
		self,
		key: str,
		condition: torch.Tensor | Callable[[], torch.Tensor],
		*,
		error: Exception,
	) -> None:
		"""Raise ``error`` when a value check selected by the policy fails."""
		if self.mode == 'minimal' or (
			self.mode == 'once' and key in self._completed
		):
			return
		if callable(condition):
			condition = condition()
		if condition.numel() != 1:
			msg = (
				'runtime check condition must be scalar; '
				f'got {tuple(condition.shape)!r}'
			)
			raise ValueError(msg)
		if not bool(condition.detach()):
			raise error
		if self.mode == 'once':
			self._completed.add(key)

	def assert_async(
		self,
		key: str,
		condition: Callable[[], torch.Tensor],
		*,
		message: str,
	) -> None:
		"""Schedule a selected assertion without synchronizing the host."""
		if self.mode == 'minimal' or (
			self.mode == 'once' and key in self._completed
		):
			return
		torch._assert_async(condition(), message)  # noqa: SLF001
		if self.mode == 'once':
			self._completed.add(key)


__all__ = [
	'SUPPORTED_RUNTIME_CHECK_MODES',
	'RuntimeCheckMode',
	'RuntimeChecks',
]
