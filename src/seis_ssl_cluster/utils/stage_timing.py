"""Low-overhead opt-in stage timing and accumulation."""

from __future__ import annotations

import json
import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable
	from types import TracebackType


@dataclass
class _StageSamples:
	durations_seconds: list[float] = field(default_factory=list)
	sample_count: int = 0
	failure_count: int = 0


class StageTimingAccumulator:
	"""Accumulate completed stage durations and produce JSON-safe summaries."""

	def __init__(self) -> None:
		"""Initialize an empty accumulator."""
		self._stages: dict[str, _StageSamples] = {}

	def add(
		self,
		name: str,
		duration_seconds: float,
		*,
		sample_count: int = 1,
		failed: bool = False,
	) -> None:
		"""Add one completed stage observation."""
		_validate_stage_path(name)
		duration = float(duration_seconds)
		if not math.isfinite(duration) or duration < 0.0:
			msg = (
				'duration_seconds must be finite and nonnegative; '
				f'got {duration_seconds!r}'
			)
			raise ValueError(msg)
		_validate_sample_count(sample_count)
		samples = self._stages.setdefault(name, _StageSamples())
		samples.durations_seconds.append(duration)
		samples.sample_count += sample_count
		samples.failure_count += int(failed)

	def to_dict(self) -> dict[str, object]:
		"""Return a deterministic JSON-compatible timing summary."""
		stages: dict[str, object] = {}
		for name in sorted(self._stages):
			samples = self._stages[name]
			durations = samples.durations_seconds
			total = float(sum(durations))
			call_count = len(durations)
			stages[name] = {
				'call_count': call_count,
				'sample_count': samples.sample_count,
				'failure_count': samples.failure_count,
				'total_seconds': total,
				'mean_seconds': total / call_count if call_count else None,
				'min_seconds': min(durations) if durations else None,
				'max_seconds': max(durations) if durations else None,
				'seconds_per_sample': (
					total / samples.sample_count if samples.sample_count else None
				),
			}
		return {'stages': stages}

	def to_json(self, *, indent: int | None = None) -> str:
		"""Serialize the accumulated summary as strict JSON."""
		return json.dumps(
			self.to_dict(),
			indent=indent,
			sort_keys=True,
			allow_nan=False,
		)

	def write_json(self, path: str | Path) -> None:
		"""Write the accumulated summary as a UTF-8 JSON file."""
		target = Path(path)
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_text(self.to_json(indent=2) + '\n', encoding='utf-8')


class StageTimer:
	"""Time nested stages when enabled and otherwise act as a cheap no-op."""

	def __init__(
		self,
		*,
		enabled: bool = False,
		clock: Callable[[], float] = time.perf_counter,
		synchronize: Callable[[], None] | None = None,
		accumulator: StageTimingAccumulator | None = None,
	) -> None:
		"""Configure optional timing, synchronization, and accumulation."""
		if not isinstance(enabled, bool):
			msg = f'enabled must be a boolean; got {enabled!r}'
			raise TypeError(msg)
		self.enabled = enabled
		self._clock = clock
		self._synchronize = synchronize
		self.accumulator = accumulator or StageTimingAccumulator()
		self._stage_stack: list[str] = []

	def stage(
		self,
		name: str,
		*,
		sample_count: int = 1,
	) -> AbstractContextManager[None]:
		"""Return a context manager recording one stage invocation."""
		if not self.enabled:
			return nullcontext()
		_validate_stage_name(name)
		_validate_sample_count(sample_count)
		return _ActiveStage(self, name, sample_count)

	def to_dict(self) -> dict[str, object]:
		"""Return the JSON-safe accumulated stage summary."""
		return self.accumulator.to_dict()

	def to_json(self, *, indent: int | None = None) -> str:
		"""Serialize accumulated stage timings as strict JSON."""
		return self.accumulator.to_json(indent=indent)

	def write_json(self, path: str | Path) -> None:
		"""Write accumulated stage timings to a JSON file."""
		self.accumulator.write_json(path)


class _ActiveStage(AbstractContextManager[None]):
	def __init__(self, timer: StageTimer, name: str, sample_count: int) -> None:
		self._timer = timer
		self._name = name
		self._sample_count = sample_count
		self._path = ''
		self._started_at = 0.0

	def __enter__(self) -> None:
		stack = self._timer._stage_stack  # noqa: SLF001
		self._path = '/'.join((*stack, self._name))
		self._sync()
		self._started_at = float(self._timer._clock())  # noqa: SLF001
		stack.append(self._name)

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_value: BaseException | None,
		traceback: TracebackType | None,
	) -> None:
		try:
			self._sync()
			finished_at = float(self._timer._clock())  # noqa: SLF001
			self._timer.accumulator.add(
				self._path,
				finished_at - self._started_at,
				sample_count=self._sample_count,
				failed=exc_type is not None,
			)
		finally:
			self._timer._stage_stack.pop()  # noqa: SLF001

	def _sync(self) -> None:
		synchronize = self._timer._synchronize  # noqa: SLF001
		if synchronize is not None:
			synchronize()


def _validate_stage_name(name: str) -> None:
	if not isinstance(name, str) or not name or '/' in name:
		msg = f"stage name must be a non-empty string without '/'; got {name!r}"
		raise ValueError(msg)


def _validate_stage_path(name: str) -> None:
	if (
		not isinstance(name, str)
		or not name
		or any(not part for part in name.split('/'))
	):
		msg = f'stage path must contain non-empty names; got {name!r}'
		raise ValueError(msg)


def _validate_sample_count(sample_count: int) -> None:
	if isinstance(sample_count, bool) or not isinstance(sample_count, int):
		msg = f'sample_count must be an integer; got {sample_count!r}'
		raise TypeError(msg)
	if sample_count < 0:
		msg = f'sample_count must be nonnegative; got {sample_count!r}'
		raise ValueError(msg)


__all__ = ['StageTimer', 'StageTimingAccumulator']
