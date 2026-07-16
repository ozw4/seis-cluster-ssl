from __future__ import annotations

import json

import pytest

from seis_ssl_cluster.utils import StageTimer, StageTimingAccumulator


class _Clock:
	def __init__(self, values: list[float]) -> None:
		self.values = iter(values)
		self.call_count = 0

	def __call__(self) -> float:
		self.call_count += 1
		return next(self.values)


def test_disabled_timer_does_not_record_synchronize_or_read_clock() -> None:
	clock = _Clock([])
	sync_calls = 0

	def synchronize() -> None:
		nonlocal sync_calls
		sync_calls += 1

	timer = StageTimer(enabled=False, clock=clock, synchronize=synchronize)
	with timer.stage('ignored', sample_count=0):
		pass

	assert clock.call_count == 0
	assert sync_calls == 0
	assert timer.to_dict() == {'stages': {}}


def test_nested_stages_accumulate_full_paths_and_sample_counts() -> None:
	clock = _Clock([1.0, 2.0, 5.0, 8.0])
	timer = StageTimer(enabled=True, clock=clock)

	with timer.stage('outer', sample_count=2), timer.stage('inner', sample_count=3):
		pass

	assert timer.to_dict() == {
		'stages': {
			'outer': {
				'call_count': 1,
				'sample_count': 2,
				'failure_count': 0,
				'total_seconds': 7.0,
				'mean_seconds': 7.0,
				'min_seconds': 7.0,
				'max_seconds': 7.0,
				'seconds_per_sample': 3.5,
			},
			'outer/inner': {
				'call_count': 1,
				'sample_count': 3,
				'failure_count': 0,
				'total_seconds': 3.0,
				'mean_seconds': 3.0,
				'min_seconds': 3.0,
				'max_seconds': 3.0,
				'seconds_per_sample': 1.0,
			},
		},
	}


def test_exception_is_recorded_and_propagated() -> None:
	timer = StageTimer(enabled=True, clock=_Clock([10.0, 10.25]))

	with pytest.raises(RuntimeError, match='failure'), timer.stage('load'):
		raise RuntimeError('failure')

	assert timer.to_dict()['stages']['load']['failure_count'] == 1


def test_zero_samples_and_empty_accumulator_are_strict_json() -> None:
	accumulator = StageTimingAccumulator()
	assert json.loads(accumulator.to_json()) == {'stages': {}}

	accumulator.add('empty_batch', 0.5, sample_count=0)
	payload = json.loads(accumulator.to_json())
	assert payload['stages']['empty_batch']['sample_count'] == 0
	assert payload['stages']['empty_batch']['seconds_per_sample'] is None


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), float('-inf')])
def test_accumulator_rejects_nonfinite_durations(duration: float) -> None:
	accumulator = StageTimingAccumulator()

	with pytest.raises(ValueError, match='finite and nonnegative'):
		accumulator.add('invalid', duration)

	assert accumulator.to_dict() == {'stages': {}}


def test_enter_synchronize_failure_does_not_corrupt_nested_paths() -> None:
	synchronize_calls = 0

	def synchronize() -> None:
		nonlocal synchronize_calls
		synchronize_calls += 1
		if synchronize_calls == 1:
			raise RuntimeError('synchronize failed')

	timer = StageTimer(
		enabled=True,
		clock=_Clock([1.0, 2.0]),
		synchronize=synchronize,
	)

	with pytest.raises(RuntimeError, match='synchronize failed'), timer.stage('failed'):
		pass
	with timer.stage('next'):
		pass

	assert set(timer.to_dict()['stages']) == {'next'}


def test_enter_clock_failure_does_not_corrupt_nested_paths() -> None:
	values = iter([1.0, 2.0])
	clock_calls = 0

	def clock() -> float:
		nonlocal clock_calls
		clock_calls += 1
		if clock_calls == 1:
			raise RuntimeError('clock failed')
		return next(values)

	timer = StageTimer(enabled=True, clock=clock)

	with pytest.raises(RuntimeError, match='clock failed'), timer.stage('failed'):
		pass
	with timer.stage('next'):
		pass

	assert set(timer.to_dict()['stages']) == {'next'}
