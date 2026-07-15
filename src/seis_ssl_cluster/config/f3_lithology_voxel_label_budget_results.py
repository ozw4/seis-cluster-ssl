"""Strict configuration for the F3 low-label voxel result summary."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_max_file_size_bytes,
	_publish_optional_bool,
	_required_absolute_path,
	_required_mapping,
	_validate_allowed_keys,
)
from seis_ssl_cluster.paths import DEFAULT_RESULTS_ROOT, ensure_under_root
from seis_ssl_cluster.results import DEFAULT_MAX_FILE_SIZE_BYTES

EXPECTED_BUDGET_COUNT = 3
EXPECTED_SEED_COUNT = 5


@dataclass(frozen=True)
class F3VoxelLabelBudgetDecisionThresholds:
	"""Preregistered descriptive decision thresholds."""

	minimum_positive_budgets: int
	minimum_primary_wins: int
	negative_budget_count: int
	monitored_class_ids: tuple[int, ...]
	major_degradation_delta: float
	systematic_degradation_budget_count: int

	def __post_init__(self) -> None:
		"""Reject thresholds outside the fixed three-budget/five-seed design."""
		_bounded_positive_int(
			self.minimum_positive_budgets,
			'decision.minimum_positive_budgets',
			maximum=EXPECTED_BUDGET_COUNT,
		)
		_bounded_positive_int(
			self.minimum_primary_wins,
			'decision.minimum_primary_wins',
			maximum=EXPECTED_SEED_COUNT,
		)
		_bounded_positive_int(
			self.negative_budget_count,
			'decision.negative_budget_count',
			maximum=EXPECTED_BUDGET_COUNT,
		)
		_bounded_positive_int(
			self.systematic_degradation_budget_count,
			'decision.systematic_degradation_budget_count',
			maximum=EXPECTED_BUDGET_COUNT,
		)
		if self.monitored_class_ids != (3, 5):
			raise ValueError('decision.monitored_class_ids must be exactly [3, 5]')
		if not math.isfinite(self.major_degradation_delta):
			raise ValueError('decision.major_degradation_delta must be finite')
		if self.major_degradation_delta >= 0.0:
			raise ValueError(
				'decision.major_degradation_delta must be strictly negative'
			)

	def to_dict(self) -> dict[str, object]:
		"""Return the exact preregistered contract for summary metadata."""
		return {
			'minimum_positive_budgets': self.minimum_positive_budgets,
			'minimum_primary_wins': self.minimum_primary_wins,
			'negative_budget_count': self.negative_budget_count,
			'monitored_class_ids': list(self.monitored_class_ids),
			'major_degradation_delta': self.major_degradation_delta,
			'systematic_degradation_budget_count': (
				self.systematic_degradation_budget_count
			),
		}


@dataclass(frozen=True)
class F3VoxelLabelBudgetResultsPublishConfig:
	"""Lightweight result publication settings."""

	enabled: bool = False
	results_root: Path = DEFAULT_RESULTS_ROOT
	output_dir: Path | None = None
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
	overwrite: bool = True

	def __post_init__(self) -> None:
		"""Keep publication below the configured results root."""
		if self.enabled and self.output_dir is None:
			raise ValueError(
				'publish.output_dir is required when publishing is enabled'
			)
		if self.output_dir is not None:
			ensure_under_root(
				self.output_dir,
				root=self.results_root,
				label='publish.output_dir',
			)
		if self.max_file_size_bytes <= 0:
			raise ValueError('publish.max_file_size_bytes must be positive')


@dataclass(frozen=True)
class F3VoxelLabelBudgetResultsConfig:
	"""Resolved inputs and outputs for the paired low-label summary."""

	artifact_root: Path
	suite_root: Path
	dataset_manifest: Path
	run_manifest: Path
	full_label_evaluations: Mapping[str, Path]
	decision: F3VoxelLabelBudgetDecisionThresholds
	overwrite: bool = False
	publish: F3VoxelLabelBudgetResultsPublishConfig = field(
		default_factory=F3VoxelLabelBudgetResultsPublishConfig
	)

	def __post_init__(self) -> None:
		"""Keep every scientific artifact input below the artifact root."""
		for label, path in (
			('suite.root', self.suite_root),
			('suite.dataset_manifest', self.dataset_manifest),
			('suite.run_manifest', self.run_manifest),
			*(
				(f'full_label_reference.{role}', path)
				for role, path in self.full_label_evaluations.items()
			),
		):
			ensure_under_root(path, root=self.artifact_root, label=label)
		if set(self.full_label_evaluations) != {'mae', 'm1', 'm2a'}:
			raise ValueError(
				'full_label_reference must define exactly mae, m1, and m2a'
			)

	@property
	def reports_dir(self) -> Path:
		"""Return the suite-owned report directory."""
		return self.suite_root / 'reports'


def f3_lithology_voxel_label_budget_results_config_from_mapping(
	config: Mapping[str, object],
) -> F3VoxelLabelBudgetResultsConfig:
	"""Validate and resolve the low-label voxel summary configuration."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'suite',
				'full_label_reference',
				'decision',
				'outputs',
				'publish',
			}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	suite = _required_mapping(config, 'suite')
	anchors = _required_mapping(config, 'full_label_reference')
	decision = _required_mapping(config, 'decision')
	outputs = _required_mapping(config, 'outputs')
	publish = _required_mapping(config, 'publish')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'results_root'}), prefix='paths'
	)
	_validate_allowed_keys(
		suite,
		frozenset({'root', 'dataset_manifest', 'run_manifest'}),
		prefix='suite',
	)
	_validate_allowed_keys(
		anchors, frozenset({'mae', 'm1', 'm2a'}), prefix='full_label_reference'
	)
	_validate_allowed_keys(
		decision,
		frozenset(
			{
				'minimum_positive_budgets',
				'minimum_primary_wins',
				'negative_budget_count',
				'monitored_class_ids',
				'major_degradation_delta',
				'systematic_degradation_budget_count',
			}
		),
		prefix='decision',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')
	_validate_allowed_keys(
		publish,
		frozenset({'enabled', 'output_dir', 'max_file_size_mb', 'overwrite'}),
		prefix='publish',
	)
	artifact_root = _required_absolute_path(
		paths, 'artifact_root', prefix='paths'
	)
	results_root = Path(_required_string(paths, 'results_root', prefix='paths'))
	publish_enabled = _publish_optional_bool(publish, 'enabled', default=False)
	publish_output = _optional_path(publish, 'output_dir')
	if publish_enabled and publish_output is None:
		raise ValueError('publish.output_dir is required when publishing is enabled')
	return F3VoxelLabelBudgetResultsConfig(
		artifact_root=artifact_root,
		suite_root=_required_absolute_path(suite, 'root', prefix='suite'),
		dataset_manifest=_required_absolute_path(
			suite, 'dataset_manifest', prefix='suite'
		),
		run_manifest=_required_absolute_path(
			suite, 'run_manifest', prefix='suite'
		),
		full_label_evaluations={
			role: _required_absolute_path(
				anchors, role, prefix='full_label_reference'
			)
			for role in ('mae', 'm1', 'm2a')
		},
		decision=F3VoxelLabelBudgetDecisionThresholds(
			minimum_positive_budgets=_required_int(
				decision, 'minimum_positive_budgets', prefix='decision'
			),
			minimum_primary_wins=_required_int(
				decision, 'minimum_primary_wins', prefix='decision'
			),
			negative_budget_count=_required_int(
				decision, 'negative_budget_count', prefix='decision'
			),
			monitored_class_ids=_integer_tuple(
				decision.get('monitored_class_ids'),
				'decision.monitored_class_ids',
			),
			major_degradation_delta=_required_float(
				decision, 'major_degradation_delta', prefix='decision'
			),
			systematic_degradation_budget_count=_required_int(
				decision,
				'systematic_degradation_budget_count',
				prefix='decision',
			),
		),
		overwrite=_required_bool(outputs, 'overwrite', prefix='outputs'),
		publish=F3VoxelLabelBudgetResultsPublishConfig(
			enabled=publish_enabled,
			results_root=results_root,
			output_dir=publish_output,
			max_file_size_bytes=_max_file_size_bytes(publish),
			overwrite=_required_bool(publish, 'overwrite', prefix='publish'),
		),
	)


def _required_string(parent: Mapping[str, object], key: str, *, prefix: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{prefix}.{key} must be a non-empty string')
	return value


def _optional_path(parent: Mapping[str, object], key: str) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		raise TypeError(f'publish.{key} must be a non-empty string or null')
	return Path(value)


def _required_int(parent: Mapping[str, object], key: str, *, prefix: str) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{prefix}.{key} must be an integer')
	return value


def _required_float(parent: Mapping[str, object], key: str, *, prefix: str) -> float:
	value = parent.get(key)
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{prefix}.{key} must be numeric')
	return float(value)


def _required_bool(parent: Mapping[str, object], key: str, *, prefix: str) -> bool:
	value = parent.get(key)
	if not isinstance(value, bool):
		raise TypeError(f'{prefix}.{key} must be boolean')
	return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError(f'{label} must be a sequence of integers')
	items = tuple(value)
	if not items or any(
		not isinstance(item, int) or isinstance(item, bool) for item in items
	):
		raise TypeError(f'{label} must contain integers')
	if len(set(items)) != len(items):
		raise ValueError(f'{label} must not contain duplicates')
	return items


def _bounded_positive_int(value: int, label: str, *, maximum: int) -> None:
	if (
		not isinstance(value, int)
		or isinstance(value, bool)
		or not 1 <= value <= maximum
	):
		raise ValueError(f'{label} must be in [1, {maximum}]')


__all__ = [
	'F3VoxelLabelBudgetDecisionThresholds',
	'F3VoxelLabelBudgetResultsConfig',
	'F3VoxelLabelBudgetResultsPublishConfig',
	'f3_lithology_voxel_label_budget_results_config_from_mapping',
]
