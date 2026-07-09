"""Contracts for F3 lithology robustness suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

F3_ROBUSTNESS_CONTRACT_VERSION = 'f3_lithology_robustness_m1_v1'
F3_M1_MODEL_ROLES = frozenset({'baseline', 'candidate'})
F3_M1_DISALLOWED_SUITE_NAMES = frozenset({'seed_sweep'})
F3_M1_DISALLOWED_CONFIG_KEYS = frozenset({'checkpoint_policy'})
F3_PAIRED_DELTA_FIELDS = (
	'delta_macro_f1',
	'delta_mean_iou',
	'delta_balanced_accuracy',
	'delta_accuracy',
)


@dataclass(frozen=True)
class F3RobustnessModelSpec:
	"""One encoder identity participating in a paired robustness comparison."""

	model_tag: str
	role: str
	embedding_spec: str
	label_set: str
	probe_spec: str

	def __post_init__(self) -> None:
		"""Validate model identity fields."""
		_validate_non_empty_str(self.model_tag, 'model_tag')
		_validate_non_empty_str(self.role, 'role')
		_validate_non_empty_str(self.embedding_spec, 'embedding_spec')
		_validate_non_empty_str(self.label_set, 'label_set')
		_validate_non_empty_str(self.probe_spec, 'probe_spec')
		if self.role not in F3_M1_MODEL_ROLES:
			msg = (
				f'role must be one of {sorted(F3_M1_MODEL_ROLES)!r}; '
				f'got {self.role!r}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class F3LabelBudgetSpec:
	"""Named small-label condition for a paired label-budget suite."""

	budget_id: str
	train_fraction: float | None = None
	max_train_tokens: int | None = None
	subsample_seeds: tuple[int, ...] = ()

	def __post_init__(self) -> None:
		"""Validate label-budget fields without generating a subsample."""
		_validate_non_empty_str(self.budget_id, 'budget_id')
		if self.train_fraction is None and self.max_train_tokens is None:
			msg = 'label budget requires train_fraction or max_train_tokens'
			raise ValueError(msg)
		if self.train_fraction is not None:
			_validate_fraction(self.train_fraction, 'train_fraction')
		if self.max_train_tokens is not None:
			_validate_positive_int(self.max_train_tokens, 'max_train_tokens')
		_validate_int_sequence(self.subsample_seeds, 'subsample_seeds')


@dataclass(frozen=True)
class F3RobustnessSuiteManifest:
	"""Resolved, versioned manifest for one F3 M1 robustness suite."""

	suite_name: str
	contract_version: str
	output_root: Path
	models: tuple[F3RobustnessModelSpec, ...]
	label_budgets: tuple[F3LabelBudgetSpec, ...] = ()
	split_ids: tuple[str, ...] = ()
	report_paths: Mapping[str, Path] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Validate suite-level invariants shared by B and C suites."""
		validate_f3_robustness_suite_name(self.suite_name)
		_validate_non_empty_str(self.contract_version, 'contract_version')
		validate_f3_robustness_output_root(self.output_root)
		validate_f3_m1_model_pair(self.models)
		_validate_str_sequence(self.split_ids, 'split_ids')
		_validate_report_paths(self.report_paths)


@dataclass(frozen=True)
class F3PairedMetricRow:
	"""One paired metric row comparing strat-HMM against MAE."""

	suite_name: str
	condition_id: str
	baseline_model_tag: str
	candidate_model_tag: str
	metric_name: str
	baseline_value: float
	candidate_value: float
	delta_value: float

	def __post_init__(self) -> None:
		"""Validate paired row identity fields."""
		validate_f3_robustness_suite_name(self.suite_name)
		_validate_non_empty_str(self.condition_id, 'condition_id')
		_validate_non_empty_str(self.baseline_model_tag, 'baseline_model_tag')
		_validate_non_empty_str(self.candidate_model_tag, 'candidate_model_tag')
		_validate_non_empty_str(self.metric_name, 'metric_name')


def validate_f3_robustness_suite_name(suite_name: str) -> None:
	"""Validate a robustness suite name."""
	_validate_non_empty_str(suite_name, 'suite_name')
	if suite_name in F3_M1_DISALLOWED_SUITE_NAMES:
		msg = (
			f'suite_name {suite_name!r} is out of scope for F3 M1 robustness; '
			'label-budget and split/index suites are the supported contracts'
		)
		raise ValueError(msg)


def validate_f3_robustness_output_root(output_root: Path) -> None:
	"""Validate that a robustness output root is absolute."""
	path = Path(output_root)
	if not path.is_absolute():
		msg = f'output_root must be an absolute path; got {path}'
		raise ValueError(msg)


def validate_f3_m1_model_pair(
	models: Sequence[F3RobustnessModelSpec],
) -> None:
	"""Validate the F3 M1 paired comparison model contract."""
	if isinstance(models, str | bytes):
		msg = f'models must be a sequence of model specs; got {models!r}'
		raise TypeError(msg)
	model_tuple = tuple(models)
	if len(model_tuple) != 2:
		msg = (
			'F3 M1 paired comparisons require exactly one baseline and one '
			f'candidate model; got {len(model_tuple)}'
		)
		raise ValueError(msg)
	roles = [model.role for model in model_tuple]
	if sorted(roles) != ['baseline', 'candidate']:
		msg = (
			'F3 M1 paired comparisons require model roles '
			f"['baseline', 'candidate']; got {roles!r}"
		)
		raise ValueError(msg)


def validate_f3_robustness_config_keys(
	config: Mapping[str, object],
	*,
	prefix: str = 'config',
) -> None:
	"""Reject suite keys that are explicitly out of scope for F3 M1 robustness."""
	if not isinstance(config, Mapping):
		msg = f'{prefix} must be a mapping; got {config!r}'
		raise TypeError(msg)
	for key, value in config.items():
		if key in F3_M1_DISALLOWED_CONFIG_KEYS:
			msg = (
				f'{prefix}.{key} is out of scope for F3 M1 robustness; '
				'checkpoint-selection policy is not part of this contract'
			)
			raise ValueError(msg)
		if key == 'suite_name' and isinstance(value, str):
			validate_f3_robustness_suite_name(value)
		if isinstance(value, Mapping):
			validate_f3_robustness_config_keys(value, prefix=f'{prefix}.{key}')


def _validate_non_empty_str(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in (0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 < fraction <= 1.0:
		msg = f'{label} must be in (0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _validate_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_int_sequence(values: object, label: str) -> tuple[int, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'{label} must be a sequence of integers; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integers; got {values!r}'
		raise TypeError(msg)
	return items


def _validate_str_sequence(values: object, label: str) -> tuple[str, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'{label} must be a sequence of strings; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	for item in items:
		_validate_non_empty_str(item, label)
	return items


def _validate_report_paths(paths: Mapping[str, Path]) -> None:
	if not isinstance(paths, Mapping):
		msg = f'report_paths must be a mapping; got {paths!r}'
		raise TypeError(msg)
	for key, value in paths.items():
		_validate_non_empty_str(key, 'report_paths key')
		if not isinstance(value, Path):
			msg = f'report_paths[{key!r}] must be a Path; got {value!r}'
			raise TypeError(msg)


__all__ = [
	'F3LabelBudgetSpec',
	'F3RobustnessModelSpec',
	'F3RobustnessSuiteManifest',
	'F3PairedMetricRow',
	'F3_M1_DISALLOWED_CONFIG_KEYS',
	'F3_M1_DISALLOWED_SUITE_NAMES',
	'F3_M1_MODEL_ROLES',
	'F3_PAIRED_DELTA_FIELDS',
	'F3_ROBUSTNESS_CONTRACT_VERSION',
	'validate_f3_m1_model_pair',
	'validate_f3_robustness_config_keys',
	'validate_f3_robustness_output_root',
	'validate_f3_robustness_suite_name',
]
