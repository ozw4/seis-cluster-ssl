"""Example F3 lithology robustness suite contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3_ROBUSTNESS_CONTRACT_VERSION,
	F3LabelBudgetConfig,
	F3LabelBudgetModelConfig,
	F3RobustnessModelSpec,
	F3RobustnessSuiteManifest,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT

if TYPE_CHECKING:
	from pathlib import Path

F3_LITHOLOGY_ROBUSTNESS_ROOT = (
	DEFAULT_ARTIFACT_ROOT / 'lithology/f3/facies_benchmark_v1/robustness'
)
F3_LABEL_BUDGET_M1_SUITE_NAME = 'label_budget_m1_v1'
F3_SPLIT_INDEX_M1_SUITE_NAME = 'split_index_m1_v1'
F3_M1_BASELINE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
F3_M1_CANDIDATE_MODEL_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
F3_M1_EMBEDDING_SPEC = 'overlap_x16'
F3_M1_LABEL_SET = 'png_slices_segy_labels_v1'
F3_M1_PROBE_SPEC = 'linear_balanced_v1'


def f3_m1_example_model_specs() -> tuple[F3RobustnessModelSpec, ...]:
	"""Return the example MAE baseline and strat-HMM candidate specs."""
	return (
		F3RobustnessModelSpec(
			model_tag=F3_M1_BASELINE_MODEL_TAG,
			role='baseline',
			embedding_spec=F3_M1_EMBEDDING_SPEC,
			label_set=F3_M1_LABEL_SET,
			probe_spec=F3_M1_PROBE_SPEC,
		),
		F3RobustnessModelSpec(
			model_tag=F3_M1_CANDIDATE_MODEL_TAG,
			role='candidate',
			embedding_spec=F3_M1_EMBEDDING_SPEC,
			label_set=F3_M1_LABEL_SET,
			probe_spec=F3_M1_PROBE_SPEC,
		),
	)


def f3_m1_robustness_suite_manifest(
	suite_name: str,
	*,
	output_root: Path | None = None,
) -> F3RobustnessSuiteManifest:
	"""Build an example resolved suite manifest for the F3 M1 contract."""
	root = output_root or F3_LITHOLOGY_ROBUSTNESS_ROOT / suite_name
	return F3RobustnessSuiteManifest(
		suite_name=suite_name,
		contract_version=F3_ROBUSTNESS_CONTRACT_VERSION,
		output_root=root,
		models=f3_m1_example_model_specs(),
		report_paths={
			'paired_metrics_csv': root / 'reports/paired_metrics.csv',
			'paired_deltas_csv': root / 'reports/paired_deltas.csv',
			'summary_markdown': root / 'reports/summary.md',
			'suite_config_resolved_json': root / 'suite_config_resolved.json',
			'suite_manifest_json': root / 'suite_manifest.json',
		},
	)


def f3_lithology_label_budget_config_from_mapping(
	config: Mapping[str, object],
) -> F3LabelBudgetConfig:
	"""Validate and normalize the F3 lithology label-budget suite config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{'paths', 'suite', 'models', 'label_budget', 'validation', 'outputs'},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	suite = _required_mapping(config, 'suite')
	models = _required_mapping(config, 'models')
	_validate_allowed_keys(paths, frozenset({'artifact_root'}), prefix='paths')
	_validate_allowed_keys(suite, frozenset({'name', 'output_root'}), prefix='suite')
	_validate_allowed_keys(
		models,
		frozenset({'baseline', 'candidate'}),
		prefix='models',
	)
	label_budget = _required_mapping(config, 'label_budget')
	validation = _required_mapping(config, 'validation')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		validation,
		frozenset({'reuse_full_validation'}),
		prefix='validation',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')
	_validate_label_budget_mapping(label_budget)
	return F3LabelBudgetConfig(
		artifact_root=_required_absolute_path(
			paths,
			'artifact_root',
			prefix='paths',
		),
		suite_name=_required_str(suite, 'name', prefix='suite'),
		output_root=_required_absolute_path(suite, 'output_root', prefix='suite'),
		models=(
			_label_budget_model_from_mapping('baseline', models),
			_label_budget_model_from_mapping('candidate', models),
		),
		per_class_caps=_per_class_caps(label_budget.get('per_class_caps')),
		subsample_seeds=_int_tuple(
			label_budget.get('subsample_seeds'),
			'label_budget.subsample_seeds',
		),
		require_all_classes=_optional_bool(
			label_budget,
			'require_all_classes',
			default=True,
			prefix='label_budget',
		),
		reuse_full_validation=_optional_bool(
			validation,
			'reuse_full_validation',
			default=True,
			prefix='validation',
		),
		overwrite=_optional_bool(outputs, 'overwrite', default=False, prefix='outputs'),
	)


def _label_budget_model_from_mapping(
	role: str,
	models: Mapping[str, object],
) -> F3LabelBudgetModelConfig:
	model = _required_mapping(models, role)
	_validate_allowed_keys(
		model,
		frozenset({'model_tag', 'token_dataset_root'}),
		prefix=f'models.{role}',
	)
	return F3LabelBudgetModelConfig(
		role=role,
		model_tag=_required_str(model, 'model_tag', prefix=f'models.{role}'),
		token_dataset_root=_required_absolute_path(
			model,
			'token_dataset_root',
			prefix=f'models.{role}',
		),
	)


def _validate_label_budget_mapping(label_budget: Mapping[str, object]) -> None:
	_validate_allowed_keys(
		label_budget,
		frozenset({'mode', 'per_class_caps', 'subsample_seeds', 'require_all_classes'}),
		prefix='label_budget',
	)
	mode = _required_str(label_budget, 'mode', prefix='label_budget')
	if mode != 'per_class_cap':
		msg = f'label_budget.mode must be "per_class_cap"; got {mode!r}'
		raise ValueError(msg)


def _per_class_caps(value: object) -> tuple[int | None, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'label_budget.per_class_caps must be a list; got {value!r}'
		raise TypeError(msg)
	caps: list[int | None] = []
	for item in value:
		if item is None:
			caps.append(None)
			continue
		if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
			msg = (
				'label_budget.per_class_caps entries must be positive integers '
				f'or null; got {item!r}'
			)
			raise ValueError(msg)
		caps.append(item)
	return tuple(caps)


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a list of integers; got {value!r}'
		raise TypeError(msg)
	items = tuple(value)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integers; got {value!r}'
		raise TypeError(msg)
	return items


def _optional_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	default: bool,
	prefix: str,
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be boolean; got {value!r}'
		raise TypeError(msg)
	return value


__all__ = [
	'F3_LABEL_BUDGET_M1_SUITE_NAME',
	'F3_LITHOLOGY_ROBUSTNESS_ROOT',
	'F3_M1_BASELINE_MODEL_TAG',
	'F3_M1_CANDIDATE_MODEL_TAG',
	'F3_M1_EMBEDDING_SPEC',
	'F3_M1_LABEL_SET',
	'F3_M1_PROBE_SPEC',
	'F3_SPLIT_INDEX_M1_SUITE_NAME',
	'f3_lithology_label_budget_config_from_mapping',
	'f3_m1_example_model_specs',
	'f3_m1_robustness_suite_manifest',
]
