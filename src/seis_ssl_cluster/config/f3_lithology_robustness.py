"""Example F3 lithology robustness suite contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_hidden_dims,
	_optional_fraction,
	_optional_int,
	_optional_mapping,
	_optional_nonnegative_float,
	_optional_nullable_str,
	_optional_positive_float,
	_optional_positive_int,
	_optional_str,
	_required_absolute_path,
	_required_fraction,
	_required_mapping,
	_required_nonnegative_int,
	_required_str,
	_string_item,
	_validate_allowed_keys,
)
from seis_ssl_cluster.f3 import (
	DEFAULT_EVALUATION_METRICS,
	F3ClassInfo,
	F3LithologyProbeConfig,
	F3LithologyProbeInputs,
	F3LithologyProbeOutputs,
	F3LithologyProbeSettings,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	F3_LABEL_BUDGET_SUITE_MANIFEST_ARTIFACT_TYPE,
	F3_ROBUSTNESS_CONTRACT_VERSION,
	F3LabelBudgetConfig,
	F3LabelBudgetModelConfig,
	F3RobustnessModelSpec,
	F3RobustnessSuiteManifest,
	F3SplitInventoryConfig,
	F3SplitInventoryInputs,
	F3SplitSweepDatasetConfig,
	F3SplitSweepDatasetModelConfig,
)
from seis_ssl_cluster.f3.lithology.tokens import (
	F3LithologyTokenPolicy,
	read_f3_lithology_class_info,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT

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


@dataclass(frozen=True)
class F3LabelBudgetProbeRunConfig:
	"""Resolved config for running probes across a label-budget suite."""

	manifest: Path
	output_root: Path
	probe: F3LithologyProbeSettings
	labels_class_info: Path
	evaluation_metrics: tuple[str, ...]
	figure_dpi: int
	overwrite: bool
	rows: tuple[Mapping[str, object], ...]
	probe_configs: tuple[F3LithologyProbeConfig, ...]


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


def f3_lithology_label_budget_probe_config_from_mapping(
	config: Mapping[str, object],
) -> F3LabelBudgetProbeRunConfig:
	"""Validate and normalize a label-budget probe runner config."""
	_validate_allowed_keys(
		config,
		frozenset({'suite', 'probe', 'labels', 'evaluation', 'outputs'}),
		prefix='config',
	)
	suite = _required_mapping(config, 'suite')
	probe_mapping = _required_mapping(config, 'probe')
	labels = _required_mapping(config, 'labels')
	evaluation = _optional_mapping(config, 'evaluation')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		suite,
		frozenset({'manifest', 'output_root'}),
		prefix='suite',
	)
	_validate_allowed_keys(labels, frozenset({'class_info'}), prefix='labels')
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')

	manifest = _required_absolute_path(suite, 'manifest', prefix='suite')
	output_root = _required_absolute_path(suite, 'output_root', prefix='suite')
	class_info = _required_absolute_path(labels, 'class_info', prefix='labels')
	manifest_payload = _read_label_budget_suite_manifest(manifest)
	rows = _manifest_rows(manifest_payload, manifest_path=manifest)
	probe = _probe_settings_from_mapping(probe_mapping)
	classes = read_f3_lithology_class_info(class_info)
	evaluation_metrics = _evaluation_metrics(evaluation)
	figure_dpi = _figure_dpi(evaluation)
	probe_configs = tuple(
		_probe_config_for_manifest_row(
			row,
			output_root=output_root,
			probe=probe,
			class_info=class_info,
			classes=classes,
			evaluation_metrics=evaluation_metrics,
			figure_dpi=figure_dpi,
		)
		for row in rows
	)
	return F3LabelBudgetProbeRunConfig(
		manifest=manifest,
		output_root=output_root,
		probe=probe,
		labels_class_info=class_info,
		evaluation_metrics=evaluation_metrics,
		figure_dpi=figure_dpi,
		overwrite=_optional_bool(outputs, 'overwrite', default=False, prefix='outputs'),
		rows=rows,
		probe_configs=probe_configs,
	)


def f3_lithology_split_inventory_config_from_mapping(
	config: Mapping[str, object],
) -> F3SplitInventoryConfig:
	"""Validate and normalize the F3 lithology split-inventory suite config."""
	_validate_allowed_keys(
		config,
		frozenset({'paths', 'inputs', 'split_sweep', 'tokenization', 'outputs'}),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	inputs = _required_mapping(config, 'inputs')
	split_sweep = _required_mapping(config, 'split_sweep')
	tokenization = _required_mapping(config, 'tokenization')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(paths, frozenset({'artifact_root'}), prefix='paths')
	_validate_allowed_keys(
		inputs,
		frozenset(
			{
				'base_png_label_inventory',
				'source_label_volume',
				'segy_geometry_json',
				'class_info',
				'reference_embedding_metadata',
			},
		),
		prefix='inputs',
	)
	_validate_allowed_keys(
		split_sweep,
		frozenset(
			{
				'name',
				'output_root',
				'split_ids',
				'random_seeds',
				'validation_slice_count',
				'require_validation_all_classes',
				'min_validation_tokens_per_class',
				'include_base_split_as_split_000',
			},
		),
		prefix='split_sweep',
	)
	_validate_allowed_keys(
		tokenization,
		frozenset(
			{
				'min_labeled_fraction',
				'min_majority_fraction',
				'ignore_z_border_samples',
				'patch_size',
			},
		),
		prefix='tokenization',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')
	return F3SplitInventoryConfig(
		artifact_root=_required_absolute_path(
			paths,
			'artifact_root',
			prefix='paths',
		),
		inputs=F3SplitInventoryInputs(
			base_png_label_inventory=_required_absolute_path(
				inputs,
				'base_png_label_inventory',
				prefix='inputs',
			),
			source_label_volume=_required_absolute_path(
				inputs,
				'source_label_volume',
				prefix='inputs',
			),
			segy_geometry_json=_required_absolute_path(
				inputs,
				'segy_geometry_json',
				prefix='inputs',
			),
			class_info=_required_absolute_path(inputs, 'class_info', prefix='inputs'),
			reference_embedding_metadata=_required_absolute_path(
				inputs,
				'reference_embedding_metadata',
				prefix='inputs',
			),
		),
		suite_name=_required_str(split_sweep, 'name', prefix='split_sweep'),
		output_root=_required_absolute_path(
			split_sweep,
			'output_root',
			prefix='split_sweep',
		),
		split_ids=_str_tuple(split_sweep.get('split_ids'), 'split_sweep.split_ids'),
		random_seeds=_int_tuple(
			split_sweep.get('random_seeds'),
			'split_sweep.random_seeds',
		),
		validation_slice_count=_validate_int_value(
			split_sweep.get('validation_slice_count'),
			'split_sweep.validation_slice_count',
		),
		require_validation_all_classes=_optional_bool(
			split_sweep,
			'require_validation_all_classes',
			default=True,
			prefix='split_sweep',
		),
		min_validation_tokens_per_class=_min_validation_tokens_per_class(
			split_sweep.get('min_validation_tokens_per_class', {'default': 1}),
		),
		include_base_split_as_split_000=_optional_bool(
			split_sweep,
			'include_base_split_as_split_000',
			default=False,
			prefix='split_sweep',
		),
		tokenization_policy=F3LithologyTokenPolicy(
			min_labeled_fraction=_required_fraction(
				tokenization,
				'min_labeled_fraction',
				prefix='tokenization',
			),
			min_majority_fraction=_required_fraction(
				tokenization,
				'min_majority_fraction',
				prefix='tokenization',
			),
			ignore_z_border_samples=_required_nonnegative_int(
				tokenization,
				'ignore_z_border_samples',
				prefix='tokenization',
			),
		),
		patch_size_xyz=_positive_int_triplet(
			tokenization.get('patch_size'),
			'tokenization.patch_size',
		),
		overwrite=_optional_bool(outputs, 'overwrite', default=False, prefix='outputs'),
	)


def f3_lithology_split_sweep_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3SplitSweepDatasetConfig:
	"""Validate and normalize the split/index paired token dataset config."""
	_validate_allowed_keys(
		config,
		frozenset({'suite', 'models', 'common', 'outputs'}),
		prefix='config',
	)
	suite = _required_mapping(config, 'suite')
	models = _required_mapping(config, 'models')
	common = _required_mapping(config, 'common')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		suite,
		frozenset({'split_inventory_manifest', 'output_root'}),
		prefix='suite',
	)
	_validate_allowed_keys(
		models,
		frozenset({'baseline', 'candidate'}),
		prefix='models',
	)
	_validate_allowed_keys(
		common,
		frozenset(
			{
				'f3_root',
				'artifact_root',
				'dataset',
				'labels',
				'registry',
				'tokenization',
			},
		),
		prefix='common',
	)
	_validate_allowed_keys(outputs, frozenset({'overwrite'}), prefix='outputs')
	dataset = _required_mapping(common, 'dataset')
	labels = _required_mapping(common, 'labels')
	registry = _required_mapping(common, 'registry')
	tokenization = _required_mapping(common, 'tokenization')
	_validate_allowed_keys(
		dataset,
		frozenset({'name', 'version'}),
		prefix='common.dataset',
	)
	_validate_allowed_keys(
		labels,
		frozenset(
			{
				'source_label_segy',
				'source_label_volume',
				'class_info',
				'segy_geometry_json',
			},
		),
		prefix='common.labels',
	)
	_validate_allowed_keys(
		registry,
		frozenset({'seismic_volume', 'label_volume', 'metadata_json'}),
		prefix='common.registry',
	)
	_validate_allowed_keys(
		tokenization,
		frozenset(
			{
				'min_labeled_fraction',
				'min_majority_fraction',
				'ignore_z_border_samples',
			},
		),
		prefix='common.tokenization',
	)
	return F3SplitSweepDatasetConfig(
		split_inventory_manifest=_required_absolute_path(
			suite,
			'split_inventory_manifest',
			prefix='suite',
		),
		output_root=_required_absolute_path(suite, 'output_root', prefix='suite'),
		models=(
			_split_sweep_dataset_model_from_mapping('baseline', models),
			_split_sweep_dataset_model_from_mapping('candidate', models),
		),
		f3_root=_required_absolute_path(common, 'f3_root', prefix='common'),
		artifact_root=_required_absolute_path(
			common,
			'artifact_root',
			prefix='common',
		),
		dataset={
			'name': _required_str(dataset, 'name', prefix='common.dataset'),
			'version': _required_str(dataset, 'version', prefix='common.dataset'),
		},
		source_label_segy=_required_absolute_path(
			labels,
			'source_label_segy',
			prefix='common.labels',
		),
		source_label_volume=_required_absolute_path(
			labels,
			'source_label_volume',
			prefix='common.labels',
		),
		class_info=_required_absolute_path(
			labels,
			'class_info',
			prefix='common.labels',
		),
		segy_geometry_json=_required_absolute_path(
			labels,
			'segy_geometry_json',
			prefix='common.labels',
		),
		seismic_volume=_required_absolute_path(
			registry,
			'seismic_volume',
			prefix='common.registry',
		),
		label_volume=_required_absolute_path(
			registry,
			'label_volume',
			prefix='common.registry',
		),
		volume_metadata_json=_required_absolute_path(
			registry,
			'metadata_json',
			prefix='common.registry',
		),
		tokenization_policy=F3LithologyTokenPolicy(
			min_labeled_fraction=_required_fraction(
				tokenization,
				'min_labeled_fraction',
				prefix='common.tokenization',
			),
			min_majority_fraction=_required_fraction(
				tokenization,
				'min_majority_fraction',
				prefix='common.tokenization',
			),
			ignore_z_border_samples=_required_nonnegative_int(
				tokenization,
				'ignore_z_border_samples',
				prefix='common.tokenization',
			),
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


def _split_sweep_dataset_model_from_mapping(
	role: str,
	models: Mapping[str, object],
) -> F3SplitSweepDatasetModelConfig:
	model = _required_mapping(models, role)
	_validate_allowed_keys(
		model,
		frozenset({'model_tag', 'embeddings_dir', 'checkpoint'}),
		prefix=f'models.{role}',
	)
	return F3SplitSweepDatasetModelConfig(
		role=role,
		model_tag=_required_str(model, 'model_tag', prefix=f'models.{role}'),
		embeddings_dir=_required_absolute_path(
			model,
			'embeddings_dir',
			prefix=f'models.{role}',
		),
		checkpoint=_required_absolute_path(
			model,
			'checkpoint',
			prefix=f'models.{role}',
		),
	)


def _read_label_budget_suite_manifest(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		msg = f'suite.manifest does not exist: {path}'
		raise FileNotFoundError(msg)
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'suite manifest must contain a JSON object: {path}'
		raise TypeError(msg)
	artifact_type = payload.get('artifact_type')
	if artifact_type != F3_LABEL_BUDGET_SUITE_MANIFEST_ARTIFACT_TYPE:
		msg = (
			'suite manifest artifact_type must be '
			f'{F3_LABEL_BUDGET_SUITE_MANIFEST_ARTIFACT_TYPE!r}; '
			f'got {artifact_type!r}'
		)
		raise ValueError(msg)
	return payload


def _manifest_rows(
	payload: Mapping[str, object],
	*,
	manifest_path: Path,
) -> tuple[Mapping[str, object], ...]:
	value = payload.get('rows')
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'suite manifest rows must be a list: {manifest_path}'
		raise TypeError(msg)
	rows: list[Mapping[str, object]] = []
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			msg = f'suite manifest row {index} must be a mapping'
			raise TypeError(msg)
		_validate_manifest_row(item, index=index)
		rows.append(item)
	if not rows:
		msg = f'suite manifest contains no rows: {manifest_path}'
		raise ValueError(msg)
	return tuple(rows)


def _validate_manifest_row(row: Mapping[str, object], *, index: int) -> None:
	required = (
		'model_role',
		'model_tag',
		'budget_id',
		'per_class_cap',
		'subsample_seed',
		'token_dataset_root',
		'train_tokens',
		'validation_tokens',
		'metadata_json',
		'selected_train_token_count',
		'validation_token_count',
		'paired_identity_hash',
	)
	missing = [key for key in required if key not in row]
	if missing:
		msg = f'suite manifest row {index} missing key(s): {missing!r}'
		raise ValueError(msg)
	_required_str(row, 'model_role', prefix=f'rows[{index}]')
	_required_str(row, 'model_tag', prefix=f'rows[{index}]')
	_required_str(row, 'budget_id', prefix=f'rows[{index}]')
	_required_str(row, 'paired_identity_hash', prefix=f'rows[{index}]')
	_validate_int_value(row['subsample_seed'], f'rows[{index}].subsample_seed')
	_validate_optional_positive_int_value(
		row['per_class_cap'],
		f'rows[{index}].per_class_cap',
	)
	_validate_nonnegative_count(
		row['selected_train_token_count'],
		f'rows[{index}].selected_train_token_count',
	)
	_validate_nonnegative_count(
		row['validation_token_count'],
		f'rows[{index}].validation_token_count',
	)
	for key in (
		'token_dataset_root',
		'train_tokens',
		'validation_tokens',
		'metadata_json',
	):
		value = Path(_required_str(row, key, prefix=f'rows[{index}]'))
		if not value.is_absolute():
			msg = f'rows[{index}].{key} must be an absolute path; got {value}'
			raise ValueError(msg)


def _probe_config_for_manifest_row(  # noqa: PLR0913
	row: Mapping[str, object],
	*,
	output_root: Path,
	probe: F3LithologyProbeSettings,
	class_info: Path,
	classes: tuple[F3ClassInfo, ...],
	evaluation_metrics: tuple[str, ...],
	figure_dpi: int,
) -> F3LithologyProbeConfig:
	token_dataset_root = Path(str(row['token_dataset_root']))
	output_dir = (
		output_root
		/ 'probes'
		/ f'model={row["model_tag"]}'
		/ f'budget={row["budget_id"]}'
		/ f'subsample_seed={row["subsample_seed"]}'
		/ probe.spec
	)
	model_tag = str(row['model_tag'])
	return F3LithologyProbeConfig(
		inputs=F3LithologyProbeInputs(
			train_tokens=Path(str(row['train_tokens'])),
			validation_tokens=Path(str(row['validation_tokens'])),
			class_info=class_info,
			token_dataset_metadata_json=Path(str(row['metadata_json'])),
		),
		outputs=F3LithologyProbeOutputs(output_dir=output_dir),
		classes=classes,
		probe=probe,
		dataset={'name': 'f3_facies_benchmark'},
		model={
			'tag': model_tag,
			'role': str(row['model_role']),
			'freeze_encoder': True,
		},
		embeddings={
			'feature_source': {
				'kind': 'label_budget_token_dataset',
				'reference_model_tag': model_tag,
			},
		},
		labels={'class_info': str(class_info)},
		token_dataset={
			'input_dir': str(token_dataset_root),
			'metadata_json': str(row['metadata_json']),
			'label_budget': {
				'budget_id': row['budget_id'],
				'per_class_cap': row['per_class_cap'],
				'subsample_seed': row['subsample_seed'],
			},
			'paired_identity_hash': row['paired_identity_hash'],
		},
		lithology={'suite_output_root': str(output_root)},
		evaluation_metrics=evaluation_metrics,
		figure_dpi=figure_dpi,
	)


def _probe_settings_from_mapping(
	probe: Mapping[str, object],
) -> F3LithologyProbeSettings:
	_validate_allowed_keys(
		probe,
		frozenset(
			{
				'spec',
				'type',
				'feature_scaling',
				'class_weight',
				'max_iter',
				'random_state',
				'hidden_dims',
				'dropout',
				'max_epochs',
				'early_stopping_patience',
				'batch_size',
				'learning_rate',
				'weight_decay',
			},
		),
		prefix='probe',
	)
	return F3LithologyProbeSettings(
		spec=_required_str(probe, 'spec', prefix='probe'),
		probe_type=_required_str(probe, 'type', prefix='probe'),
		feature_scaling=_optional_str(
			probe,
			'feature_scaling',
			default='standard',
			prefix='probe',
		),
		class_weight=_optional_nullable_str(
			probe,
			'class_weight',
			default='balanced',
			prefix='probe',
		),
		max_iter=_optional_positive_int(probe.get('max_iter', 2000), 'probe.max_iter'),
		hidden_dims=_hidden_dims(probe.get('hidden_dims', (256, 128))),
		dropout=_optional_fraction(probe.get('dropout', 0.2), 'probe.dropout'),
		max_epochs=_optional_positive_int(
			probe.get('max_epochs', 200),
			'probe.max_epochs',
		),
		early_stopping_patience=_optional_positive_int(
			probe.get('early_stopping_patience', 20),
			'probe.early_stopping_patience',
		),
		batch_size=_optional_positive_int(
			probe.get('batch_size', 1024),
			'probe.batch_size',
		),
		learning_rate=_optional_positive_float(
			probe.get('learning_rate', 1.0e-3),
			'probe.learning_rate',
		),
		weight_decay=_optional_nonnegative_float(
			probe.get('weight_decay', 0.0),
			'probe.weight_decay',
		),
		random_state=_optional_int(
			probe.get('random_state', 42),
			'probe.random_state',
		),
	)


def _evaluation_metrics(evaluation: Mapping[str, object]) -> tuple[str, ...]:
	value = evaluation.get('metrics', DEFAULT_EVALUATION_METRICS)
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'evaluation.metrics must be a list of metric names; got {value!r}'
		raise TypeError(msg)
	metrics = tuple(_string_item(item, 'evaluation.metrics') for item in value)
	if not metrics:
		msg = 'evaluation.metrics must contain at least one metric name'
		raise ValueError(msg)
	return metrics


def _figure_dpi(evaluation: Mapping[str, object]) -> int:
	figure = evaluation.get('figure')
	if figure is None:
		return 300
	if not isinstance(figure, Mapping):
		msg = f'evaluation.figure must be a mapping; got {figure!r}'
		raise TypeError(msg)
	_validate_allowed_keys(figure, frozenset({'dpi'}), prefix='evaluation.figure')
	return _optional_positive_int(figure.get('dpi', 300), 'evaluation.figure.dpi')


def _validate_int_value(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{label} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_optional_positive_int_value(value: object, label: str) -> int | None:
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer or null; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_nonnegative_count(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{label} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


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


def _str_tuple(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a list of strings; got {value!r}'
		raise TypeError(msg)
	items = tuple(value)
	if not all(isinstance(item, str) and item for item in items):
		msg = f'{label} must contain only non-empty strings; got {value!r}'
		raise TypeError(msg)
	return items


def _positive_int_triplet(value: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 3
	):
		msg = f'{label} must contain three positive integers; got {value!r}'
		raise TypeError(msg)
	triplet = tuple(value)
	if not all(
		isinstance(item, int) and not isinstance(item, bool) and item > 0
		for item in triplet
	):
		msg = f'{label} must contain three positive integers; got {value!r}'
		raise ValueError(msg)
	return (int(triplet[0]), int(triplet[1]), int(triplet[2]))


def _min_validation_tokens_per_class(value: object) -> dict[str, int]:
	if not isinstance(value, Mapping):
		msg = (
			'split_sweep.min_validation_tokens_per_class must be a mapping; '
			f'got {value!r}'
		)
		raise TypeError(msg)
	result: dict[str, int] = {}
	for raw_key, raw_count in value.items():
		key = str(raw_key)
		if key != 'default':
			try:
				int(key)
			except ValueError as exc:
				msg = (
					'split_sweep.min_validation_tokens_per_class keys must be '
					f'class ids or "default"; got {raw_key!r}'
				)
				raise ValueError(msg) from exc
		result[key] = _validate_nonnegative_count(
			raw_count,
			f'split_sweep.min_validation_tokens_per_class[{key!r}]',
		)
	return result


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
	'F3LabelBudgetProbeRunConfig',
	'f3_lithology_label_budget_config_from_mapping',
	'f3_lithology_label_budget_probe_config_from_mapping',
	'f3_lithology_split_inventory_config_from_mapping',
	'f3_lithology_split_sweep_dataset_config_from_mapping',
	'f3_m1_example_model_specs',
	'f3_m1_robustness_suite_manifest',
]
