"""Contracts and result summaries for the F3 strat-HMM M1 guardrails."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

DEFAULT_RESULTS_ROOT = Path('results')
F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME = 'strat_hmm_m1_guardrails_v1'
F3_STRAT_HMM_M1_BASELINE_MODEL_TAG = (
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
)
F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG = 'strat_hmm_m1_guardrail_distill_only'
F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG = 'strat_hmm_m1_guardrail_shuffled_hmm_seed42'
F3_STRAT_HMM_M1_GUARDRAIL_ROLES = (
	'baseline',
	'candidate',
	'distillation_only',
	'shuffled_hmm',
)
F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS = {
	'baseline': F3_STRAT_HMM_M1_BASELINE_MODEL_TAG,
	'candidate': F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG,
	'distillation_only': F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
	'shuffled_hmm': F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
}
F3_STRAT_HMM_M1_GUARDRAIL_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)
F3_STRAT_HMM_M1_PRIMARY_METRIC = 'macro_f1'
F3_STRAT_HMM_M1_LOW_BUDGET_IDS = ('cap25', 'cap100', 'cap500', 'full')
F3_STRAT_HMM_M1_CLASS_F1_METRICS = tuple(
	f'class_{class_id}_f1' for class_id in range(6)
)
F3_STRAT_HMM_M1_GUARDRAIL_JOB_STAGES = frozenset(
	{
		'extract_guardrail_embeddings',
		'build_guardrail_token_datasets',
		'run_guardrail_probes',
	},
)
F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_DIR = (
	DEFAULT_RESULTS_ROOT / 'f3/facies_benchmark_v1/strat_hmm_m1_guardrails'
)
F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_SUFFIXES = frozenset(
	{'.md', '.json', '.csv', '.png'}
)


@dataclass(frozen=True)
class F3ShuffledHMMTargetConfig:
	"""Resolved contract for a future deterministic target shuffle."""

	suite_name: str
	source_root: Path
	output_root: Path
	k: int
	seed: int
	shuffle_scope: str
	overwrite: bool


@dataclass(frozen=True)
class F3GuardrailJob:
	"""One downstream job with an isolated guardrail artifact root."""

	role: str
	model_tag: str
	input_path: Path
	output_path: Path


@dataclass(frozen=True)
class F3GuardrailJobsConfig:
	"""Resolved paired downstream-job contract for both guardrails."""

	suite_name: str
	stage: str
	jobs: tuple[F3GuardrailJob, ...]


@dataclass(frozen=True)
class F3GuardrailResultInput:
	"""Result artifacts for one baseline, candidate, or guardrail role."""

	role: str
	model_tag: str
	metrics_json: Path
	label_budget_summary_csv: Path | None = None
	label_budget_suite_manifest_json: Path | None = None
	split_index_summary_json: Path | None = None


@dataclass(frozen=True)
class F3GuardrailPublishConfig:
	"""Settings for publishing a lightweight guardrail summary."""

	enabled: bool = False
	output_dir: Path = F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_DIR
	max_file_size_bytes: int = _DEFAULT_PUBLISH_MAX_FILE_SIZE_BYTES


@dataclass(frozen=True)
class F3GuardrailSummaryConfig:
	"""Resolved guardrail result-summary contract."""

	suite_name: str
	strict: bool
	models: tuple[F3GuardrailResultInput, ...]
	output_dir: Path
	publish: F3GuardrailPublishConfig = field(default_factory=F3GuardrailPublishConfig)


@dataclass(frozen=True)
class F3GuardrailSummaryResult:
	"""Paths written by guardrail result consolidation."""

	comparison_table: Path
	summary_json: Path
	summary_markdown: Path
	pending_roles: tuple[str, ...]
	warnings: tuple[str, ...]
	published_files: tuple[Path, ...] = ()


def f3_shuffled_hmm_target_config_from_mapping(
	config: Mapping[str, object],
) -> F3ShuffledHMMTargetConfig:
	"""Validate the deterministic shuffled-HMM pseudo-target contract."""
	_validate_keys(config, {'suite', 'source', 'shuffle', 'outputs'}, 'config')
	suite = _mapping(config, 'suite')
	source = _mapping(config, 'source')
	shuffle = _mapping(config, 'shuffle')
	outputs = _mapping(config, 'outputs')
	_validate_keys(suite, {'name'}, 'suite')
	_validate_keys(source, {'pseudo_target_root', 'k'}, 'source')
	_validate_keys(
		shuffle,
		{
			'seed',
			'scope',
			'preserve_valid_token_mask',
			'preserve_global_label_histogram',
			'preserve_confidence_distribution',
			'preserve_artifact_schema',
		},
		'shuffle',
	)
	_validate_keys(outputs, {'pseudo_target_root', 'overwrite'}, 'outputs')
	suite_name = _stable_suite_name(suite)
	source_root = _absolute_path(source, 'pseudo_target_root', 'source')
	output_root = _absolute_path(outputs, 'pseudo_target_root', 'outputs')
	if source_root == output_root:
		raise ValueError('shuffled pseudo-target output must not overwrite its source')
	k = _positive_int(source, 'k', 'source')
	seed = _nonnegative_int(shuffle, 'seed', 'shuffle')
	scope = _string(shuffle, 'scope', 'shuffle')
	if scope != 'global_valid_tokens':
		raise ValueError(
			'shuffle.scope must be "global_valid_tokens" to preserve the global '
			'label histogram',
		)
	for key in (
		'preserve_valid_token_mask',
		'preserve_global_label_histogram',
		'preserve_confidence_distribution',
		'preserve_artifact_schema',
	):
		if shuffle.get(key) is not True:
			raise ValueError(f'shuffle.{key} must be true for the M1 contract')
	overwrite = outputs['overwrite']
	if not isinstance(overwrite, bool):
		raise TypeError(
			f'outputs.overwrite must be a boolean; got {overwrite!r}',
		)
	return F3ShuffledHMMTargetConfig(
		suite_name=suite_name,
		source_root=source_root,
		output_root=output_root,
		k=k,
		seed=seed,
		shuffle_scope=scope,
		overwrite=overwrite,
	)


def f3_guardrail_jobs_config_from_mapping(
	config: Mapping[str, object],
) -> F3GuardrailJobsConfig:
	"""Validate a two-guardrail downstream artifact-routing contract."""
	_validate_keys(config, {'suite', 'stage', 'jobs'}, 'config')
	suite_name = _stable_suite_name(_mapping(config, 'suite'))
	stage = config.get('stage')
	if not isinstance(stage, str) or stage not in F3_STRAT_HMM_M1_GUARDRAIL_JOB_STAGES:
		raise ValueError(
			f'stage must be one of {sorted(F3_STRAT_HMM_M1_GUARDRAIL_JOB_STAGES)!r}; '
			f'got {stage!r}',
		)
	rows = config.get('jobs')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'jobs must be a list; got {rows!r}')
	jobs: list[F3GuardrailJob] = []
	for index, raw in enumerate(rows):
		if not isinstance(raw, Mapping):
			raise TypeError(f'jobs[{index}] must be a mapping; got {raw!r}')
		_validate_keys(
			raw,
			{'role', 'model_tag', 'input_path', 'output_path'},
			f'jobs[{index}]',
		)
		role = _string(raw, 'role', f'jobs[{index}]')
		if role not in {'distillation_only', 'shuffled_hmm'}:
			raise ValueError(f'jobs[{index}].role is not an M1 guardrail: {role!r}')
		model_tag = _stable_model_tag(role, raw, prefix=f'jobs[{index}]')
		input_path = _absolute_path(raw, 'input_path', f'jobs[{index}]')
		output_path = _absolute_path(raw, 'output_path', f'jobs[{index}]')
		if F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG in output_path.parts:
			raise ValueError(
				f'jobs[{index}].output_path must not use the milestone-1 '
				'candidate root',
			)
		if model_tag not in output_path.parts:
			raise ValueError(
				f'jobs[{index}].output_path must contain stable model tag '
				f'{model_tag!r}',
			)
		jobs.append(F3GuardrailJob(role, model_tag, input_path, output_path))
	if tuple(job.role for job in jobs) != ('distillation_only', 'shuffled_hmm'):
		raise ValueError(
			'jobs must contain distillation_only then shuffled_hmm exactly once',
		)
	if len({job.output_path for job in jobs}) != len(jobs):
		raise ValueError('guardrail job output paths must be distinct')
	return F3GuardrailJobsConfig(suite_name=suite_name, stage=stage, jobs=tuple(jobs))


def f3_guardrail_summary_config_from_mapping(
	config: Mapping[str, object],
) -> F3GuardrailSummaryConfig:
	"""Validate and normalize an M1 guardrail summary config."""
	_validate_keys(config, {'suite', 'models', 'outputs', 'publish'}, 'config')
	suite = _mapping(config, 'suite')
	_validate_keys(suite, {'name', 'strict'}, 'suite')
	suite_name = _stable_suite_name(suite)
	strict = suite.get('strict', False)
	if not isinstance(strict, bool):
		raise TypeError(f'suite.strict must be a boolean; got {strict!r}')
	models = _mapping(config, 'models')
	_validate_keys(models, set(F3_STRAT_HMM_M1_GUARDRAIL_ROLES), 'models')
	resolved_models: list[F3GuardrailResultInput] = []
	for role in F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		raw = _mapping(models, role)
		_validate_keys(
			raw,
			{
				'model_tag',
				'metrics_json',
				'label_budget_summary_csv',
				'label_budget_suite_manifest_json',
				'split_index_summary_json',
			},
			f'models.{role}',
			allow_missing={
				'label_budget_summary_csv',
				'label_budget_suite_manifest_json',
				'split_index_summary_json',
			},
		)
		resolved_models.append(
			F3GuardrailResultInput(
				role=role,
				model_tag=_stable_model_tag(role, raw, prefix=f'models.{role}'),
				metrics_json=_absolute_path(raw, 'metrics_json', f'models.{role}'),
				label_budget_summary_csv=_optional_absolute_path(
					raw,
					'label_budget_summary_csv',
					f'models.{role}',
				),
				label_budget_suite_manifest_json=_optional_absolute_path(
					raw,
					'label_budget_suite_manifest_json',
					f'models.{role}',
				),
				split_index_summary_json=_optional_absolute_path(
					raw,
					'split_index_summary_json',
					f'models.{role}',
				),
			),
		)
	outputs = _mapping(config, 'outputs')
	_validate_keys(outputs, {'output_dir'}, 'outputs')
	output_dir = _absolute_path(outputs, 'output_dir', 'outputs')
	if F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG in output_dir.parts:
		raise ValueError(
			'guardrail summary must not use the milestone-1 candidate root',
		)
	publish = _publish_config(config.get('publish'))
	if publish.enabled and not strict:
		raise ValueError('guardrail publishing requires suite.strict: true')
	return F3GuardrailSummaryConfig(
		suite_name=suite_name,
		strict=strict,
		models=tuple(resolved_models),
		output_dir=output_dir,
		publish=publish,
	)


def summarize_f3_strat_hmm_m1_guardrails(
	config: F3GuardrailSummaryConfig,
) -> F3GuardrailSummaryResult:
	"""Write a deterministic summary, reporting absent artifacts as pending."""
	rows: list[dict[str, object]] = []
	pending: list[str] = []
	warnings: list[str] = []
	for model in config.models:
		if not model.metrics_json.is_file():
			if config.strict:
				raise FileNotFoundError(
					f'missing guardrail metrics for {model.role}: {model.metrics_json}',
				)
			pending.append(model.role)
			warnings.append(
				f'{model.role}: full-budget metrics are pending: {model.metrics_json}',
			)
			rows.append(
				{
					'role': model.role,
					'model_tag': model.model_tag,
					'status': 'pending',
					'metrics': None,
					'robustness': _robustness_payload(
						model,
						strict=False,
						warnings=warnings,
					),
				},
			)
			continue
		metrics = _read_metrics(
			model.metrics_json,
			strict=config.strict,
			warnings=warnings,
		)
		status = (
			'complete'
			if all(
				metrics[metric] is not None
				for metric in F3_STRAT_HMM_M1_GUARDRAIL_METRICS
			)
			else 'incomplete'
		)
		rows.append(
			{
				'role': model.role,
				'model_tag': model.model_tag,
				'status': status,
				'metrics': metrics,
				'robustness': _robustness_payload(
					model,
					strict=config.strict,
					warnings=warnings,
				),
			},
		)
	decision = _decision_payload(rows)
	low_budget = _low_budget_payload(rows, strict=config.strict, warnings=warnings)
	_validate_guardrail_publish_readiness(config, rows, low_budget)
	payload = {
		'schema_version': 2,
		'suite_name': config.suite_name,
		'strict': config.strict,
		'models': rows,
		'pending_roles': pending,
		'low_budget': low_budget,
		'decision': decision,
		'warnings': warnings,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	table_path = config.output_dir / 'guardrail_comparison_table.csv'
	json_path = config.output_dir / 'guardrail_comparison_summary.json'
	markdown_path = config.output_dir / 'guardrail_comparison_report.md'
	_write_comparison_csv(table_path, rows)
	json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	markdown_path.write_text(_render_markdown(payload), encoding='utf-8')
	published_files = publish_f3_strat_hmm_m1_guardrails(
		comparison_table=table_path,
		summary_json=json_path,
		summary_markdown=markdown_path,
		publish_config=config.publish,
	)
	return F3GuardrailSummaryResult(
		table_path,
		json_path,
		markdown_path,
		tuple(pending),
		tuple(warnings),
		published_files,
	)


def publish_f3_strat_hmm_m1_guardrails(
	*,
	comparison_table: Path,
	summary_json: Path,
	summary_markdown: Path,
	publish_config: F3GuardrailPublishConfig,
) -> tuple[Path, ...]:
	"""Publish only lightweight guardrail summary formats into ``results/``."""
	if not publish_config.enabled:
		return ()
	_validate_guardrail_publish_payload(summary_json)
	entries = tuple(
		(source, publish_config.output_dir / source.name)
		for source in (summary_markdown, summary_json, comparison_table)
	)
	_preflight_guardrail_publish_entries(
		entries,
		max_file_size_bytes=publish_config.max_file_size_bytes,
	)
	for source, target in entries:
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, target)
	return tuple(target for _, target in entries)


def _preflight_guardrail_publish_entries(
	entries: Sequence[tuple[Path, Path]], *, max_file_size_bytes: int
) -> None:
	if (
		isinstance(max_file_size_bytes, bool)
		or not isinstance(max_file_size_bytes, int)
		or max_file_size_bytes <= 0
	):
		raise ValueError('max_file_size_bytes must be a positive integer')
	targets: set[Path] = set()
	for source, target_path in entries:
		if source.is_symlink() or not source.is_file():
			raise FileNotFoundError(
				f'required guardrail publish source must be a regular file: {source}'
			)
		target = target_path.resolve(strict=False)
		if source.resolve(strict=False) == target:
			raise ValueError(f'publish target must differ from source: {target_path}')
		if target in targets:
			raise ValueError(f'duplicate publish target: {target_path}')
		targets.add(target)
		if target_path.is_symlink():
			raise ValueError(f'publish target must not be a symlink: {target_path}')
		if target_path.exists() and not target_path.is_file():
			raise IsADirectoryError(f'publish target is not a file: {target_path}')
		if source.stat().st_size > max_file_size_bytes:
			raise ValueError(f'guardrail publish source exceeds size limit: {source}')


def _validate_guardrail_publish_readiness(
	config: F3GuardrailSummaryConfig,
	rows: Sequence[Mapping[str, object]],
	low_budget: Mapping[str, object],
) -> None:
	if not config.publish.enabled:
		return
	if not config.strict:
		raise ValueError('guardrail publishing requires suite.strict: true')
	incomplete_roles = [
		str(row.get('role')) for row in rows if row.get('status') != 'complete'
	]
	if incomplete_roles:
		raise ValueError(
			'guardrail publishing requires all roles to be complete; '
			f'incomplete roles: {incomplete_roles!r}'
		)
	if low_budget.get('status') != 'complete':
		raise ValueError('guardrail publishing requires complete robustness evidence')


def _validate_guardrail_publish_payload(path: Path) -> None:
	payload = _read_json_object(path)
	if payload.get('strict') is not True:
		raise ValueError('guardrail publishing requires a strict summary payload')
	pending = payload.get('pending_roles')
	if not isinstance(pending, Sequence) or isinstance(pending, str | bytes):
		raise TypeError('guardrail summary pending_roles must be a list')
	if pending:
		raise ValueError(
			f'guardrail publishing requires no pending roles; got {list(pending)!r}'
		)
	models = payload.get('models')
	if not isinstance(models, Sequence) or isinstance(models, str | bytes):
		raise TypeError('guardrail summary models must be a list')
	model_rows = [row for row in models if isinstance(row, Mapping)]
	roles = tuple(row.get('role') for row in model_rows)
	if len(model_rows) != len(models) or roles != F3_STRAT_HMM_M1_GUARDRAIL_ROLES:
		raise ValueError(
			'guardrail publishing requires the exact registered role inventory; '
			f'got {roles!r}'
		)
	incomplete = [
		str(row['role']) for row in model_rows if row.get('status') != 'complete'
	]
	if incomplete:
		raise ValueError(
			'guardrail publishing requires all roles to be complete; '
			f'incomplete roles: {incomplete!r}'
		)
	low_budget = payload.get('low_budget')
	if not isinstance(low_budget, Mapping) or low_budget.get('status') != 'complete':
		raise ValueError('guardrail publishing requires complete robustness evidence')


def _publish_config(value: object) -> F3GuardrailPublishConfig:
	if value is None:
		return F3GuardrailPublishConfig()
	if not isinstance(value, Mapping):
		raise TypeError(f'publish must be a mapping; got {value!r}')
	_publish_keys = {'enabled', 'output_dir', 'max_file_size_mb'}
	_validate_keys(value, _publish_keys, 'publish', allow_missing=_publish_keys)
	enabled = value.get('enabled', False)
	if not isinstance(enabled, bool):
		raise TypeError(f'publish.enabled must be a boolean; got {enabled!r}')
	output_raw = value.get('output_dir', F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_DIR)
	if not isinstance(output_raw, str | Path) or not str(output_raw):
		raise TypeError('publish.output_dir must be a non-empty path string')
	output_dir = Path(output_raw)
	max_size = value.get('max_file_size_mb', 10)
	if (
		isinstance(max_size, bool)
		or not isinstance(max_size, int | float)
		or max_size <= 0
	):
		raise ValueError(
			f'publish.max_file_size_mb must be positive; got {max_size!r}'
		)
	return F3GuardrailPublishConfig(
		enabled=enabled,
		output_dir=output_dir,
		max_file_size_bytes=int(max_size * 1024 * 1024),
	)


def _read_metrics(  # noqa: C901
	path: Path,
	*,
	strict: bool,
	warnings: list[str],
) -> dict[str, float | None]:
	payload = _read_json_object(path)
	metrics: dict[str, float | None] = {}
	for metric in F3_STRAT_HMM_M1_GUARDRAIL_METRICS:
		value = payload.get(metric)
		if value is None:
			message = f'{path}: missing required full-budget metric {metric}'
			if strict:
				raise ValueError(message)
			warnings.append(message)
			metrics[metric] = None
			continue
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError(f'{path}: {metric} must be numeric; got {value!r}')
		value = float(value)
		if not math.isfinite(value):
			raise ValueError(f'{path}: {metric} must be finite; got {value!r}')
		metrics[metric] = value
	per_class = payload.get('per_class_f1')
	if per_class is not None and not isinstance(per_class, Mapping):
		raise TypeError(f'{path}: per_class_f1 must be a mapping; got {per_class!r}')
	for class_id, metric in enumerate(F3_STRAT_HMM_M1_CLASS_F1_METRICS):
		value = None if per_class is None else per_class.get(str(class_id))
		if value is None:
			continue
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError(
				f'{path}: per_class_f1[{class_id!r}] must be numeric; got {value!r}',
			)
		value = float(value)
		if not math.isfinite(value):
			raise ValueError(
				f'{path}: per_class_f1[{class_id!r}] must be finite; got {value!r}',
			)
		metrics[metric] = value
	return metrics


def _robustness_payload(
	model: F3GuardrailResultInput,
	*,
	strict: bool,
	warnings: list[str],
) -> dict[str, object]:
	return {
		'label_budget': _optional_label_budget_csv_artifact(
			model.label_budget_summary_csv,
			strict=strict,
			label=f'{model.role} label-budget summary',
			warnings=warnings,
		),
		'label_budget_suite_manifest': _optional_json_artifact(
			model.label_budget_suite_manifest_json,
			strict=strict,
			label=f'{model.role} label-budget suite manifest',
			warnings=warnings,
		),
		'split_index': _optional_json_artifact(
			model.split_index_summary_json,
			strict=strict,
			label=f'{model.role} split/index summary',
			warnings=warnings,
		),
	}


def _optional_label_budget_csv_artifact(
	path: Path | None,
	*,
	strict: bool,
	label: str,
	warnings: list[str],
) -> dict[str, object]:
	if path is None:
		return {'status': 'not_configured', 'summary': None}
	if not path.is_file():
		if strict:
			raise FileNotFoundError(f'missing {label}: {path}')
		warnings.append(f'{label} is pending: {path}')
		return {'status': 'pending', 'summary': None}
	return {'status': 'complete', 'summary': _read_label_budget_csv(path)}


def _read_label_budget_csv(path: Path) -> dict[str, object]:
	with path.open(newline='', encoding='utf-8') as handle:
		reader = csv.DictReader(handle)
		fieldnames = set(reader.fieldnames or ())
		required = {'budget_id', 'mean_delta_macro_f1'}
		missing = sorted(required - fieldnames)
		if missing:
			raise ValueError(
				f'{path}: missing label-budget summary column(s): {missing!r}',
			)
		rows: list[dict[str, object]] = []
		for index, raw in enumerate(reader):
			row: dict[str, object] = {'budget_id': raw.get('budget_id', '')}
			for metric in ('macro_f1', 'mean_iou', 'balanced_accuracy'):
				key = f'mean_delta_{metric}'
				value = raw.get(key)
				if value in (None, ''):
					continue
				try:
					row[key] = float(value)
				except ValueError as exc:
					raise ValueError(
						f'{path}: row {index + 2} column {key} must be numeric',
					) from exc
			rows.append(row)
	return {'budgets': rows}


def _optional_json_artifact(
	path: Path | None,
	*,
	strict: bool,
	label: str,
	warnings: list[str],
) -> dict[str, object]:
	if path is None:
		return {'status': 'not_configured', 'summary': None}
	if not path.is_file():
		if strict:
			raise FileNotFoundError(f'missing {label}: {path}')
		warnings.append(f'{label} is pending: {path}')
		return {'status': 'pending', 'summary': None}
	return {'status': 'complete', 'summary': _read_json_object(path)}


def _read_json_object(path: Path) -> dict[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'{path} must contain valid JSON') from exc
	if not isinstance(payload, dict):
		raise TypeError(f'{path} must contain a JSON object')
	return payload


def _write_comparison_csv(
	path: Path,
	rows: Sequence[Mapping[str, object]],
) -> None:
	class_metrics = tuple(
		metric
		for metric in F3_STRAT_HMM_M1_CLASS_F1_METRICS
		if any(
			isinstance(row.get('metrics'), Mapping) and metric in row['metrics']
			for row in rows
		)
	)
	fieldnames = (
		'role',
		'model_tag',
		'status',
		*F3_STRAT_HMM_M1_GUARDRAIL_METRICS,
		*class_metrics,
	)
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			metrics = row.get('metrics')
			writer.writerow(
				{
					'role': row['role'],
					'model_tag': row['model_tag'],
					'status': row['status'],
					**(
						{
							metric: metrics.get(metric, '')
							for metric in (
								*F3_STRAT_HMM_M1_GUARDRAIL_METRICS,
								*class_metrics,
							)
						}
						if isinstance(metrics, Mapping)
						else {}
					),
				},
			)


def _decision_payload(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
	by_role = {str(row['role']): row for row in rows}
	values: dict[str, float] = {}
	for role in ('candidate', 'distillation_only', 'shuffled_hmm'):
		metrics = by_role[role].get('metrics')
		value = (
			metrics.get(F3_STRAT_HMM_M1_PRIMARY_METRIC)
			if isinstance(metrics, Mapping)
			else None
		)
		if not isinstance(value, int | float) or isinstance(value, bool):
			return {
				'status': 'pending',
				'primary_metric': F3_STRAT_HMM_M1_PRIMARY_METRIC,
				'interpretation': (
					'Guardrail evidence is pending until candidate, distillation-only, '
					'and shuffled-HMM full-budget metrics are available.'
				),
			}
		values[role] = float(value)
	distillation_matches = values['distillation_only'] >= values['candidate']
	shuffled_matches = values['shuffled_hmm'] >= values['candidate']
	if not distillation_matches and not shuffled_matches:
		interpretation = (
			'Strat-HMM beats both guardrails; evidence supports an ordered structured '
			'HMM pretext contribution.'
		)
	elif distillation_matches and not shuffled_matches:
		interpretation = (
			'Distillation-only matches or exceeds strat-HMM; improvement may be '
			'generic top-block adaptation, not HMM structure.'
		)
	elif shuffled_matches and not distillation_matches:
		interpretation = (
			'Shuffled-HMM matches or exceeds strat-HMM; improvement may be extra CE '
			'regularization or label histogram effects, not ordered structure.'
		)
	else:
		interpretation = (
			'Both guardrails match or exceed strat-HMM; the result does not isolate '
			'an ordered structured HMM pretext contribution.'
		)
	return {
		'status': 'complete',
		'primary_metric': F3_STRAT_HMM_M1_PRIMARY_METRIC,
		'interpretation': interpretation,
	}


def _low_budget_payload(  # noqa: C901, PLR0912, PLR0915
	rows: Sequence[Mapping[str, object]],
	*,
	strict: bool,
	warnings: list[str],
) -> dict[str, object]:
	summaries: dict[str, Mapping[str, object]] = {}
	manifests: dict[str, Mapping[str, object]] = {}
	configured = False
	for row in rows:
		robustness = row.get('robustness')
		label_budget = (
			robustness.get('label_budget') if isinstance(robustness, Mapping) else None
		)
		manifest = (
			robustness.get('label_budget_suite_manifest')
			if isinstance(robustness, Mapping)
			else None
		)
		if not isinstance(label_budget, Mapping) or not isinstance(manifest, Mapping):
			continue
		role = str(row['role'])
		status = label_budget.get('status')
		manifest_status = manifest.get('status')
		configured = configured or any(
			value != 'not_configured' for value in (status, manifest_status)
		)
		summary = label_budget.get('summary')
		if status == 'complete' and isinstance(summary, Mapping):
			summaries[role] = summary
		manifest_payload = manifest.get('summary')
		if manifest_status == 'complete' and isinstance(manifest_payload, Mapping):
			manifests[role] = manifest_payload
		if status == 'complete' and manifest_status != 'complete':
			message = (
				f'{role}: label-budget comparison requires a suite manifest to '
				'validate shared baseline subsampling indices'
			)
			if strict:
				raise ValueError(message)
			warnings.append(message)
	if not configured:
		if strict:
			raise ValueError(
				'strict guardrail summary requires label-budget comparisons for '
				+ ', '.join(F3_STRAT_HMM_M1_LOW_BUDGET_IDS),
			)
		return {
			'status': 'not_configured',
			'pairing_provenance': {'status': 'not_configured'},
			'budgets': [],
		}
	if 'candidate' not in summaries or 'candidate' not in manifests:
		if strict:
			raise ValueError(
				'strict guardrail summary requires candidate label-budget '
				'comparisons for '
				+ ', '.join(F3_STRAT_HMM_M1_LOW_BUDGET_IDS),
			)
		return {
			'status': 'pending',
			'pairing_provenance': {'status': 'pending'},
			'budgets': [],
		}

	candidate = _budget_delta_rows(
		summaries['candidate'],
		role='candidate',
		strict=strict,
		warnings=warnings,
	)
	candidate_identity = _label_budget_identity_by_budget(
		manifests['candidate'],
		role='candidate',
		expected_model_tag=F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG,
	)
	comparison_rows: dict[str, dict[str, object]] = {}
	verified_comparisons: list[dict[str, object]] = []
	verified_comparison_keys: set[tuple[str, str]] = set()
	provenance_failed = False
	for guardrail_role in ('distillation_only', 'shuffled_hmm'):
		summary = summaries.get(guardrail_role)
		manifest = manifests.get(guardrail_role)
		if summary is None or manifest is None:
			continue
		guardrail = _budget_delta_rows(
			summary,
			role=guardrail_role,
			strict=strict,
			warnings=warnings,
		)
		guardrail_identity = _label_budget_identity_by_budget(
			manifest,
			role=guardrail_role,
			expected_model_tag=F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS[
				guardrail_role
			],
		)
		for budget_id in sorted(
			set(candidate) & set(guardrail),
			key=_budget_sort_key,
		):
			candidate_conditions = candidate_identity.get(budget_id)
			guardrail_conditions = guardrail_identity.get(budget_id)
			if (
				candidate_conditions is None
				or guardrail_conditions is None
				or candidate_conditions != guardrail_conditions
			):
				message = (
					'label-budget pairing provenance mismatch for '
					f'candidate vs {guardrail_role}, budget_id={budget_id!r}; '
					f'candidate={candidate_conditions!r}, '
					f'{guardrail_role}={guardrail_conditions!r}'
				)
				if strict:
					raise ValueError(message)
				warnings.append(message)
				provenance_failed = True
				continue
			candidate_row = candidate[budget_id]
			guardrail_row = guardrail[budget_id]
			deltas = {
				metric: candidate_row[metric] - guardrail_row[metric]
				for metric in candidate_row.keys() & guardrail_row.keys()
			}
			comparison_rows.setdefault(budget_id, {'budget_id': budget_id})[
				f'candidate_minus_{guardrail_role}'
			] = dict(sorted(deltas.items()))
			verified_comparisons.append(
				{
					'budget_id': budget_id,
					'guardrail_role': guardrail_role,
					'conditions': [
						{
							'subsample_seed': seed,
							'paired_identity_hash': identity,
						}
						for seed, identity in candidate_conditions
					],
				},
			)
			verified_comparison_keys.add((budget_id, guardrail_role))
	budgets = [
		comparison_rows[budget_id]
		for budget_id in sorted(comparison_rows, key=_budget_sort_key)
	]
	expected_comparison_keys = {
		(budget_id, guardrail_role)
		for budget_id in F3_STRAT_HMM_M1_LOW_BUDGET_IDS
		for guardrail_role in ('distillation_only', 'shuffled_hmm')
	}
	missing_comparison_keys = expected_comparison_keys - verified_comparison_keys
	coverage_complete = bool(expected_comparison_keys) and not missing_comparison_keys
	if missing_comparison_keys:
		formatted = ', '.join(
			f'{budget_id}:{guardrail_role}'
			for budget_id, guardrail_role in sorted(
				missing_comparison_keys,
				key=lambda item: (_budget_sort_key(item[0]), item[1]),
			)
		)
		message = (
			'label-budget comparisons are missing for expected candidate budgets: '
			f'{formatted}'
		)
		if strict:
			raise ValueError(message)
		warnings.append(message)
	status = (
		'complete'
		if all(
			role in summaries and role in manifests
			for role in ('distillation_only', 'shuffled_hmm')
		)
		and not provenance_failed
		and coverage_complete
		else 'partial'
	)
	return {
		'status': status,
		'basis': 'difference_of_shared_baseline_paired_mean_deltas',
		'pairing_provenance': {
			'status': 'verified' if verified_comparisons else 'unverified',
			'baseline_model_tag': F3_STRAT_HMM_M1_BASELINE_MODEL_TAG,
			'identity_field': 'paired_identity_hash',
			'comparisons': verified_comparisons,
		},
		'budgets': budgets,
	}


def _label_budget_identity_by_budget(  # noqa: C901, PLR0912
	payload: Mapping[str, object],
	*,
	role: str,
	expected_model_tag: str,
) -> dict[str, tuple[tuple[int, str], ...]]:
	if payload.get('artifact_type') != 'f3_lithology_label_budget_suite_manifest':
		raise ValueError(
			f'{role} label-budget suite manifest has unexpected artifact_type',
		)
	rows = payload.get('rows')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'{role} label-budget suite manifest rows must be a list')
	by_condition: dict[tuple[str, int], set[str]] = {}
	baseline_conditions: set[tuple[str, int]] = set()
	comparison_conditions: set[tuple[str, int]] = set()
	for index, raw in enumerate(rows):
		if not isinstance(raw, Mapping):
			raise TypeError(
				f'{role} label-budget manifest rows[{index}] must be a mapping',
			)
		budget_id = raw.get('budget_id')
		if not isinstance(budget_id, str) or not budget_id:
			raise TypeError(
				f'{role} label-budget manifest rows[{index}].budget_id must be '
				'a string',
			)
		seed = raw.get('subsample_seed')
		if not isinstance(seed, int) or isinstance(seed, bool):
			raise TypeError(
				f'{role} label-budget manifest rows[{index}].subsample_seed must '
				'be an integer',
			)
		identity = raw.get('paired_identity_hash')
		if not isinstance(identity, str) or not identity:
			raise TypeError(
				f'{role} label-budget manifest rows[{index}].paired_identity_hash '
				'must be a string',
			)
		condition = (budget_id, seed)
		by_condition.setdefault(condition, set()).add(identity)
		if raw.get('model_role') == 'baseline':
			model_tag = raw.get('model_tag')
			if model_tag != F3_STRAT_HMM_M1_BASELINE_MODEL_TAG:
				raise ValueError(
					f'{role} label-budget manifest baseline model_tag must be '
					f'{F3_STRAT_HMM_M1_BASELINE_MODEL_TAG!r}; got {model_tag!r}',
				)
			baseline_conditions.add(condition)
		elif raw.get('model_role') == 'candidate':
			model_tag = raw.get('model_tag')
			if model_tag != expected_model_tag:
				raise ValueError(
					f'{role} label-budget manifest candidate model_tag must be '
					f'{expected_model_tag!r}; got {model_tag!r}',
				)
			comparison_conditions.add(condition)
	result: dict[str, list[tuple[int, str]]] = {}
	for (budget_id, seed), identities in sorted(by_condition.items()):
		if (budget_id, seed) not in baseline_conditions:
			raise ValueError(
				f'{role} label-budget manifest has no common baseline row for '
				f'budget_id={budget_id!r}, subsample_seed={seed}',
			)
		if (budget_id, seed) not in comparison_conditions:
			raise ValueError(
				f'{role} label-budget manifest has no candidate row for '
				f'model_tag={expected_model_tag!r}, budget_id={budget_id!r}, '
				f'subsample_seed={seed}',
			)
		if len(identities) != 1:
			raise ValueError(
				f'{role} label-budget manifest has inconsistent paired identity for '
				f'budget_id={budget_id!r}, subsample_seed={seed}',
			)
		result.setdefault(budget_id, []).append((seed, next(iter(identities))))
	return {budget_id: tuple(conditions) for budget_id, conditions in result.items()}


def _budget_delta_rows(  # noqa: C901, PLR0912
	payload: Mapping[str, object],
	*,
	role: str,
	strict: bool,
	warnings: list[str],
) -> dict[str, dict[str, float]]:
	container = payload.get('label_budget', payload)
	if not isinstance(container, Mapping):
		raise TypeError(f'{role} label-budget summary label_budget must be a mapping')
	rows = container.get('budgets')
	if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
		raise TypeError(f'{role} label-budget summary budgets must be a list')
	result: dict[str, dict[str, float]] = {}
	for index, raw in enumerate(rows):
		if not isinstance(raw, Mapping):
			raise TypeError(f'{role} label-budget budgets[{index}] must be a mapping')
		budget_id = raw.get('budget_id')
		if not isinstance(budget_id, str) or not budget_id:
			raise TypeError(
				f'{role} label-budget budgets[{index}].budget_id must be a string',
			)
		if budget_id not in F3_STRAT_HMM_M1_LOW_BUDGET_IDS:
			continue
		if budget_id in result:
			raise ValueError(f'{role} label-budget has duplicate budget {budget_id!r}')
		metrics: dict[str, float] = {}
		for metric in ('macro_f1', 'mean_iou', 'balanced_accuracy'):
			key = f'mean_delta_{metric}'
			value = raw.get(key)
			if value is None:
				continue
			if not isinstance(value, int | float) or isinstance(value, bool):
				raise TypeError(f'{role} {budget_id} {key} must be numeric')
			value = float(value)
			if not math.isfinite(value):
				raise ValueError(f'{role} {budget_id} {key} must be finite')
			metrics[metric] = value
		if F3_STRAT_HMM_M1_PRIMARY_METRIC not in metrics:
			message = (
				f'{role} label-budget {budget_id} is missing '
				f'mean_delta_{F3_STRAT_HMM_M1_PRIMARY_METRIC}'
			)
			if strict:
				raise ValueError(message)
			warnings.append(message)
			continue
		result[budget_id] = metrics
	return result


def _budget_sort_key(budget_id: str) -> tuple[int, int, str]:
	if budget_id == 'full':
		return (2, 0, budget_id)
	if budget_id.startswith('cap') and budget_id[3:].isdigit():
		return (0, int(budget_id[3:]), budget_id)
	return (1, 0, budget_id)


def _render_markdown(payload: Mapping[str, object]) -> str:
	models = payload['models']
	if not isinstance(models, list):
		raise TypeError('summary models must be a list')
	class_metrics = tuple(
		metric
		for metric in F3_STRAT_HMM_M1_CLASS_F1_METRICS
		if any(
			isinstance(row, Mapping)
			and isinstance(row.get('metrics'), Mapping)
			and metric in row['metrics']
			for row in models
		)
	)
	table_metrics = (*F3_STRAT_HMM_M1_GUARDRAIL_METRICS, *class_metrics)
	lines = [
		'# F3 Strat-HMM Milestone-1 Guardrails',
		'',
		'## Full-budget comparison',
		'',
		'| role | status | ' + ' | '.join(table_metrics) + ' |',
		'|---|---|' + '|'.join('---:' for _ in table_metrics) + '|',
	]
	for row in models:
		if not isinstance(row, Mapping):
			raise TypeError('summary model row must be a mapping')
		metrics = row['metrics']
		values = (
			['pending'] * len(table_metrics)
			if metrics is None
			else [
				(
					'pending'
					if metrics.get(name) is None
					else f'{float(metrics[name]):.6f}'
				)
				for name in table_metrics
			]
		)
		lines.append(
			f'| {row["role"]} | {row["status"]} | ' + ' | '.join(values) + ' |',
		)
	lines.extend(['', '## Low-budget guardrail deltas', ''])
	low_budget = payload.get('low_budget')
	if not isinstance(low_budget, Mapping) or not low_budget.get('budgets'):
		status = (
			low_budget.get('status', 'not_configured')
			if isinstance(
				low_budget,
				Mapping,
			)
			else 'not_configured'
		)
		lines.append(f'Low-budget evaluation status: {status}.')
	else:
		lines.extend(
			[
				'Positive values favor strat-HMM. Deltas subtract each guardrail from '
				'the candidate after aggregation over verified identical baseline '
				'subsampling selections.',
				'',
				'| budget_id | candidate - distillation-only macro_f1 | candidate - '
				'shuffled-HMM macro_f1 |',
				'|---|---:|---:|',
			],
		)
		for budget in low_budget['budgets']:
			if not isinstance(budget, Mapping):
				continue
			distillation = budget.get('candidate_minus_distillation_only')
			shuffled = budget.get('candidate_minus_shuffled_hmm')
			lines.append(
				f'| {budget["budget_id"]} | '
				f'{_markdown_delta(distillation, F3_STRAT_HMM_M1_PRIMARY_METRIC)} | '
				f'{_markdown_delta(shuffled, F3_STRAT_HMM_M1_PRIMARY_METRIC)} |',
			)
	decision = payload.get('decision')
	if not isinstance(decision, Mapping):
		raise TypeError('summary decision must be a mapping')
	lines.extend(
		[
			'',
			'## Decision',
			'',
			str(decision['interpretation']),
			'',
			'Interpretation guide:',
			'',
			'- If distillation-only matches strat-HMM, improvement may be generic '
			'top-block adaptation, not HMM structure.',
			'- If shuffled-HMM matches strat-HMM, improvement may be extra CE '
			'regularization or label histogram effects, not ordered structure.',
			'- If strat-HMM beats both, evidence supports an ordered structured HMM '
			'pretext contribution.',
		],
	)
	warnings = payload.get('warnings')
	if isinstance(warnings, Sequence) and warnings:
		lines.extend(['', '## Warnings', ''])
		lines.extend(f'- {warning}' for warning in warnings)
	return '\n'.join(lines) + '\n'


def _markdown_delta(value: object, metric: str) -> str:
	if not isinstance(value, Mapping):
		return 'pending'
	metric_value = value.get(metric)
	if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
		return 'pending'
	return f'{float(metric_value):.6f}'


def _stable_suite_name(config: Mapping[str, object]) -> str:
	_validate_keys(config, {'name', 'strict'}, 'suite', allow_missing={'strict'})
	name = _string(config, 'name', 'suite')
	if name != F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME:
		raise ValueError(
			f'suite.name must be '
			f'{F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME!r}; got {name!r}',
		)
	return name


def _stable_model_tag(
	role: str,
	config: Mapping[str, object],
	*,
	prefix: str,
) -> str:
	model_tag = _string(config, 'model_tag', prefix)
	expected = F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS[role]
	if model_tag != expected:
		raise ValueError(f'{prefix}.model_tag must be {expected!r}; got {model_tag!r}')
	return model_tag


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping; got {value!r}')
	return value


def _string(parent: Mapping[str, object], key: str, prefix: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{prefix}.{key} must be a non-empty string; got {value!r}')
	return value


def _absolute_path(parent: Mapping[str, object], key: str, prefix: str) -> Path:
	path = Path(_string(parent, key, prefix))
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be an absolute path; got {path}')
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path | None:
	if parent.get(key) is None:
		return None
	return _absolute_path(parent, key, prefix)


def _positive_int(parent: Mapping[str, object], key: str, prefix: str) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise TypeError(f'{prefix}.{key} must be a positive integer; got {value!r}')
	return value


def _nonnegative_int(parent: Mapping[str, object], key: str, prefix: str) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		raise TypeError(f'{prefix}.{key} must be a nonnegative integer; got {value!r}')
	return value


def _validate_keys(
	parent: Mapping[str, object],
	allowed: set[str],
	prefix: str,
	*,
	allow_missing: set[str] | None = None,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		raise ValueError(f'{prefix} key(s) not allowed: {unexpected!r}')
	missing = sorted(allowed - set(parent) - (allow_missing or set()))
	if missing:
		raise ValueError(f'{prefix} missing required key(s): {missing!r}')


__all__ = [
	'F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG',
	'F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG',
	'F3_STRAT_HMM_M1_GUARDRAIL_METRICS',
	'F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS',
	'F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_DIR',
	'F3_STRAT_HMM_M1_GUARDRAIL_PUBLISH_SUFFIXES',
	'F3_STRAT_HMM_M1_GUARDRAIL_ROLES',
	'F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME',
	'F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG',
	'F3GuardrailJob',
	'F3GuardrailJobsConfig',
	'F3GuardrailPublishConfig',
	'F3GuardrailResultInput',
	'F3GuardrailSummaryConfig',
	'F3GuardrailSummaryResult',
	'F3ShuffledHMMTargetConfig',
	'f3_guardrail_jobs_config_from_mapping',
	'f3_guardrail_summary_config_from_mapping',
	'f3_shuffled_hmm_target_config_from_mapping',
	'publish_f3_strat_hmm_m1_guardrails',
	'summarize_f3_strat_hmm_m1_guardrails',
]
