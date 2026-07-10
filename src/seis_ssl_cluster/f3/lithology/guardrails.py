"""Contracts and result summaries for the F3 strat-HMM M1 guardrails."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME = 'strat_hmm_m1_guardrails_v1'
F3_STRAT_HMM_M1_BASELINE_MODEL_TAG = (
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
)
F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG = (
	'strat_hmm_pretext_m1_k6_topblock1_distill'
)
F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG = (
	'strat_hmm_m1_guardrail_distill_only'
)
F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG = (
	'strat_hmm_m1_guardrail_shuffled_hmm'
)
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
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'accuracy',
	'weighted_f1',
)
F3_STRAT_HMM_M1_GUARDRAIL_JOB_STAGES = frozenset(
	{
		'extract_guardrail_embeddings',
		'build_guardrail_token_datasets',
		'run_guardrail_probes',
	},
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
	label_budget_summary_json: Path | None = None
	split_index_summary_json: Path | None = None


@dataclass(frozen=True)
class F3GuardrailSummaryConfig:
	"""Resolved guardrail result-summary contract."""

	suite_name: str
	strict: bool
	models: tuple[F3GuardrailResultInput, ...]
	output_dir: Path


@dataclass(frozen=True)
class F3GuardrailSummaryResult:
	"""Paths written by guardrail result consolidation."""

	summary_json: Path
	summary_markdown: Path
	pending_roles: tuple[str, ...]


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
	_validate_keys(config, {'suite', 'models', 'outputs'}, 'config')
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
				'label_budget_summary_json',
				'split_index_summary_json',
			},
			f'models.{role}',
			allow_missing={
				'label_budget_summary_json',
				'split_index_summary_json',
			},
		)
		resolved_models.append(
			F3GuardrailResultInput(
				role=role,
				model_tag=_stable_model_tag(role, raw, prefix=f'models.{role}'),
				metrics_json=_absolute_path(raw, 'metrics_json', f'models.{role}'),
				label_budget_summary_json=_optional_absolute_path(
					raw,
					'label_budget_summary_json',
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
	return F3GuardrailSummaryConfig(
		suite_name=suite_name,
		strict=strict,
		models=tuple(resolved_models),
		output_dir=output_dir,
	)


def summarize_f3_strat_hmm_m1_guardrails(
	config: F3GuardrailSummaryConfig,
) -> F3GuardrailSummaryResult:
	"""Write a deterministic summary, reporting absent artifacts as pending."""
	rows: list[dict[str, object]] = []
	pending: list[str] = []
	for model in config.models:
		if not model.metrics_json.is_file():
			if config.strict:
				raise FileNotFoundError(
					f'missing guardrail metrics for {model.role}: {model.metrics_json}',
				)
			pending.append(model.role)
			rows.append(
				{
					'role': model.role,
					'model_tag': model.model_tag,
					'status': 'pending',
					'metrics': None,
					'robustness': _robustness_payload(model, strict=False),
				},
			)
			continue
		metrics = _read_metrics(model.metrics_json)
		rows.append(
			{
				'role': model.role,
				'model_tag': model.model_tag,
				'status': 'complete',
				'metrics': metrics,
				'robustness': _robustness_payload(model, strict=config.strict),
			},
		)
	payload = {
		'schema_version': 1,
		'suite_name': config.suite_name,
		'strict': config.strict,
		'models': rows,
		'pending_roles': pending,
	}
	config.output_dir.mkdir(parents=True, exist_ok=True)
	json_path = config.output_dir / 'guardrail_summary.json'
	markdown_path = config.output_dir / 'guardrail_summary.md'
	json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
	markdown_path.write_text(_render_markdown(payload), encoding='utf-8')
	return F3GuardrailSummaryResult(json_path, markdown_path, tuple(pending))


def _read_metrics(path: Path) -> dict[str, float]:
	payload = _read_json_object(path)
	metrics: dict[str, float] = {}
	for metric in F3_STRAT_HMM_M1_GUARDRAIL_METRICS:
		value = payload.get(metric)
		if not isinstance(value, int | float) or isinstance(value, bool):
			raise TypeError(f'{path}: {metric} must be numeric; got {value!r}')
		value = float(value)
		if not math.isfinite(value):
			raise ValueError(f'{path}: {metric} must be finite; got {value!r}')
		metrics[metric] = value
	return metrics


def _robustness_payload(
	model: F3GuardrailResultInput,
	*,
	strict: bool,
) -> dict[str, object]:
	return {
		'label_budget': _optional_json_artifact(
			model.label_budget_summary_json,
			strict=strict,
			label=f'{model.role} label-budget summary',
		),
		'split_index': _optional_json_artifact(
			model.split_index_summary_json,
			strict=strict,
			label=f'{model.role} split/index summary',
		),
	}


def _optional_json_artifact(
	path: Path | None,
	*,
	strict: bool,
	label: str,
) -> dict[str, object]:
	if path is None:
		return {'status': 'not_configured', 'summary': None}
	if not path.is_file():
		if strict:
			raise FileNotFoundError(f'missing {label}: {path}')
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


def _render_markdown(payload: Mapping[str, object]) -> str:
	lines = [
		'# F3 Strat-HMM Milestone-1 Guardrails',
		'',
		'| role | status | macro_f1 | mean_iou | balanced_accuracy | accuracy '
		'| weighted_f1 |',
		'|---|---|---:|---:|---:|---:|---:|',
	]
	models = payload['models']
	if not isinstance(models, list):
		raise TypeError('summary models must be a list')
	for row in models:
		if not isinstance(row, Mapping):
			raise TypeError('summary model row must be a mapping')
		metrics = row['metrics']
		values = (
			['pending'] * len(F3_STRAT_HMM_M1_GUARDRAIL_METRICS)
			if metrics is None
			else [
				f'{float(metrics[name]):.6f}'
				for name in F3_STRAT_HMM_M1_GUARDRAIL_METRICS
			]
		)
		lines.append(
			f'| {row["role"]} | {row["status"]} | ' + ' | '.join(values) + ' |',
		)
	return '\n'.join(lines) + '\n'


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
	'F3_STRAT_HMM_M1_GUARDRAIL_ROLES',
	'F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME',
	'F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG',
	'F3GuardrailJob',
	'F3GuardrailJobsConfig',
	'F3GuardrailResultInput',
	'F3GuardrailSummaryConfig',
	'F3GuardrailSummaryResult',
	'F3ShuffledHMMTargetConfig',
	'f3_guardrail_jobs_config_from_mapping',
	'f3_guardrail_summary_config_from_mapping',
	'f3_shuffled_hmm_target_config_from_mapping',
	'summarize_f3_strat_hmm_m1_guardrails',
]
