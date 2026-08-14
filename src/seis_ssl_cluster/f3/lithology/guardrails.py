"""Contracts for retained F3 strat-HMM M1 guardrail artifact producers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME = 'strat_hmm_m1_guardrails_v1'
F3_STRAT_HMM_M1_CANDIDATE_MODEL_TAG = 'strat_hmm_pretext_m1_k6_topblock1_distill'
F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG = 'strat_hmm_m1_guardrail_distill_only'
F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG = 'strat_hmm_m1_guardrail_shuffled_hmm_seed42'
F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS = {
	'distillation_only': F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG,
	'shuffled_hmm': F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG,
}
F3_STRAT_HMM_M1_GUARDRAIL_JOB_STAGES = frozenset(
	{'extract_guardrail_embeddings'},
)


@dataclass(frozen=True)
class F3ShuffledHMMTargetConfig:
	"""Resolved contract for deterministic target shuffling."""

	suite_name: str
	source_root: Path
	output_root: Path
	k: int
	seed: int
	shuffle_scope: str
	overwrite: bool


@dataclass(frozen=True)
class F3GuardrailJob:
	"""One retained artifact job with an isolated guardrail root."""

	role: str
	model_tag: str
	input_path: Path
	output_path: Path


@dataclass(frozen=True)
class F3GuardrailJobsConfig:
	"""Resolved artifact-routing contract for both retained guardrails."""

	suite_name: str
	stage: str
	jobs: tuple[F3GuardrailJob, ...]


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
	"""Validate retained guardrail artifact routing."""
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
		if role not in F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS:
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


def _stable_suite_name(config: Mapping[str, object]) -> str:
	_validate_keys(config, {'name'}, 'suite')
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
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		raise ValueError(f'{prefix} key(s) not allowed: {unexpected!r}')
	missing = sorted(allowed - set(parent))
	if missing:
		raise ValueError(f'{prefix} missing required key(s): {missing!r}')


__all__ = [
	'F3_STRAT_HMM_M1_DISTILL_ONLY_MODEL_TAG',
	'F3_STRAT_HMM_M1_GUARDRAIL_MODEL_TAGS',
	'F3_STRAT_HMM_M1_GUARDRAIL_SUITE_NAME',
	'F3_STRAT_HMM_M1_SHUFFLED_HMM_MODEL_TAG',
	'F3GuardrailJob',
	'F3GuardrailJobsConfig',
	'F3ShuffledHMMTargetConfig',
	'f3_guardrail_jobs_config_from_mapping',
	'f3_shuffled_hmm_target_config_from_mapping',
]
