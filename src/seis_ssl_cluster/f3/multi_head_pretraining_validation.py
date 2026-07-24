"""Validate and publish the F3 K=6/8/10 pretraining handoff.

This is deliberately upstream of the voxel-label-budget stages: a PASS
handoff is evidence for an already complete paired pretraining run, not an
input manufactured by a downstream aggregation command.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_components,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	scientific_identity_sha256,
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'control_full_config',
		'nocons_full_config',
		'cons010_full_config',
	}
)
_CANDIDATES = (
	('nocons', 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1', 0.0),
	('cons010', 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1', 0.1),
)
_HANDOFF_TYPE = 'f3_multi_head_pretraining_handoff'
_CURRENT_K6_MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'


@dataclass(frozen=True)
class F3MultiHeadPretrainingValidationConfig:
	"""Resolved paths for the canonical two-candidate validation."""

	artifact_root: Path
	experiment_root: Path
	target_manifest: Path
	control_full_config: Path
	nocons_full_config: Path
	cons010_full_config: Path


@dataclass(frozen=True)
class F3MultiHeadPretrainingValidationResult:
	"""Candidate validation outcomes and any published handoffs."""

	phase: str
	candidates: Mapping[str, Mapping[str, object]]
	published_handoffs: tuple[Path, ...]


def f3_multi_head_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3MultiHeadPretrainingValidationConfig:
	"""Resolve the small, intentionally closed validation config schema."""
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown validation config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing validation config keys: {sorted(missing)!r}')

	def path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3MultiHeadPretrainingValidationConfig(
		artifact_root=path('artifact_root'),
		experiment_root=path('experiment_root'),
		target_manifest=path('target_manifest'),
		control_full_config=path('control_full_config'),
		nocons_full_config=path('nocons_full_config'),
		cons010_full_config=path('cons010_full_config'),
	)
	if (
		not result.artifact_root.is_absolute()
		or not result.experiment_root.is_absolute()
	):
		raise ValueError('artifact_root and experiment_root must be absolute')
	ensure_under_root(
		result.experiment_root, root=result.artifact_root, label='experiment_root'
	)
	ensure_under_root(
		result.target_manifest, root=result.artifact_root, label='target_manifest'
	)
	for key, value in (
		('control_full_config', result.control_full_config),
		('nocons_full_config', result.nocons_full_config),
		('cons010_full_config', result.cons010_full_config),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{key} is missing: {value}')
	if not result.target_manifest.is_file():
		raise FileNotFoundError(f'target_manifest is missing: {result.target_manifest}')
	return result


def load_f3_multi_head_pretraining_validation_config(
	path: str | Path,
) -> F3MultiHeadPretrainingValidationConfig:
	"""Load the canonical YAML config through the repository YAML loader."""
	return f3_multi_head_pretraining_validation_config_from_mapping(load_config(path))


def load_f3_multi_head_pretraining_handoff(  # noqa: C901, PLR0912
	path: str | Path,
) -> Mapping[str, object]:
	"""Load the single strict public schema consumed by downstream stages."""
	payload = _json(Path(path))
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError('multi-head pretraining handoff type/status mismatch')
	for key in ('model_tag', 'variant'):
		if not isinstance(payload.get(key), str) or not payload[key]:
			raise TypeError(f'handoff {key} is missing')
	if payload['variant'] not in {'nocons', 'cons010'}:
		raise ValueError('handoff variant mismatch')
	stratigraphy = _mapping(payload.get('stratigraphy_pretext'), 'handoff stratigraphy')
	if stratigraphy.get(
		'head_spec'
	) != 'multi_resolution_ordered_prototypes_v1' or stratigraphy.get('head_ks') != [
		6,
		8,
		10,
	]:
		raise ValueError('handoff multi-head specification/K mismatch')
	for key in (
		'target_manifest_path',
		'target_manifest_sha256',
		'consistency_policy',
		'scientific_identity_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	):
		if not _is_sha256(stratigraphy.get(key)) and key.endswith('sha256'):
			raise TypeError(f'handoff stratigraphy_pretext.{key} is missing')
		if key.endswith('path') and (
			not isinstance(stratigraphy.get(key), str) or not stratigraphy[key]
		):
			raise TypeError(f'handoff stratigraphy_pretext.{key} is missing')
	if stratigraphy.get('consistency_policy') != 'normalized_order_smooth_l1_v1':
		raise ValueError('handoff consistency policy mismatch')
	for key in ('consistency_weight', 'consistency_beta'):
		if not _finite_number(stratigraphy.get(key)):
			raise TypeError(f'handoff stratigraphy_pretext.{key} is missing')
	if stratigraphy['consistency_beta'] != 0.1:
		raise ValueError('handoff consistency beta mismatch')
	_per_head_target_hashes(stratigraphy.get('per_head_target_sha256'))
	checkpoint = _mapping(payload.get('checkpoint'), 'handoff checkpoint')
	for key in ('path', 'latest_path', 'selection_metric'):
		if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
			raise TypeError(f'handoff checkpoint.{key} is missing')
	for key in ('sha256', 'latest_sha256'):
		if not _is_sha256(checkpoint.get(key)):
			raise TypeError(f'handoff checkpoint.{key} is missing')
	if checkpoint['selection_metric'] != 'metrics.loss':
		raise ValueError('handoff checkpoint selection metric mismatch')
	for key in ('best_epoch', 'best_global_step'):
		if isinstance(checkpoint.get(key), bool) or not isinstance(
			checkpoint.get(key), int
		) or checkpoint[key] < 0:
			raise TypeError(f'handoff checkpoint.{key} must be an integer')
	embedding = _mapping(payload.get('embedding'), 'handoff embedding')
	for key in ('root', 'metadata_path'):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	for key in ('metadata_sha256', 'embeddings_sha256', 'valid_tokens_sha256'):
		if not _is_sha256(embedding.get(key)):
			raise TypeError(f'handoff embedding.{key} is missing')
	if payload.get('embedding_metadata_sha256') != embedding['metadata_sha256']:
		raise ValueError('handoff embedding metadata SHA-256 mismatch')
	return payload


def validate_f3_multi_head_pretraining(  # noqa: C901, PLR0912, PLR0915
	config: F3MultiHeadPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3MultiHeadPretrainingValidationResult:
	"""Validate checkpoints and, after extraction, atomically publish PASS files."""
	if phase not in {'checkpoints', 'complete'}:
		raise ValueError('phase must be checkpoints or complete')
	try:
		target = load_multi_head_target_manifest(config.target_manifest)
		if target.get('head_ks') != [6, 8, 10]:
			raise ValueError('target manifest K identity mismatch')  # noqa: TRY301
		expected_per_head_targets = _manifest_per_head_target_hashes(target)
	except (OSError, TypeError, ValueError) as error:
		if not dry_run:
			raise
		return _dry_run_failures(phase, error)
	try:
		control = _training_config(config.control_full_config)
	except (OSError, TypeError, ValueError) as error:
		if not dry_run:
			raise
		return _dry_run_failures(phase, error)
	configs: dict[str, Mapping[str, object]] = {'control': control}
	results: dict[str, Mapping[str, object]] = {}
	for variant, model_tag, _weight in _CANDIDATES:
		path = getattr(config, f'{variant}_full_config')
		try:
			configs[variant] = _training_config(path)
		except (OSError, TypeError, ValueError) as error:
			if not dry_run:
				raise
			results[variant] = _dry_run_failure(phase, model_tag, error)
	try:
		_validate_control_config(config, control)
		canonical_valid_tokens_sha256 = (
			_canonical_k6_valid_tokens_sha256(config, control)
			if phase == 'complete'
			else None
		)
	except (OSError, TypeError, ValueError) as error:
		if not dry_run:
			raise
		return _dry_run_failures(phase, error)
	for variant, model_tag, weight in _CANDIDATES:
		if variant in results:
			continue
		try:
			_validate_candidate_config_contract(
				config,
				configs[variant],
				variant=variant,
				model_tag=model_tag,
				weight=weight,
			)
			result = _checkpoint_evidence(
				config,
				configs[variant],
				variant=variant,
				model_tag=model_tag,
				weight=weight,
				expected_per_head_targets=expected_per_head_targets,
			)
			if phase == 'complete':
				if canonical_valid_tokens_sha256 is None:
					raise RuntimeError('current K6 valid-token identity is unavailable')
				result = {
					**result,
					'embedding': _embedding_evidence(
						config,
						result,
						model_tag,
						canonical_valid_tokens_sha256=canonical_valid_tokens_sha256,
					),
				}
			results[variant] = {
				'status': 'PASS',
				'planned_action': _planned_action(phase, model_tag),
				**result,
			}
		except (OSError, TypeError, ValueError) as error:
			if not dry_run:
				raise
			results[variant] = _dry_run_failure(phase, model_tag, error)
	if all(result['status'] == 'PASS' for result in results.values()):
		try:
			_validate_pair_config_contract(configs)
			_validate_pair(results['nocons'], results['cons010'], configs)
		except (TypeError, ValueError) as error:
			if not dry_run:
				raise
			for variant, model_tag, _weight in _CANDIDATES:
				results[variant] = _dry_run_failure(phase, model_tag, error)
	if dry_run and any(result['status'] != 'PASS' for result in results.values()):
		return F3MultiHeadPretrainingValidationResult(phase, results, ())
	published: list[Path] = []
	for variant, model_tag, _weight in _CANDIDATES:
		candidate_root = config.experiment_root / model_tag
		preflight = candidate_root / 'preflight'
		checkpoint_report = preflight / 'checkpoint_validation.json'
		if not dry_run:
			_atomic_json(checkpoint_report, _report('checkpoints', results[variant]))
		if phase != 'complete':
			continue
		embedding_report = preflight / 'embedding_validation.json'
		if not dry_run:
			_atomic_json(embedding_report, _report('complete', results[variant]))
		handoff = _handoff(variant, model_tag, results[variant])
		handoff_path = preflight / 'multi_head_handoff.json'
		if dry_run:
			continue
		if _publish_handoff(
			handoff_path,
			handoff,
			only_missing=only_missing,
			_quarantine_invalid=quarantine_invalid,
		):
			published.append(handoff_path)
	return F3MultiHeadPretrainingValidationResult(phase, results, tuple(published))


def _dry_run_failures(
	phase: str, error: Exception
) -> F3MultiHeadPretrainingValidationResult:
	"""Return one explicit dry-run failure for each planned candidate action."""
	return F3MultiHeadPretrainingValidationResult(
		phase,
		{
			variant: _dry_run_failure(phase, model_tag, error)
			for variant, model_tag, _weight in _CANDIDATES
		},
		(),
	)


def _dry_run_failure(
	phase: str, model_tag: str, error: Exception
) -> Mapping[str, object]:
	return {
		'status': 'FAIL',
		'planned_action': _planned_action(phase, model_tag),
		'error': f'{type(error).__name__}: {error}',
	}


def _planned_action(phase: str, model_tag: str) -> str:
	action = 'validate checkpoints'
	if phase == 'complete':
		action += ', validate embeddings, publish PASS handoff'
	return f'{model_tag}: {action}'


def _training_config(path: Path) -> Mapping[str, object]:
	value = resolve_strat_hmm_pretext_config(load_config(path))
	for key in ('paths', 'identity', 'pseudo_targets', 'head', 'loss', 'train'):
		_mapping(value.get(key), f'{path.name}:{key}')
	return value


def _validate_control_config(
	config: F3MultiHeadPretrainingValidationConfig,
	control: Mapping[str, object],
) -> None:
	paths = _mapping(control['paths'], 'control paths')
	identity = _mapping(control['identity'], 'control identity')
	model_tag = identity.get('model_tag')
	if model_tag != _CURRENT_K6_MODEL_TAG:
		raise ValueError('control model tag mismatch')
	output = Path(str(paths.get('output_root', ''))).resolve()
	if output != (config.experiment_root / model_tag).resolve():
		raise ValueError('control output root mismatch')
	ensure_under_root(output, root=config.artifact_root, label='control.output_root')
	if _mapping(control['pseudo_targets'], 'control pseudo_targets').get('k') != 6:
		raise ValueError('control pseudo-target K mismatch')
	if _mapping(control['head'], 'control head').get('num_prototypes') != 6:
		raise ValueError('control head K mismatch')


def _validate_candidate_config_contract(
	config: F3MultiHeadPretrainingValidationConfig,
	candidate: Mapping[str, object],
	*,
	variant: str,
	model_tag: str,
	weight: float,
) -> None:
	paths = _mapping(candidate['paths'], 'paths')
	identity = _mapping(candidate['identity'], 'identity')
	if identity.get('model_tag') != model_tag:
		raise ValueError(f'{variant} model tag mismatch')
	output = Path(str(paths.get('output_root', ''))).resolve()
	if output != (config.experiment_root / model_tag).resolve():
		raise ValueError(f'{variant} output root mismatch')
	ensure_under_root(output, root=config.artifact_root, label=f'{variant}.output_root')
	if _mapping(candidate['loss'], 'loss').get('consistency_weight') != weight:
		raise ValueError(f'{variant} consistency weight mismatch')
	if (
		Path(
			str(
				_mapping(candidate['pseudo_targets'], 'pseudo_targets').get(
					'manifest', ''
				)
			)
		).resolve()
		!= config.target_manifest
	):
		raise ValueError(f'{variant} target manifest path mismatch')


def _validate_pair_config_contract(configs: Mapping[str, Mapping[str, object]]) -> None:
	left, right = (
		_pair_comparable(configs['nocons']),
		_pair_comparable(configs['cons010']),
	)
	if left != right:
		raise ValueError('pretraining scientific config drift outside allowed fields')


def _pair_comparable(value: Mapping[str, object]) -> object:
	"""Remove four pair-specific fields after binding the mirrored weight."""
	copy = json.loads(json.dumps(value))
	loss = _mapping(copy['loss'], 'loss')
	scientific = _mapping(
		_mapping(copy['identity'], 'identity')['scientific_identity'], 'scientific'
	)
	if scientific.get('consistency_weight') != loss.get('consistency_weight'):
		raise ValueError(
			'scientific consistency weight must match loss consistency weight'
		)
	loss.pop('consistency_weight', None)
	_mapping(copy['identity'], 'identity').pop('model_tag', None)
	scientific.pop('variant', None)
	# This is a required mirror of the permitted loss field, not a fifth
	# independently variable field.
	scientific.pop('consistency_weight', None)
	_mapping(copy['paths'], 'paths').pop('output_root', None)
	return copy


def _checkpoint_evidence(  # noqa: PLR0913
	config: F3MultiHeadPretrainingValidationConfig,
	training: Mapping[str, object],
	*,
	variant: str,
	model_tag: str,
	weight: float,
	expected_per_head_targets: Mapping[str, object],
) -> Mapping[str, object]:
	root = Path(str(_mapping(training['paths'], 'paths')['output_root']))
	latest_path, best_path = root / 'latest.pt', root / 'best.pt'
	if not latest_path.is_file() or not best_path.is_file():
		raise FileNotFoundError(f'{variant} requires latest.pt and best.pt')
	latest, best = _torch_mapping(latest_path), _torch_mapping(best_path)
	for payload in (latest, best):
		validate_stratigraphy_checkpoint_payload(payload)
		_identity_contract(
			config,
			training,
			payload,
			model_tag,
			weight,
			expected_per_head_targets,
		)
		_validate_initial_states(training, payload)
		_metrics_finite(payload)
	if latest.get('epoch') != 25 or latest.get('global_step') != 25600:
		raise ValueError(f'{variant} full run must finish epoch 25/global step 25600')
	if (
		_mapping(latest.get('training_state'), 'training_state').get('checkpoint_kind')
		!= 'epoch'
	):
		raise ValueError(f'{variant} latest checkpoint is not an epoch checkpoint')
	rows = _epoch_rows(root / 'multi_head_epoch_metrics.csv')
	if [row['epoch'] for row in rows] != list(range(1, 26)) or rows[-1][
		'global_step'
	] != 25600:
		raise ValueError(f'{variant} epoch metrics coverage is incomplete')
	_validate_best_selection(best, rows, variant=variant)
	_validate_freeze_contract(best, training)
	identity = _mapping(best['stratigraphy_checkpoint'], 'stratigraphy_checkpoint')
	return {
		'root': root,
		'best_path': best_path,
		'latest_path': latest_path,
		'best': best,
		'latest': latest,
		'identity': identity,
		'epoch_rows': rows,
	}


def _identity_contract(  # noqa: PLR0913
	config: F3MultiHeadPretrainingValidationConfig,
	training: Mapping[str, object],
	payload: Mapping[str, object],
	model_tag: str,
	weight: float,
	expected_per_head_targets: Mapping[str, object],
) -> None:
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'stratigraphy_checkpoint'
	)
	if (
		identity.get('model_tag') != model_tag
		or Path(str(identity.get('output_root', ''))).resolve()
		!= Path(str(_mapping(training['paths'], 'paths')['output_root'])).resolve()
	):
		raise ValueError('checkpoint model tag/output root mismatch')
	if identity.get(
		'head_spec'
	) != 'multi_resolution_ordered_prototypes_v1' or identity.get('head_ks') != [
		6,
		8,
		10,
	]:
		raise ValueError('checkpoint head specification/K mismatch')
	if (
		identity.get('consistency_weight') != weight
		or identity.get('consistency_policy') != 'normalized_order_smooth_l1_v1'
		or identity.get('consistency_beta') != 0.1
	):
		raise ValueError('checkpoint consistency identity mismatch')
	target = _mapping(identity.get('target_manifest'), 'checkpoint target manifest')
	if Path(
		str(target.get('path', ''))
	).resolve() != config.target_manifest or target.get('sha256') != file_sha256(
		config.target_manifest
	):
		raise ValueError('checkpoint target manifest mismatch')
	scientific_identity = _mapping(
		_mapping(training['identity'], 'identity').get('scientific_identity'),
		'scientific identity',
	)
	if scientific_identity.get('target_head_hashes') != expected_per_head_targets:
		raise ValueError(
			'scientific identity per-head target hashes do not match target manifest'
		)
	if identity.get('per_head_targets') != scientific_identity['target_head_hashes']:
		raise ValueError(
			'checkpoint per-head targets do not match scientific target hashes'
		)
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific_identity
	):
		raise ValueError(
			'checkpoint scientific identity does not match training config'
		)


def _validate_initial_states(
	training: Mapping[str, object], payload: Mapping[str, object]
) -> None:
	"""Bind recorded initialization hashes to the reproducible initial states."""
	seed = _mapping(training['train'], 'train').get('seed', 42)
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('train.seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('multi-head validation requires initialized multi-heads')
	_validate_initial_state_hashes(
		_mapping(payload['stratigraphy_checkpoint'], 'stratigraphy_checkpoint'),
		student_state=components.student.state_dict(),
		head_state=heads.state_dict(),
	)


def _validate_initial_state_hashes(
	identity: Mapping[str, object],
	*,
	student_state: Mapping[str, object],
	head_state: Mapping[str, object],
) -> None:
	"""Require identity hashes to describe the actual pre-optimization states."""
	if identity.get('initial_student_state_sha256') != _state_sha256(student_state):
		raise ValueError('checkpoint initial student state SHA-256 mismatch')
	if identity.get('initial_head_state_sha256') != _state_sha256(head_state):
		raise ValueError('checkpoint initial head state SHA-256 mismatch')


def _validate_pair(
	left: Mapping[str, object],
	right: Mapping[str, object],
	configs: Mapping[str, Mapping[str, object]],
) -> None:
	left_id, right_id = (
		_mapping(left['identity'], 'nocons identity'),
		_mapping(right['identity'], 'cons010 identity'),
	)
	for variant, identity in (('nocons', left_id), ('cons010', right_id)):
		scientific_identity = _mapping(
			_mapping(configs[variant]['identity'], f'{variant} identity').get(
				'scientific_identity'
			),
			f'{variant} scientific identity',
		)
		if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
			scientific_identity
		):
			raise ValueError(
				f'paired pretraining scientific identity mismatch: {variant}'
			)
	for key in (
		'teacher_checkpoint_sha256',
		'student_init_checkpoint_sha256',
		'target_manifest',
		'per_head_targets',
		'head_spec',
		'head_ks',
		'consistency_policy',
		'consistency_beta',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'optimizer_group_identity',
	):
		if left_id.get(key) != right_id.get(key):
			raise ValueError(f'paired pretraining identity mismatch: {key}')


def _validate_freeze_contract(
	payload: Mapping[str, object], training: Mapping[str, object]
) -> None:
	summary = _mapping(payload.get('trainability_summary'), 'trainability_summary')
	names = summary.get('trainable_names')
	if (
		not isinstance(names, list)
		or not names
		or any(
			not isinstance(name, str) or not name.startswith('encoder.layers.7.')
			for name in names
		)
		or len(names) != len(set(names))
	):
		raise ValueError('freeze contract requires only the student top block to train')
	identity = _mapping(payload['stratigraphy_checkpoint'], 'stratigraphy_checkpoint')
	state = _mapping(payload.get('stratigraphy_state_dict'), 'stratigraphy_state_dict')
	initial_head = _initial_head_parameter_hashes(payload)
	_validate_optimizer_contract(
		identity=identity,
		trainable_names=names,
		initial_head=initial_head,
	)
	_validate_head_updates(state, initial_head)
	if _mapping(training['student'], 'student').get('unfreeze_top_blocks') != 1:
		raise ValueError('freeze contract requires unfreeze_top_blocks=1')
	initial = _torch_mapping(
		Path(str(_mapping(training['student'], 'student')['init_checkpoint']))
	)
	initial_state = _mapping(initial.get('model_state_dict'), 'student init state')
	student_state = _mapping(payload.get('model_state_dict'), 'student state')
	if set(initial_state) != set(student_state):
		raise ValueError('freeze contract student state keys mismatch')
	top_changed = False
	for name, initial_value in initial_state.items():
		current = student_state[name]
		if not isinstance(initial_value, torch.Tensor) or not isinstance(
			current, torch.Tensor
		):
			raise TypeError('freeze contract student state must contain tensors')
		changed = not torch.equal(initial_value, current)
		if name.startswith('encoder.layers.7.'):
			top_changed = top_changed or changed
		elif changed:
			raise ValueError('freeze contract frozen student block was updated')
	if not top_changed:
		raise ValueError('freeze contract top student block was not updated')


def _initial_head_parameter_hashes(
	payload: Mapping[str, object],
) -> Mapping[str, object]:
	control = _mapping(payload.get('control_identity'), 'control_identity')
	initial = _mapping(
		control.get('initial_parameter_sha256'), 'initial_parameter_sha256'
	)
	head = _mapping(initial.get('prototype_head'), 'prototype_head')
	if not head or any(
		not isinstance(name, str) or not isinstance(digest, str) or not digest
		for name, digest in head.items()
	):
		raise ValueError('freeze contract initial prototype-head hashes are invalid')
	return head


def _validate_optimizer_contract(
	*,
	identity: Mapping[str, object],
	trainable_names: list[object],
	initial_head: Mapping[str, object],
) -> None:
	groups = identity.get('optimizer_group_identity')
	if not isinstance(groups, list) or len(groups) != 2:
		raise ValueError('freeze contract optimizer groups are invalid')
	if not all(isinstance(group, Mapping) for group in groups) or [
		group.get('name') for group in groups
	] != ['head', 'encoder']:
		raise ValueError('freeze contract optimizer groups are invalid')
	head_names = groups[0].get('parameter_names')
	encoder_names = groups[1].get('parameter_names')
	if not isinstance(head_names, list) or not isinstance(encoder_names, list):
		raise TypeError(
			'freeze contract optimizer parameters must appear exactly once'
		)
	parameter_names = [*head_names, *encoder_names]
	if any(not isinstance(name, str) for name in parameter_names) or len(
		parameter_names
	) != len(set(parameter_names)):
		raise ValueError(
			'freeze contract optimizer parameters must appear exactly once'
		)
	expected_head_names = {f'head.{name}' for name in initial_head}
	if set(head_names) != expected_head_names:
		raise ValueError(
			'freeze contract optimizer must contain every head parameter exactly once'
		)
	if set(encoder_names) != {f'student.{name}' for name in trainable_names}:
		raise ValueError(
			'freeze contract optimizer must contain every top-block parameter '
			'exactly once'
		)


def _validate_head_updates(
	state: Mapping[str, object], initial_head: Mapping[str, object]
) -> None:
	for head_k in (6, 8, 10):
		prefix = f'heads.k{head_k}.'
		head_names = [name for name in initial_head if name.startswith(prefix)]
		if not head_names:
			raise ValueError(f'freeze contract initial K={head_k} head is missing')
		if not any(
			name in state
			and isinstance(state[name], torch.Tensor)
			and _tensor_sha256(name, state[name]) != initial_head[name]
			for name in head_names
		):
			raise ValueError(f'freeze contract requires K={head_k} head to update')


def _embedding_evidence(
	config: F3MultiHeadPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
	model_tag: str,
	*,
	canonical_valid_tokens_sha256: str,
) -> Mapping[str, object]:
	root = (
		config.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ model_tag
		/ 'overlap_x16'
	)
	files = output_paths(root, 'f3_facies_benchmark')
	if (
		not files.embeddings.is_file()
		or not files.valid_tokens.is_file()
		or not files.metadata.is_file()
	):
		raise FileNotFoundError(f'{model_tag} complete embedding artifacts are missing')
	metadata = _json(files.metadata)
	best_path = Path(checkpoint['best_path'])
	if Path(
		str(metadata.get('checkpoint_path', ''))
	).resolve() != best_path.resolve() or metadata.get(
		'checkpoint_sha256'
	) != file_sha256(best_path):
		raise ValueError('embedding metadata does not bind selected best.pt')
	embeddings, valid = (
		np.load(files.embeddings, mmap_mode='r'),
		np.load(files.valid_tokens, mmap_mode='r'),
	)
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or int(valid.sum()) <= 0
	):
		raise ValueError('embedding shape/dtype/valid-token contract mismatch')
	if not np.isfinite(embeddings[valid]).all():
		raise ValueError('embeddings contain non-finite valid values')
	if file_sha256(files.valid_tokens) != canonical_valid_tokens_sha256:
		raise ValueError('embedding valid-token identity differs from current K6')
	identity = _mapping(checkpoint['identity'], 'checkpoint identity')
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'embedding stratigraphy identity'
	)
	for key in (
		'model_tag',
		'head_spec',
		'head_ks',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'scientific_identity_sha256',
	):
		if stratigraphy.get(key) != identity.get(key):
			raise ValueError(f'embedding stratigraphy identity mismatch: {key}')
	if stratigraphy.get('target_manifest_sha256') != _mapping(
		identity['target_manifest'], 'checkpoint target manifest'
	).get('sha256'):
		raise ValueError('embedding target manifest identity mismatch')
	return {
		'root': root,
		'metadata_path': files.metadata,
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_sha256': file_sha256(files.valid_tokens),
		'valid_token_count': int(valid.sum()),
	}


def _canonical_k6_valid_tokens_sha256(
	config: F3MultiHeadPretrainingValidationConfig,
	control: Mapping[str, object],
) -> str:
	"""Resolve the canonical K=6 extraction from the validated control config."""
	model_tag = _mapping(control['identity'], 'control identity').get('model_tag')
	if model_tag != _CURRENT_K6_MODEL_TAG:
		raise ValueError('control model tag mismatch')
	root = (
		config.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ model_tag
		/ 'overlap_x16'
	)
	path = output_paths(root, 'f3_facies_benchmark').valid_tokens
	if not path.is_file():
		raise FileNotFoundError(f'current K6 valid-token artifact is missing: {path}')
	return file_sha256(path)


def _handoff(
	variant: str,
	model_tag: str,
	evidence: Mapping[str, object],
) -> dict[str, object]:
	identity, embedding = (
		_mapping(evidence['identity'], 'identity'),
		_mapping(evidence['embedding'], 'embedding'),
	)
	best, latest = Path(evidence['best_path']), Path(evidence['latest_path'])
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': model_tag,
		'variant': variant,
		'checkpoint': {
			'path': str(best),
			'sha256': file_sha256(best),
			'latest_path': str(latest),
			'latest_sha256': file_sha256(latest),
			'best_epoch': evidence['best']['epoch'],
			'best_global_step': evidence['best']['global_step'],
			'selection_metric': 'metrics.loss',
		},
		'embedding': {
			'root': str(embedding['root']),
			'metadata_path': str(embedding['metadata_path']),
			'metadata_sha256': embedding['metadata_sha256'],
			'embeddings_sha256': embedding['embeddings_sha256'],
			'valid_tokens_sha256': embedding['valid_tokens_sha256'],
		},
		# Retained as the existing downstream field name; it is also bound above.
		'embedding_metadata_sha256': embedding['metadata_sha256'],
		'stratigraphy_pretext': {
			'head_spec': identity['head_spec'],
			'head_ks': identity['head_ks'],
			'target_manifest_path': identity['target_manifest']['path'],
			'target_manifest_sha256': identity['target_manifest']['sha256'],
			'per_head_target_sha256': identity['per_head_targets'],
			'consistency_policy': identity['consistency_policy'],
			'consistency_weight': identity['consistency_weight'],
			'consistency_beta': identity['consistency_beta'],
			'scientific_identity_sha256': identity['scientific_identity_sha256'],
			'initial_student_state_sha256': identity['initial_student_state_sha256'],
			'initial_head_state_sha256': identity['initial_head_state_sha256'],
		},
	}


def _publish_handoff(
	path: Path,
	handoff: Mapping[str, object],
	*,
	only_missing: bool,
	_quarantine_invalid: bool,
) -> bool:
	"""Publish a handoff, quarantining an invalid predecessor on request.

	``--only-missing`` is the sole reuse mode: it retains an exact live handoff
	without rewriting it.  A stale, partial, hash-mismatched, or wrong-variant
	predecessor requires ``--quarantine-invalid`` before it is preserved under a
	timestamped quarantine name and replaced.
	"""
	if path.is_file():
		try:
			existing = load_f3_multi_head_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff:
			if only_missing:
				return False
		else:
			if not _quarantine_invalid:
				raise ValueError(
					'existing handoff is stale or invalid; '
					'pass --quarantine-invalid to replace it'
				)
			_quarantine(path)
	_atomic_json(path, handoff)
	return True


def _epoch_rows(path: Path) -> list[dict[str, float | int]]:
	if not path.is_file():
		raise FileNotFoundError(f'epoch metrics are missing: {path}')
	with path.open(newline='', encoding='utf-8') as handle:
		rows = list(csv.DictReader(handle))
	if not rows or not {'epoch', 'global_step', 'loss'} <= set(rows[0]):
		raise ValueError('multi-head epoch metrics schema is invalid')
	result = []
	for row in rows:
		try:
			parsed = {
				'epoch': int(row['epoch']),
				'global_step': int(row['global_step']),
				**{
					key: float(value)
					for key, value in row.items()
					if key not in {'epoch', 'global_step'}
				},
			}
		except (TypeError, ValueError) as exc:
			raise ValueError('multi-head epoch metrics contain invalid values') from exc
		if not all(
			math.isfinite(value)
			for key, value in parsed.items()
			if key not in {'epoch', 'global_step'}
		):
			raise ValueError('multi-head epoch metrics contain non-finite values')
		result.append(parsed)
	return result


def _validate_best_selection(
	best: Mapping[str, object],
	rows: list[Mapping[str, float | int]],
	*,
	variant: str,
) -> None:
	"""Bind best.pt to the lowest-loss row recorded for the completed run."""
	best_epoch, best_global_step = best.get('epoch'), best.get('global_step')
	if (
		isinstance(best_epoch, bool)
		or not isinstance(best_epoch, int)
		or isinstance(best_global_step, bool)
		or not isinstance(best_global_step, int)
	):
		raise TypeError(f'{variant} best.pt epoch/global step must be integers')
	best_loss = _mapping(best.get('metrics'), 'best metrics').get('loss')
	if not _finite_number(best_loss):
		raise ValueError(f'{variant} best.pt metrics.loss must be finite')
	matching_rows = [
		row
		for row in rows
		if row['epoch'] == best_epoch and row['global_step'] == best_global_step
	]
	if (
		len(matching_rows) != 1
		or best_loss != matching_rows[0]['loss']
		or best_loss != min(row['loss'] for row in rows)
	):
		raise ValueError(
			f'{variant} best.pt is not selected by lowest finite metrics.loss'
		)


def _torch_mapping(path: Path) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'checkpoint must be a mapping: {path}')
	return payload


def _metrics_finite(payload: Mapping[str, object]) -> None:
	metrics = _mapping(payload.get('metrics'), 'checkpoint metrics')
	if not metrics or not all(
		isinstance(value, int | float)
		and not isinstance(value, bool)
		and math.isfinite(float(value))
		for value in metrics.values()
	):
		raise ValueError('checkpoint metrics must all be finite')


def _state_sha256(state: Mapping[str, object]) -> str:
	digest = hashlib.sha256()
	for name in sorted(state):
		value = state[name]
		if not isinstance(value, torch.Tensor):
			raise TypeError('state contains a non-tensor')
		cpu = value.detach().cpu().contiguous()
		digest.update(name.encode())
		digest.update(str(cpu.dtype).encode())
		digest.update(str(tuple(cpu.shape)).encode())
		digest.update(cpu.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _tensor_sha256(name: str, value: torch.Tensor) -> str:
	"""Match the per-parameter identity written before optimization."""
	cpu = value.detach().cpu().contiguous()
	digest = hashlib.sha256()
	digest.update(name.encode())
	digest.update(str(cpu.dtype).encode())
	digest.update(str(tuple(cpu.shape)).encode())
	digest.update(cpu.view(torch.uint8).numpy().tobytes())
	return digest.hexdigest()


def _report(phase: str, evidence: Mapping[str, object]) -> dict[str, object]:
	return {
		'artifact_type': 'f3_multi_head_pretraining_validation',
		'schema_version': 1,
		'phase': phase,
		'status': 'PASS',
		'validated_at': datetime.now(timezone.utc).isoformat(),
		'checkpoint_sha256': file_sha256(Path(evidence['best_path'])),
	}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as handle:
			json.dump(value, handle, indent=2, sort_keys=True)
			handle.write('\n')
			handle.flush()
			os.fsync(handle.fileno())
		Path(temporary).replace(path)
	finally:
		if Path(temporary).exists():
			Path(temporary).unlink()


def _quarantine(path: Path) -> Path:
	target = path.with_name(
		f'{path.name}.quarantine.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
	)
	# Retain the canonical predecessor until its fully fsynced replacement is
	# atomically installed.  The quarantine link keeps the stale evidence after
	# replacement without a window in which a write failure removes the only
	# canonical handoff.
	os.link(path, target)
	return target


def _json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as handle:
		value = json.load(handle)
	return _mapping(value, str(path))


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _finite_number(value: object) -> bool:
	return (
		isinstance(value, int | float)
		and not isinstance(value, bool)
		and math.isfinite(float(value))
	)


def _is_sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _per_head_target_hashes(value: object) -> None:
	"""Require every K=6/8/10 target hash recorded by the checkpoint identity."""
	per_head = _mapping(value, 'handoff per_head_target_sha256')
	if set(per_head) != {'6', '8', '10'}:
		raise ValueError('handoff per_head_target_sha256 K keys mismatch')
	for head_k, surveys in per_head.items():
		survey_hashes = _mapping(
			surveys, f'handoff per_head_target_sha256.{head_k}'
		)
		if not survey_hashes:
			raise ValueError(
				f'handoff per_head_target_sha256.{head_k} must not be empty'
			)
		for survey_id, artifacts in survey_hashes.items():
			digests = _mapping(
				artifacts,
				f'handoff per_head_target_sha256.{head_k}.{survey_id}',
			)
			if not digests or not all(
				_is_sha256(digest) for digest in digests.values()
			):
				raise ValueError(
					'handoff per_head_target_sha256 contains an empty or invalid '
					'digest'
				)


def _manifest_per_head_target_hashes(
	manifest: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract the checkpoint identity hashes from the canonical manifest."""
	heads = _mapping(manifest.get('heads'), 'target manifest heads')
	result: dict[str, dict[str, dict[str, str]]] = {}
	for head_k in (6, 8, 10):
		head = _mapping(heads.get(str(head_k)), f'target manifest K={head_k}')
		surveys = _mapping(head.get('surveys'), f'target manifest K={head_k} surveys')
		result[str(head_k)] = {}
		for survey_id, entry in surveys.items():
			target = _mapping(entry, f'target manifest K={head_k} survey {survey_id}')
			result[str(head_k)][str(survey_id)] = {}
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				digest = _mapping(
					target.get(name),
					f'target manifest K={head_k} survey {survey_id} {name}',
				).get('sha256')
				if not _is_sha256(digest):
					raise ValueError(
						f'target manifest K={head_k} survey {survey_id} {name} '
						'must have a SHA-256'
					)
				result[str(head_k)][str(survey_id)][name] = digest
	return result
