"""Validate the independent F3 XY-neighbour hard-label pretraining handoff.

The validator deliberately binds only the immutable consensus hard-target
publication. It does not accept posterior, affinity, emission, Viterbi, or
beta-calibration evidence from M5-U or M5-LS.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.pretraining import (
	_multi_head_target_hashes,
	_xy_neighbor_consensus_smoothing_identity,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
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
		'xy_neighbor_consensus_smoke_config',
		'xy_neighbor_consensus_full_config',
	}
)
_TARGET_REPRESENTATION = 'xy_neighbor_consensus_hard_labels_v1'
_TARGET_SEMANTICS = 'xy_neighbor_consensus_hard_label_smoothing_v1'
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
_VARIANT = 'xycons1_nocons'
_HANDOFF_TYPE = 'f3_xy_neighbor_consensus_pretraining_handoff'
_CONSISTENCY_POLICY = 'disabled_for_xy_neighbor_consensus_v1'
_EMBEDDING_STRATIGRAPHY_FIELDS = frozenset(
	{
		'method',
		'base_objective',
		'head_spec',
		'head_ks',
		'head_count',
		'unfreeze_top_blocks',
		'distillation_weight',
		'prototype_weight',
		'prototype_weight_semantics',
		'usage_weight',
		'usage_weight_semantics',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'scientific_identity_sha256',
		'checkpoint_stratigraphy_state_sha256',
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_path',
		'xy_neighbor_consensus_target_manifest_sha256',
		'per_head_xy_neighbor_consensus_target_sha256',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
	}
)


@dataclass(frozen=True)
class F3XYNeighborConsensusPretrainingValidationConfig:
	"""All fixed locations required by the closed successor validation schema."""

	artifact_root: Path
	experiment_root: Path
	target_manifest: Path
	xy_neighbor_consensus_smoke_config: Path
	xy_neighbor_consensus_full_config: Path


@dataclass(frozen=True)
class F3XYNeighborConsensusPretrainingValidationResult:
	"""One preflight phase and an optional final immutable handoff path."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3XYNeighborConsensusPretrainingValidationConfig:
	"""Resolve the intentionally non-extensible F3 consensus validator config."""
	if not isinstance(config, Mapping):
		raise TypeError('XY-neighbour consensus validation config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			'unknown XY-neighbour consensus validation config keys: '
			f'{sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			'missing XY-neighbour consensus validation config keys: '
			f'{sorted(missing)!r}'
		)

	def path(name: str) -> Path:
		value = config[name]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path string')
		return Path(value).resolve()

	result = F3XYNeighborConsensusPretrainingValidationConfig(
		artifact_root=path('artifact_root'),
		experiment_root=path('experiment_root'),
		target_manifest=path('target_manifest'),
		xy_neighbor_consensus_smoke_config=path('xy_neighbor_consensus_smoke_config'),
		xy_neighbor_consensus_full_config=path('xy_neighbor_consensus_full_config'),
	)
	if (
		not result.artifact_root.is_absolute()
		or not result.experiment_root.is_absolute()
	):
		raise ValueError('artifact_root and experiment_root must be absolute')
	for name, value in (
		('target_manifest', result.target_manifest),
		(
			'xy_neighbor_consensus_smoke_config',
			result.xy_neighbor_consensus_smoke_config,
		),
		('xy_neighbor_consensus_full_config', result.xy_neighbor_consensus_full_config),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{name} is missing: {value}')
	return result


def load_f3_xy_neighbor_consensus_pretraining_validation_config(
	path: str | Path,
) -> F3XYNeighborConsensusPretrainingValidationConfig:
	"""Load the F3 consensus preflight configuration from YAML."""
	return f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping(
		load_config(path)
	)


def validate_f3_xy_neighbor_consensus_pretraining(
	config: F3XYNeighborConsensusPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3XYNeighborConsensusPretrainingValidationResult:
	"""Validate target, smoke, full run, extraction, and publish a PASS handoff."""
	if phase not in {'targets', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be targets, smoke, checkpoints, or complete')
	try:
		target = load_multi_head_xy_neighbor_consensus_target_manifest(
			config.target_manifest
		)
		full = _training_config(config.xy_neighbor_consensus_full_config)
		target_evidence = _target_evidence(config, target=target, full=full)
		if phase == 'targets':
			return F3XYNeighborConsensusPretrainingValidationResult(
				phase, {'status': 'PASS', **target_evidence}, None
			)

		smoke = _training_config(config.xy_neighbor_consensus_smoke_config)
		_smoke_config_contract(config, full=full, smoke=smoke)
		if phase == 'smoke':
			smoke_evidence = _checkpoint_evidence(
				smoke, expected_global_step=2, require_best=False
			)
			return F3XYNeighborConsensusPretrainingValidationResult(
				phase,
				{'status': 'PASS', **target_evidence, 'smoke': smoke_evidence},
				None,
			)

		checkpoint = _checkpoint_evidence(
			full, expected_global_step=25600, require_best=True
		)
		evidence: dict[str, object] = {
			'status': 'PASS',
			**target_evidence,
			**checkpoint,
		}
		if phase == 'checkpoints':
			if not dry_run:
				_atomic_json(
					Path(checkpoint['root'])
					/ 'preflight'
					/ 'xy_neighbor_consensus_checkpoint_validation.json',
					{
						'artifact_type': 'f3_xy_neighbor_consensus_validation',
						'schema_version': 1,
						'phase': phase,
						'status': 'PASS',
						'target_manifest_sha256': target_evidence['target_manifest'][
							'sha256'
						],
					},
				)
			return F3XYNeighborConsensusPretrainingValidationResult(
				phase, evidence, None
			)

		evidence['embedding'] = _embedding_evidence(config, checkpoint)
		handoff = _handoff(evidence)
		path = (
			Path(checkpoint['root'])
			/ 'preflight'
			/ 'xy_neighbor_consensus_handoff.json'
		)
		if dry_run:
			return F3XYNeighborConsensusPretrainingValidationResult(
				phase, evidence, None
			)
		published = _publish_handoff(
			path,
			handoff,
			only_missing=only_missing,
			quarantine_invalid=quarantine_invalid,
		)
		return F3XYNeighborConsensusPretrainingValidationResult(
			phase, evidence, path if published else None
		)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if not dry_run:
			raise
		return F3XYNeighborConsensusPretrainingValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def load_f3_xy_neighbor_consensus_pretraining_handoff(  # noqa: C901, PLR0912
	path: str | Path,
) -> Mapping[str, object]:
	"""Load a complete successor PASS handoff without accepting M5 artifacts."""
	payload = _mapping(_json(Path(path)), 'XY-neighbour consensus handoff')
	if set(payload) != {
		'artifact_type',
		'schema_version',
		'status',
		'model_tag',
		'variant',
		'targets',
		'checkpoint',
		'embedding',
	}:
		raise ValueError('XY-neighbour consensus handoff keys mismatch')
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
		or payload.get('model_tag') != _MODEL_TAG
		or payload.get('variant') != _VARIANT
	):
		raise ValueError('XY-neighbour consensus handoff identity mismatch')
	targets = _mapping(payload.get('targets'), 'handoff targets')
	if (
		targets.get('target_representation') != _TARGET_REPRESENTATION
		or targets.get('target_semantics') != _TARGET_SEMANTICS
		or targets.get('consistency_policy') != _CONSISTENCY_POLICY
	):
		raise ValueError('XY-neighbour consensus handoff target contract mismatch')
	if set(targets) != {
		'target_representation',
		'target_semantics',
		'consistency_policy',
		'target_manifest',
		'xy_neighbor_consensus_target_head_hashes',
		'source_hard_manifest',
		'xy_neighbor_consensus_smoothing',
		'temporal_transition_counts',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	}:
		raise ValueError('XY-neighbour consensus handoff target keys mismatch')
	for key in ('target_manifest', 'source_hard_manifest'):
		_reference(targets.get(key), f'handoff targets.{key}')
	if not isinstance(targets.get('xy_neighbor_consensus_smoothing'), Mapping):
		raise TypeError('handoff consensus smoothing is missing')
	_validate_handoff_temporal_transition_counts(
		targets.get('temporal_transition_counts')
	)
	if not _sha256(targets.get('initial_student_state_sha256')) or not _sha256(
		targets.get('initial_head_state_sha256')
	):
		raise TypeError('handoff initial-state hashes are missing')
	_validate_handoff_target_head_hashes(
		targets.get('xy_neighbor_consensus_target_head_hashes')
	)
	checkpoint = _mapping(payload.get('checkpoint'), 'handoff checkpoint')
	if set(checkpoint) != {
		'path',
		'sha256',
		'selected_checkpoint_kind',
		'selected_epoch',
		'selected_global_step',
		'selected_loss',
	}:
		raise ValueError('XY-neighbour consensus handoff checkpoint keys mismatch')
	for key in ('path', 'sha256', 'selected_checkpoint_kind'):
		if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
			raise TypeError(f'handoff checkpoint.{key} is missing')
	if not _sha256(checkpoint['sha256']):
		raise TypeError('handoff checkpoint.sha256 is invalid')
	if checkpoint['selected_checkpoint_kind'] not in {'step', 'epoch'}:
		raise ValueError('handoff checkpoint kind is invalid')
	for key in ('selected_epoch', 'selected_global_step'):
		value = checkpoint.get(key)
		if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
			raise TypeError(f'handoff checkpoint.{key} must be a positive integer')
	selected_loss = checkpoint.get('selected_loss')
	if (
		isinstance(selected_loss, bool)
		or not isinstance(selected_loss, int | float)
		or not math.isfinite(float(selected_loss))
	):
		raise TypeError('handoff checkpoint.selected_loss must be finite')
	embedding = _mapping(payload.get('embedding'), 'handoff embedding')
	if set(embedding) != {
		'root',
		'metadata_path',
		'metadata_sha256',
		'embeddings_sha256',
		'valid_tokens_sha256',
		'valid_token_count',
	}:
		raise ValueError('XY-neighbour consensus handoff embedding keys mismatch')
	for key in (
		'root',
		'metadata_path',
		'metadata_sha256',
		'embeddings_sha256',
		'valid_tokens_sha256',
	):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	for key in ('metadata_sha256', 'embeddings_sha256', 'valid_tokens_sha256'):
		if not _sha256(embedding[key]):
			raise TypeError(f'handoff embedding.{key} is invalid')
	if (
		isinstance(embedding.get('valid_token_count'), bool)
		or not isinstance(embedding.get('valid_token_count'), int)
		or embedding['valid_token_count'] <= 0
	):
		raise TypeError('handoff embedding.valid_token_count must be positive')
	return payload


def _training_config(path: Path) -> Mapping[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(path))


def _target_evidence(  # noqa: C901, PLR0912
	config: F3XYNeighborConsensusPretrainingValidationConfig,
	*,
	target: Mapping[str, object],
	full: Mapping[str, object],
) -> dict[str, object]:
	if target.get('head_ks') != [6, 8, 10]:
		raise ValueError('XY-neighbour consensus target K identity mismatch')
	if target.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('XY-neighbour consensus target semantics mismatch')
	full_identity = _training_identity(full, 'full')
	if _manifest_path(full) != config.target_manifest:
		raise ValueError('full training target manifest path mismatch')
	if _model_tag(full, 'full') != _MODEL_TAG:
		raise ValueError('full training model tag mismatch')
	if (
		Path(str(_mapping(full['paths'], 'full paths')['output_root'])).resolve()
		!= (config.experiment_root / _MODEL_TAG).resolve()
	):
		raise ValueError('full training output root mismatch')
	if full_identity.get('target_representation') != _TARGET_REPRESENTATION:
		raise ValueError('full training target representation mismatch')
	if full_identity.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('full training target semantics mismatch')
	if full_identity.get('consistency_policy') != _CONSISTENCY_POLICY:
		raise ValueError('full training consistency policy mismatch')
	if (
		full_identity.get('consistency_weight') != 0.0
		or _mapping(full['loss'], 'full loss').get('consistency_weight') != 0.0
	):
		raise ValueError('XY-neighbour consensus consistency must be disabled')
	if full_identity.get('supervised_loss') != 'structured_hmm_hard_categorical_v1':
		raise ValueError('full training hard-loss identity mismatch')
	if (
		_mapping(full['pseudo_targets'], 'full pseudo targets').get('min_confidence')
		!= 0.0
	):
		raise ValueError('full training min_confidence must be zero')
	manifest_sha256 = file_sha256(config.target_manifest)
	if (
		full_identity.get('xy_neighbor_consensus_target_manifest_sha256')
		!= manifest_sha256
	):
		raise ValueError('consensus target manifest SHA-256 mismatch')
	if full_identity.get(
		'xy_neighbor_consensus_target_head_hashes'
	) != _multi_head_target_hashes(target):
		raise ValueError('consensus target head hashes mismatch')
	source_hard = _reference(target.get('source_hard_manifest'), 'source hard manifest')
	if full_identity.get('source_hard_manifest_sha256') != source_hard['sha256']:
		raise ValueError('source hard manifest SHA-256 mismatch')
	if full_identity.get(
		'xy_neighbor_consensus_smoothing'
	) != _xy_neighbor_consensus_smoothing_identity(target):
		raise ValueError('consensus smoothing policy identity mismatch')
	temporal_transition_counts = _target_temporal_transition_counts(target)
	for forbidden in ('source_posterior_manifest_sha256', 'lateral_smoothing'):
		if forbidden in full_identity:
			raise ValueError(f'consensus training must not carry {forbidden}')
	return {
		'target_representation': _TARGET_REPRESENTATION,
		'target_semantics': _TARGET_SEMANTICS,
		'consistency_policy': _CONSISTENCY_POLICY,
		'target_manifest': {
			'path': str(config.target_manifest),
			'sha256': manifest_sha256,
		},
		'xy_neighbor_consensus_target_head_hashes': _multi_head_target_hashes(target),
		'source_hard_manifest': source_hard,
		'xy_neighbor_consensus_smoothing': _xy_neighbor_consensus_smoothing_identity(
			target
		),
		'temporal_transition_counts': temporal_transition_counts,
	}


def _target_temporal_transition_counts(
	target: Mapping[str, object],
) -> dict[str, dict[str, int]]:
	"""Read per-head source/output transition diagnostics from a target manifest.

	The counts are recorded for observation only.  In particular, this function
	does not compare source and output values because an increase is permitted.
	"""
	if target.get('head_ks') != [6, 8, 10]:
		raise ValueError('XY-neighbour consensus target K identity mismatch')
	heads = _mapping(target.get('heads'), 'XY-neighbour consensus target heads')
	counts: dict[str, dict[str, int]] = {}
	for k in (6, 8, 10):
		head = _mapping(heads.get(str(k)), f'XY-neighbour consensus target head k={k}')
		diagnostics = _mapping(
			head.get('diagnostics'),
			f'XY-neighbour consensus target diagnostics k={k}',
		)
		aggregate = _mapping(
			diagnostics.get('aggregate'),
			f'XY-neighbour consensus aggregate diagnostics k={k}',
		)
		counts[str(k)] = _temporal_transition_count_pair(
			aggregate.get('temporal_transition_counts'),
			f'XY-neighbour consensus aggregate transition counts k={k}',
		)
	return counts


def _validate_handoff_temporal_transition_counts(value: object) -> None:
	"""Validate diagnostic-only source/output transition counts in a handoff."""
	counts = _mapping(value, 'handoff temporal transition counts')
	if set(counts) != {'6', '8', '10'}:
		raise ValueError('handoff temporal transition counts must contain K=6/8/10')
	for k in (6, 8, 10):
		_temporal_transition_count_pair(
			counts.get(str(k)),
			f'handoff temporal transition counts k={k}',
		)


def _temporal_transition_count_pair(
	value: object,
	label: str,
) -> dict[str, int]:
	counts = _mapping(value, label)
	if set(counts) != {'source', 'output'}:
		raise ValueError(f'{label} must contain source and output')
	result: dict[str, int] = {}
	for name in ('source', 'output'):
		count = counts[name]
		if isinstance(count, bool) or not isinstance(count, int) or count < 0:
			raise TypeError(f'{label}.{name} must be a nonnegative integer')
		result[name] = count
	return result


def _smoke_config_contract(
	config: F3XYNeighborConsensusPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	if _manifest_path(smoke) != config.target_manifest:
		raise ValueError('smoke target manifest path mismatch')
	if _model_tag(smoke, 'smoke') != _MODEL_TAG:
		raise ValueError('smoke model tag mismatch')
	if _mapping(smoke['loss'], 'smoke loss').get('consistency_weight') != 0.0:
		raise ValueError('smoke consistency must be disabled')
	if _mapping(smoke['train'], 'smoke train').get('device') != 'cpu':
		raise ValueError('smoke train device must be cpu')
	runtime = _mapping(
		_mapping(smoke['identity'], 'smoke identity').get('runtime_identity'),
		'smoke runtime identity',
	)
	if runtime.get('device') != 'cpu':
		raise ValueError('smoke runtime identity device must be cpu')
	output = Path(str(_mapping(smoke['paths'], 'smoke paths')['output_root'])).resolve()
	if (
		output
		== Path(str(_mapping(full['paths'], 'full paths')['output_root'])).resolve()
	):
		raise ValueError('smoke and full output roots must differ')
	if _mapping(smoke['train'], 'smoke train').get('max_steps') != 2:
		raise ValueError('smoke max_steps must be exactly 2')
	_validate_smoke_full_config_equivalence(full, smoke)


def _validate_smoke_full_config_equivalence(
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	"""Allow only isolated root, CPU device, and two-step execution differences."""
	left, right = json.loads(json.dumps(full)), json.loads(json.dumps(smoke))
	for value in (left, right):
		paths = _mapping(value['paths'], 'paths')
		paths.pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		runtime = identity.get('runtime_identity')
		if runtime is not None:
			_mapping(runtime, 'runtime identity').pop('device', None)
		train = _mapping(value['train'], 'train')
		train.pop('device', None)
		train.pop('max_steps', None)
		_mapping(
			_mapping(identity['scientific_identity'], 'scientific identity')['train'],
			'scientific train identity',
		).pop('max_steps', None)
	if left != right:
		raise ValueError(
			'smoke/full consensus config drift outside CPU two-step settings'
		)


def _checkpoint_evidence(
	training: Mapping[str, object], *, expected_global_step: int, require_best: bool
) -> dict[str, object]:
	root = Path(str(_mapping(training['paths'], 'paths')['output_root']))
	latest_path = root / 'latest.pt'
	best_path = root / 'best.pt'
	if not latest_path.is_file():
		raise FileNotFoundError(f'latest checkpoint is missing: {latest_path}')
	if require_best and not best_path.is_file():
		raise FileNotFoundError(f'best checkpoint is missing: {best_path}')
	latest = _checkpoint(latest_path, expected_config=training)
	if latest.get('global_step') != expected_global_step:
		raise ValueError(
			f'latest global_step must be {expected_global_step}; '
			f'got {latest.get("global_step")!r}'
		)
	selected_path, selected = latest_path, latest
	if best_path.is_file():
		best = _checkpoint(best_path, expected_config=training)
		selected_path, selected = best_path, best
	else:
		best = None
	for payload in (latest, best):
		if payload is None:
			continue
		hard_validation._metrics_finite(payload)  # noqa: SLF001
		_validate_hard_label_checkpoint_metrics(payload)
	if require_best:
		if best is None:  # pragma: no cover - checked before checkpoint loading
			raise AssertionError('full consensus validation requires best.pt')
		epoch_rows = _validate_full_checkpoint_progress(
			latest,
			root=root,
			expected_global_step=expected_global_step,
		)
		checkpoint_selection = hard_validation._validate_best_selection(  # noqa: SLF001
			best,
			latest,
			variant=_VARIANT,
		)
	else:
		_validate_smoke_checkpoint_progress(latest, expected_global_step)
		epoch_rows = []
		checkpoint_selection = None
	identity = _mapping(selected.get('stratigraphy_checkpoint'), 'checkpoint identity')
	_scientific_checkpoint_contract(training, identity)
	return {
		'root': str(root),
		'latest_path': str(latest_path),
		'latest_sha256': file_sha256(latest_path),
		'best_path': None if best is None else str(best_path),
		'best_sha256': None if best is None else file_sha256(best_path),
		'selected_path': str(selected_path),
		'selected_sha256': file_sha256(selected_path),
		'selected_checkpoint_kind': _checkpoint_kind(selected),
		'selected_epoch': selected.get('epoch'),
		'selected_global_step': selected.get('global_step'),
		'selected_loss': _mapping(selected.get('metrics'), 'checkpoint metrics').get(
			'loss'
		),
		'initial_student_state_sha256': identity.get('initial_student_state_sha256'),
		'initial_head_state_sha256': identity.get('initial_head_state_sha256'),
		'identity': identity,
		'epoch_rows': epoch_rows,
		'checkpoint_selection': checkpoint_selection,
	}


def _validate_full_checkpoint_progress(
	latest: Mapping[str, object],
	*,
	root: Path,
	expected_global_step: int,
) -> list[dict[str, float | int]]:
	if latest.get('epoch') != 25 or latest.get('global_step') != expected_global_step:
		raise ValueError('consensus full run must finish epoch 25/global step 25600')
	if _checkpoint_kind(latest) != 'epoch':
		raise ValueError('consensus full latest checkpoint must be an epoch checkpoint')
	rows = hard_validation._epoch_rows(root / 'multi_head_epoch_metrics.csv')  # noqa: SLF001
	if [row['epoch'] for row in rows] != list(range(1, 26)) or rows[-1][
		'global_step'
	] != expected_global_step:
		raise ValueError('consensus epoch metrics coverage is incomplete')
	return rows


def _validate_smoke_checkpoint_progress(
	latest: Mapping[str, object],
	expected_global_step: int,
) -> None:
	if latest.get('global_step') != expected_global_step:
		raise ValueError(
			f'consensus smoke must finish at global step {expected_global_step}'
		)
	if latest.get('epoch') != 1 or _checkpoint_kind(latest) != 'step':
		raise ValueError('consensus smoke must end with a two-step partial checkpoint')


def _validate_hard_label_checkpoint_metrics(payload: Mapping[str, object]) -> None:
	metrics = _mapping(payload.get('metrics'), 'checkpoint metrics')
	if 'loss_consistency' not in metrics:
		raise ValueError(
			'consensus checkpoint did not use the hard multi-head loss path'
		)
	if any('posterior' in str(name) for name in metrics):
		raise ValueError('consensus checkpoint contains posterior loss-path metrics')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'),
		'checkpoint identity',
	)
	if identity.get('consistency_weight') != 0.0:
		raise ValueError('consensus checkpoint consistency weight must be zero')


def _checkpoint(
	path: Path,
	*,
	expected_config: Mapping[str, object],
) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError(f'checkpoint must be a mapping: {path}')
	_validate_xy_neighbor_consensus_checkpoint_payload(
		payload,
		expected_config=expected_config,
	)
	return payload


def _validate_xy_neighbor_consensus_checkpoint_payload(
	payload: Mapping[str, object],
	*,
	expected_config: Mapping[str, object],
) -> None:
	"""Require every saved checkpoint to match this successor's resolved config."""
	validate_stratigraphy_checkpoint_payload(
		payload,
		expected_config=expected_config,
	)
	if (
		_mapping(payload.get('stratigraphy_config'), 'checkpoint config')
		!= expected_config
	):
		raise ValueError('consensus checkpoint config differs from resolved config')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'),
		'checkpoint identity',
	)
	_scientific_checkpoint_contract(expected_config, identity)


def _scientific_checkpoint_contract(
	training: Mapping[str, object], identity: Mapping[str, object]
) -> None:
	scientific = _training_identity(training, 'checkpoint')
	if identity.get('schema_version') != 5:
		raise ValueError('XY-neighbour consensus checkpoint requires schema_version 5')
	for key in (
		'target_representation',
		'target_semantics',
		'xy_neighbor_consensus_target_manifest_sha256',
		'source_hard_manifest_sha256',
		'xy_neighbor_consensus_smoothing',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'model_tag',
		'output_root',
	):
		expected = (
			_mapping(training['paths'], 'paths').get('output_root')
			if key == 'output_root'
			else _mapping(training['identity'], 'identity').get('model_tag')
			if key == 'model_tag'
			else scientific.get(key)
		)
		if identity.get(key) != expected:
			raise ValueError(f'consensus checkpoint identity mismatch: {key}')
	if identity.get('per_head_xy_neighbor_consensus_targets') != scientific.get(
		'xy_neighbor_consensus_target_head_hashes'
	):
		raise ValueError('consensus checkpoint head hashes mismatch')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		scientific
	):
		raise ValueError('consensus checkpoint scientific identity hash mismatch')


def _embedding_evidence(
	config: F3XYNeighborConsensusPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	root = (
		config.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ _MODEL_TAG
		/ 'overlap_x16'
	)
	files = output_paths(root, 'f3_facies_benchmark')
	if not (
		files.embeddings.is_file()
		and files.valid_tokens.is_file()
		and files.metadata.is_file()
	):
		raise FileNotFoundError('consensus embedding artifacts are incomplete')
	metadata = _mapping(_json(files.metadata), 'embedding metadata')
	selected_path = Path(str(checkpoint['selected_path']))
	if (
		Path(str(metadata.get('checkpoint_path', ''))).resolve()
		!= selected_path.resolve()
	):
		raise ValueError('embedding metadata checkpoint path mismatch')
	if metadata.get('checkpoint_sha256') != file_sha256(selected_path):
		raise ValueError('embedding metadata checkpoint SHA-256 mismatch')
	_validate_embedding_stratigraphy_identity(
		metadata,
		_mapping(checkpoint['identity'], 'checkpoint identity'),
	)
	embeddings = np.load(files.embeddings, mmap_mode='r', allow_pickle=False)
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or not int(valid.sum())
		or not np.isfinite(embeddings[valid]).all()
	):
		raise ValueError('consensus embedding array contract mismatch')
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_sha256': file_sha256(files.valid_tokens),
		'valid_token_count': int(valid.sum()),
	}


def _validate_embedding_stratigraphy_identity(
	metadata: Mapping[str, object],
	checkpoint_identity: Mapping[str, object],
) -> None:
	"""Bind extraction metadata to the schema-v5 consensus checkpoint identity."""
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'),
		'embedding stratigraphy identity',
	)
	unknown = set(stratigraphy) - _EMBEDDING_STRATIGRAPHY_FIELDS
	missing = _EMBEDDING_STRATIGRAPHY_FIELDS - set(stratigraphy)
	if unknown or missing:
		raise ValueError(
			'consensus embedding metadata keys mismatch; '
			f'missing={sorted(missing)!r}; unknown={sorted(unknown)!r}'
		)
	head_ks = checkpoint_identity.get('head_ks')
	if not isinstance(head_ks, list):
		raise TypeError('checkpoint XY-neighbour consensus head_ks must be a list')
	for metadata_key, expected in (
		('method', 'strat_hmm_multi_head_pretext'),
		('base_objective', 'amp_mae3d'),
		('head_spec', checkpoint_identity.get('head_spec')),
		('head_ks', head_ks),
		('head_count', len(head_ks)),
		('consistency_policy', checkpoint_identity.get('consistency_policy')),
		('consistency_weight', checkpoint_identity.get('consistency_weight')),
		('consistency_beta', checkpoint_identity.get('consistency_beta')),
	):
		if stratigraphy.get(metadata_key) != expected:
			raise ValueError(
				f'embedding stratigraphy identity mismatch: {metadata_key}'
			)
	target_manifest = _mapping(
		checkpoint_identity.get('xy_neighbor_consensus_target_manifest'),
		'checkpoint XY-neighbour consensus target manifest',
	)
	target_manifest_path = target_manifest.get('path')
	if (
		stratigraphy.get('xy_neighbor_consensus_target_manifest_path')
		!= target_manifest_path
	):
		raise ValueError(
			'embedding stratigraphy identity mismatch: '
			'xy_neighbor_consensus_target_manifest_path'
		)
	checkpoint_state_sha256 = checkpoint_identity.get('stratigraphy_state_sha256')
	embedding_state_sha256 = stratigraphy.get('checkpoint_stratigraphy_state_sha256')
	if embedding_state_sha256 != checkpoint_state_sha256:
		raise ValueError(
			'embedding stratigraphy identity mismatch: '
			'checkpoint_stratigraphy_state_sha256'
		)
	for metadata_key, checkpoint_key in (
		('model_tag', 'model_tag'),
		('target_representation', 'target_representation'),
		('target_semantics', 'target_semantics'),
		(
			'xy_neighbor_consensus_target_manifest_sha256',
			'xy_neighbor_consensus_target_manifest_sha256',
		),
		(
			'per_head_xy_neighbor_consensus_target_sha256',
			'per_head_xy_neighbor_consensus_targets',
		),
		('source_hard_manifest_sha256', 'source_hard_manifest_sha256'),
		('xy_neighbor_consensus_smoothing', 'xy_neighbor_consensus_smoothing'),
		('scientific_identity_sha256', 'scientific_identity_sha256'),
	):
		if stratigraphy.get(metadata_key) != checkpoint_identity.get(checkpoint_key):
			raise ValueError(
				f'embedding stratigraphy identity mismatch: {metadata_key}'
			)


def _validate_handoff_target_head_hashes(value: object) -> None:
	"""Validate the complete K=6/8/10 artifact-hash mapping in a handoff."""
	head_hashes = _mapping(value, 'handoff target head hashes')
	if set(head_hashes) != {'6', '8', '10'}:
		raise ValueError('handoff target head hashes must contain K=6/8/10')
	for k, surveys_value in head_hashes.items():
		surveys = _mapping(surveys_value, f'handoff target head k={k}')
		if not surveys:
			raise ValueError(f'handoff target head k={k} must contain surveys')
		for survey_id, artifacts_value in surveys.items():
			if not isinstance(survey_id, str) or not survey_id:
				raise TypeError(
					'handoff target survey identifiers must be non-empty strings'
				)
			artifacts = _mapping(
				artifacts_value,
				f'handoff target head k={k} survey={survey_id}',
			)
			if set(artifacts) != {'labels', 'confidence', 'valid_tokens', 'metadata'}:
				raise ValueError(
					'handoff target artifact hashes must contain labels, confidence, '
					'valid_tokens, and metadata'
				)
			for name, digest in artifacts.items():
				if not _sha256(digest):
					raise TypeError(
						f'handoff target artifact hash is invalid: '
						f'k={k}, survey={survey_id}, artifact={name}'
					)


def _handoff(evidence: Mapping[str, object]) -> dict[str, object]:
	target = _mapping(evidence['target_manifest'], 'target evidence')
	checkpoint = _mapping(evidence, 'checkpoint evidence')
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': _VARIANT,
		'targets': {
			'target_representation': evidence['target_representation'],
			'target_semantics': evidence['target_semantics'],
			'consistency_policy': evidence['consistency_policy'],
			'target_manifest': target,
			'xy_neighbor_consensus_target_head_hashes': evidence[
				'xy_neighbor_consensus_target_head_hashes'
			],
			'source_hard_manifest': evidence['source_hard_manifest'],
			'xy_neighbor_consensus_smoothing': evidence[
				'xy_neighbor_consensus_smoothing'
			],
			'temporal_transition_counts': evidence['temporal_transition_counts'],
			'initial_student_state_sha256': checkpoint['initial_student_state_sha256'],
			'initial_head_state_sha256': checkpoint['initial_head_state_sha256'],
		},
		'checkpoint': {
			'path': checkpoint['selected_path'],
			'sha256': checkpoint['selected_sha256'],
			'selected_checkpoint_kind': checkpoint['selected_checkpoint_kind'],
			'selected_epoch': checkpoint['selected_epoch'],
			'selected_global_step': checkpoint['selected_global_step'],
			'selected_loss': checkpoint['selected_loss'],
		},
		'embedding': evidence['embedding'],
	}


def _publish_handoff(
	path: Path,
	handoff: Mapping[str, object],
	*,
	only_missing: bool,
	quarantine_invalid: bool,
) -> bool:
	if path.is_file():
		try:
			existing = load_f3_xy_neighbor_consensus_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff:
			if only_missing:
				return False
			raise FileExistsError(f'complete handoff already exists: {path}')
		if not quarantine_invalid:
			raise ValueError(
				'existing handoff is stale or invalid; use --quarantine-invalid'
			)
		path.replace(path.with_name(f'{path.name}.quarantine'))
	_atomic_json(path, handoff)
	return True


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


def _training_identity(
	training: Mapping[str, object], label: str
) -> Mapping[str, object]:
	return _mapping(
		_mapping(training['identity'], f'{label} identity').get('scientific_identity'),
		f'{label} scientific identity',
	)


def _manifest_path(training: Mapping[str, object]) -> Path:
	return Path(
		str(_mapping(training['pseudo_targets'], 'pseudo targets')['manifest'])
	).resolve()


def _model_tag(training: Mapping[str, object], label: str) -> str:
	value = _mapping(training['identity'], f'{label} identity').get('model_tag')
	if not isinstance(value, str):
		raise TypeError(f'{label} model_tag must be a string')
	return value


def _checkpoint_kind(payload: Mapping[str, object]) -> str:
	state = _mapping(payload.get('training_state'), 'checkpoint training state')
	value = state.get('checkpoint_kind')
	if value not in {'step', 'epoch'}:
		raise ValueError('checkpoint kind is invalid')
	return str(value)


def _reference(value: object, label: str) -> dict[str, str]:
	mapping = _mapping(value, label)
	path, sha256 = mapping.get('path'), mapping.get('sha256')
	if not isinstance(path, str) or not path or not _sha256(sha256):
		raise TypeError(f'{label} must contain path and SHA-256')
	return {'path': path, 'sha256': str(sha256)}


def _sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value)
	)


def _json(path: Path) -> object:
	return json.loads(path.read_text(encoding='utf-8'))


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


__all__ = [
	'F3XYNeighborConsensusPretrainingValidationConfig',
	'F3XYNeighborConsensusPretrainingValidationResult',
	'f3_xy_neighbor_consensus_pretraining_validation_config_from_mapping',
	'load_f3_xy_neighbor_consensus_pretraining_handoff',
	'load_f3_xy_neighbor_consensus_pretraining_validation_config',
	'validate_f3_xy_neighbor_consensus_pretraining',
]
