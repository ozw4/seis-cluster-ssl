"""Strict M5-LS lateral-hard pretraining validation and PASS publication.

M5-LS is deliberately a hard-label run.  This validator binds its immutable
selected lateral manifest to the target-only calibration handoff before it
accepts a CPU smoke checkpoint or a later full pretraining publication.
"""
# ruff: noqa: E501, SLF001

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.pretraining import (
	_lateral_smoothing_identity,
	_multi_head_target_hashes,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.lateral_targets import (
	load_multi_head_lateral_target_manifest,
)
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
		'calibration_handoff',
		'selected_manifest',
		'hard_full_config',
		'hard_handoff',
		'lateral_smoke_config',
		'lateral_full_config',
	}
)
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1'
_HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_MAE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
_CURRENT_K6_MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
_TARGET_REPRESENTATION = 'lateral_mean_field_hard_labels_v1'
_TARGET_SEMANTICS = 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'
_SELECTION_POLICY = 'target_only_smallest_eligible_beta_v1'
_HANDOFF_TYPE = 'f3_m5_lateral_smoothing_pretraining_handoff'
_CANONICAL_BETAS = (0.10, 0.25, 0.50)
_ALLOWED_CONFIG_DELTA = {
	'paths': ['output_root'],
	'identity': ['model_tag', 'scientific_identity.representation_specific_fields'],
	'pseudo_targets': ['manifest', 'target_representation'],
}


@dataclass(frozen=True)
class F3M5LateralSmoothingPretrainingValidationConfig:
	"""Resolved paths required by the closed M5-LS validator contract."""

	artifact_root: Path
	experiment_root: Path
	calibration_handoff: Path
	selected_manifest: Path
	hard_full_config: Path
	hard_handoff: Path
	lateral_smoke_config: Path
	lateral_full_config: Path


@dataclass(frozen=True)
class F3M5LateralSmoothingPretrainingValidationResult:
	"""Validation evidence for one phase and any final PASS handoff."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3M5LateralSmoothingPretrainingValidationConfig:
	"""Resolve the deliberately non-extensible M5-LS validator schema."""
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown M5-LS validation config keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing M5-LS validation config keys: {sorted(missing)!r}'
		)

	def required_path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3M5LateralSmoothingPretrainingValidationConfig(
		artifact_root=required_path('artifact_root'),
		experiment_root=required_path('experiment_root'),
		calibration_handoff=required_path('calibration_handoff'),
		selected_manifest=required_path('selected_manifest'),
		hard_full_config=required_path('hard_full_config'),
		hard_handoff=required_path('hard_handoff'),
		lateral_smoke_config=required_path('lateral_smoke_config'),
		lateral_full_config=required_path('lateral_full_config'),
	)
	if not result.artifact_root.is_absolute() or not result.experiment_root.is_absolute():
		raise ValueError('artifact_root and experiment_root must be absolute')
	ensure_under_root(
		result.experiment_root, root=result.artifact_root, label='experiment_root'
	)
	for label, path in (
		('calibration_handoff', result.calibration_handoff),
		('selected_manifest', result.selected_manifest),
		('hard_handoff', result.hard_handoff),
	):
		ensure_under_root(path, root=result.artifact_root, label=label)
	for label, path in (
		('calibration_handoff', result.calibration_handoff),
		('selected_manifest', result.selected_manifest),
		('hard_full_config', result.hard_full_config),
		('hard_handoff', result.hard_handoff),
		('lateral_smoke_config', result.lateral_smoke_config),
		('lateral_full_config', result.lateral_full_config),
	):
		if not path.is_file():
			raise FileNotFoundError(f'{label} is missing: {path}')
	return result


def load_f3_m5_lateral_smoothing_pretraining_validation_config(
	path: str | Path,
) -> F3M5LateralSmoothingPretrainingValidationConfig:
	"""Load the closed M5-LS validator configuration from YAML."""
	return f3_m5_lateral_smoothing_pretraining_validation_config_from_mapping(
		load_config(path)
	)


def load_f3_m5_lateral_smoothing_pretraining_handoff(
	path: str | Path,
) -> Mapping[str, object]:
	"""Load only a complete, versioned M5-LS PASS handoff."""
	payload = _json(Path(path))
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError('M5-LS handoff type/status mismatch')
	if payload.get('model_tag') != _MODEL_TAG or payload.get('variant') != 'latmf1_nocons':
		raise ValueError('M5-LS handoff model identity mismatch')
	_validate_handoff_targets(_mapping(payload.get('targets'), 'handoff targets'))
	_validate_handoff_checkpoint(
		_mapping(payload.get('checkpoint'), 'handoff checkpoint')
	)
	_validate_handoff_embedding(_mapping(payload.get('embedding'), 'handoff embedding'))
	return payload


def _validate_handoff_targets(targets: Mapping[str, object]) -> None:
	"""Validate the complete target lineage in a final M5-LS handoff."""
	_validate_handoff_target_identity(targets)
	_require_sha256s(
		targets,
		('initial_student_state_sha256', 'initial_head_state_sha256'),
		label='handoff targets',
	)
	_validate_handoff_hard_baseline(targets)


def _validate_handoff_target_identity(targets: Mapping[str, object]) -> None:
	"""Validate the selected target identity and immutable source references."""
	if targets.get('target_representation') != _TARGET_REPRESENTATION:
		raise ValueError('M5-LS handoff target representation mismatch')
	if targets.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('M5-LS handoff target semantics mismatch')
	if not _canonical_beta(targets.get('selected_beta')):
		raise ValueError('M5-LS handoff selected beta mismatch')
	for key in (
		'calibration_handoff',
		'lateral_target_manifest',
		'source_hard_manifest',
		'source_posterior_manifest',
	):
		_reference(targets.get(key), f'handoff targets.{key}')
	_validate_lateral_head_hashes(
		targets.get('lateral_target_head_hashes'),
		label='handoff lateral target head hashes',
	)
	if not isinstance(targets.get('lateral_smoothing'), Mapping):
		raise TypeError('handoff targets.lateral_smoothing is missing')
	if targets.get('allowed_config_diff') != _ALLOWED_CONFIG_DELTA:
		raise ValueError('M5-LS handoff allowed config diff mismatch')


def _validate_handoff_hard_baseline(targets: Mapping[str, object]) -> None:
	"""Validate the hard baseline evidence paired to the lateral candidate."""
	_require_nonempty_strings(
		targets,
		(
			'hard_baseline_config',
			'hard_baseline_handoff',
			'hard_baseline_checkpoint',
		),
		label='handoff targets',
	)
	_require_sha256s(
		targets, ('hard_baseline_checkpoint_sha256',), label='handoff targets'
	)


def _validate_handoff_checkpoint(checkpoint: Mapping[str, object]) -> None:
	"""Validate final checkpoint selection and paired optimizer evidence."""
	_require_nonempty_strings(
		checkpoint, ('path', 'selected_checkpoint_kind'), label='handoff checkpoint'
	)
	_require_sha256s(
		checkpoint, ('sha256', 'selection_history_sha256'), label='handoff checkpoint'
	)
	_require_nonnegative_ints(
		checkpoint,
		('selected_epoch', 'selected_global_step'),
		label='handoff checkpoint',
	)
	if checkpoint['selected_checkpoint_kind'] not in {'step', 'epoch'}:
		raise ValueError('M5-LS handoff selected checkpoint kind mismatch')
	if not _finite_number(checkpoint.get('selected_loss')):
		raise TypeError('handoff checkpoint.selected_loss must be finite')
	trainability = _trainability_summary(
		checkpoint.get('trainability_summary'), 'handoff checkpoint trainability summary'
	)
	if checkpoint.get('trainability_summary_sha256') != scientific_identity_sha256(
		trainability
	):
		raise ValueError('M5-LS handoff trainability identity mismatch')
	optimizer_groups = checkpoint.get('optimizer_group_identity')
	if not isinstance(optimizer_groups, list) or not optimizer_groups:
		raise TypeError('handoff checkpoint.optimizer_group_identity is missing')


def _validate_handoff_embedding(embedding: Mapping[str, object]) -> None:
	"""Validate final extracted embedding and canonical-mask evidence."""
	for key in ('root', 'metadata_path'):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	for key in ('metadata_sha256', 'embeddings_sha256', 'valid_tokens_sha256'):
		if not _sha256(embedding.get(key)):
			raise TypeError(f'handoff embedding.{key} is missing')
	if (
		embedding.get('embeddings_shape') != [76, 113, 32, 384]
		or embedding.get('embeddings_dtype') != 'float16'
		or embedding.get('valid_tokens_shape') != [76, 113, 32]
		or embedding.get('valid_tokens_dtype') != 'bool'
		or not _positive_int(embedding.get('finite_valid_count'))
	):
		raise ValueError('M5-LS handoff embedding identity mismatch')
	canonical = _mapping(
		embedding.get('canonical_valid_token_identities'),
		'handoff canonical valid-token identities',
	)
	if set(canonical) != {'mae', 'current_k6', 'mh_nocons'}:
		raise ValueError('M5-LS handoff canonical valid-token identities mismatch')
	for role, value in canonical.items():
		reference = _reference(value, f'handoff canonical valid-token {role}')
		if reference['sha256'] != embedding['valid_tokens_sha256']:
			raise ValueError('M5-LS handoff canonical valid-token identity mismatch')


def validate_f3_m5_lateral_smoothing_pretraining(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3M5LateralSmoothingPretrainingValidationResult:
	"""Validate M5-LS targets, smoke, full checkpoints, and extraction."""
	if phase not in {'targets', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be targets, smoke, checkpoints, or complete')
	try:
		calibration = _load_calibration_handoff(config.calibration_handoff)
		selected = load_multi_head_lateral_target_manifest(config.selected_manifest)
		hard = _training_config(config.hard_full_config)
		lateral = _training_config(config.lateral_full_config)
		target_evidence = _validate_target_contract(
			config, calibration, selected, hard, lateral
		)
		if phase == 'targets':
			return F3M5LateralSmoothingPretrainingValidationResult(
				phase, {'status': 'PASS', **target_evidence}, None
			)
		if phase == 'smoke':
			smoke = _training_config(config.lateral_smoke_config)
			smoke_evidence = _smoke_evidence(
				config, full=lateral, smoke=smoke, target_evidence=target_evidence
			)
			return F3M5LateralSmoothingPretrainingValidationResult(
				phase,
				{'status': 'PASS', **target_evidence, 'smoke': smoke_evidence},
				None,
			)
		checkpoint = _checkpoint_evidence(
			lateral,
			hard_trainability_summary=_mapping(
				target_evidence['hard_baseline_trainability_summary'],
				'hard baseline trainability summary',
			),
			hard_optimizer_group_identity=target_evidence[
				'hard_baseline_optimizer_group_identity'
			],
			expected_global_step=25600,
			require_full_epoch_history=True,
		)
		evidence: dict[str, object] = {'status': 'PASS', **target_evidence, **checkpoint}
		if phase == 'checkpoints':
			if not dry_run:
				hard_validation._atomic_json(
					Path(checkpoint['root'])
					/ 'preflight'
					/ 'lateral_smoothing_checkpoint_validation.json',
					{
						'artifact_type': 'f3_m5_lateral_smoothing_validation',
						'schema_version': 1,
						'phase': 'checkpoints',
						'status': 'PASS',
					},
				)
			return F3M5LateralSmoothingPretrainingValidationResult(
				phase, evidence, None
			)
		evidence['embedding'] = _embedding_evidence(config, checkpoint)
		handoff = _handoff(evidence)
		path = Path(checkpoint['root']) / 'preflight' / 'lateral_smoothing_handoff.json'
		if dry_run:
			return F3M5LateralSmoothingPretrainingValidationResult(
				phase, evidence, None
			)
		published = _publish_handoff(
			path,
			handoff,
			only_missing=only_missing,
			quarantine_invalid=quarantine_invalid,
		)
		return F3M5LateralSmoothingPretrainingValidationResult(
			phase, evidence, path if published else None
		)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if not dry_run:
			raise
		return F3M5LateralSmoothingPretrainingValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def _load_calibration_handoff(path: Path) -> Mapping[str, object]:
	"""Import the calibration consumer lazily to avoid a reverse dependency."""
	from seis_ssl_cluster.f3.lateral_smoothing_target_calibration import (  # noqa: PLC0415
		load_f3_m5_lateral_target_calibration_handoff,
	)

	return load_f3_m5_lateral_target_calibration_handoff(path)


def _training_config(path: Path) -> Mapping[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(path))


def _validate_target_contract(  # noqa: C901, PLR0912, PLR0915
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	calibration: Mapping[str, object],
	selected: Mapping[str, object],
	hard: Mapping[str, object],
	lateral: Mapping[str, object],
) -> dict[str, object]:
	"""Bind the selected target, calibration evidence, and paired configs."""
	selection = _validate_calibration_selection(config, calibration, selected)
	if selected.get('head_ks') != [6, 8, 10]:
		raise ValueError('selected lateral manifest K identity mismatch')
	if selected.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('selected lateral manifest semantics mismatch')
	lateral_identity = _scientific_identity(lateral, 'lateral')
	if _model_tag(lateral, 'lateral') != _MODEL_TAG:
		raise ValueError('lateral model tag mismatch')
	if _model_tag(hard, 'hard') != _HARD_MODEL_TAG:
		raise ValueError('hard baseline model tag mismatch')
	if lateral_identity.get('target_representation') != _TARGET_REPRESENTATION:
		raise ValueError('lateral target representation mismatch')
	if lateral_identity.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('lateral target semantics mismatch')
	if lateral_identity.get('supervised_loss') != 'structured_hmm_hard_categorical_v1':
		raise ValueError('lateral supervised loss identity mismatch')
	if lateral_identity.get('consistency_policy') != 'disabled_for_m5_ls_v1':
		raise ValueError('lateral consistency policy mismatch')
	if lateral_identity.get('consistency_weight') != 0.0:
		raise ValueError('lateral consistency weight must be zero')
	if _mapping(lateral['loss'], 'lateral loss').get('consistency_weight') != 0.0:
		raise ValueError('lateral loss consistency weight must be zero')
	for label, training, model_tag in (
		('hard', hard, _HARD_MODEL_TAG),
		('lateral', lateral, _MODEL_TAG),
	):
		output = _output_root(training, label)
		if output != (config.experiment_root / model_tag).resolve():
			raise ValueError(f'{label} output root mismatch')
		ensure_under_root(output, root=config.artifact_root, label=f'{label}.output_root')
	if Path(str(_mapping(lateral['pseudo_targets'], 'lateral pseudo targets')['manifest'])).resolve() != config.selected_manifest:
		raise ValueError('lateral selected manifest path mismatch')
	if lateral_identity.get('lateral_target_manifest_sha256') != file_sha256(
		config.selected_manifest
	):
		raise ValueError('lateral selected manifest SHA-256 mismatch')
	if lateral_identity.get('lateral_target_head_hashes') != _multi_head_target_hashes(
		selected
	):
		raise ValueError('lateral selected target head hashes mismatch')
	if lateral_identity.get('lateral_smoothing') != _lateral_smoothing_identity(
		selected
	):
		raise ValueError('lateral smoothing identity mismatch')
	if _mapping(selected['smoothing'], 'selected smoothing').get(
		'pairwise_strength_ratio'
	) != selection['selected_beta']:
		raise ValueError('selected beta does not match selected smoothing identity')
	_validate_allowed_config_delta(hard, lateral)

	hard_manifest_path = Path(
		str(_mapping(hard['pseudo_targets'], 'hard pseudo targets')['manifest'])
	).resolve()
	hard_manifest = load_multi_head_target_manifest(hard_manifest_path)
	source_hard = _reference(
		selected.get('source_hard_manifest'), 'selected source hard manifest'
	)
	source_posterior = _reference(
		selected.get('source_posterior_manifest'), 'selected source posterior manifest'
	)
	if source_hard != selection['source_hard_manifest'] or source_posterior != selection[
		'source_posterior_manifest'
	]:
		raise ValueError('selected manifest source identity differs from calibration')
	if source_hard['path'] != str(hard_manifest_path) or source_hard['sha256'] != file_sha256(
		hard_manifest_path
	):
		raise ValueError('selected source hard manifest does not match hard baseline')
	if lateral_identity.get('source_hard_manifest_sha256') != source_hard['sha256']:
		raise ValueError('lateral source hard manifest SHA-256 mismatch')
	if lateral_identity.get('source_posterior_manifest_sha256') != source_posterior[
		'sha256'
	]:
		raise ValueError('lateral source posterior manifest SHA-256 mismatch')
	_validate_canonical_valid_masks(selected, hard_manifest)

	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(
		config.hard_handoff
	)
	if hard_handoff.get('model_tag') != _HARD_MODEL_TAG:
		raise ValueError('hard baseline handoff model tag mismatch')
	hard_checkpoint = _hard_baseline_checkpoint_evidence(hard, hard_handoff)
	hard_runtime = _runtime_contract(hard)
	lateral_runtime = _runtime_contract(lateral)
	hard_handoff_identity = _mapping(
		hard_handoff.get('stratigraphy_pretext'), 'hard handoff identity'
	)
	hard_initial = (
		str(hard_runtime['initial_student_state_sha256']),
		str(hard_runtime['initial_head_state_sha256']),
	)
	if (
		hard_handoff_identity.get('initial_student_state_sha256'),
		hard_handoff_identity.get('initial_head_state_sha256'),
	) != hard_initial:
		raise ValueError('hard baseline handoff initial state hashes are stale')
	if (
		hard_checkpoint['hard_checkpoint_identity'].get('initial_student_state_sha256'),
		hard_checkpoint['hard_checkpoint_identity'].get('initial_head_state_sha256'),
	) != hard_initial:
		raise ValueError('hard baseline checkpoint initial state hashes are stale')
	lateral_initial = (
		str(lateral_runtime['initial_student_state_sha256']),
		str(lateral_runtime['initial_head_state_sha256']),
	)
	if lateral_initial != hard_initial:
		raise ValueError('lateral initial student/head hashes differ from hard baseline')
	if lateral_runtime['trainability_summary'] != hard_runtime['trainability_summary']:
		raise ValueError('lateral trainability differs from hard baseline config')
	if lateral_runtime['optimizer_group_identity'] != hard_runtime[
		'optimizer_group_identity'
	]:
		raise ValueError('lateral optimizer groups differ from hard baseline config')
	if hard_checkpoint['hard_baseline_trainability_summary'] != hard_runtime[
		'trainability_summary'
	]:
		raise ValueError('hard baseline checkpoint trainability is stale')
	if hard_checkpoint['hard_baseline_optimizer_group_identity'] != hard_runtime[
		'optimizer_group_identity'
	]:
		raise ValueError('hard baseline checkpoint optimizer groups are stale')
	return {
		'target_representation': _TARGET_REPRESENTATION,
		'target_semantics': _TARGET_SEMANTICS,
		'selected_beta': selection['selected_beta'],
		'calibration_handoff': selection['calibration_handoff'],
		'lateral_target_manifest': selection['selected_manifest'],
		'lateral_target_head_hashes': _multi_head_target_hashes(selected),
		'source_hard_manifest': source_hard,
		'source_posterior_manifest': source_posterior,
		'lateral_smoothing': _lateral_smoothing_identity(selected),
		'initial_student_state_sha256': lateral_initial[0],
		'initial_head_state_sha256': lateral_initial[1],
		'hard_baseline_config': str(config.hard_full_config),
		'hard_baseline_handoff': str(config.hard_handoff),
		'allowed_config_diff': dict(_ALLOWED_CONFIG_DELTA),
		'hard_baseline_trainability_summary': hard_runtime['trainability_summary'],
		'hard_baseline_optimizer_group_identity': hard_runtime[
			'optimizer_group_identity'
		],
		**hard_checkpoint,
	}


def _validate_calibration_selection(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	calibration: Mapping[str, object],
	selected: Mapping[str, object],
) -> dict[str, object]:
	"""Require a complete selected target-only calibration, never a HOLD."""
	selected_beta = _validate_calibration_header(calibration)
	selected_manifest, candidate_manifest = _validate_selected_manifest_records(
		config, calibration
	)
	_validate_selected_candidate(
		calibration,
		selected_beta=selected_beta,
		candidate_manifest=candidate_manifest,
		selected=selected,
	)
	return {
		'selected_beta': selected_beta,
		'calibration_handoff': {
			'path': str(config.calibration_handoff),
			'sha256': file_sha256(config.calibration_handoff),
		},
		'selected_manifest': selected_manifest,
		'source_hard_manifest': _reference(
			calibration.get('source_hard_manifest'),
			'calibration source hard manifest',
		),
		'source_posterior_manifest': _reference(
			calibration.get('source_posterior_manifest'),
			'calibration source posterior manifest',
		),
	}


def _validate_calibration_header(calibration: Mapping[str, object]) -> float:
	"""Validate the immutable target-only policy and beta-zero evidence."""
	if calibration.get('artifact_type') != 'f3_m5_lateral_target_calibration':
		raise ValueError('M5-LS calibration artifact type mismatch')
	if calibration.get('schema_version') != 1:
		raise ValueError('M5-LS calibration schema version mismatch')
	if calibration.get('status') != 'M5_LS_TARGET_SELECTED':
		raise ValueError('M5-LS calibration did not select a target')
	if calibration.get('selection_policy') != _SELECTION_POLICY:
		raise ValueError('M5-LS calibration selection policy mismatch')
	if tuple(calibration.get('candidate_betas', ())) != _CANONICAL_BETAS:
		raise ValueError('M5-LS calibration candidate beta set mismatch')
	parity = _mapping(calibration.get('beta_zero_parity'), 'beta-zero parity')
	if parity.get('status') != 'PASS' or not isinstance(parity.get('heads'), Mapping):
		raise ValueError('M5-LS beta-zero parity evidence is not PASS')
	for k in ('6', '8', '10'):
		if k not in parity['heads']:
			raise ValueError('M5-LS beta-zero parity head evidence is incomplete')
	selected_beta = calibration.get('selected_beta')
	if not _canonical_beta(selected_beta):
		raise ValueError('M5-LS calibration selected beta mismatch')
	return float(selected_beta)


def _validate_selected_manifest_records(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	calibration: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
	"""Require the fixed publication to be a byte-exact candidate copy."""
	selected_manifest = _reference(
		calibration.get('selected_manifest'), 'calibration selected manifest'
	)
	if selected_manifest['path'] != str(config.selected_manifest):
		raise ValueError('calibration selected manifest path mismatch')
	if selected_manifest['sha256'] != file_sha256(config.selected_manifest):
		raise ValueError('calibration selected manifest SHA-256 mismatch')
	candidate_manifest = _reference(
		calibration.get('selected_candidate_manifest'),
		'calibration selected candidate manifest',
	)
	if candidate_manifest['sha256'] != selected_manifest['sha256']:
		raise ValueError('selected manifest is not byte-identical to candidate manifest')
	if file_sha256(Path(candidate_manifest['path'])) != selected_manifest['sha256']:
		raise ValueError('selected candidate manifest SHA-256 mismatch')
	return selected_manifest, candidate_manifest


def _validate_selected_candidate(
	calibration: Mapping[str, object],
	*,
	selected_beta: float,
	candidate_manifest: Mapping[str, str],
	selected: Mapping[str, object],
) -> None:
	"""Bind the selected beta to the eligible canonical candidate record."""
	candidates = _mapping(calibration.get('candidates'), 'calibration candidates')
	candidate_key = {0.10: 'beta010', 0.25: 'beta025', 0.50: 'beta050'}[
		selected_beta
	]
	candidate = _mapping(candidates.get(candidate_key), f'calibration {candidate_key}')
	if candidate.get('beta') != selected_beta:
		raise ValueError('calibration selected candidate beta mismatch')
	if _reference(candidate.get('manifest'), 'selected candidate manifest') != candidate_manifest:
		raise ValueError('calibration selected candidate manifest mismatch')
	eligibility = _mapping(candidate.get('eligibility'), 'selected candidate eligibility')
	if eligibility.get('eligible') is not True:
		raise ValueError('calibration selected candidate is not eligible')
	expected_hashes = _multi_head_target_hashes(selected)
	if candidate.get('head_hashes') != expected_hashes:
		raise ValueError('calibration selected candidate head hashes mismatch')


def _validate_canonical_valid_masks(
	selected: Mapping[str, object], hard: Mapping[str, object]
) -> None:
	"""Require every selected target mask to remain the frozen hard mask."""
	selected_heads = _mapping(selected.get('heads'), 'selected lateral heads')
	hard_heads = _mapping(hard.get('heads'), 'hard target heads')
	for k in (6, 8, 10):
		selected_surveys = _mapping(
			_mapping(selected_heads.get(str(k)), f'selected K{k}')['surveys'],
			f'selected K{k} surveys',
		)
		hard_surveys = _mapping(
			_mapping(hard_heads.get(str(k)), f'hard K{k}')['surveys'],
			f'hard K{k} surveys',
		)
		if set(selected_surveys) != set(hard_surveys):
			raise ValueError('selected valid-mask survey set mismatch')
		for survey_id, selected_survey in selected_surveys.items():
			selected_valid = _reference(
				_mapping(selected_survey, 'selected survey').get('valid_tokens'),
				'selected valid tokens',
			)
			hard_valid = _reference(
				_mapping(hard_surveys[survey_id], 'hard survey').get('valid_tokens'),
				'hard valid tokens',
			)
			if selected_valid['sha256'] != hard_valid['sha256']:
				raise ValueError('selected valid mask differs from canonical hard mask')


def _validate_allowed_config_delta(
	hard: Mapping[str, object], lateral: Mapping[str, object]
) -> None:
	"""Allow only representation identity and target publication changes."""
	left, right = json.loads(json.dumps(hard)), json.loads(json.dumps(lateral))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		identity.pop('model_tag', None)
		pseudo_targets = _mapping(value['pseudo_targets'], 'pseudo targets')
		pseudo_targets.pop('manifest', None)
		pseudo_targets.pop('target_representation', None)
		scientific = _mapping(
			identity['scientific_identity'], 'scientific identity'
		)
		for key in (
			'experiment_role',
			'variant',
			'target_representation',
			'target_manifest_sha256',
			'target_head_hashes',
			'lateral_target_manifest_sha256',
			'lateral_target_head_hashes',
			'target_semantics',
			'source_hard_manifest_sha256',
			'source_posterior_manifest_sha256',
			'lateral_smoothing',
			'supervised_loss',
			'consistency_policy',
		):
			scientific.pop(key, None)
	if left != right:
		raise ValueError('hard/lateral scientific config drift outside allowed fields')


def _runtime_contract(training: Mapping[str, object]) -> dict[str, object]:
	"""Build reproducible initial states and the configured optimizer partition."""
	seed = _mapping(training['train'], 'train').get('seed')
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('train.seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('M5-LS validation requires multi-head initialization')
	student = components.student
	parameter_names = {
		id(parameter): f'{prefix}.{name}'
		for prefix, module in (('student', student), ('head', heads))
		for name, parameter in module.named_parameters()
	}
	optimizer_groups: list[dict[str, object]] = []
	for group in components.optimizer.param_groups:
		parameters = group.get('params')
		if not isinstance(parameters, list):
			raise TypeError('optimizer parameter group must be a list')
		try:
			names = [parameter_names[id(parameter)] for parameter in parameters]
		except KeyError as error:
			raise ValueError('optimizer contains an unknown parameter') from error
		optimizer_groups.append(
			{
				'name': group.get('name'),
				'parameter_names': names,
				'lr': float(group.get('lr', 0.0)),
			}
		)
	summary = components.trainability_summary
	return {
		'initial_student_state_sha256': hard_validation._state_sha256(
			student.state_dict()
		),
		'initial_head_state_sha256': hard_validation._state_sha256(heads.state_dict()),
		'trainability_summary': {
			'trainable_parameter_count': int(summary.trainable_parameter_count),
			'frozen_parameter_count': int(summary.frozen_parameter_count),
			'trainable_names': list(summary.trainable_names),
		},
		'optimizer_group_identity': optimizer_groups,
	}


def _initial_hashes(training: Mapping[str, object]) -> tuple[str, str]:
	"""Return only the two initial hashes for focused callers and tests."""
	runtime = _runtime_contract(training)
	return (
		str(runtime['initial_student_state_sha256']),
		str(runtime['initial_head_state_sha256']),
	)


def _hard_baseline_checkpoint_evidence(
	hard: Mapping[str, object], hard_handoff: Mapping[str, object]
) -> dict[str, object]:
	"""Bind the supplied hard config to its canonical hard PASS checkpoint."""
	record = _mapping(hard_handoff.get('checkpoint'), 'hard handoff checkpoint')
	path = Path(str(record.get('path', ''))).resolve()
	if not path.is_file():
		raise FileNotFoundError('hard baseline handoff checkpoint is missing')
	if record.get('sha256') != file_sha256(path):
		raise ValueError('hard baseline handoff checkpoint SHA-256 mismatch')
	payload = _torch_mapping(path)
	validate_stratigraphy_checkpoint_payload(payload, expected_config=hard)
	if _mapping(payload.get('stratigraphy_config'), 'hard checkpoint config') != hard:
		raise ValueError('hard baseline config does not match canonical hard checkpoint')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'hard checkpoint identity'
	)
	handoff_identity = _mapping(
		hard_handoff.get('stratigraphy_pretext'), 'hard handoff identity'
	)
	for key in (
		'head_spec',
		'head_ks',
		'consistency_policy',
		'consistency_weight',
		'consistency_beta',
		'scientific_identity_sha256',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
	):
		if identity.get(key) != handoff_identity.get(key):
			raise ValueError(f'hard baseline handoff checkpoint identity mismatch: {key}')
	if _mapping(identity.get('target_manifest'), 'hard checkpoint target manifest').get(
		'sha256'
	) != handoff_identity.get('target_manifest_sha256'):
		raise ValueError('hard baseline handoff checkpoint target manifest mismatch')
	if identity.get('per_head_targets') != handoff_identity.get(
		'per_head_target_sha256'
	):
		raise ValueError('hard baseline handoff checkpoint target heads mismatch')
	trainability = _trainability_summary(
		payload.get('trainability_summary'), 'hard baseline trainability summary'
	)
	optimizer_groups = identity.get('optimizer_group_identity')
	if not isinstance(optimizer_groups, list) or not optimizer_groups:
		raise TypeError('hard baseline optimizer group identity is missing')
	return {
		'hard_baseline_checkpoint': str(path),
		'hard_baseline_checkpoint_sha256': record['sha256'],
		'hard_baseline_trainability_summary': trainability,
		'hard_baseline_optimizer_group_identity': optimizer_groups,
		'hard_checkpoint_identity': identity,
	}


def _checkpoint_evidence(
	training: Mapping[str, object],
	*,
	hard_trainability_summary: Mapping[str, object],
	hard_optimizer_group_identity: object,
	expected_global_step: int,
	require_full_epoch_history: bool,
) -> dict[str, object]:
	"""Validate schema-v4 M5-LS checkpoints against paired hard identities."""
	root = _output_root(training, 'lateral')
	latest_path, best_path = root / 'latest.pt', root / 'best.pt'
	if not latest_path.is_file() or not best_path.is_file():
		raise FileNotFoundError('lateral run requires latest.pt and best.pt')
	latest, best = _torch_mapping(latest_path), _torch_mapping(best_path)
	for payload in (latest, best):
		_validate_lateral_checkpoint_payload(
			payload,
			training=training,
			hard_trainability_summary=hard_trainability_summary,
			hard_optimizer_group_identity=hard_optimizer_group_identity,
		)
	runtime = _runtime_contract(training)
	identity = _mapping(best['stratigraphy_checkpoint'], 'lateral checkpoint identity')
	if (
		identity.get('initial_student_state_sha256'),
		identity.get('initial_head_state_sha256'),
	) != (
		runtime['initial_student_state_sha256'],
		runtime['initial_head_state_sha256'],
	):
		raise ValueError('lateral checkpoint initial state hash mismatch')
	rows = _validate_checkpoint_progress(
		latest,
		root=root,
		expected_global_step=expected_global_step,
		require_full_epoch_history=require_full_epoch_history,
	)
	selection = hard_validation._validate_best_selection(
		best, latest, variant='latmf1_nocons'
	)
	return {
		'root': root,
		'best_path': best_path,
		'latest_path': latest_path,
		'best': best,
		'latest': latest,
		'identity': identity,
		'epoch_rows': rows,
		'selection': selection,
	}


def _validate_lateral_checkpoint_payload(
	payload: Mapping[str, object],
	*,
	training: Mapping[str, object],
	hard_trainability_summary: Mapping[str, object],
	hard_optimizer_group_identity: object,
) -> None:
	"""Validate one M5-LS checkpoint against its config and hard baseline."""
	validate_stratigraphy_checkpoint_payload(payload, expected_config=training)
	hard_validation._metrics_finite(payload)
	_validate_finite_checkpoint_tensors(payload)
	if _mapping(payload.get('stratigraphy_config'), 'lateral checkpoint config') != training:
		raise ValueError('lateral checkpoint config differs from resolved config')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'lateral checkpoint identity'
	)
	if identity.get('schema_version') != 4 or identity.get('model_tag') != _MODEL_TAG:
		raise ValueError('lateral checkpoint schema/model identity mismatch')
	if identity.get('target_representation') != _TARGET_REPRESENTATION:
		raise ValueError('lateral checkpoint target representation mismatch')
	if identity.get('target_semantics') != _TARGET_SEMANTICS:
		raise ValueError('lateral checkpoint target semantics mismatch')
	if identity.get('consistency_weight') != 0.0:
		raise ValueError('lateral checkpoint consistency weight must be zero')
	if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
		_scientific_identity(training, 'lateral')
	):
		raise ValueError('lateral checkpoint scientific identity mismatch')
	if _trainability_summary(
		payload.get('trainability_summary'), 'lateral checkpoint trainability summary'
	) != hard_trainability_summary:
		raise ValueError('lateral checkpoint trainability differs from hard baseline')
	if identity.get('optimizer_group_identity') != hard_optimizer_group_identity:
		raise ValueError('lateral checkpoint optimizer groups differ from hard baseline')
	_validate_hard_loss_metrics(payload)


def _validate_checkpoint_progress(
	latest: Mapping[str, object],
	*,
	root: Path,
	expected_global_step: int,
	require_full_epoch_history: bool,
) -> list[dict[str, float | int]]:
	"""Require either the fixed full completion or exactly two smoke steps."""
	if not require_full_epoch_history:
		if latest.get('global_step') != expected_global_step:
			raise ValueError(
				f'lateral smoke must finish at global step {expected_global_step}'
			)
		return []
	if latest.get('epoch') != 25 or latest.get('global_step') != expected_global_step:
		raise ValueError('lateral full run must finish epoch 25/global step 25600')
	if _mapping(latest.get('training_state'), 'lateral training state').get(
		'checkpoint_kind'
	) != 'epoch':
		raise ValueError('lateral full latest checkpoint must be an epoch checkpoint')
	rows = hard_validation._epoch_rows(root / 'multi_head_epoch_metrics.csv')
	if (
		[row['epoch'] for row in rows] != list(range(1, 26))
		or rows[-1]['global_step'] != expected_global_step
	):
		raise ValueError('lateral epoch metrics coverage is incomplete')
	return rows


def _validate_finite_checkpoint_tensors(payload: Mapping[str, object]) -> None:
	"""Make smoke finite-gradient evidence explicit through saved state tensors."""
	for label in ('model_state_dict', 'stratigraphy_state_dict', 'optimizer_state_dict'):
		value = payload.get(label)
		if value is None:
			raise TypeError(f'checkpoint {label} is missing')
		for tensor in _walk_tensors(value):
			if tensor.is_floating_point() and not torch.isfinite(tensor).all():
				raise ValueError(f'checkpoint {label} contains non-finite values')


def _walk_tensors(value: object) -> Iterator[torch.Tensor]:
	if isinstance(value, Mapping):
		for child in value.values():
			yield from _walk_tensors(child)
	elif isinstance(value, list | tuple):
		for child in value:
			yield from _walk_tensors(child)
	elif isinstance(value, torch.Tensor):
		yield value


def _validate_hard_loss_metrics(payload: Mapping[str, object]) -> None:
	"""Require the persisted metrics shape of the existing hard loss path."""
	metrics = _mapping(payload.get('metrics'), 'lateral checkpoint metrics')
	if 'loss_consistency' not in metrics:
		raise ValueError('lateral checkpoint did not use the hard multi-head loss path')
	if any('posterior' in str(key) for key in metrics):
		raise ValueError('lateral checkpoint contains posterior loss-path metrics')
	consistency_contribution = metrics.get('loss_consistency')
	if not _finite_number(consistency_contribution):
		raise ValueError('lateral hard loss consistency metric must be finite')
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'lateral checkpoint identity'
	)
	if identity.get('consistency_weight') != 0.0:
		raise ValueError('lateral consistency contribution is not disabled')
	if float(identity['consistency_weight']) * float(consistency_contribution) != 0.0:
		raise ValueError('lateral consistency contribution must be zero')


def _smoke_evidence(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
	target_evidence: Mapping[str, object],
) -> dict[str, object]:
	"""Validate the isolated CPU two-step M5-LS smoke contract."""
	_validate_smoke_config(config, full=full, smoke=smoke)
	hard_initial_hashes = (
		str(target_evidence['initial_student_state_sha256']),
		str(target_evidence['initial_head_state_sha256']),
	)
	if _initial_hashes(smoke) != _initial_hashes(full):
		raise ValueError('lateral smoke initial state hashes differ from lateral full')
	if _initial_hashes(smoke) != hard_initial_hashes:
		raise ValueError('lateral smoke initial state hashes differ from hard baseline')
	evidence = _checkpoint_evidence(
		smoke,
		hard_trainability_summary=_mapping(
			target_evidence['hard_baseline_trainability_summary'],
			'hard baseline trainability summary',
		),
		hard_optimizer_group_identity=target_evidence[
			'hard_baseline_optimizer_group_identity'
		],
		expected_global_step=2,
		require_full_epoch_history=False,
	)
	latest = _mapping(evidence['latest'], 'lateral smoke latest checkpoint')
	state = _mapping(latest.get('training_state'), 'lateral smoke training state')
	if latest.get('epoch') != 1 or state.get('checkpoint_kind') != 'step':
		raise ValueError('lateral smoke must end with a two-step partial checkpoint')
	identity = _mapping(evidence['identity'], 'lateral smoke checkpoint identity')
	if (
		identity.get('schema_version') != 4
		or identity.get('target_representation') != _TARGET_REPRESENTATION
		or identity.get('target_semantics') != _TARGET_SEMANTICS
		or identity.get('consistency_weight') != 0.0
	):
		raise ValueError('lateral smoke target/schema/consistency identity mismatch')
	return {
		**evidence,
		'hard_multi_head_loss_path_used': True,
		'posterior_loss_path_used': False,
		'consistency_contribution': 0.0,
		'gradients_finite': True,
	}


def _validate_smoke_config(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	"""Allow only the isolated CPU two-step execution delta from full config."""
	full_root, smoke_root = _output_root(full, 'lateral full'), _output_root(
		smoke, 'lateral smoke'
	)
	ensure_under_root(
		full_root, root=config.artifact_root, label='lateral full output root'
	)
	ensure_under_root(
		smoke_root, root=config.artifact_root, label='lateral smoke output root'
	)
	if smoke_root == full_root:
		raise ValueError('lateral smoke output root must differ from lateral full root')
	if full_root.exists():
		raise ValueError('lateral full output root must remain unmodified during smoke')
	smoke_train = _mapping(smoke['train'], 'lateral smoke train')
	if smoke_train.get('device') != 'cpu' or smoke_train.get('max_steps') != 2:
		raise ValueError('lateral smoke must use device=cpu and max_steps=2')
	runtime = _mapping(_mapping(smoke['identity'], 'smoke identity').get(
		'runtime_identity'
	), 'lateral smoke runtime identity')
	if runtime.get('device') != 'cpu':
		raise ValueError('lateral smoke runtime identity must use cpu')
	left, right = json.loads(json.dumps(full)), json.loads(json.dumps(smoke))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		identity = _mapping(value['identity'], 'identity')
		runtime_identity = identity.get('runtime_identity')
		if runtime_identity is not None:
			_mapping(runtime_identity, 'runtime identity').pop('device', None)
		train = _mapping(value['train'], 'train')
		train.pop('device', None)
		train.pop('max_steps', None)
		_mapping(identity['scientific_identity'], 'scientific identity')['train'].pop(
			'max_steps', None
		)
	if left != right:
		raise ValueError('lateral smoke config drift outside CPU two-step settings')


def _embedding_evidence(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	"""Validate the later M5-LS extraction and canonical mask identity."""
	root = (
		config.artifact_root
		/ 'embeddings/f3/facies_benchmark_v1'
		/ _MODEL_TAG
		/ 'overlap_x16'
	)
	files = output_paths(root, 'f3_facies_benchmark')
	if not all(
		path.is_file()
		for path in (files.embeddings, files.valid_tokens, files.metadata)
	):
		raise FileNotFoundError('M5-LS complete embedding artifacts are missing')
	metadata = _json(files.metadata)
	best_path = Path(checkpoint['best_path'])
	if Path(str(metadata.get('checkpoint_path', ''))).resolve() != best_path.resolve() or metadata.get('checkpoint_sha256') != file_sha256(best_path):
		raise ValueError('embedding metadata does not bind selected best.pt')
	stratigraphy = _mapping(
		metadata.get('stratigraphy_pretext'), 'embedding stratigraphy identity'
	)
	identity = _mapping(checkpoint['identity'], 'checkpoint identity')
	for key in (
		'model_tag',
		'target_representation',
		'target_semantics',
		'lateral_target_manifest_sha256',
		'source_hard_manifest_sha256',
		'source_posterior_manifest_sha256',
		'lateral_smoothing',
		'scientific_identity_sha256',
	):
		if stratigraphy.get(key) != identity.get(key):
			raise ValueError(f'embedding stratigraphy identity mismatch: {key}')
	embeddings = np.load(files.embeddings, mmap_mode='r')
	valid = np.load(files.valid_tokens, mmap_mode='r')
	if (
		embeddings.shape != (76, 113, 32, 384)
		or embeddings.dtype != np.float16
		or valid.shape != (76, 113, 32)
		or valid.dtype != np.bool_
		or not int(valid.sum())
		or not np.isfinite(embeddings[valid]).all()
	):
		raise ValueError('M5-LS embedding shape/dtype/finite contract mismatch')
	canonical = _canonical_valid_token_identities(config)
	valid_sha = file_sha256(files.valid_tokens)
	if any(reference['sha256'] != valid_sha for reference in canonical.values()):
		raise ValueError('M5-LS embedding valid-token identity differs from canonical')
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_sha256': valid_sha,
		'canonical_valid_token_identities': canonical,
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
	}


def _canonical_valid_token_identities(
	config: F3M5LateralSmoothingPretrainingValidationConfig,
) -> dict[str, dict[str, str]]:
	"""Return the canonical bitwise-identical valid-token references."""
	identities: dict[str, dict[str, str]] = {}
	for role, model_tag in (
		('mae', _MAE_MODEL_TAG),
		('current_k6', _CURRENT_K6_MODEL_TAG),
		('mh_nocons', _HARD_MODEL_TAG),
	):
		path = output_paths(
			config.artifact_root
			/ 'embeddings/f3/facies_benchmark_v1'
			/ model_tag
			/ 'overlap_x16',
			'f3_facies_benchmark',
		).valid_tokens
		if not path.is_file():
			raise FileNotFoundError(
				f'{role} canonical valid-token artifact is missing: {path}'
			)
		identities[role] = {'path': str(path), 'sha256': file_sha256(path)}
	if len({identity['sha256'] for identity in identities.values()}) != 1:
		raise ValueError('canonical valid-token identities do not match')
	return identities


def _handoff(
	evidence: Mapping[str, object],
) -> dict[str, object]:
	"""Build the final-only complete M5-LS PASS handoff."""
	selection = _mapping(evidence['selection'], 'checkpoint selection')
	selected = _mapping(selection['selected'], 'selected checkpoint')
	checkpoint_identity = _mapping(evidence['identity'], 'checkpoint identity')
	best = _mapping(evidence['best'], 'best checkpoint')
	target_keys = (
		'target_representation',
		'target_semantics',
		'selected_beta',
		'calibration_handoff',
		'lateral_target_manifest',
		'lateral_target_head_hashes',
		'source_hard_manifest',
		'source_posterior_manifest',
		'lateral_smoothing',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'hard_baseline_config',
		'hard_baseline_handoff',
		'hard_baseline_checkpoint',
		'hard_baseline_checkpoint_sha256',
		'allowed_config_diff',
	)
	trainability = _trainability_summary(
		best.get('trainability_summary'), 'best checkpoint trainability summary'
	)
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': 'latmf1_nocons',
		'targets': {key: evidence[key] for key in target_keys},
		'checkpoint': {
			'path': str(evidence['best_path']),
			'sha256': file_sha256(Path(evidence['best_path'])),
			'selected_epoch': selected['epoch'],
			'selected_global_step': selected['global_step'],
			'selected_checkpoint_kind': selected['checkpoint_kind'],
			'selected_loss': selected['loss'],
			'selection_history_sha256': selection['sha256'],
			'optimizer_group_identity': checkpoint_identity['optimizer_group_identity'],
			'trainability_summary': trainability,
			'trainability_summary_sha256': scientific_identity_sha256(trainability),
		},
		'embedding': dict(_mapping(evidence['embedding'], 'embedding')),
	}


def _publish_handoff(
	path: Path,
	handoff: Mapping[str, object],
	*,
	only_missing: bool,
	quarantine_invalid: bool,
) -> bool:
	"""Publish a final handoff only after complete evidence is available."""
	if path.is_file():
		try:
			existing = load_f3_m5_lateral_smoothing_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff and only_missing:
			return False
		if existing != handoff:
			if not quarantine_invalid:
				raise ValueError(
					'existing handoff is stale or invalid; pass --quarantine-invalid '
					'to replace it'
				)
			hard_validation._quarantine(path)
	hard_validation._atomic_json(path, handoff)
	return True


def _output_root(training: Mapping[str, object], label: str) -> Path:
	return Path(
		str(_mapping(training['paths'], f'{label} paths')['output_root'])
	).resolve()


def _scientific_identity(training: Mapping[str, object], label: str) -> Mapping[str, object]:
	return _mapping(
		_mapping(training['identity'], f'{label} identity').get('scientific_identity'),
		f'{label} scientific identity',
	)


def _model_tag(training: Mapping[str, object], label: str) -> object:
	return _mapping(training['identity'], f'{label} identity').get('model_tag')


def _torch_mapping(path: Path) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	return _mapping(payload, f'checkpoint {path}')


def _json(path: Path) -> Mapping[str, object]:
	return _mapping(json.loads(path.read_text(encoding='utf-8')), str(path))


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _reference(value: object, label: str) -> dict[str, str]:
	reference = _mapping(value, label)
	path = reference.get('path')
	sha256 = reference.get('sha256')
	if not isinstance(path, str) or not path:
		raise TypeError(f'{label}.path is missing')
	if not _sha256(sha256):
		raise TypeError(f'{label}.sha256 is missing')
	return {'path': str(Path(path).resolve()), 'sha256': sha256}


def _sha256(value: object) -> bool:
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in '0123456789abcdef' for character in value.lower())
	)


def _require_nonempty_strings(
	value: Mapping[str, object], keys: tuple[str, ...], *, label: str
) -> None:
	for key in keys:
		if not isinstance(value.get(key), str) or not value[key]:
			raise TypeError(f'{label}.{key} is missing')


def _require_sha256s(
	value: Mapping[str, object], keys: tuple[str, ...], *, label: str
) -> None:
	for key in keys:
		if not _sha256(value.get(key)):
			raise TypeError(f'{label}.{key} is missing')


def _require_nonnegative_ints(
	value: Mapping[str, object], keys: tuple[str, ...], *, label: str
) -> None:
	for key in keys:
		if not _nonnegative_int(value.get(key)):
			raise TypeError(f'{label}.{key} must be a nonnegative integer')


def _finite_number(value: object) -> bool:
	return (
		isinstance(value, int | float)
		and not isinstance(value, bool)
		and math.isfinite(float(value))
	)


def _nonnegative_int(value: object) -> bool:
	return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
	return _nonnegative_int(value) and bool(value)


def _canonical_beta(value: object) -> bool:
	return (
		isinstance(value, int | float)
		and not isinstance(value, bool)
		and float(value) in _CANONICAL_BETAS
	)


def _trainability_summary(value: object, label: str) -> dict[str, object]:
	summary = _mapping(value, label)
	result = {
		'trainable_parameter_count': summary.get('trainable_parameter_count'),
		'frozen_parameter_count': summary.get('frozen_parameter_count'),
		'trainable_names': summary.get('trainable_names'),
	}
	if not _nonnegative_int(result['trainable_parameter_count']) or not _nonnegative_int(
		result['frozen_parameter_count']
	):
		raise TypeError(f'{label} parameter counts are invalid')
	names = result['trainable_names']
	if (
		not isinstance(names, list)
		or not names
		or any(not isinstance(name, str) or not name for name in names)
		or len(names) != len(set(names))
	):
		raise TypeError(f'{label} trainable names are invalid')
	return result


def _validate_lateral_head_hashes(value: object, *, label: str) -> None:
	head_hashes = _mapping(value, label)
	if set(head_hashes) != {'6', '8', '10'}:
		raise ValueError(f'{label} K keys mismatch')
	for k, surveys in head_hashes.items():
		if not isinstance(surveys, Mapping) or not surveys:
			raise TypeError(f'{label}.K{k} must contain surveys')
		for survey_id, artifacts in surveys.items():
			items = _mapping(artifacts, f'{label}.K{k}.{survey_id}')
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				if not _sha256(items.get(name)):
					raise TypeError(f'{label}.K{k}.{survey_id}.{name} is missing')
