"""Strict M5-U soft-posterior pretraining validation and handoff publication."""
# ruff: noqa: E501, SLF001

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.pretraining import _multi_head_posterior_hashes
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3 import multi_head_pretraining_validation as hard_validation
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.stratigraphy.state_posterior import (
	load_multi_head_state_posterior_manifest,
)
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
		'posterior_manifest',
		'hard_full_config',
		'hard_handoff',
		'soft_smoke_config',
		'soft_full_config',
	}
)
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1'
_HARD_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
_MAE_MODEL_TAG = 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
_CURRENT_K6_MODEL_TAG = 'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1'
_HANDOFF_TYPE = 'f3_m5_soft_posterior_pretraining_handoff'


@dataclass(frozen=True)
class F3M5SoftPosteriorPretrainingValidationConfig:
	"""Resolved paths required to validate the single M5-U candidate."""

	artifact_root: Path
	experiment_root: Path
	posterior_manifest: Path
	hard_full_config: Path
	hard_handoff: Path
	soft_smoke_config: Path
	soft_full_config: Path


@dataclass(frozen=True)
class F3M5SoftPosteriorPretrainingValidationResult:
	"""One phase result and the optional atomically published handoff."""

	phase: str
	evidence: Mapping[str, object]
	published_handoff: Path | None


def f3_m5_soft_posterior_pretraining_validation_config_from_mapping(
	config: Mapping[str, object],
) -> F3M5SoftPosteriorPretrainingValidationConfig:
	"""Resolve the deliberately closed M5-U validator configuration."""
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown M5-U validation config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing M5-U validation config keys: {sorted(missing)!r}')

	def required_path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3M5SoftPosteriorPretrainingValidationConfig(
		artifact_root=required_path('artifact_root'),
		experiment_root=required_path('experiment_root'),
		posterior_manifest=required_path('posterior_manifest'),
		hard_full_config=required_path('hard_full_config'),
		hard_handoff=required_path('hard_handoff'),
		soft_smoke_config=required_path('soft_smoke_config'),
		soft_full_config=required_path('soft_full_config'),
	)
	for label, path in (
		('posterior_manifest', result.posterior_manifest),
		('hard_handoff', result.hard_handoff),
	):
		ensure_under_root(path, root=result.artifact_root, label=label)
	if not result.experiment_root.is_absolute() or not result.artifact_root.is_absolute():
		raise ValueError('artifact_root and experiment_root must be absolute')
	ensure_under_root(
		result.experiment_root, root=result.artifact_root, label='experiment_root'
	)
	for label, path in (
		('posterior_manifest', result.posterior_manifest),
		('hard_full_config', result.hard_full_config),
		('hard_handoff', result.hard_handoff),
		('soft_smoke_config', result.soft_smoke_config),
		('soft_full_config', result.soft_full_config),
	):
		if not path.is_file():
			raise FileNotFoundError(f'{label} is missing: {path}')
	return result


def load_f3_m5_soft_posterior_pretraining_validation_config(
	path: str | Path,
) -> F3M5SoftPosteriorPretrainingValidationConfig:
	"""Load the closed M5-U validation schema from YAML."""
	return f3_m5_soft_posterior_pretraining_validation_config_from_mapping(
		load_config(path)
	)


def load_f3_m5_soft_posterior_pretraining_handoff(  # noqa: C901, PLR0912
	path: str | Path,
) -> Mapping[str, object]:
	"""Load only a complete, versioned M5-U PASS handoff."""
	payload = _mapping(json.loads(Path(path).read_text(encoding='utf-8')), 'handoff')
	if (
		payload.get('artifact_type') != _HANDOFF_TYPE
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError('M5-U handoff type/status mismatch')
	if payload.get('model_tag') != _MODEL_TAG or payload.get('variant') != 'soft_nocons':
		raise ValueError('M5-U handoff model identity mismatch')
	targets = _mapping(payload.get('targets'), 'handoff targets')
	for key in (
		'posterior_manifest_path',
		'hard_baseline_config',
		'hard_baseline_handoff',
	):
		if not isinstance(targets.get(key), str) or not targets[key]:
			raise TypeError(f'handoff targets.{key} is missing')
	for key in ('posterior_manifest_sha256', 'initial_student_state_sha256', 'initial_head_state_sha256'):
		if not _sha256(targets.get(key)):
			raise TypeError(f'handoff targets.{key} is missing')
	_validate_posterior_head_hashes(targets.get('posterior_head_hashes'))
	if targets.get('target_representation') != 'ordered_path_state_posterior_v1':
		raise ValueError('M5-U handoff target representation mismatch')
	if (
		targets.get('posterior_semantics')
		!= 'ordered_path_cost_gibbs_state_marginal_v1'
		or targets.get('posterior_cost_temperature') != 1.0
	):
		raise ValueError('M5-U handoff posterior semantics mismatch')
	checkpoint = _mapping(payload.get('checkpoint'), 'handoff checkpoint')
	for key in ('path', 'selected_checkpoint_kind'):
		if not isinstance(checkpoint.get(key), str) or not checkpoint[key]:
			raise TypeError(f'handoff checkpoint.{key} is missing')
	for key in ('sha256', 'selection_history_sha256'):
		if not _sha256(checkpoint.get(key)):
			raise TypeError(f'handoff checkpoint.{key} is missing')
	for key in ('selected_epoch', 'selected_global_step'):
		if (
			isinstance(checkpoint.get(key), bool)
			or not isinstance(checkpoint.get(key), int)
			or checkpoint[key] < 0
		):
			raise TypeError(f'handoff checkpoint.{key} must be a nonnegative integer')
	if checkpoint['selected_checkpoint_kind'] not in {'step', 'epoch'}:
		raise ValueError('M5-U handoff selected checkpoint kind mismatch')
	selected_loss = checkpoint.get('selected_loss')
	if (
		isinstance(selected_loss, bool)
		or not isinstance(selected_loss, int | float)
		or not np.isfinite(selected_loss)
	):
		raise TypeError('handoff checkpoint.selected_loss must be finite')
	trainability = _mapping(
		checkpoint.get('trainability_summary'), 'handoff checkpoint trainability summary'
	)
	if checkpoint.get('trainability_summary_sha256') != scientific_identity_sha256(
		trainability
	):
		raise ValueError('M5-U handoff trainability identity mismatch')
	optimizer_groups = checkpoint.get('optimizer_group_identity')
	if not isinstance(optimizer_groups, list) or not optimizer_groups:
		raise TypeError('handoff checkpoint.optimizer_group_identity is missing')
	embedding = _mapping(payload.get('embedding'), 'handoff embedding')
	for key in ('metadata_sha256', 'embeddings_sha256', 'valid_tokens_sha256'):
		if not _sha256(embedding.get(key)):
			raise TypeError(f'handoff embedding.{key} is missing')
	for key in ('root', 'metadata_path'):
		if not isinstance(embedding.get(key), str) or not embedding[key]:
			raise TypeError(f'handoff embedding.{key} is missing')
	if (
		embedding.get('embeddings_shape') != [76, 113, 32, 384]
		or embedding.get('embeddings_dtype') != 'float16'
		or embedding.get('valid_tokens_shape') != [76, 113, 32]
		or embedding.get('valid_tokens_dtype') != 'bool'
		or not isinstance(embedding.get('finite_valid_count'), int)
		or embedding['finite_valid_count'] <= 0
	):
		raise ValueError('M5-U handoff embedding identity mismatch')
	canonical_valid_tokens = _mapping(
		embedding.get('canonical_valid_token_identities'),
		'handoff embedding canonical valid-token identities',
	)
	if set(canonical_valid_tokens) != {'mae', 'current_k6', 'mh_nocons'}:
		raise ValueError('M5-U handoff canonical valid-token identities mismatch')
	for role, identity in canonical_valid_tokens.items():
		reference = _mapping(identity, f'handoff embedding {role} valid-token identity')
		if not isinstance(reference.get('path'), str) or not _sha256(reference.get('sha256')):
			raise TypeError(
				f'handoff embedding {role} canonical valid-token identity is missing'
			)
		if reference['sha256'] != embedding['valid_tokens_sha256']:
			raise ValueError('M5-U handoff canonical valid-token identity mismatch')
	return payload


def validate_f3_m5_soft_posterior_pretraining(
	config: F3M5SoftPosteriorPretrainingValidationConfig,
	*,
	phase: str,
	dry_run: bool = False,
	only_missing: bool = False,
	quarantine_invalid: bool = False,
) -> F3M5SoftPosteriorPretrainingValidationResult:
	"""Validate targets, full checkpoints, and extraction before PASS publication."""
	if phase not in {'targets', 'smoke', 'checkpoints', 'complete'}:
		raise ValueError('phase must be targets, smoke, checkpoints, or complete')
	try:
		posterior = load_multi_head_state_posterior_manifest(config.posterior_manifest)
		hard, soft = _training_config(config.hard_full_config), _training_config(
			config.soft_full_config
		)
		target_evidence = _validate_target_contract(config, posterior, hard, soft)
		if phase == 'targets':
			return F3M5SoftPosteriorPretrainingValidationResult(
				phase, {'status': 'PASS', **target_evidence}, None
			)
		if phase == 'smoke':
			smoke = _training_config(config.soft_smoke_config)
			smoke_evidence = _smoke_evidence(
				config,
				full=soft,
				smoke=smoke,
				hard_trainability_summary=_mapping(
					target_evidence['hard_baseline_trainability_summary'],
					'hard baseline trainability summary',
				),
				hard_optimizer_group_identity=target_evidence[
					'hard_baseline_optimizer_group_identity'
				],
			)
			return F3M5SoftPosteriorPretrainingValidationResult(
				phase,
				{'status': 'PASS', **target_evidence, 'smoke': smoke_evidence},
				None,
			)
		checkpoint = _checkpoint_evidence(
			soft,
			hard_trainability_summary=_mapping(
				target_evidence['hard_baseline_trainability_summary'],
				'hard baseline trainability summary',
			),
			hard_optimizer_group_identity=target_evidence[
				'hard_baseline_optimizer_group_identity'
			],
		)
		evidence: dict[str, object] = {'status': 'PASS', **target_evidence, **checkpoint}
		if phase == 'checkpoints':
			if not dry_run:
				hard_validation._atomic_json(
					Path(checkpoint['root']) / 'preflight' / 'checkpoint_validation.json',
					{'artifact_type': 'f3_m5_soft_posterior_validation', 'schema_version': 1, 'status': 'PASS'},
				)
			return F3M5SoftPosteriorPretrainingValidationResult(phase, evidence, None)
		evidence['embedding'] = _embedding_evidence(config, checkpoint)
		handoff = _handoff(evidence)
		path = Path(checkpoint['root']) / 'preflight' / 'soft_posterior_handoff.json'
		if dry_run:
			return F3M5SoftPosteriorPretrainingValidationResult(phase, evidence, None)
		published = _publish_handoff(
			path, handoff, only_missing=only_missing, quarantine_invalid=quarantine_invalid
		)
		return F3M5SoftPosteriorPretrainingValidationResult(
			phase, evidence, path if published else None
		)
	except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
		if not dry_run:
			raise
		return F3M5SoftPosteriorPretrainingValidationResult(
			phase,
			{'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'},
			None,
		)


def _training_config(path: Path) -> Mapping[str, object]:
	return resolve_strat_hmm_pretext_config(load_config(path))


def _validate_target_contract(  # noqa: C901
	config: F3M5SoftPosteriorPretrainingValidationConfig,
	posterior: Mapping[str, object],
	hard: Mapping[str, object],
	soft: Mapping[str, object],
) -> dict[str, object]:
	if posterior.get('head_ks') != [6, 8, 10]:
		raise ValueError('posterior manifest K identity mismatch')
	if Path(str(_mapping(soft['pseudo_targets'], 'soft pseudo_targets')['manifest'])).resolve() != config.posterior_manifest:
		raise ValueError('soft posterior manifest path mismatch')
	soft_identity = _mapping(_mapping(soft['identity'], 'soft identity')['scientific_identity'], 'soft scientific identity')
	if soft_identity.get('posterior_manifest_sha256') != file_sha256(config.posterior_manifest):
		raise ValueError('soft posterior manifest SHA-256 mismatch')
	if _mapping(soft['identity'], 'soft identity').get('model_tag') != _MODEL_TAG:
		raise ValueError('soft model tag mismatch')
	if _mapping(hard['identity'], 'hard identity').get('model_tag') != _HARD_MODEL_TAG:
		raise ValueError('hard baseline model tag mismatch')
	hard_manifest = Path(
		str(_mapping(hard['pseudo_targets'], 'hard pseudo targets')['manifest'])
	).resolve()
	posterior_source_hard_manifest = _mapping(
		posterior['source_hard_manifest'], 'posterior source hard manifest'
	)
	if (
		Path(str(posterior_source_hard_manifest.get('path', ''))).resolve()
		!= hard_manifest
		or posterior_source_hard_manifest.get('sha256') != file_sha256(hard_manifest)
	):
		raise ValueError(
			'posterior source hard manifest does not match hard baseline target manifest'
		)
	for label, training, model_tag in (
		('soft', soft, _MODEL_TAG),
		('hard', hard, _HARD_MODEL_TAG),
	):
		output = Path(
			str(_mapping(training['paths'], f'{label} paths')['output_root'])
		).resolve()
		if output != (config.experiment_root / model_tag).resolve():
			raise ValueError(f'{label} output root mismatch')
		ensure_under_root(output, root=config.artifact_root, label=f'{label}.output_root')
	_validate_allowed_config_delta(hard, soft)
	hard_handoff = hard_validation.load_f3_multi_head_pretraining_handoff(config.hard_handoff)
	if hard_handoff.get('model_tag') != _HARD_MODEL_TAG:
		raise ValueError('hard baseline handoff model tag mismatch')
	hard_checkpoint = _hard_baseline_checkpoint_evidence(hard, hard_handoff)
	student_hash, head_hash = _initial_hashes(soft)
	hard_targets = _mapping(hard_handoff['stratigraphy_pretext'], 'hard handoff targets')
	if (hard_targets.get('initial_student_state_sha256'), hard_targets.get('initial_head_state_sha256')) != (student_hash, head_hash):
		raise ValueError('soft initial student/head hashes differ from hard baseline')
	posterior_head_hashes = _multi_head_posterior_hashes(posterior)
	_validate_posterior_head_hashes(posterior_head_hashes)
	return {
		'target_representation': soft_identity['target_representation'],
		'posterior_manifest_path': str(config.posterior_manifest),
		'posterior_manifest_sha256': file_sha256(config.posterior_manifest),
		'posterior_semantics': posterior['posterior_semantics'],
		'posterior_cost_temperature': posterior['cost_temperature'],
		'posterior_head_hashes': posterior_head_hashes,
		'posterior_source_hard_manifest': posterior_source_hard_manifest,
		'posterior_source_embedding': posterior['source_embedding'],
		'posterior_head_diagnostics': {
			str(k): _mapping(
				_mapping(posterior['heads'], 'posterior heads')[str(k)],
				f'posterior head {k}',
			)['diagnostics']
			for k in (6, 8, 10)
		},
		'initial_student_state_sha256': student_hash,
		'initial_head_state_sha256': head_hash,
		'hard_baseline_config': str(config.hard_full_config),
		'hard_baseline_handoff': str(config.hard_handoff),
		**hard_checkpoint,
	}


def _validate_allowed_config_delta(hard: Mapping[str, object], soft: Mapping[str, object]) -> None:
	left, right = json.loads(json.dumps(hard)), json.loads(json.dumps(soft))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		_mapping(value['identity'], 'identity').pop('model_tag', None)
		pseudo_targets = _mapping(value['pseudo_targets'], 'pseudo targets')
		pseudo_targets.pop('manifest', None)
		pseudo_targets.pop('target_representation', None)
		scientific = _mapping(_mapping(value['identity'], 'identity')['scientific_identity'], 'scientific identity')
		for key in (
			'experiment_role', 'variant', 'target_representation', 'target_manifest_sha256',
			'target_head_hashes', 'posterior_manifest_sha256', 'posterior_semantics',
			'posterior_cost_temperature', 'posterior_head_hashes', 'supervised_loss',
			'consistency_policy',
		):
			scientific.pop(key, None)
	if left != right:
		raise ValueError('hard/soft scientific config drift outside target representation')


def _initial_hashes(training: Mapping[str, object]) -> tuple[str, str]:
	seed = _mapping(training['train'], 'train').get('seed')
	if isinstance(seed, bool) or not isinstance(seed, int):
		raise TypeError('train.seed must be an integer')
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		components = build_strat_hmm_components(training, device='cpu')
	heads = getattr(components, 'heads', None)
	if not isinstance(heads, torch.nn.Module):
		raise TypeError('soft validation requires multi-head initialization')
	return (
		hard_validation._state_sha256(components.student.state_dict()),
		hard_validation._state_sha256(heads.state_dict()),
	)


def _hard_baseline_checkpoint_evidence(
	hard: Mapping[str, object], hard_handoff: Mapping[str, object]
) -> dict[str, object]:
	"""Bind the supplied hard config to the checkpoint named by its handoff."""
	record = _mapping(hard_handoff.get('checkpoint'), 'hard handoff checkpoint')
	path = Path(str(record.get('path', ''))).resolve()
	if not path.is_file():
		raise FileNotFoundError('hard baseline handoff checkpoint is missing')
	if record.get('sha256') != file_sha256(path):
		raise ValueError('hard baseline handoff checkpoint SHA-256 mismatch')
	payload = _torch_mapping(path)
	validate_stratigraphy_checkpoint_payload(payload, expected_config=hard)
	checkpoint_config = _mapping(
		payload.get('stratigraphy_config'), 'hard baseline checkpoint config'
	)
	if checkpoint_config != hard:
		raise ValueError(
			'hard baseline config does not match canonical hard handoff checkpoint'
		)
	identity = _mapping(
		payload.get('stratigraphy_checkpoint'), 'hard baseline checkpoint identity'
	)
	handoff_identity = _mapping(
		hard_handoff.get('stratigraphy_pretext'), 'hard handoff stratigraphy identity'
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
			raise ValueError(
				f'hard baseline handoff checkpoint identity mismatch: {key}'
			)
	if _mapping(identity.get('target_manifest'), 'hard checkpoint target manifest').get(
		'sha256'
	) != handoff_identity.get('target_manifest_sha256'):
		raise ValueError('hard baseline handoff checkpoint target manifest mismatch')
	if identity.get('per_head_targets') != handoff_identity.get(
		'per_head_target_sha256'
	):
		raise ValueError('hard baseline handoff checkpoint target heads mismatch')
	trainability = _mapping(
		payload.get('trainability_summary'), 'hard baseline trainability summary'
	)
	optimizer_groups = identity.get('optimizer_group_identity')
	if not isinstance(optimizer_groups, list) or not optimizer_groups:
		raise TypeError('hard baseline optimizer group identity is missing')
	return {
		'hard_baseline_checkpoint': str(path),
		'hard_baseline_checkpoint_sha256': record['sha256'],
		'hard_baseline_trainability_summary': dict(trainability),
		'hard_baseline_optimizer_group_identity': optimizer_groups,
	}


def _checkpoint_evidence(  # noqa: C901
	training: Mapping[str, object],
	*,
	hard_trainability_summary: Mapping[str, object],
	hard_optimizer_group_identity: object,
	expected_global_step: int = 25600,
	require_full_epoch_history: bool = True,
) -> dict[str, object]:
	root = Path(str(_mapping(training['paths'], 'paths')['output_root']))
	latest_path, best_path = root / 'latest.pt', root / 'best.pt'
	if not latest_path.is_file() or not best_path.is_file():
		raise FileNotFoundError('soft run requires latest.pt and best.pt')
	latest, best = _torch_mapping(latest_path), _torch_mapping(best_path)
	for payload in (latest, best):
		validate_stratigraphy_checkpoint_payload(payload, expected_config=training)
		hard_validation._metrics_finite(payload)
		identity = _mapping(payload['stratigraphy_checkpoint'], 'soft checkpoint identity')
		if identity.get('schema_version') != 3 or identity.get('model_tag') != _MODEL_TAG:
			raise ValueError('soft checkpoint identity mismatch')
		if identity.get('scientific_identity_sha256') != scientific_identity_sha256(
			_mapping(_mapping(training['identity'], 'identity')['scientific_identity'], 'scientific identity')
		):
			raise ValueError('soft checkpoint scientific identity mismatch')
		if _mapping(
			payload.get('trainability_summary'), 'soft checkpoint trainability summary'
		) != hard_trainability_summary:
			raise ValueError(
				'soft checkpoint trainability differs from hard baseline'
			)
		if identity.get('optimizer_group_identity') != hard_optimizer_group_identity:
			raise ValueError(
				'soft checkpoint optimizer groups differ from hard baseline'
			)
	student_hash, head_hash = _initial_hashes(training)
	identity = _mapping(best['stratigraphy_checkpoint'], 'soft checkpoint identity')
	if (identity.get('initial_student_state_sha256'), identity.get('initial_head_state_sha256')) != (student_hash, head_hash):
		raise ValueError('soft checkpoint initial state hash mismatch')
	if require_full_epoch_history:
		if latest.get('epoch') != 25 or latest.get('global_step') != expected_global_step:
			raise ValueError('soft full run must finish epoch 25/global step 25600')
		rows = hard_validation._epoch_rows(root / 'multi_head_epoch_metrics.csv')
		if (
			[row['epoch'] for row in rows] != list(range(1, 26))
			or rows[-1]['global_step'] != expected_global_step
		):
			raise ValueError('soft epoch metrics coverage is incomplete')
	elif latest.get('global_step') != expected_global_step:
		raise ValueError(
			f'soft smoke must finish at global step {expected_global_step}'
		)
	selection = hard_validation._validate_best_selection(best, latest, variant='soft_nocons')
	hard_validation._validate_freeze_contract(best, training)
	return {
		'root': root,
		'best_path': best_path,
		'latest_path': latest_path,
		'best': best,
		'latest': latest,
		'identity': identity,
		'selection': selection,
	}


def _smoke_evidence(
	config: F3M5SoftPosteriorPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
	hard_trainability_summary: Mapping[str, object],
	hard_optimizer_group_identity: object,
) -> dict[str, object]:
	"""Validate the isolated CPU two-step M5-U smoke without full-run rules."""
	_validate_smoke_config(config, full=full, smoke=smoke)
	full_initial_hashes = _initial_hashes(full)
	if _initial_hashes(smoke) != full_initial_hashes:
		raise ValueError('soft smoke initial state hashes differ from soft full config')
	evidence = _checkpoint_evidence(
		smoke,
		hard_trainability_summary=hard_trainability_summary,
		hard_optimizer_group_identity=hard_optimizer_group_identity,
		expected_global_step=2,
		require_full_epoch_history=False,
	)
	latest = _mapping(evidence['latest'], 'soft smoke latest checkpoint')
	state = _mapping(latest.get('training_state'), 'soft smoke training state')
	if latest.get('epoch') != 1 or state.get('checkpoint_kind') != 'step':
		raise ValueError('soft smoke must end with a two-step partial epoch checkpoint')
	identity = _mapping(evidence['identity'], 'soft smoke checkpoint identity')
	if (
		identity.get('target_representation') != 'ordered_path_state_posterior_v1'
		or identity.get('consistency_weight') != 0.0
	):
		raise ValueError('soft smoke target representation or consistency mismatch')
	return evidence


def _validate_smoke_config(
	config: F3M5SoftPosteriorPretrainingValidationConfig,
	*,
	full: Mapping[str, object],
	smoke: Mapping[str, object],
) -> None:
	"""Allow only the isolated CPU two-step execution delta from the full config."""
	full_root = Path(
		str(_mapping(full['paths'], 'soft full paths')['output_root'])
	).resolve()
	smoke_root = Path(
		str(_mapping(smoke['paths'], 'soft smoke paths')['output_root'])
	).resolve()
	ensure_under_root(full_root, root=config.artifact_root, label='soft full output root')
	ensure_under_root(smoke_root, root=config.artifact_root, label='soft smoke output root')
	if smoke_root == full_root:
		raise ValueError('soft smoke output root must differ from soft full output root')
	if full_root.exists():
		raise ValueError('soft full output root must remain unmodified during smoke')
	smoke_train = _mapping(smoke['train'], 'soft smoke train')
	if smoke_train.get('device') != 'cpu' or smoke_train.get('max_steps') != 2:
		raise ValueError('soft smoke must use device=cpu and max_steps=2')
	left, right = json.loads(json.dumps(full)), json.loads(json.dumps(smoke))
	for value in (left, right):
		_mapping(value['paths'], 'paths').pop('output_root', None)
		runtime = _mapping(value['identity'], 'identity').get('runtime_identity')
		if runtime is not None:
			_mapping(runtime, 'runtime identity').pop('device', None)
		train = _mapping(value['train'], 'train')
		train.pop('device', None)
		train.pop('max_steps', None)
		_mapping(
			_mapping(value['identity'], 'identity')['scientific_identity'],
			'scientific identity',
		)['train'].pop('max_steps', None)
	if left != right:
		raise ValueError('soft smoke config drift outside CPU two-step execution settings')


def _embedding_evidence(
	config: F3M5SoftPosteriorPretrainingValidationConfig,
	checkpoint: Mapping[str, object],
) -> dict[str, object]:
	root = config.artifact_root / 'embeddings/f3/facies_benchmark_v1' / _MODEL_TAG / 'overlap_x16'
	files = output_paths(root, 'f3_facies_benchmark')
	if not all(path.is_file() for path in (files.embeddings, files.valid_tokens, files.metadata)):
		raise FileNotFoundError('soft complete embedding artifacts are missing')
	metadata = _mapping(
		json.loads(files.metadata.read_text(encoding='utf-8')), 'embedding metadata'
	)
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
		'posterior_semantics',
		'posterior_cost_temperature',
		'scientific_identity_sha256',
	):
		if stratigraphy.get(key) != identity.get(key):
			raise ValueError(f'embedding stratigraphy identity mismatch: {key}')
	if stratigraphy.get('posterior_manifest_sha256') != _mapping(
		identity['posterior_manifest'], 'checkpoint posterior manifest'
	).get('sha256'):
		raise ValueError('embedding posterior manifest identity mismatch')
	embeddings, valid = np.load(files.embeddings, mmap_mode='r'), np.load(files.valid_tokens, mmap_mode='r')
	if embeddings.shape != (76, 113, 32, 384) or embeddings.dtype != np.float16 or valid.shape != (76, 113, 32) or valid.dtype != np.bool_ or not int(valid.sum()) or not np.isfinite(embeddings[valid]).all():
		raise ValueError('soft embedding shape/dtype/finite contract mismatch')
	canonical_valid_tokens = _canonical_valid_token_identities(config)
	valid_tokens_sha256 = file_sha256(files.valid_tokens)
	if any(
		identity['sha256'] != valid_tokens_sha256
		for identity in canonical_valid_tokens.values()
	):
		raise ValueError(
			'soft embedding valid-token identity differs from a canonical baseline'
		)
	return {
		'root': str(root),
		'metadata_path': str(files.metadata),
		'metadata_sha256': file_sha256(files.metadata),
		'embeddings_sha256': file_sha256(files.embeddings),
		'valid_tokens_sha256': valid_tokens_sha256,
		'canonical_valid_token_identities': canonical_valid_tokens,
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'finite_valid_count': int(valid.sum()),
	}


def _handoff(evidence: Mapping[str, object]) -> dict[str, object]:
	selection = _mapping(evidence['selection'], 'checkpoint selection')
	selected = _mapping(selection['selected'], 'selected checkpoint')
	embedding = _mapping(evidence['embedding'], 'embedding')
	target_keys = (
		'target_representation',
		'posterior_manifest_path',
		'posterior_manifest_sha256',
		'posterior_semantics',
		'posterior_cost_temperature',
		'posterior_head_hashes',
		'posterior_source_hard_manifest',
		'posterior_source_embedding',
		'posterior_head_diagnostics',
		'initial_student_state_sha256',
		'initial_head_state_sha256',
		'hard_baseline_config',
		'hard_baseline_handoff',
	)
	return {
		'artifact_type': _HANDOFF_TYPE,
		'schema_version': 1,
		'status': 'PASS',
		'model_tag': _MODEL_TAG,
		'variant': 'soft_nocons',
		'targets': {key: evidence[key] for key in target_keys},
		'checkpoint': {
			'path': str(evidence['best_path']),
			'sha256': file_sha256(Path(evidence['best_path'])),
			'selected_epoch': selected['epoch'],
			'selected_global_step': selected['global_step'],
			'selected_checkpoint_kind': selected['checkpoint_kind'],
			'selected_loss': selected['loss'],
			'selection_history_sha256': selection['sha256'],
			'optimizer_group_identity': _mapping(evidence['identity'], 'identity')[
				'optimizer_group_identity'
			],
			'trainability_summary': _mapping(
				_mapping(evidence['best'], 'best checkpoint').get('trainability_summary'),
				'best checkpoint trainability summary',
			),
			'trainability_summary_sha256': scientific_identity_sha256(
				_mapping(
					_mapping(evidence['best'], 'best checkpoint').get(
						'trainability_summary'
					),
					'best checkpoint trainability summary',
				)
			),
		},
		'embedding': dict(embedding),
	}


def _canonical_valid_token_identities(
	config: F3M5SoftPosteriorPretrainingValidationConfig,
) -> dict[str, dict[str, str]]:
	"""Return the three bitwise-identical F3 valid-token references."""
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


def _publish_handoff(path: Path, handoff: Mapping[str, object], *, only_missing: bool, quarantine_invalid: bool) -> bool:
	if path.is_file():
		try:
			existing = load_f3_m5_soft_posterior_pretraining_handoff(path)
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			existing = None
		if existing == handoff and only_missing:
			return False
		if existing != handoff:
			if not quarantine_invalid:
				raise ValueError('existing handoff is stale or invalid; pass --quarantine-invalid to replace it')
			hard_validation._quarantine(path)
	hard_validation._atomic_json(path, handoff)
	return True


def _torch_mapping(path: Path) -> Mapping[str, object]:
	payload = torch.load(path, map_location='cpu', weights_only=False)
	return _mapping(payload, f'checkpoint {path}')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _sha256(value: object) -> bool:
	return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value.lower())


def _validate_posterior_head_hashes(value: object) -> None:
	"""Require the complete K6/K8/K10 posterior artifact hash evidence."""
	head_hashes = _mapping(value, 'posterior head hashes')
	if set(head_hashes) != {'6', '8', '10'}:
		raise ValueError('posterior head hashes K keys mismatch')
	for head_k, surveys in head_hashes.items():
		if not isinstance(surveys, Mapping) or not surveys:
			raise TypeError(f'posterior head hashes K{head_k} must contain surveys')
		for survey_id, artifacts in surveys.items():
			artifact_hashes = _mapping(
				artifacts, f'posterior head hashes K{head_k} survey {survey_id}'
			)
			for name in ('posterior', 'valid_tokens', 'metadata'):
				if not _sha256(artifact_hashes.get(name)):
					raise TypeError(
					f'posterior head hashes K{head_k} survey {survey_id}.{name} is missing'
				)
