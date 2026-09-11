"""Strict read-only audit of final F3 HMM checkpoints and live input lineage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import load_config, resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_STAGE2_EPOCHS,
	FIVE_WAY_STAGE2_GLOBAL_STEPS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.stratigraphy import discover_pseudo_target_inputs
from seis_ssl_cluster.training.random_checkpoint import (
	load_checkpoint_metadata_without_weights,
)


def audit_f3_hmm_final_sources(config_paths: Sequence[Path]) -> dict[str, object]:
	"""Audit each full training recipe; fail if any source is incomplete or stale."""
	if not config_paths:
		raise ValueError('at least one full training config is required')
	sources = [audit_f3_hmm_final_source(path) for path in config_paths]
	checkpoints = [source['checkpoint']['path'] for source in sources]
	if len(set(checkpoints)) != len(checkpoints):
		raise ValueError('full training configs must select distinct checkpoints')
	return {'status': 'complete', 'source_count': len(sources), 'sources': sources}


def audit_f3_hmm_final_source(config_path: Path) -> dict[str, object]:
	"""Verify a final epoch checkpoint against its full config and current inputs."""
	expected = resolve_strat_hmm_pretext_config(load_config(config_path))
	train = _mapping(expected, 'train')
	if (
		train.get('epochs') != FIVE_WAY_STAGE2_EPOCHS
		or train.get('samples_per_epoch') != 10_000
		or train.get('batch_size') != 16
		or train.get('max_steps') is not None
	):
		raise ValueError('audit requires the full F3 25-epoch / 15,625-step budget')
	checkpoint = Path(_mapping(expected, 'paths')['output_root']) / 'latest.pt'
	# A byte identity before and after metadata loading prevents a racing rolling
	# checkpoint replacement from mixing metadata and the reported source digest.
	checkpoint_identity = _file_identity(checkpoint)
	payload = load_checkpoint_metadata_without_weights(checkpoint)
	state = _validate_final_boundary(payload, checkpoint)
	actual = _mapping(payload, 'stratigraphy_config')
	config_sha256 = _config_sha256(expected)
	if _config_sha256(actual) != config_sha256:
		raise ValueError(f'{checkpoint}: resolved configuration differs from recipe')
	control = _mapping(payload, 'control_identity')
	identity = _mapping(expected, 'identity')
	if control.get('schema_version') != 1:
		raise ValueError(f'{checkpoint}: expected file-based control identity schema 1')
	if control.get('resolved_training_config_sha256') != config_sha256:
		raise ValueError(f'{checkpoint}: recorded resolved configuration SHA mismatch')
	for key in ('model_tag', 'scientific_identity'):
		if control.get(key) != identity.get(key, {}):
			raise ValueError(f'{checkpoint}: control identity {key} mismatch')
	if payload.get('amp_enabled') is not train.get('amp'):
		raise ValueError(f'{checkpoint}: checkpoint AMP state differs from recipe')
	live_inputs = _live_input_identities(expected)
	if _mapping(control, 'input_identities') != live_inputs:
		raise ValueError(f'{checkpoint}: live parent/target input identities mismatch')
	if _file_identity(checkpoint) != checkpoint_identity:
		raise ValueError(f'{checkpoint}: checkpoint changed during final-source audit')
	return {
		'config': str(config_path),
		'checkpoint': checkpoint_identity,
		'model_tag': identity['model_tag'],
		'epoch': payload['epoch'],
		'global_step': payload['global_step'],
		'training_state': dict(state),
		'resolved_training_config_sha256': config_sha256,
		'scientific_identity': identity.get('scientific_identity', {}),
		'input_identities': live_inputs,
		'status': 'complete',
	}


def _validate_final_boundary(
	payload: Mapping[str, object], checkpoint: Path
) -> Mapping[str, object]:
	state = _mapping(payload, 'training_state')
	if state.get('stage') != 'train_strat_hmm_pretext':
		raise ValueError(f'{checkpoint}: training_state.stage is not HMM pretraining')
	if state.get('checkpoint_kind') != 'epoch':
		raise ValueError(f'{checkpoint}: final checkpoint_kind must be epoch')
	if state.get('batch_index') is not None:
		raise ValueError(f'{checkpoint}: final epoch has a partial batch_index')
	if (
		payload.get('epoch') != FIVE_WAY_STAGE2_EPOCHS
		or payload.get('global_step') != FIVE_WAY_STAGE2_GLOBAL_STEPS
	):
		raise ValueError(f'{checkpoint}: final epoch/step budget is incomplete')
	return state


def _live_input_identities(config: Mapping[str, object]) -> dict[str, object]:
	teacher = Path(_mapping(config, 'teacher')['checkpoint'])
	student = Path(_mapping(config, 'student')['init_checkpoint'])
	if teacher != student:
		raise ValueError('F3 HMM teacher and student initialization must be identical')
	parent_identity = _file_identity(teacher)
	target_config = _mapping(config, 'pseudo_targets')
	target_inputs = discover_pseudo_target_inputs(
		Path(target_config['input_dir']), k=int(target_config['k'])
	)
	if not target_inputs:
		raise ValueError('live pseudo-target survey set is empty')
	targets: list[dict[str, object]] = []
	for item in target_inputs:
		entry: dict[str, object] = {
			'survey_id': item.survey_id,
			'labels': _file_identity(item.labels_path),
			'confidence': _file_identity(item.confidence_path),
			'valid_tokens': _file_identity(item.valid_tokens_path),
			'metadata': _file_identity(item.metadata_path),
			'boundary_weight_present': item.boundary_weight_path is not None,
		}
		if item.boundary_weight_path is not None:
			entry['boundary_weight'] = _file_identity(item.boundary_weight_path)
		targets.append(entry)
	return {
		'teacher_checkpoint': parent_identity,
		'student_init_checkpoint': parent_identity,
		'pseudo_targets': targets,
	}


def _file_identity(path: Path) -> dict[str, str]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {'path': str(path), 'sha256': file_sha256(path)}


def _config_sha256(config: Mapping[str, object]) -> str:
	serialized = json.dumps(
		config, sort_keys=True, separators=(',', ':'), allow_nan=False
	).encode('utf-8')
	return hashlib.sha256(serialized).hexdigest()


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = config.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value
