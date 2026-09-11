"""Read-only audit of completed hard Multi-Head continuation sources."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.config.io import _expand_environment_variables
from seis_ssl_cluster.stratigraphy.multi_head import (
	compare_k6_replay,
	load_multi_head_target_manifest,
)
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

if TYPE_CHECKING:
	from collections.abc import Mapping


def audit_multi_head_source(config_path: Path) -> dict[str, object]:
	"""Validate a final latest checkpoint without repairing or publishing artifacts."""
	raw = yaml.safe_load(config_path.read_text())
	manifest_path = Path(
		_expand_environment_variables(raw['pseudo_targets']['manifest'])
	)
	# Each candidate has its own live manifest digest; no shared shell hash is used.
	scientific = raw['identity']['scientific_identity']
	if (
		scientific['target_manifest_sha256']
		== '${SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256}'
	):
		scientific['target_manifest_sha256'] = file_sha256(manifest_path)
	config = resolve_strat_hmm_pretext_config(_expand_environment_variables(raw))
	train = config['train']
	if train['max_steps'] is not None:
		raise ValueError('completed source requires an unlimited full training recipe')
	if config['pseudo_targets']['target_representation'] != 'hard_viterbi_labels_v1':
		raise ValueError('source audit requires hard Viterbi targets')
	manifest = load_multi_head_target_manifest(manifest_path)
	if 6 in manifest['head_ks']:
		parity = compare_k6_replay(
			historical_root=manifest['heads']['6']['pseudo_target_root'],
			replay_root=manifest['k6_replay_parity']['replay_root'],
		)
		if not parity['exact']:
			raise ValueError('live K6 replay parity mismatch')
	checkpoint = Path(config['paths']['output_root']) / 'latest.pt'
	before = file_sha256(checkpoint)
	payload = load_checkpoint(checkpoint, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)
	if payload.get('stratigraphy_config') != config:
		raise ValueError('checkpoint resolved configuration differs from recipe')
	steps = train['epochs'] * math.ceil(
		train['samples_per_epoch'] / train['batch_size']
	)
	state = payload.get('training_state', {})
	if (
		payload.get('epoch') != train['epochs']
		or payload.get('global_step') != steps
		or state.get('stage') != 'train_strat_hmm_pretext'
		or state.get('checkpoint_kind') != 'epoch'
		or state.get('batch_index') is not None
	):
		raise ValueError('checkpoint is not a completed final epoch')
	if payload.get('amp_enabled') is not train['amp']:
		raise ValueError('checkpoint AMP state differs from recipe')
	_validate_frozen_state(payload, config)
	if file_sha256(checkpoint) != before:
		raise ValueError('checkpoint changed during source audit')
	return {
		'status': 'complete',
		'config': str(config_path),
		'model_tag': config['identity']['model_tag'],
		'checkpoint': {'path': str(checkpoint), 'sha256': before},
		'epoch': payload['epoch'],
		'global_step': payload['global_step'],
		'head_ks': manifest['head_ks'],
		'source_embedding': manifest['source_embedding'],
		'target_manifest': {
			'path': str(manifest_path),
			'sha256': file_sha256(manifest_path),
		},
		'stratigraphy_checkpoint': payload['stratigraphy_checkpoint'],
	}


def _validate_frozen_state(
	payload: Mapping[str, object], config: Mapping[str, object]
) -> None:
	student = config['student']
	if student['unfreeze_top_blocks'] != 1:
		raise ValueError('source audit requires top-1 continuation')
	parent = load_checkpoint(student['init_checkpoint'], map_location='cpu')
	initial, current = parent['model_state_dict'], payload['model_state_dict']
	prefix = f'encoder.layers.{config["model"]["encoder_depth"] - 1}.'
	names = payload.get('trainability_summary', {}).get('trainable_names')
	if not isinstance(names, list) or not names or len(names) != len(set(names)):
		raise ValueError('missing or invalid trainability evidence')
	if any(not name.startswith(prefix) or name not in current for name in names):
		raise ValueError('trainability evidence exceeds top-1 block')
	if set(initial) != set(current):
		raise ValueError('parent and student state keys differ')
	changed = False
	for name, value in current.items():
		if not torch.isfinite(value).all():
			raise ValueError('nonfinite student state')
		if not torch.equal(value, initial[name]):
			if name not in names:
				raise ValueError('frozen student parameter changed')
			changed = True
	if not changed:
		raise ValueError('top block did not update')
