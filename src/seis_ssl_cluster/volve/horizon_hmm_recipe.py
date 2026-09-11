"""Audit fixed-budget HMM continuation of a declared Volve encoder recipe."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.volve.horizon_five_way_sources import (
	_reject_trace_drop,
	_validate_hmm_checkpoint,
	_validate_pseudo_target_lineage,
	_validate_recorded_parent_sha,
	_validate_stage2_budget,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_recipe_arm import VolveHorizonRecipeArmConfig


def audit_hmm_recipe_checkpoint(
	config: VolveHorizonRecipeArmConfig,
) -> dict[str, object]:
	"""Verify complete HMM training and its live parent and target identities."""
	from seis_ssl_cluster.volve.horizon_recipe_arm import (  # noqa: PLC0415
		_audit_recipe_arm_checkpoint,
		_checkpoint_sha256,
		_load_checkpoint,
		_recipe_identity,
		_required_mapping,
		_validate_declared_mapping,
		as_five_way_config,
	)

	if config.hmm is None:
		raise ValueError('HMM recipe audit requires arm.hmm')
	hmm = config.hmm
	label = config.arm_id
	payload = _load_checkpoint(label, config.arm_checkpoint)
	base_config = _required_mapping(payload, 'config', label)
	_validate_stage2_budget(label, payload, config=base_config)
	model = as_five_way_config(config).models[0]
	parent, pseudo = _validate_hmm_checkpoint(model, payload, base_config=base_config)
	if parent.resolve() != Path(str(hmm['init_checkpoint'])).resolve():
		raise ValueError('HMM parent differs from arm.hmm.init_checkpoint')
	parent_report = _audit_recipe_arm_checkpoint(
		replace(config, arm_checkpoint=parent, hmm=None)
	)
	parent_sha = str(parent_report['checkpoint_sha256'])
	_validate_recorded_parent_sha(
		label, payload, parent_path=parent, parent_sha=parent_sha, required=True
	)
	if (
		Path(str(pseudo['input_dir'])).resolve()
		!= Path(str(hmm['pseudo_targets_dir'])).resolve()
	):
		raise ValueError('HMM pseudo targets differ from the declared input')
	pseudo_identity = _validate_pseudo_target_lineage(
		label,
		pseudo,
		checkpoint_payload=payload,
		parent_path=parent,
		parent_sha=parent_sha,
		survey_id=config.survey_id,
	)
	stratigraphy = _required_mapping(payload, 'stratigraphy_config', label)
	_reject_trace_drop(stratigraphy, label)
	for section, expected in (
		(
			'loss',
			{
				'prototype_weight': 1.0,
				'usage_weight': 0.005,
				'entropy_floor': None,
				'distillation_weight': hmm['distillation_weight'],
			},
		),
		(
			'head',
			{
				'num_prototypes': 6,
				'projection_dim': 128,
				'temperature': 0.1,
				'normalize': True,
			},
		),
		(
			'train',
			{
				'epochs': 25,
				'batch_size': 4,
				'samples_per_epoch': 10000,
				'lr': 1.0e-5,
				'encoder_lr': 1.0e-5,
				'weight_decay': 0.05,
				'amp': False,
				'seed': 42,
				'shuffle': True,
				'grad_clip_norm': 1.0,
			},
		),
	):
		_validate_declared_mapping(
			f'{label} {section}',
			_required_mapping(stratigraphy, section, label),
			expected,
			subset=True,
		)
	cluster_identity = _audit_declared_clustering(config, parent_sha=parent_sha)
	return {
		'model_id': label,
		'role': 'recipe_arm',
		'checkpoint': str(config.arm_checkpoint),
		'checkpoint_sha256': _checkpoint_sha256(label, config.arm_checkpoint),
		'objective': model.expected['objective'],
		'recipe': _recipe_identity(config.recipe),
		'parent_checkpoint': str(parent),
		'parent_checkpoint_sha256': parent_sha,
		'hmm': dict(hmm),
		'pseudo_targets': pseudo_identity,
		'clustering': cluster_identity,
		'stage_2': {'epochs': 25, 'global_steps': 62500, 'unfreeze_top_blocks': 1},
	}


def _audit_declared_clustering(
	config: VolveHorizonRecipeArmConfig, *, parent_sha: str
) -> dict[str, object]:
	from seis_ssl_cluster.volve.horizon_recipe_arm import (  # noqa: PLC0415
		_required_mapping,
		_validate_declared_mapping,
	)

	if config.hmm is None:
		raise ValueError('HMM recipe audit requires arm.hmm')
	hmm = config.hmm
	declared = load_config(Path(str(hmm['clustering_config'])))
	embeddings = _required_mapping(declared, 'embeddings', 'clustering config')
	if (
		Path(str(embeddings['input_dir'])).resolve()
		!= Path(str(hmm['source_embeddings_dir'])).resolve()
	):
		raise ValueError('clustering source differs from arm.hmm.source_embeddings_dir')
	clustering = _required_mapping(declared, 'clustering', 'clustering config')
	output = Path(str(clustering['output_dir']))
	metadata_path = output / 'models/k6/clustering_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	_validate_declared_mapping(
		'clustering metadata',
		metadata,
		{
			'method': 'stratigraphic_hmm_kmeans',
			'k': 6,
			'normalization': 'l2',
			'random_seed': 42,
		},
		subset=True,
	)
	for section in ('pca', 'residualization', 'stratigraphic_hmm'):
		actual = _required_mapping(metadata, section, 'clustering metadata')
		expected = _required_mapping(clustering, section, 'clustering config')
		_validate_subset(f'clustering {section}', actual, expected)
	signature = _required_mapping(metadata, 'embedding_compatibility_signature', 'HMM')
	if signature.get('checkpoint_sha256') != parent_sha:
		raise ValueError('clustering embedding checkpoint SHA-256 differs from parent')
	_audit_source_embedding_metadata(config, metadata, parent_sha=parent_sha)
	pseudo_path = (
		Path(str(hmm['pseudo_targets_dir']))
		/ 'k6'
		/ (f'{config.survey_id}.pseudo_target_metadata.json')
	)
	pseudo = json.loads(pseudo_path.read_text(encoding='utf-8'))
	source = _required_mapping(pseudo, 'source', 'pseudo targets')
	if Path(str(source['source_clustering_output_dir'])).resolve() != output.resolve():
		raise ValueError('pseudo targets do not come from the declared clustering')
	return {
		'metadata': str(metadata_path),
		'metadata_sha256': file_sha256(metadata_path),
	}


def _audit_source_embedding_metadata(
	config: VolveHorizonRecipeArmConfig,
	metadata: Mapping[str, object],
	*,
	parent_sha: str,
) -> None:
	from seis_ssl_cluster.volve.horizon_recipe_arm import (  # noqa: PLC0415
		_required_mapping,
	)

	if config.hmm is None:
		raise ValueError('HMM recipe audit requires arm.hmm')
	inputs = metadata.get('embedding_inputs')
	if not isinstance(inputs, list) or len(inputs) != 1:
		raise ValueError('clustering must record exactly one Volve embedding input')
	source = _required_mapping({'source': inputs[0]}, 'source', 'clustering')
	directory = Path(str(config.hmm['source_embeddings_dir']))
	for key, suffix in (
		('metadata_path', 'embedding_metadata.json'),
		('embeddings_path', 'embeddings.npy'),
		('valid_tokens_path', 'valid_tokens.npy'),
	):
		expected_path = directory / f'{config.survey_id}.{suffix}'
		if Path(str(source.get(key))).resolve() != expected_path.resolve():
			raise ValueError(f'clustering source embedding {key} differs from config')
	path = Path(str(source['metadata_path']))
	if file_sha256(path) != source.get('metadata_sha256'):
		raise ValueError('clustering source embedding metadata SHA-256 mismatch')
	live = json.loads(path.read_text(encoding='utf-8'))
	_validate_subset(
		'source embedding metadata',
		live,
		{
			'survey_id': config.survey_id,
			'checkpoint_sha256': parent_sha,
			'window_size': [128, 128, 128],
			'overlap': [64, 64, 64],
			'output_dtype': 'float16',
			'min_token_valid_fraction': 1.0,
			'precision': {'amp_enabled': True, 'resolved_dtype': 'bfloat16'},
		},
	)


def _validate_subset(
	label: str, actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
	from seis_ssl_cluster.volve.horizon_recipe_arm import (  # noqa: PLC0415
		_required_mapping,
		_validate_declared_mapping,
	)

	for key, value in expected.items():
		if isinstance(value, Mapping):
			_validate_subset(
				f'{label}.{key}', _required_mapping(actual, key, label), value
			)
		else:
			_validate_declared_mapping(label, actual, {key: value}, subset=True)
