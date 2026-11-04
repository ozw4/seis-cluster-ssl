"""Categorical target, order invariance, and checkpoint regression tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from seis_ssl_cluster.clustering.kmeans_targets import (
	CONSISTENCY_POLICY,
	TARGET_REPRESENTATION,
	build_kmeans_targets,
	file_reference,
	load_unordered_target_manifest,
)
from seis_ssl_cluster.config import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.data.target_providers import load_strat_multi_head_target_manifest
from seis_ssl_cluster.embedding.extractor import _stratigraphy_pretext_metadata
from seis_ssl_cluster.stratigraphy import MultiResolutionOrderedPrototypeHeads
from seis_ssl_cluster.stratigraphy.multi_head import load_multi_head_target_manifest
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm import (
	compute_strat_hmm_multi_head_losses,
	run_strat_hmm_pretext_training,
	runner,
)
from seis_ssl_cluster.training.strat_hmm import losses as losses_module
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)
from tests.seis_ssl_cluster.test_minibatch_kmeans import (
	_config,
	_write_embedding_artifacts,
)
from tests.seis_ssl_cluster.test_strat_hmm_pretraining_head_only import _raw_config


def _targets(tmp_path, *, shape=(4, 4, 4), margins=(1, 1, 0)):
	inputs = tmp_path / 'embeddings'
	inputs.mkdir()
	features = np.random.default_rng(9).normal(size=(*shape, 4)).astype(np.float32)
	_write_embedding_artifacts(
		inputs, 'survey', embeddings=features, valid=np.ones(shape, dtype=bool)
	)
	config = _config(inputs, tmp_path / 'outputs' / 'clustering')
	config.update(
		output_root=str(tmp_path / 'outputs'), edge_margin_tokens=list(margins)
	)
	path = tmp_path / 'targets.yaml'
	path.write_text(yaml.safe_dump(config))
	return build_kmeans_targets(config, path), config


def _training_config(tmp_path, manifest):
	config = _raw_config(
		tmp_path, encoder_depth=2, unfreeze_top_blocks=1, distillation_weight=0.2
	)
	config['student']['init_checkpoint'] = config['teacher']['checkpoint']
	config['pseudo_targets'] = {
		'manifest': str(manifest),
		'min_confidence': 0.0,
		'target_representation': TARGET_REPRESENTATION,
	}
	config['head'] = {
		'spec': 'multi_resolution_ordered_prototypes_v1',
		'ks': [2, 3],
		'projection_dim': 6,
		'temperature': 0.5,
		'normalize': True,
	}
	config['loss'].update(consistency_weight=0.0, consistency_beta=0.1)
	config['identity'] = {
		'model_tag': 'kmeans_fixture',
		'scientific_identity': {
			'experiment_role': 'multi_head_unordered_kmeans_pretext',
			'variant': 'kmeans_nocons',
			'head_spec': config['head']['spec'],
			'head_ks': [2, 3],
			'target_representation': TARGET_REPRESENTATION,
			'target_manifest_sha256': file_reference(manifest)['sha256'],
			'consistency_policy': CONSISTENCY_POLICY,
		},
	}
	return config


def test_kmeans_roundtrip_preserves_support_and_accepts_reverse_paths(tmp_path):
	manifest, config = _targets(tmp_path)
	payload = load_unordered_target_manifest(manifest)
	for k in (2, 3):
		entry = payload['heads'][str(k)]['surveys']['survey']
		labels = np.load(entry['labels']['path'])
		valid = np.load(entry['valid_tokens']['path'])
		assert valid.sum() == 16
		assert np.all(labels[~valid] == -1)
		assert np.any(np.diff(labels[1:3, 1:3], axis=2) < 0)
	adapted = load_strat_multi_head_target_manifest(manifest)
	assert adapted.by_survey['survey'][0].target_representation == TARGET_REPRESENTATION
	assert build_kmeans_targets(config, tmp_path / 'targets.yaml') == manifest
	with pytest.raises(ValueError, match=r'schema|manifest|fields'):
		load_multi_head_target_manifest(manifest)


@pytest.mark.parametrize('corruption', ['hash', 'range', 'support'])
def test_unordered_manifest_rejects_corruption(tmp_path, corruption):
	manifest, _ = _targets(tmp_path)
	payload = json.loads(manifest.read_text())
	entry = payload['heads']['2']['surveys']['survey']
	path = Path(entry['labels']['path'])
	labels = np.load(path)
	labels[1, 1, 0] = 99 if corruption == 'range' else 1 - labels[1, 1, 0]
	if corruption == 'support':
		valid_path = Path(entry['valid_tokens']['path'])
		valid = np.load(valid_path)
		valid[1, 1, 0] = False
		np.save(valid_path, valid)
		entry['valid_tokens'] = file_reference(valid_path)
	np.save(path, labels)
	if corruption != 'hash':
		entry['labels'] = file_reference(path)
		manifest.write_text(json.dumps(payload))
	with pytest.raises(ValueError, match=r'hash|range|valid|invalid|mask'):
		load_unordered_target_manifest(manifest)


def test_unordered_loss_is_invariant_to_independent_cluster_permutations(monkeypatch):

	def forbidden(*_args, **_kwargs):
		raise AssertionError('ordered consistency must not be evaluated')

	monkeypatch.setattr(losses_module, '_multi_head_consistency_losses', forbidden)
	torch.manual_seed(4)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=4, ks=[2, 3], projection_dim=3, temperature=0.1, normalize=True
	)
	tokens = torch.randn(1, 6, 4)
	batch = {
		'strat_multi_targets': {
			f'k{k}': {
				'labels': torch.arange(6).reshape(1, 6) % k,
				'confidence': torch.ones(1, 6),
				'boundary_weight': torch.ones(1, 6),
				'valid_mask': torch.ones(1, 6, dtype=torch.bool),
			}
			for k in (2, 3)
		}
	}
	kwargs = {
		'heads': heads,
		'encoded': {'tokens': tokens},
		'teacher_encoded': {'tokens': tokens + 0.1},
		'batch': batch,
		'loss_config': {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
			'consistency_weight': 0.0,
		},
		'pseudo_target_config': {'target_representation': TARGET_REPRESENTATION},
	}
	before = compute_strat_hmm_multi_head_losses(**kwargs)
	with torch.no_grad():
		for k, permutation in ((2, torch.tensor([1, 0])), (3, torch.tensor([2, 0, 1]))):
			heads.heads[f'k{k}'].prototypes.copy_(
				heads.heads[f'k{k}'].prototypes[permutation]
			)
			target = batch['strat_multi_targets'][f'k{k}']
			target['labels'] = torch.argsort(permutation)[target['labels']]
	after = compute_strat_hmm_multi_head_losses(**kwargs)
	assert torch.allclose(before['loss'], after['loss'])
	assert before['loss_consistency'].item() == 0
	assert not any(key.startswith('loss_consistency_k') for key in before)
	kwargs['loss_config']['consistency_weight'] = 0.1
	with pytest.raises(ValueError, match='zero consistency'):
		compute_strat_hmm_multi_head_losses(**kwargs)


def test_unordered_cpu_training_checkpoint_and_resume(tmp_path, monkeypatch):
	manifest, _ = _targets(tmp_path, shape=(2, 2, 2), margins=(0, 0, 0))
	raw = _training_config(tmp_path, manifest)
	config = resolve_strat_hmm_pretext_config(raw)
	config['train']['checkpoint_every_steps'] = 1

	original_epoch = runner.train_strat_hmm_multi_head_one_epoch

	def interrupted_epoch(**kwargs):
		original_callback = kwargs['step_callback']

		def interrupt_after_checkpoint(state):
			original_callback(state)
			raise RuntimeError('simulated interruption')

		kwargs['step_callback'] = interrupt_after_checkpoint
		return original_epoch(**kwargs)

	monkeypatch.setattr(
		runner, 'train_strat_hmm_multi_head_one_epoch', interrupted_epoch
	)
	with pytest.raises(RuntimeError, match='simulated interruption'):
		run_strat_hmm_pretext_training(config)
	checkpoint = Path(config['paths']['output_root']) / 'latest.pt'
	assert load_checkpoint(checkpoint, map_location='cpu')['global_step'] == 1
	monkeypatch.setattr(runner, 'train_strat_hmm_multi_head_one_epoch', original_epoch)
	assert run_strat_hmm_pretext_training(config, resume=checkpoint) == checkpoint
	payload = load_checkpoint(checkpoint, map_location='cpu')
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)
	assert payload['stratigraphy_checkpoint']['schema_version'] == 9
	metadata = _stratigraphy_pretext_metadata(payload)
	assert metadata['target_representation'] == TARGET_REPRESENTATION
	assert metadata['clustering_method'] == 'minibatch_kmeans'
	assert payload['global_step'] == 2
	assert payload['metrics']['loss_consistency'] == 0
	changed = copy.deepcopy(config)
	changed['loss']['consistency_weight'] = 0.1
	with pytest.raises(ValueError, match=r'incompatible|consistency|identity'):
		validate_stratigraphy_checkpoint_payload(payload, expected_config=changed)
	with pytest.raises(ValueError, match='zero consistency'):
		resolve_strat_hmm_pretext_config(
			{**raw, 'loss': {**raw['loss'], 'consistency_weight': 0.1}}
		)
