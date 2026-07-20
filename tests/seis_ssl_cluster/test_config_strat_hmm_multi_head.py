from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_config_strat_hmm_pretext import _minimal_config
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import _artifacts

if TYPE_CHECKING:
	from pathlib import Path


def test_multi_head_config_resolves_with_manifest_and_scientific_identity(
	tmp_path: Path,
) -> None:
	config = _multi_head_config(tmp_path)

	resolved = resolve_strat_hmm_pretext_config(config)

	assert resolved['head']['ks'] == [6, 8, 10]
	assert resolved['loss']['consistency_weight'] == 0.0
	assert resolved['loss']['consistency_beta'] == 0.1
	scientific = resolved['identity']['scientific_identity']
	assert set(scientific['target_head_hashes']) == {
		'6',
		'8',
		'10',
	}
	assert scientific['head_projection_dim'] == resolved['head']['projection_dim']
	assert scientific['head_temperature'] == resolved['head']['temperature']
	assert scientific['head_normalize'] == resolved['head']['normalize']
	assert scientific['prototype_weight'] == resolved['loss']['prototype_weight']
	assert scientific['usage_weight'] == resolved['loss']['usage_weight']
	assert scientific['consistency_weight'] == resolved['loss']['consistency_weight']
	assert scientific['distillation_weight'] == resolved['loss']['distillation_weight']
	assert scientific['model'] == resolved['model']
	assert scientific['data'] == resolved['data']
	assert scientific['zero_mask'] == resolved['zero_mask']


@pytest.mark.parametrize(
	'mutate',
	[
		lambda config: config['pseudo_targets'].update(input_dir='/invalid', k=6),
		lambda config: config['head'].update(num_prototypes=6),
		lambda config: config['loss'].update(consistency_weight=-0.1),
		lambda config: config['loss'].update(consistency_beta=0.0),
		lambda config: config['identity']['scientific_identity'].update(
			target_manifest_sha256='0' * 64
		),
		lambda config: config['identity']['scientific_identity'].update(
			head_projection_dim=256
		),
		lambda config: config['identity']['scientific_identity'].update(
			prototype_weight=0.5
		),
	],
)
def test_multi_head_config_rejects_incompatible_or_invalid_values(
	tmp_path: Path,
	mutate: object,
) -> None:
	config = _multi_head_config(tmp_path)
	mutate(config)  # type: ignore[operator]

	with pytest.raises((TypeError, ValueError)):
		resolve_strat_hmm_pretext_config(config)


@pytest.mark.parametrize('ks', [[8, 6, 10], [6, 6, 10], [True, 8], [1, 8], [6]])
def test_multi_head_config_rejects_invalid_head_ks(
	tmp_path: Path,
	ks: list[object],
) -> None:
	config = _multi_head_config(tmp_path)
	config['head']['ks'] = ks

	with pytest.raises((TypeError, ValueError), match=r'head\.ks'):
		resolve_strat_hmm_pretext_config(config)


def test_multi_head_identity_and_loss_policy_are_strict(tmp_path: Path) -> None:
	config = _multi_head_config(tmp_path)
	distillation_only = deepcopy(config)
	del config['identity']

	with pytest.raises(ValueError, match='identity is required'):
		resolve_strat_hmm_pretext_config(config)

	distillation_only['loss'].update(
		prototype_weight=0.0,
		usage_weight=0.0,
		consistency_weight=0.0,
		distillation_weight=0.2,
	)
	resolved = resolve_strat_hmm_pretext_config(distillation_only)
	assert resolved['loss']['distillation_weight'] == 0.2


def test_k6810_no_consistency_and_main_configs_have_a_pure_scientific_diff(
	tmp_path: Path,
) -> None:
	no_consistency = _multi_head_config(tmp_path)
	main = deepcopy(no_consistency)
	main['paths']['output_root'] = str(tmp_path / 'artifacts' / 'pretraining' / 'main')
	main['identity']['model_tag'] = 'strat_hmm_multi_k6810_main_v1'
	main['loss']['consistency_weight'] = 0.1

	no_consistency_resolved = resolve_strat_hmm_pretext_config(no_consistency)
	main_resolved = resolve_strat_hmm_pretext_config(main)

	for config in (no_consistency_resolved, main_resolved):
		assert config['head']['ks'] == [6, 8, 10]
		assert config['head']['projection_dim'] == 128
		assert config['head']['temperature'] == 0.1
		assert config['loss']['prototype_weight'] == 1.0
		assert config['loss']['usage_weight'] == 0.005
		assert config['loss']['distillation_weight'] == 0.2
		assert config['loss']['consistency_beta'] == 0.1
	assert no_consistency_resolved['loss']['consistency_weight'] == 0.0
	assert main_resolved['loss']['consistency_weight'] == 0.1

	main_resolved['loss']['consistency_weight'] = 0.0
	main_resolved['identity']['scientific_identity']['consistency_weight'] = 0.0
	main_resolved['identity']['model_tag'] = no_consistency_resolved['identity'][
		'model_tag'
	]
	main_resolved['paths']['output_root'] = no_consistency_resolved['paths'][
		'output_root'
	]
	assert main_resolved == no_consistency_resolved


def _multi_head_config(tmp_path: Path) -> dict[str, object]:
	embeddings, heads = _artifacts(tmp_path)
	manifest = tmp_path / 'multi_head_target_manifest.json'
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots={6: heads[6], 8: heads[8], 10: heads[10]},
		replay_k6_root=heads[6],
	)
	config = deepcopy(_minimal_config(tmp_path))
	config['pseudo_targets'] = {'manifest': str(manifest), 'min_confidence': 0.0}
	config['head'] = {
		'spec': 'multi_resolution_ordered_prototypes_v1',
		'ks': [6, 8, 10],
		'projection_dim': 128,
		'temperature': 0.1,
		'normalize': True,
	}
	config['loss'] = {
		'prototype_weight': 1.0,
		'usage_weight': 0.005,
		'entropy_floor': None,
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'distillation_weight': 0.2,
	}
	config['identity'] = {
		'model_tag': 'strat_hmm_multi_k6810_no_consistency_v1',
		'scientific_identity': {
			'experiment_role': 'multi_head_ordered_pretext',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_sha256': file_sha256(manifest),
			'consistency_policy': 'normalized_order_smooth_l1_v1',
		},
		'runtime_identity': {'device': 'cpu', 'workers': 0},
	}
	return config
