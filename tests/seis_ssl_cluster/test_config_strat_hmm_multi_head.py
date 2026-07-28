from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.stratigraphy import multi_head, state_posterior
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_config_strat_hmm_pretext import _minimal_config
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
	_write_positive_preflight,
)

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


def test_multi_head_config_resolution_does_not_load_target_arrays(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _multi_head_config(tmp_path)

	def fail_array_load(*_args: object, **_kwargs: object) -> object:
		raise AssertionError('config resolution must not load pseudo-target arrays')

	monkeypatch.setattr(multi_head.np, 'load', fail_array_load)

	resolved = resolve_strat_hmm_pretext_config(config)

	assert resolved['head']['ks'] == [6, 8, 10]


def test_soft_multi_head_config_resolves_with_posterior_identity(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _multi_head_config(tmp_path)
	manifest_sha256 = file_sha256(str(config['pseudo_targets']['manifest']))
	posterior_hashes = {
		str(k): {
			'survey': {
				'posterior': f'{k:064x}',
				'valid_tokens': f'{k + 10:064x}',
				'metadata': f'{k + 20:064x}',
			}
		}
		for k in (6, 8, 10)
	}
	posterior_manifest = {
		'head_ks': [6, 8, 10],
		'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
		'cost_temperature': 1.0,
		'heads': {
			str(k): {
				'surveys': {
					'survey': {
						name: {'sha256': value}
						for name, value in posterior_hashes[str(k)]['survey'].items()
					}
				}
			}
			for k in (6, 8, 10)
		},
	}
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path: posterior_manifest,
	)
	config['pseudo_targets']['target_representation'] = (
		'ordered_path_state_posterior_v1'
	)
	config['identity']['model_tag'] = (
		'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1'
	)
	scientific = config['identity']['scientific_identity']
	scientific.update(
		{
			'experiment_role': 'multi_head_ordered_soft_posterior_pretext',
			'variant': 'soft_nocons',
			'target_representation': 'ordered_path_state_posterior_v1',
			'posterior_manifest_sha256': manifest_sha256,
			'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
			'posterior_cost_temperature': 1.0,
			'posterior_head_hashes': posterior_hashes,
			'supervised_loss': 'soft_categorical_cross_entropy_v1',
			'consistency_policy': 'disabled_for_m5_u_v1',
			'consistency_weight': 0.0,
		}
	)
	del scientific['target_manifest_sha256']

	resolved = resolve_strat_hmm_pretext_config(config)

	assert resolved['identity']['scientific_identity']['posterior_head_hashes'] == (
		posterior_hashes
	)


def test_soft_multi_head_config_requires_posterior_hashes(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _multi_head_config(tmp_path)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path: {
			'head_ks': [6, 8, 10],
			'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
			'cost_temperature': 1.0,
			'heads': {},
		},
	)
	config['pseudo_targets']['target_representation'] = (
		'ordered_path_state_posterior_v1'
	)
	config['identity']['scientific_identity']['experiment_role'] = (
		'multi_head_ordered_soft_posterior_pretext'
	)
	config['identity']['scientific_identity']['variant'] = 'soft_nocons'

	with pytest.raises(ValueError, match='target_representation'):
		resolve_strat_hmm_pretext_config(config)


@pytest.mark.parametrize(
	'mutate',
	[
		lambda config: config['pseudo_targets'].update(input_dir='/invalid', k=6),
		lambda config: config['head'].update(num_prototypes=6),
		lambda config: config['loss'].update(consistency_weight=-0.1),
		lambda config: config['loss'].update(consistency_weight=0.1),
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


@pytest.mark.parametrize(
	('variant', 'consistency_weight'),
	[
		('nocons', 0.1),
		('cons010', 0.0),
	],
)
def test_multi_head_variant_requires_its_exact_consistency_weight(
	tmp_path: Path,
	variant: str,
	consistency_weight: float,
) -> None:
	config = _multi_head_config(tmp_path)
	config['identity']['scientific_identity']['variant'] = variant
	config['identity']['model_tag'] = {
		'nocons': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'cons010': 'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
	}[variant]
	config['loss']['consistency_weight'] = consistency_weight

	with pytest.raises(ValueError, match=r'consistency_weight.*variant'):
		resolve_strat_hmm_pretext_config(config)


def test_k6810_no_consistency_and_main_configs_have_a_pure_scientific_diff(
	tmp_path: Path,
) -> None:
	no_consistency = _multi_head_config(tmp_path)
	main = deepcopy(no_consistency)
	main['paths']['output_root'] = str(tmp_path / 'artifacts' / 'pretraining' / 'main')
	main['identity']['model_tag'] = (
		'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1'
	)
	main['identity']['scientific_identity']['variant'] = 'cons010'
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
	main_resolved['identity']['scientific_identity']['variant'] = 'nocons'
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
	migration, control = _write_positive_preflight(tmp_path)
	build_multi_head_target_manifest(
		manifest_path=manifest,
		source_embedding_dir=embeddings,
		head_roots={6: heads[6], 8: heads[8], 10: heads[10]},
		replay_k6_root=_replay_k6_root(tmp_path, heads[6]),
		migration_decision=migration,
		control_summary=control,
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
		'model_tag': 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'scientific_identity': {
			'experiment_role': 'multi_head_ordered_pretext',
			'variant': 'nocons',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_sha256': file_sha256(manifest),
			'consistency_policy': 'normalized_order_smooth_l1_v1',
		},
		'runtime_identity': {'device': 'cpu', 'workers': 0},
	}
	return config
