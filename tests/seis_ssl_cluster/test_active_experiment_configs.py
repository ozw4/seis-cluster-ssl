
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.clustering.kmeans import clustering_settings_from_config
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	stratigraphic_hmm_settings_from_config,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_f3_facies_inspection_config,
	resolve_mae_training_config,
	resolve_strat_hmm_pretext_config,
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.config.f3_lithology import (
	f3_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.config.pretraining import _multi_head_target_hashes
from seis_ssl_cluster.config.schema import (
	STAGE_F3_INSPECT_FILES,
	STAGE_F3_LABEL_CONSISTENCY,
	STAGE_F3_PNG_LABELS,
	STAGE_F3_QUICKLOOK,
	STAGE_F3_SEGY_GEOMETRY,
	STAGE_F3_TOKENIZATION_PREVIEW,
)
from seis_ssl_cluster.embedding.extractor import extraction_settings_from_config
from seis_ssl_cluster.f3.center_trace_masked_periodic_refresh_validation import (
	load_f3_center_trace_masked_periodic_refresh_validation_config,
)
from seis_ssl_cluster.f3.lithology.guardrails import (
	f3_shuffled_hmm_target_config_from_mapping,
)
from seis_ssl_cluster.stratigraphy import (
	lateral_targets,
	state_posterior,
	xy_neighbor_consensus_targets,
	xy_neighbor_unanimous_targets,
)
from seis_ssl_cluster.stratigraphy.multi_head import build_multi_head_target_manifest
from tests.seis_ssl_cluster.test_strat_multi_head_target_manifest import (
	_artifacts,
	_replay_k6_root,
)

ALL_CONFIGS = sorted(
	[
		*Path('proc/configs/seis_ssl_cluster').rglob('*.yaml'),
		*Path('experiments/nopims').rglob('*.yaml'),
		*Path('experiments/f3').rglob('*.yaml'),
	],
)

CORE_CONFIG_RESOLVERS = {
	frozenset(
		{
			'paths',
			'manifests',
			'data',
			'zero_mask',
			'model',
			'masking',
			'loss',
			'train',
			'visualization',
		}
	): resolve_mae_training_config,
	frozenset({'paths', 'manifests', 'embeddings', 'embedding'}): (
		resolve_embedding_extraction_config
	),
	frozenset({'paths', 'embeddings', 'clustering'}): resolve_clustering_config,
	frozenset({'paths', 'clustering', 'visualization'}): (
		resolve_cluster_visualization_config
	),
}

NOPIMS_ROOT = Path('experiments/nopims/pretrain_v1')
NOPIMS_PRETRAINING_CONFIGS = sorted((NOPIMS_ROOT / '10_pretrain').rglob('*.yaml'))
NOPIMS_EMBEDDING_CONFIGS = sorted((NOPIMS_ROOT / '20_embedding').rglob('*.yaml'))
NOPIMS_CLUSTERING_CONFIGS = sorted((NOPIMS_ROOT / '30_clustering').rglob('*.yaml'))
NOPIMS_VISUALIZATION_CONFIGS = sorted(
	(NOPIMS_ROOT / '40_visualization').rglob('*.yaml'),
)
PARIHAKA_FULL_MAE_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/20_pretrain/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/02_full_100ep.yaml',
)
STABLE_NOPIMS_FULL_MAE_CONFIG = (
	NOPIMS_ROOT
	/ '10_pretrain'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ '03_full_100ep.yaml'
)
F3_ROOT = Path('experiments/f3/facies_benchmark_v1')
F3_INSPECTION_STAGES = {
	'01_inspect_files.yaml': STAGE_F3_INSPECT_FILES,
	'02_inspect_segy_geometry.yaml': STAGE_F3_SEGY_GEOMETRY,
	'03_inspect_png_labels.yaml': STAGE_F3_PNG_LABELS,
	'04_make_quicklook_figures.yaml': STAGE_F3_QUICKLOOK,
	'05_check_label_consistency.yaml': STAGE_F3_LABEL_CONSISTENCY,
	'06_make_tokenization_preview.yaml': STAGE_F3_TOKENIZATION_PREVIEW,
}
F3_INSPECTION_CONFIGS = [
	(path, F3_INSPECTION_STAGES[path.name])
	for path in sorted((F3_ROOT / '00_inspection').rglob('*.yaml'))
]
F3_PREPARE_CONFIGS = sorted((F3_ROOT / '10_prepare').rglob('*.yaml'))
F3_EMBEDDING_CONFIGS = sorted((F3_ROOT / '20_embedding').rglob('*.yaml'))
F3_STRATIGRAPHIC_CLUSTERING_CONFIGS = sorted(
	(F3_ROOT / '60_stratigraphic_clustering').rglob('*.yaml'),
)
F3_STRAT_HMM_PRETRAINING_M1_ROOT = F3_ROOT / '80_strat_hmm_pretraining_m1'
F3_STRAT_HMM_M1_GUARDRAIL_ROOT = F3_ROOT / '83_strat_hmm_m1_guardrails'
F3_STRAT_HMM_PRETRAINING_M2A_ROOT = F3_ROOT / '84_strat_hmm_pretraining_m2a_boundary'
F3_CURRENT_K6_CONTROL_ROOT = F3_ROOT / '93_strat_hmm_m1_current_k6_control'
F3_STRAT_HMM_MULTI_HEAD_ROOT = F3_ROOT / '94_strat_hmm_multi_head_k6810_v1'
F3_STRAT_HMM_SOFT_POSTERIOR_ROOT = (
	F3_ROOT / '97_strat_hmm_multi_head_k6810_soft_posterior_v1'
)
F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT = (
	F3_ROOT / '99_strat_hmm_multi_head_k6810_lateral_smoothing_v1'
)
F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT = (
	F3_ROOT / '100_strat_hmm_multi_head_k6810_xy_neighbor_consensus_v1'
)
F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT = (
	F3_ROOT / '102_strat_hmm_multi_head_k6810_xy_neighbor_unanimous_v1'
)
F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT = (
	F3_ROOT / '104_strat_hmm_multi_head_k6810_center_trace_masked_v1'
)
F3_STRAT_HMM_PERIODIC_REFRESH_ROOT = (
	F3_ROOT
	/ '107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1'
)
F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT / '01_train_periodic_refresh_smoke.yaml',
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT / '02_train_periodic_refresh_full.yaml',
]
F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT
	/ '03_extract_periodic_refresh_embeddings.yaml',
]
F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS = [
	F3_STRAT_HMM_PERIODIC_REFRESH_ROOT
	/ '04_validate_periodic_refresh_pretraining.yaml',
]
F3_STRAT_HMM_PRETEXT_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '02_train_single_head_topblock_distill_smoke.yaml',
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '03_train_single_head_topblock_distill_full.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '01_train_distillation_only_smoke.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '02_train_distillation_only_full.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '07_train_shuffled_hmm_smoke.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '08_train_shuffled_hmm_full.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '03_train_boundary_smoke.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '04_train_boundary_full.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '01_train_current_k6_smoke.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '02_train_current_k6_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '02_train_nocons_smoke.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '03_train_cons010_smoke.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '04_train_nocons_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '05_train_cons010_full.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '02_train_soft_smoke.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '03_train_soft_full.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '05_train_lateral_smoke.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '06_train_lateral_full.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '02_train_xy_neighbor_consensus_smoke.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '03_train_xy_neighbor_consensus_full.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '03_train_xy_neighbor_unanimous_smoke.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '04_train_xy_neighbor_unanimous_full.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '01_train_center_trace_masked_smoke.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '02_train_center_trace_masked_full.yaml',
	],
)
F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '04_extract_student_embeddings.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '03_extract_distillation_only_embeddings.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '09_extract_shuffled_hmm_embeddings.yaml',
		F3_STRAT_HMM_PRETRAINING_M2A_ROOT / '05_extract_student_embeddings.yaml',
		F3_CURRENT_K6_CONTROL_ROOT / '03_extract_current_k6_embeddings.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '06_extract_nocons_embeddings.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '07_extract_cons010_embeddings.yaml',
		F3_STRAT_HMM_SOFT_POSTERIOR_ROOT / '04_extract_soft_embeddings.yaml',
		F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT / '07_extract_lateral_embeddings.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_CONSENSUS_ROOT
		/ '04_extract_xy_neighbor_consensus_embeddings.yaml',
		F3_STRAT_HMM_XY_NEIGHBOR_UNANIMOUS_ROOT
		/ '05_extract_xy_neighbor_unanimous_embeddings.yaml',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
		/ '03_extract_center_trace_masked_embeddings.yaml',
	],
)
F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS = [
	F3_STRAT_HMM_CENTER_TRACE_MASKED_ROOT
	/ '04_validate_center_trace_masked_pretraining.yaml',
]
F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS = [
	F3_STRAT_HMM_M1_GUARDRAIL_ROOT / '03_build_shuffled_hmm_pseudo_targets.yaml',
]
F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '08_refresh_pseudo_targets_from_logits_smoke.yaml',
	],
)
REQUIRED_ACTIVE_CONFIG_GROUPS = (
	('nopims pretraining', NOPIMS_PRETRAINING_CONFIGS),
	('nopims embedding', NOPIMS_EMBEDDING_CONFIGS),
	('nopims clustering', NOPIMS_CLUSTERING_CONFIGS),
	('nopims visualization', NOPIMS_VISUALIZATION_CONFIGS),
	('f3 inspection', F3_INSPECTION_CONFIGS),
	('f3 prepare', F3_PREPARE_CONFIGS),
	('f3 embedding', F3_EMBEDDING_CONFIGS),
	('f3 stratigraphic clustering', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS),
	('f3 strat hmm pretext', F3_STRAT_HMM_PRETEXT_CONFIGS),
	(
		'f3 strat hmm center-trace masked validation',
		F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS,
	),
	(
		'f3 strat hmm periodic refresh pretext',
		F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS,
	),
	(
		'f3 strat hmm periodic refresh validation',
		F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS,
	),
	('f3 strat hmm shuffled targets', F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS),
	(
		'f3 strat hmm student embedding',
		F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
	),
	(
		'f3 strat hmm pseudo-target refresh',
		F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS,
	),
)

@pytest.mark.parametrize('config_path', ALL_CONFIGS, ids=str)
def test_all_repository_configs_load_and_resolve_supported_stages(
	config_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')
	monkeypatch.setenv('F3_ROOT', '/test/f3')
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_MANIFEST_SHA256',
		'0' * 64,
	)
	for k in (6, 8, 10):
		for name in ('POSTERIOR', 'VALID_TOKENS', 'METADATA'):
			monkeypatch.setenv(
				f'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_K{k}_{name}_SHA256',
				'0' * 64,
			)
	config = load_config(config_path)

	assert isinstance(config, dict)
	assert config
	resolver = CORE_CONFIG_RESOLVERS.get(frozenset(config))
	if resolver is not None:
		resolved = resolver(config)
		assert resolved['stage']


def test_f3_frozen_report_is_separate_from_active_configs() -> None:
	legacy_root = Path('reports/f3/legacy/facies_benchmark_v1')
	old_active_root = Path('reports/f3/facies_benchmark_v1')
	assert legacy_root.is_dir()
	assert not old_active_root.exists()

	for config_path in ALL_CONFIGS:
		config_text = config_path.read_text(encoding='utf-8')
		assert 'reports/f3/legacy/facies_benchmark_v1' not in config_text
		assert 'reports/f3/facies_benchmark_v1' not in config_text


def test_repository_configs_preserve_legacy_optimization_defaults() -> None:
	training = resolve_mae_training_config(
		load_config(Path('proc/configs/seis_ssl_cluster/train_amp_mae.yaml')),
	)
	assert training['data']['finite_check_mode'] == 'strict'
	assert {
		key: training['train'][key]
		for key in (
			'prefetch_factor',
			'persistent_workers',
			'amp_dtype',
			'runtime_check_mode',
			'stage_timing',
		)
	} == {
		'prefetch_factor': 2,
		'persistent_workers': True,
		'amp_dtype': 'auto',
		'runtime_check_mode': 'once',
		'stage_timing': False,
	}

	embedding_raw = load_config(
		Path('proc/configs/seis_ssl_cluster/extract_embeddings.yaml'),
	)
	for key in (
		'prefetch_queue_depth',
		'amp',
		'amp_dtype',
		'stage_timing',
		'preprocessing_cache',
	):
		embedding_raw['embedding'].pop(key)
	embedding = resolve_embedding_extraction_config(embedding_raw)
	embedding_settings = extraction_settings_from_config(
		embedding,
		checkpoint_config=training,
	)
	assert embedding_settings.average_chunk_size_x == 16
	assert embedding_settings.prefetch_queue_depth == 0
	assert embedding_settings.amp is False
	assert embedding_settings.amp_dtype == 'auto'
	assert embedding_settings.stage_timing is False
	assert embedding_settings.preprocessing_cache.mode == 'off'
	assert embedding_settings.preprocessing_cache.chunk_size_x == 16
	assert embedding_settings.preprocessing_cache.reuse is True
	assert embedding_settings.preprocessing_cache.cleanup is False

	hmm_raw = load_config(F3_STRATIGRAPHIC_CLUSTERING_CONFIGS[0])
	hmm_raw['clustering'].pop('stage_timing', None)
	hmm_raw['clustering']['stratigraphic_hmm'].pop(
		'prepared_feature_cache',
		None,
	)
	hmm = resolve_clustering_config(hmm_raw)
	assert clustering_settings_from_config(hmm).stage_timing is False
	prepared_cache = stratigraphic_hmm_settings_from_config(
		hmm,
	).prepared_feature_cache
	assert prepared_cache.chunk_size_tokens == 65_536
	assert prepared_cache.reuse is True
	assert prepared_cache.force_rebuild is False
	assert prepared_cache.cleanup is False
	assert prepared_cache.persist is True


@pytest.mark.parametrize(('group_name', 'configs'), REQUIRED_ACTIVE_CONFIG_GROUPS)
def test_active_config_groups_are_not_empty(
	group_name: str,
	configs: list[Path] | list[tuple[Path, str]],
) -> None:
	assert configs, f'{group_name} active config list must not be empty'


@pytest.mark.parametrize('config_path', NOPIMS_PRETRAINING_CONFIGS)
def test_active_nopims_pretraining_configs_resolve(config_path: Path) -> None:
	resolve_mae_training_config(load_config(config_path))


def test_parihaka_full_mae_matches_stable_nopims_scientific_contract(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	parihaka = resolve_mae_training_config(load_config(PARIHAKA_FULL_MAE_CONFIG))
	nopims = resolve_mae_training_config(load_config(STABLE_NOPIMS_FULL_MAE_CONFIG))

	for section in ('data', 'zero_mask', 'model', 'masking', 'loss', 'visualization'):
		assert parihaka[section] == nopims[section]
	parihaka_train = dict(parihaka['train'])
	nopims_train = dict(nopims['train'])
	nopims_train['amp'] = True
	assert parihaka_train == nopims_train


@pytest.mark.parametrize('config_path', NOPIMS_EMBEDDING_CONFIGS)
def test_active_nopims_embedding_configs_resolve(config_path: Path) -> None:
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_CLUSTERING_CONFIGS)
def test_active_nopims_clustering_configs_resolve(config_path: Path) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_VISUALIZATION_CONFIGS)
def test_active_nopims_cluster_visualization_configs_resolve(
	config_path: Path,
) -> None:
	resolve_cluster_visualization_config(load_config(config_path))


@pytest.mark.parametrize(('config_path', 'stage'), F3_INSPECTION_CONFIGS)
def test_active_f3_inspection_configs_resolve(
	config_path: Path,
	stage: str,
) -> None:
	resolve_f3_facies_inspection_config(load_config(config_path), stage=stage)


@pytest.mark.parametrize('config_path', F3_PREPARE_CONFIGS)
def test_active_f3_prepare_configs_resolve(config_path: Path) -> None:
	f3_prepare_volume_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_EMBEDDING_CONFIGS,
		*F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
		*F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS,
	],
)
def test_active_f3_embedding_configs_resolve(
	config_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS)
def test_active_f3_stratigraphic_clustering_configs_resolve(
	config_path: Path,
) -> None:
	resolve_clustering_config(load_config(config_path))



@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PRETEXT_CONFIGS)
def test_active_f3_strat_hmm_pretext_configs_resolve(
	config_path: Path,
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_MANIFEST_SHA256',
		'0' * 64,
	)
	for k in (6, 8, 10):
		for name, offset in (
			('POSTERIOR', 0),
			('VALID_TOKENS', 10),
			('METADATA', 20),
		):
			monkeypatch.setenv(
				f'SEIS_SSL_CLUSTER_MULTI_HEAD_POSTERIOR_K{k}_{name}_SHA256',
				f'{k + offset:064x}',
			)
	monkeypatch.setattr(
		state_posterior,
		'load_multi_head_state_posterior_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_posterior_manifest()
			if not validate_array_semantics
			else pytest.fail('config validation requested full posterior arrays')
		),
	)
	monkeypatch.setattr(
		lateral_targets,
		'load_multi_head_lateral_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_lateral_manifest()
			if not validate_array_semantics
			else pytest.fail('config validation requested full lateral arrays')
		),
	)
	monkeypatch.setattr(
		xy_neighbor_consensus_targets,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_xy_neighbor_consensus_manifest()
			if not validate_array_semantics
			else pytest.fail(
				'config validation requested full XY-neighbour consensus arrays'
			)
		),
	)
	monkeypatch.setattr(
		xy_neighbor_unanimous_targets,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda _path, *, validate_array_semantics: (
			_active_xy_neighbor_unanimous_manifest()
			if not validate_array_semantics
			else pytest.fail(
				'config validation requested full XY-neighbour unanimous arrays'
			)
		),
	)
	resolve_strat_hmm_pretext_config(
		_config_with_existing_strat_hmm_pretext_inputs(config_path, tmp_path),
	)


def test_active_f3_multi_head_pretext_config_contract(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		'0' * 64,
	)
	config_paths = [
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '04_train_nocons_full.yaml',
		F3_STRAT_HMM_MULTI_HEAD_ROOT / '05_train_cons010_full.yaml',
	]
	for config_path, consistency_weight, model_tag in zip(
		config_paths,
		(0.0, 0.1),
		(
			'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
			'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
		),
		strict=True,
	):
		raw = load_config(config_path)
		assert raw['identity']['model_tag'] == model_tag
		assert raw['identity']['scientific_identity'] == {
			'experiment_role': 'multi_head_ordered_pretext',
			'variant': 'nocons' if consistency_weight == 0.0 else 'cons010',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'target_manifest_sha256': '0' * 64,
			'consistency_policy': 'normalized_order_smooth_l1_v1',
		}
		assert raw['head'] == {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 128,
			'temperature': 0.1,
			'normalize': True,
		}
		assert raw['loss'] == {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'entropy_floor': None,
			'consistency_weight': consistency_weight,
			'consistency_beta': 0.1,
			'distillation_weight': 0.2,
		}

	no_consistency, main = [
		resolve_strat_hmm_pretext_config(
			_config_with_existing_strat_hmm_pretext_inputs(config_path, tmp_path)
		)
		for config_path in config_paths
	]
	comparison = deepcopy(main)
	comparison['loss']['consistency_weight'] = 0.0
	comparison['identity']['scientific_identity']['consistency_weight'] = 0.0
	comparison['identity']['scientific_identity']['variant'] = 'nocons'
	comparison['identity']['model_tag'] = no_consistency['identity']['model_tag']
	comparison['paths']['output_root'] = no_consistency['paths']['output_root']
	comparison['pseudo_targets']['manifest'] = no_consistency['pseudo_targets'][
		'manifest'
	]
	comparison['identity']['scientific_identity']['target_manifest_sha256'] = (
		no_consistency['identity']['scientific_identity']['target_manifest_sha256']
	)
	assert comparison == no_consistency


@pytest.mark.parametrize(
	'config_path', F3_STRAT_HMM_CENTER_TRACE_MASKED_VALIDATION_CONFIGS
)
def test_active_f3_center_trace_masked_validation_configs_load(
	config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', '/test/artifacts/seis_ssl_cluster'
	)
	config = load_config(config_path)
	assert set(config) == {
		'artifact_root',
		'experiment_root',
		'target_manifest',
		'hard_full_config',
		'hard_handoff',
		'center_trace_masked_smoke_config',
		'center_trace_masked_full_config',
		'center_trace_masked_embedding_config',
	}


def test_active_f3_periodic_refresh_configs_resolve(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact_root = Path('/workspace/artifacts/seis_ssl_cluster')
	target = artifact_root / (
		'pseudo_targets/f3/facies_benchmark_v1/'
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1/'
		'multi_head_target_manifest.json'
	)
	if not target.is_file():
		pytest.skip('real F3 periodic-refresh artifacts are not available')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256',
		file_sha256(target),
	)
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_PRETEXT_CONFIGS:
		resolved = resolve_strat_hmm_pretext_config(load_config(path))
		assert resolved['identity']['model_tag'].endswith('distill_v1')
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_EMBEDDING_CONFIGS:
		resolved = resolve_embedding_extraction_config(load_config(path))
		assert Path(resolved['embeddings']['checkpoint']).name == 'selected.pt'
	for path in F3_STRAT_HMM_PERIODIC_REFRESH_VALIDATION_CONFIGS:
		resolved = load_f3_center_trace_masked_periodic_refresh_validation_config(path)
		assert resolved.target_manifest == target.resolve()



def test_active_f3_m5_ls_config_contract(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Keep M5-LS changes limited to its target representation identity."""
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'/test/artifacts/seis_ssl_cluster',
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', '0' * 64)
	root = F3_STRAT_HMM_LATERAL_SMOOTHING_ROOT
	candidates = [
		load_config(root / '01_export_lateral_beta010.yaml'),
		load_config(root / '02_export_lateral_beta025.yaml'),
		load_config(root / '03_export_lateral_beta050.yaml'),
	]
	assert [
		candidate['smoothing']['pairwise_strength_ratio'] for candidate in candidates
	] == [0.10, 0.25, 0.50]
	assert all(candidate['outputs'] == {'overwrite': False} for candidate in candidates)
	assert [
		Path(candidate['handoff_manifest']).parent.name for candidate in candidates
	] == [
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta010_v1',
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta025_v1',
		'strat_hmm_multi_k6810_pca64_resid_token_phase_edge8_expected3_iter10_v1_lateral_mean_field_beta050_v1',
	]

	for baseline_name, lateral_name in (
		('02_train_nocons_smoke.yaml', '05_train_lateral_smoke.yaml'),
		('04_train_nocons_full.yaml', '06_train_lateral_full.yaml'),
	):
		baseline = load_config(F3_STRAT_HMM_MULTI_HEAD_ROOT / baseline_name)
		lateral = load_config(root / lateral_name)
		assert lateral['identity']['model_tag'] == (
			'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1'
		)
		assert lateral['pseudo_targets']['target_representation'] == (
			'lateral_mean_field_hard_labels_v1'
		)
		assert lateral['pseudo_targets']['min_confidence'] == 0.0
		assert lateral['identity']['scientific_identity']['target_semantics'] == (
			'ordered_hmm_edge_aware_lateral_mean_field_hard_v1'
		)
		assert lateral['loss']['consistency_weight'] == 0.0

		baseline['paths']['output_root'] = lateral['paths']['output_root']
		baseline['identity']['model_tag'] = lateral['identity']['model_tag']
		baseline['pseudo_targets']['manifest'] = lateral['pseudo_targets']['manifest']
		baseline['pseudo_targets']['target_representation'] = lateral['pseudo_targets'][
			'target_representation'
		]
		baseline_scientific = baseline['identity']['scientific_identity']
		lateral_scientific = lateral['identity']['scientific_identity']
		assert isinstance(baseline_scientific, dict)
		assert isinstance(lateral_scientific, dict)
		for key in (
			'experiment_role',
			'variant',
			'target_manifest_sha256',
			'target_representation',
			'target_semantics',
			'supervised_loss',
			'consistency_policy',
			'consistency_weight',
		):
			baseline_scientific.pop(key, None)
			lateral_scientific.pop(key, None)
		assert baseline == lateral

	calibration = load_config(root / '04_calibrate_lateral_targets.yaml')
	assert set(calibration) == {
		'artifact_root',
		'source_hard_manifest',
		'source_posterior_manifest',
		'candidate_manifests',
		'selected_manifest',
		'calibration_handoff',
		'calibration_report',
		'hard_full_config',
		'lateral_smoke_config',
		'lateral_full_config',
	}
	assert list(calibration['candidate_manifests']) == ['beta010', 'beta025', 'beta050']
	assert calibration['selected_manifest'].endswith(
		'strat_hmm_multi_k6810_lateral_mean_field_selected_v1/'
		'multi_head_lateral_target_handoff.json'
	)
	validator = load_config(root / '08_validate_lateral_pretraining.yaml')
	assert set(validator) == {
		'artifact_root',
		'experiment_root',
		'calibration_handoff',
		'selected_manifest',
		'hard_full_config',
		'hard_handoff',
		'lateral_smoke_config',
		'lateral_full_config',
	}
	embedding = load_config(root / '07_extract_lateral_embeddings.yaml')
	assert embedding['embeddings']['checkpoint'].endswith(
		'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1/best.pt'
	)
	assert embedding['embeddings']['output_dir'].endswith(
		'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1/overlap_x16'
	)



@pytest.mark.parametrize('config_path', F3_STRAT_HMM_SHUFFLED_TARGET_CONFIGS)
def test_active_f3_strat_hmm_shuffled_target_configs_resolve(
	config_path: Path,
) -> None:
	f3_shuffled_hmm_target_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS)
def test_active_f3_strat_hmm_pseudo_target_refresh_configs_resolve(
	config_path: Path,
	tmp_path: Path,
) -> None:
	resolve_strat_hmm_pseudo_target_config(
		_config_with_existing_strat_hmm_refresh_inputs(config_path, tmp_path),
	)



def test_active_nopims_overlap_x16_paths_are_explicit() -> None:
	model_tag = 'amp_mae_m075_mse_g0_patchnorm_clip8_vis01_v1'

	embedding = load_config(
		NOPIMS_ROOT / '20_embedding' / model_tag / '01_ten_surveys_overlap_x16.yaml',
	)
	clustering = load_config(
		NOPIMS_ROOT
		/ '30_clustering'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_k6_8_whiten.yaml',
	)
	visualization = load_config(
		NOPIMS_ROOT
		/ '40_visualization'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_whiten.yaml',
	)

	resolved_embedding = resolve_embedding_extraction_config(embedding)
	resolved_clustering = resolve_clustering_config(clustering)
	resolved_visualization = resolve_cluster_visualization_config(visualization)

	assert resolved_embedding['embeddings']['output_dir'] == (
		embedding['embeddings']['output_dir']
	)
	assert resolved_clustering['embeddings']['input_dir'] == (
		clustering['embeddings']['input_dir']
	)
	assert resolved_clustering['clustering']['output_dir'] == (
		clustering['clustering']['output_dir']
	)
	assert resolved_visualization['clustering']['input_dir'] == (
		visualization['clustering']['input_dir']
	)
	assert resolved_visualization['visualization']['output_dir'] == (
		visualization['visualization']['output_dir']
	)



def _config_with_existing_strat_hmm_pretext_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	pseudo_target_dir = tmp_path / 'pseudo_targets'
	pseudo_target_dir.mkdir(exist_ok=True)
	checkpoint = tmp_path / 'mae_best.pt'
	checkpoint.touch()

	config['paths']['artifact_root'] = str(artifact_root)
	config['paths']['output_root'] = str(
		artifact_root / 'pretraining' / 'f3' / config_path.stem,
	)
	if 'manifest' in config['pseudo_targets']:
		fixture_root = tmp_path / config_path.stem
		fixture_root.mkdir(exist_ok=True)
		embeddings, heads = _artifacts(
			fixture_root,
			source_root=tmp_path / 'shared_multi_head_sources',
		)
		manifest = fixture_root / 'multi_head_target_manifest.json'
		build_multi_head_target_manifest(
			manifest_path=manifest,
			source_embedding_dir=embeddings,
			head_roots={6: heads[6], 8: heads[8], 10: heads[10]},
			replay_k6_root=_replay_k6_root(fixture_root, heads[6]),
		)
		config['pseudo_targets']['manifest'] = str(manifest)
		if 'target_head_hashes' in config['identity']['scientific_identity']:
			config['identity']['scientific_identity']['target_head_hashes'] = (
				_multi_head_target_hashes(
					json.loads(manifest.read_text(encoding='utf-8'))
				)
			)
		if config['pseudo_targets'].get('target_representation') == (
			'ordered_path_state_posterior_v1'
		):
			config['identity']['scientific_identity']['posterior_manifest_sha256'] = (
				file_sha256(manifest)
			)
		elif config['pseudo_targets'].get('target_representation') not in {
			'lateral_mean_field_hard_labels_v1',
			'xy_neighbor_consensus_hard_labels_v1',
			'xy_neighbor_unanimous_hard_labels_v1',
		}:
			config['identity']['scientific_identity']['target_manifest_sha256'] = (
				file_sha256(manifest)
			)
	else:
		config['pseudo_targets']['input_dir'] = str(pseudo_target_dir)
	config['teacher']['checkpoint'] = str(checkpoint)
	config['student']['init_checkpoint'] = str(checkpoint)
	return config


def _active_posterior_manifest() -> dict[str, object]:
	return {
		'head_ks': [6, 8, 10],
		'posterior_semantics': 'ordered_path_cost_gibbs_state_marginal_v1',
		'cost_temperature': 1.0,
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'posterior': {'sha256': f'{k:064x}'},
						'valid_tokens': {'sha256': f'{k + 10:064x}'},
						'metadata': {'sha256': f'{k + 20:064x}'},
					}
				}
			}
			for k in (6, 8, 10)
		},
	}


def _active_lateral_manifest() -> dict[str, object]:
	"""Return the reference-only lateral identity used by active config tests."""
	return {
		'head_ks': [6, 8, 10],
		'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		'source_hard_manifest': {'sha256': 'a' * 64},
		'source_posterior_manifest': {'sha256': 'b' * 64},
		'smoothing': {
			'neighborhood': 'xy_4_connected_v1',
			'affinity': 'source_embedding_cosine_rbf_v1',
			'affinity_scale_policy': (
				'global_valid_xy_edge_distance_median_floor_1e-6_v1'
			),
			'emission_scale_policy': 'per_head_valid_second_gap_median_floor_1e-6_v1',
			'pairwise_strength_ratio': 0.10,
			'iterations': 1,
			'projection': 'original_ordered_viterbi_v1',
		},
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'labels': {'sha256': f'{k:064x}'},
						'confidence': {'sha256': f'{k + 10:064x}'},
						'valid_tokens': {'sha256': f'{k + 20:064x}'},
						'metadata': {'sha256': f'{k + 30:064x}'},
					}
				},
				'diagnostics': {
					'resolved_scales': {
						'affinity': {'resolved_scale': 1.0},
						'emission_gap': {'resolved_scale': 1.0},
					}
				},
			}
			for k in (6, 8, 10)
		},
	}


def _active_xy_neighbor_consensus_manifest() -> dict[str, object]:
	"""Return the reference-only XY identity used by active config tests."""
	return {
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'source_hard_manifest': {'sha256': 'a' * 64},
		'smoothing': {
			'neighborhood': 'same_z_xy_four_neighbors',
			'neighbor_order': ['x_minus', 'x_plus', 'y_minus', 'y_plus'],
			'four_valid_neighbors_minimum_agreement': 3,
			'three_valid_neighbors_minimum_agreement': 3,
			'fewer_than_three_valid_neighbors': 'unchanged',
			'tied_or_nonunique_consensus': 'unchanged',
			'center_matching_consensus': 'unchanged',
			'temporal_guard': 'internal_valid_token_source_label_bounds',
			'application': 'single_pass_synchronous_source_labels',
		},
		'heads': {
			str(k): {
				'surveys': {
					'f3_facies_benchmark': {
						'labels': {'sha256': f'{k:064x}'},
						'confidence': {'sha256': f'{k + 10:064x}'},
						'valid_tokens': {'sha256': f'{k + 20:064x}'},
						'metadata': {'sha256': f'{k + 30:064x}'},
					}
				}
			}
			for k in (6, 8, 10)
		},
	}


def _active_xy_neighbor_unanimous_manifest() -> dict[str, object]:
	"""Return the reference-only unanimous identity used by active configs."""
	manifest = _active_xy_neighbor_consensus_manifest()
	manifest['target_representation'] = 'xy_neighbor_unanimous_hard_labels_v1'
	manifest['target_semantics'] = 'xy_neighbor_unanimous_outlier_correction_v1'
	manifest['smoothing']['four_valid_neighbors_minimum_agreement'] = 4
	return manifest


def _config_with_existing_strat_hmm_refresh_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	checkpoint = tmp_path / 'latest.pt'
	torch.save(
		{
			'stratigraphy_config': {
				'head': {'num_prototypes': config['hmm']['k']},
			},
		},
		checkpoint,
	)

	config['paths']['artifact_root'] = str(artifact_root)
	config['checkpoint']['path'] = str(checkpoint)
	config['outputs']['pseudo_target_root'] = str(
		artifact_root / 'pseudo_targets' / 'f3' / config_path.stem,
	)
	return config



def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
