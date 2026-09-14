"""Frozen source lineage and portable resolution for all eight K6810 arms."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
)
from seis_ssl_cluster.hmm.multi_source_definition import (
	validate_multi_source_experiment_definition,
)
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	as_five_way_config,
	volve_horizon_recipe_arm_config_from_mapping,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_multi_head import _multi_head_config

WORKSPACE = Path(__file__).resolve().parents[2]
ROOTS = tuple(sorted(WORKSPACE.glob('experiments/*/*/*hmm_v2_k6810_multi_source_v1')))


def _read(path: Path) -> dict:
	return yaml.safe_load(path.read_text())


@pytest.fixture
def configured_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	for root in ROOTS:
		for path in root.rglob('*.yaml'):
			for variable in re.findall(r'\$\{([^}]+)\}', path.read_text()):
				monkeypatch.setenv(variable, str(tmp_path / variable))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(WORKSPACE))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', 'a' * 64)


@pytest.mark.usefixtures('configured_environment')
@pytest.mark.parametrize('root', ROOTS, ids=lambda p: p.parts[-3])
def test_all_source_contracts_and_config_resolution(root: Path, tmp_path: Path) -> None:
	validate_multi_source_experiment_definition(root)
	definition = _read(root / 'execution.yaml')
	survey = definition['survey']
	fixture = _multi_head_config(tmp_path)
	for family, entry in definition['arms'].items():
		cid = entry['candidate_id']
		fam = entry['target_family']
		resolve_clustering_config(
			load_config(root / '10_targets' / fam / '01_cluster_hmm_k8k10.yaml')
		)
		for filename in ('01_gpu_feasibility_1step.yaml', '02_full_25ep.yaml'):
			path = root / '30_pretraining' / cid / filename
			raw = load_config(path)
			raw['pseudo_targets']['manifest'] = fixture['pseudo_targets']['manifest']
			raw['identity']['scientific_identity']['target_manifest_sha256'] = fixture[
				'identity'
			]['scientific_identity']['target_manifest_sha256']
			raw['teacher']['checkpoint'] = fixture['teacher']['checkpoint']
			raw['student']['init_checkpoint'] = fixture['teacher']['checkpoint']
			resolved = resolve_strat_hmm_pretext_config(raw)
			assert resolved['head']['ks'] == [6, 8, 10]
			assert resolved['loss']['distillation_weight'] == 0.2
			assert resolved['loss']['consistency_weight'] == 0.0
		embpath = root / '40_embeddings' / f'{cid}.yaml'
		resolve_embedding_extraction_config(load_config(embpath))
		down = load_config(root / '50_downstream' / f'{cid}.yaml')
		if survey == 'f3':
			f3_lithology_candidate_config_from_mapping(down)
		elif survey == 'parihaka':
			channel_decoder_config_from_mapping(down)
		else:
			config = volve_horizon_recipe_arm_config_from_mapping(down)
			assert config.multi_head_training_config.name == '02_full_25ep.yaml'
			source = as_five_way_config(config).models[0]
			assert source.expected['head_ks'] == [6, 8, 10]
			assert source.expected['objective'] == (
				'local_barlow_twins_3d' if family == 'local_bt' else 'amp_mae3d'
			)
