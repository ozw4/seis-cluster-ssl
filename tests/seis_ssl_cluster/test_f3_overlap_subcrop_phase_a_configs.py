from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import load_config, resolve_barlow_twins_training_config
from seis_ssl_cluster.config.f3_lithology_five_way import (
	EXPECTED_MODEL_IDENTITIES,
	f3_lithology_five_way_config_from_mapping,
)

ROOT = Path('experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1')
INITIAL_ID = 'shift04_proj384_pairs128_lambda005'
INITIAL_PRETRAINING = ROOT / '10_pretraining' / f'{INITIAL_ID}.yaml'
INITIAL_EMBEDDING = ROOT / '20_embeddings' / f'{INITIAL_ID}.yaml'
INITIAL_DOWNSTREAM = ROOT / '30_downstream' / f'{INITIAL_ID}_medium.yaml'
CANDIDATE_SHIFTS = {
	'shift02_proj384_pairs128_lambda005': [2, 2, 0],
	'shift06_proj384_pairs128_lambda005': [6, 6, 0],
}
V3_FIVE_WAY = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)


@pytest.fixture(autouse=True)
def _config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', '/workspace/artifacts/seis_ssl_cluster'
	)
	monkeypatch.setenv('F3_ROOT', '/home/dcuser/data/public_data/field/F3')
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', '/workspace')


def _yaml_mapping(path: Path) -> dict[str, object]:
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		raise TypeError(f'expected YAML mapping in {path}')
	return payload


def _candidate_path(value: object, candidate_id: str) -> str:
	if not isinstance(value, str):
		raise TypeError(f'expected path string; got {value!r}')
	return value.replace(INITIAL_ID, candidate_id)


@pytest.mark.parametrize(('candidate_id', 'shift'), CANDIDATE_SHIFTS.items())
def test_phase_a_pretraining_changes_only_shift_and_output_namespace(
	candidate_id: str,
	shift: list[int],
) -> None:
	candidate_path = ROOT / '10_pretraining' / f'{candidate_id}.yaml'
	initial = _yaml_mapping(INITIAL_PRETRAINING)
	candidate = _yaml_mapping(candidate_path)
	expected = deepcopy(initial)
	expected['paths']['output_root'] = _candidate_path(
		expected['paths']['output_root'], candidate_id
	)
	expected['augmentations']['max_subcrop_shift_tokens'] = shift

	assert candidate == expected
	resolved = resolve_barlow_twins_training_config(load_config(candidate_path))
	assert resolved['train']['seed'] == 42
	assert resolved['train']['epochs'] == 10
	assert 'continuation' not in resolved
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 6_250
	)


@pytest.mark.parametrize('candidate_id', CANDIDATE_SHIFTS)
def test_phase_a_embedding_changes_only_candidate_artifact_paths(
	candidate_id: str,
) -> None:
	candidate_path = ROOT / '20_embeddings' / f'{candidate_id}.yaml'
	initial = _yaml_mapping(INITIAL_EMBEDDING)
	candidate = _yaml_mapping(candidate_path)
	expected = deepcopy(initial)
	for key in ('checkpoint', 'output_dir'):
		expected['embeddings'][key] = _candidate_path(
			expected['embeddings'][key], candidate_id
		)

	assert candidate == expected
	assert 'facies_benchmark_v2' in candidate['manifests']['input']
	assert candidate['embeddings']['output_dir'].endswith(
		f'/{candidate_id}/local_barlow_twins/overlap_x64'
	)


@pytest.mark.parametrize('candidate_id', CANDIDATE_SHIFTS)
def test_phase_a_downstream_changes_only_candidate_artifact_paths(
	candidate_id: str,
) -> None:
	candidate_path = ROOT / '30_downstream' / f'{candidate_id}_medium.yaml'
	initial = _yaml_mapping(INITIAL_DOWNSTREAM)
	candidate = _yaml_mapping(candidate_path)
	expected = deepcopy(initial)
	local = expected['models'][2]
	for key in ('checkpoint', 'embeddings_dir'):
		local[key] = _candidate_path(local[key], candidate_id)
	for key in ('runs_root', 'summary_root'):
		expected['outputs'][key] = _candidate_path(
			expected['outputs'][key], candidate_id
		)

	assert candidate == expected


@pytest.mark.parametrize('candidate_id', CANDIDATE_SHIFTS)
def test_phase_a_downstream_uses_v3_layout_v2_sources_and_fixed_identity(
	candidate_id: str,
) -> None:
	candidate_path = ROOT / '30_downstream' / f'{candidate_id}_medium.yaml'
	candidate = f3_lithology_five_way_config_from_mapping(load_config(candidate_path))
	v3 = f3_lithology_five_way_config_from_mapping(load_config(V3_FIVE_WAY))

	assert (
		candidate.dataset
		== v3.dataset
		== {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v2',
		}
	)
	assert candidate.labels == v3.labels
	assert candidate.section_layout_dataset_root == v3.section_layout_dataset_root
	assert candidate.section_layout_dataset_root.name == 'voxel_section_layout_v3'
	for model_id in ('mae', 'mae_hmm_k6', 'local_barlow_twins_hmm_k6', 'random'):
		assert candidate.model_by_id(model_id) == v3.model_by_id(model_id)
	local = candidate.model_by_id('local_barlow_twins')
	assert local.expected == EXPECTED_MODEL_IDENTITIES['local_barlow_twins']
	assert local.checkpoint.parts[-2:] == (candidate_id, 'latest.pt')
	assert local.embeddings_dir.parts[-3:] == (
		candidate_id,
		'local_barlow_twins',
		'overlap_x64',
	)
	assert 'facies_benchmark_v2' in local.embeddings_dir.parts
