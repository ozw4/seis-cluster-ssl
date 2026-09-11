from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_decoder import (
	channel_decoder_config_from_mapping,
)
from seis_ssl_cluster.parihaka.channel_recipe_transfer_results import (
	channel_recipe_transfer_summary_config_from_mapping,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

EXPERIMENT_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/39_channel_local_bt_rot90_asym_g060_v1'
)
F3_RECIPE_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1'
)
PARIHAKA_REFERENCE_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/33_channel_mae_local_bt_four_way_v1'
)
PRETRAIN_CONFIG = EXPERIMENT_ROOT / '10_pretraining/01_full_3ep.yaml'
EMBEDDING_CONFIG = EXPERIMENT_ROOT / '20_embeddings/01_extract_embeddings.yaml'
CHANNEL_CONFIG = EXPERIMENT_ROOT / '30_downstream/01_channel_candidate.yaml'
F3_PRETRAIN_CONFIG = F3_RECIPE_ROOT / '10_pretraining/rot90_asym_g060_3ep.yaml'
F3_EMBEDDING_CONFIG = F3_RECIPE_ROOT / '20_embeddings/rot90_asym_g060_3ep.yaml'
PARIHAKA_EMBEDDING_REFERENCE = (
	PARIHAKA_REFERENCE_ROOT / '01_extract_local_barlow_twins_embeddings.yaml'
)
PARIHAKA_CHANNEL_REFERENCE = (
	PARIHAKA_REFERENCE_ROOT / '03_channel_mae_local_bt_four_way.yaml'
)
CANDIDATE_MODEL_ID = 'local_barlow_twins_rot90_asym_g060_3ep'
REFERENCE_MODEL_ID = 'local_barlow_twins'


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	return root


def test_experiment_owns_exactly_three_yaml_configs() -> None:
	assert {
		path.relative_to(EXPERIMENT_ROOT).as_posix()
		for path in EXPERIMENT_ROOT.rglob('*.yaml')
	} == {
		'10_pretraining/01_full_3ep.yaml',
		'20_embeddings/01_extract_embeddings.yaml',
		'30_downstream/01_channel_candidate.yaml',
	}


def test_pretraining_changes_only_f3_survey_lineage_paths(
	artifact_root: Path,
) -> None:
	candidate = resolve_barlow_twins_training_config(load_config(PRETRAIN_CONFIG))
	f3_recipe = resolve_barlow_twins_training_config(load_config(F3_PRETRAIN_CONFIG))
	candidate_paths = _mapping(candidate, 'paths')
	f3_paths = _mapping(f3_recipe, 'paths')

	assert candidate_paths['artifact_root'] == f3_paths['artifact_root']
	assert candidate_paths['output_root'] == str(
		artifact_root
		/ 'pretraining/parihaka/facies_benchmark_v1'
		/ 'local_bt_rot90_asym_g060_v1/full_3ep'
	)
	assert _mapping(candidate, 'manifests') == {
		'train': str(
			artifact_root
			/ 'data/parihaka/facies_benchmark_v1/parihaka_amplitude_manifest.json'
		),
		'train_path_list': str(
			artifact_root / 'data/parihaka/facies_benchmark_v1/parihaka_npy_paths.txt'
		),
	}

	comparison = deepcopy(candidate)
	_mapping(comparison, 'paths')['output_root'] = f3_paths['output_root']
	comparison['manifests'] = deepcopy(f3_recipe['manifests'])
	assert comparison == f3_recipe


def test_embedding_changes_only_lineage_paths_and_reuses_parihaka_contract(
	artifact_root: Path,
) -> None:
	candidate = resolve_embedding_extraction_config(load_config(EMBEDDING_CONFIG))
	f3_recipe = resolve_embedding_extraction_config(load_config(F3_EMBEDDING_CONFIG))
	parihaka_reference = resolve_embedding_extraction_config(
		load_config(PARIHAKA_EMBEDDING_REFERENCE)
	)
	embeddings = _mapping(candidate, 'embeddings')
	pretraining = resolve_barlow_twins_training_config(load_config(PRETRAIN_CONFIG))

	assert candidate['embedding'] == f3_recipe['embedding']
	assert candidate['embedding'] == parihaka_reference['embedding']
	assert candidate['manifests'] == parihaka_reference['manifests']
	assert embeddings == {
		'checkpoint': str(
			Path(str(_mapping(pretraining, 'paths')['output_root'])) / 'latest.pt'
		),
		'output_dir': str(
			artifact_root
			/ 'embeddings/parihaka/facies_benchmark_v1'
			/ 'local_bt_rot90_asym_g060_v1'
			/ f'{CANDIDATE_MODEL_ID}/overlap_x64'
		),
	}

	f3_comparison = deepcopy(candidate)
	f3_comparison['manifests'] = deepcopy(f3_recipe['manifests'])
	f3_comparison['embeddings'] = deepcopy(f3_recipe['embeddings'])
	assert f3_comparison == f3_recipe

	parihaka_comparison = deepcopy(candidate)
	parihaka_comparison['embeddings'] = deepcopy(parihaka_reference['embeddings'])
	assert parihaka_comparison == parihaka_reference


def test_channel_changes_only_models_summary_and_comparison_identity(
	artifact_root: Path,
) -> None:
	candidate = load_config(CHANNEL_CONFIG)
	parihaka_reference = load_config(PARIHAKA_CHANNEL_REFERENCE)
	extraction = resolve_embedding_extraction_config(load_config(EMBEDDING_CONFIG))
	models = _mapping(_mapping(candidate, 'embeddings'), 'models')
	reference_models = _mapping(
		_mapping(parihaka_reference, 'embeddings'),
		'models',
	)
	outputs = _mapping(candidate, 'outputs')
	comparison_config = _mapping(candidate, 'comparison')

	assert tuple(models) == (CANDIDATE_MODEL_ID, REFERENCE_MODEL_ID)
	assert models[REFERENCE_MODEL_ID] == reference_models[REFERENCE_MODEL_ID]
	assert models[CANDIDATE_MODEL_ID] == {
		'dir': _mapping(extraction, 'embeddings')['output_dir'],
		'checkpoint': _mapping(extraction, 'embeddings')['checkpoint'],
	}
	assert comparison_config == {
		'candidate_model_id': CANDIDATE_MODEL_ID,
		'reference_model_id': REFERENCE_MODEL_ID,
		'legacy_random_runs_root': str(artifact_root / 'channel_benchmark/runs'),
		'legacy_random_model_id': 'random',
	}
	assert outputs['output_dir'] == str(
		artifact_root / 'channel_benchmark/local_bt_rot90_asym_g060_v1/summary'
	)

	normalized = deepcopy(candidate)
	normalized.pop('comparison')
	normalized['embeddings'] = deepcopy(parihaka_reference['embeddings'])
	_mapping(normalized, 'outputs')['output_dir'] = _mapping(
		parihaka_reference, 'outputs'
	)['output_dir']
	assert normalized == parihaka_reference

	decoder_config = channel_decoder_config_from_mapping(candidate)
	summary_config = channel_recipe_transfer_summary_config_from_mapping(candidate)
	assert tuple(decoder_config.models) == (CANDIDATE_MODEL_ID, REFERENCE_MODEL_ID)
	assert len(LAYOUT_IDS) * len(DATA_SIZE_PREFIX) == 15
	assert summary_config.learned_runs_root == decoder_config.runs_root
	assert summary_config.output_dir == Path(str(outputs['output_dir']))


def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
	child = value.get(key)
	if not isinstance(child, dict):
		raise TypeError(f'{key} must be a dict')
	return child
