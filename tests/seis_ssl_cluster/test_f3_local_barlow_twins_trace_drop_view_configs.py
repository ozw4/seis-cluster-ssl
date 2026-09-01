"""Producer and runbook contracts for F3 trace-drop experiment 112."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)

ROOT = Path(
	'experiments/f3/facies_benchmark_v2/'
	'112_local_barlow_twins_trace_drop_view_v1'
)
GAUSSIAN_BASE1 = Path(
	'experiments/f3/facies_benchmark_v2/'
	'111_local_barlow_twins_gaussian_view_v1/50_base1ep'
)
BASE_CONFIG = (
	ROOT
	/ '10_stage1/horizontal_trace_drop_p001_base1ep/01_screen_1ep.yaml'
)
CONTINUATION_CONFIG = (
	ROOT
	/ '15_stage2/horizontal_trace_drop_p001_base1ep/01_continue_25ep.yaml'
)
EXTRACTION_CONFIG = (
	ROOT / '20_embeddings/01_extract_horizontal_trace_drop_p001_base1ep.yaml'
)
VALIDATION_CONFIG = ROOT / '30_validation/01_candidate.yaml'
EXPECTED_AUGMENTATIONS = {
	'policy': 'horizontal_flip_trace_drop_v1',
	'horizontal_flip_probability': 0.5,
	'trace_drop_probability': 0.01,
}


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts')
	)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))


def test_base_changes_only_augmentation_and_namespace() -> None:
	parent = load_config(
		GAUSSIAN_BASE1
		/ '10_stage1/gaussian_noise_std010_base1ep/01_screen_1ep.yaml'
	)
	candidate = load_config(BASE_CONFIG)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = parent['paths']['output_root']
	comparison['augmentations'] = parent['augmentations']

	assert comparison == parent
	assert candidate['augmentations'] == EXPECTED_AUGMENTATIONS
	resolved = resolve_barlow_twins_training_config(candidate)
	assert resolved['train']['epochs'] == 1
	assert resolved['train']['lr'] == 1e-4
	assert resolved['train']['seed'] == 42
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 625
	)


def test_continuation_changes_only_augmentation_and_namespace() -> None:
	parent = load_config(
		GAUSSIAN_BASE1
		/ '15_stage2/gaussian_noise_std010_base1ep/01_continue_25ep.yaml'
	)
	candidate = load_config(CONTINUATION_CONFIG)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = parent['paths']['output_root']
	comparison['continuation']['init_checkpoint'] = parent['continuation'][
		'init_checkpoint'
	]
	comparison['augmentations'] = parent['augmentations']

	assert comparison == parent
	base = resolve_barlow_twins_training_config(load_config(BASE_CONFIG))
	resolved = resolve_barlow_twins_training_config(candidate)
	assert resolved['augmentations'] == EXPECTED_AUGMENTATIONS
	assert resolved['continuation'] == {
		'init_checkpoint': str(Path(base['paths']['output_root']) / 'latest.pt'),
		'unfreeze_top_blocks': 1,
	}
	assert resolved['train']['epochs'] == 25
	assert resolved['train']['lr'] == 1e-5
	assert resolved['train']['weight_decay'] == 0.05
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 15_625
	)


def test_extraction_changes_only_checkpoint_and_namespace() -> None:
	parent = load_config(
		GAUSSIAN_BASE1
		/ '20_embeddings/01_extract_gaussian_noise_std010_base1ep.yaml'
	)
	candidate = load_config(EXTRACTION_CONFIG)
	comparison = deepcopy(candidate)
	comparison['embeddings'] = parent['embeddings']

	assert comparison == parent
	continuation = resolve_barlow_twins_training_config(
		load_config(CONTINUATION_CONFIG)
	)
	resolved = resolve_embedding_extraction_config(candidate)
	assert Path(resolved['embeddings']['checkpoint']) == (
		Path(continuation['paths']['output_root']) / 'latest.pt'
	)
	assert resolved['embedding']['window_size'] == [128, 128, 128]
	assert resolved['embedding']['overlap'] == [64, 64, 64]
	assert resolved['embedding']['min_token_valid_fraction'] == 0.5


def test_validation_config_is_one_candidate_without_selection_lock() -> None:
	config = load_config(VALIDATION_CONFIG)

	assert set(config) == {'parent', 'benchmark', 'candidate', 'outputs'}
	assert config['parent']['final_result_sha256'] == (
		'6ab9cb2ae8bf89eb5dea9ab34244b60f41499f44cc046995e41ec0350854e1fa'
	)
	assert config['candidate'] == {
		'candidate_id': 'local_barlow_twins_horizontal_trace_drop_p001_base1ep',
		'role': 'preregistered_trace_drop_view',
		'base_checkpoint': config['candidate']['base_checkpoint'],
		'final_checkpoint': config['candidate']['final_checkpoint'],
		'embeddings_dir': config['candidate']['embeddings_dir'],
		'augmentations': EXPECTED_AUGMENTATIONS,
		'base_pretraining_epochs': 1,
		'continuation_epochs': 25,
	}
	assert set(config['outputs']) == {'runs_root', 'protocol_lock', 'final_result'}
	assert 'selection_lock' not in str(config)
	assert '/local_barlow_twins_trace_drop_view_v1/' in config['outputs']['runs_root']


def test_readme_uses_only_existing_producer_clis_in_protocol_order() -> None:
	readme = (ROOT / 'README.md').read_text(encoding='utf-8')
	producer_paths = re.findall(r'python (proc/seis_ssl_cluster/[^ \\\n]+\.py)', readme)

	assert producer_paths
	assert all(Path(path).is_file() for path in producer_paths)
	assert 'continue_amp_barlow_twins.py' not in readme
	base_position = readme.index(
		'10_stage1/horizontal_trace_drop_p001_base1ep/01_screen_1ep.yaml'
	)
	lock_position = readme.index('--create-protocol-lock')
	continuation_position = readme.index(
		'15_stage2/horizontal_trace_drop_p001_base1ep/01_continue_25ep.yaml'
	)
	medium_position = readme.index('--size medium')
	assert base_position < lock_position < continuation_position < medium_position
	assert 'No test label or test metric is read.' in readme
