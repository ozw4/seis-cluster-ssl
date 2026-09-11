"""Training and embedding config contracts for F3 trace-drop experiment 113."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)

ROOT = Path(
	'experiments/f3/facies_benchmark_v2/113_local_barlow_twins_trace_drop_p002_view_v1'
)
P001_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/112_local_barlow_twins_trace_drop_view_v1'
)
BASE_CONFIG = ROOT / '10_stage1/horizontal_trace_drop_p002_base1ep/01_screen_1ep.yaml'
CONTINUATION_CONFIG = (
	ROOT / '15_stage2/horizontal_trace_drop_p002_base1ep/01_continue_25ep.yaml'
)
EXTRACTION_CONFIG = (
	ROOT / '20_embeddings/01_extract_horizontal_trace_drop_p002_base1ep.yaml'
)
EXPECTED_AUGMENTATIONS = {
	'policy': 'horizontal_flip_trace_drop_v1',
	'horizontal_flip_probability': 0.5,
	'trace_drop_probability': 0.02,
}


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))


def test_base_changes_only_augmentation_and_namespace() -> None:
	parent = load_config(
		P001_ROOT / '10_stage1/horizontal_trace_drop_p001_base1ep/01_screen_1ep.yaml'
	)
	candidate = load_config(BASE_CONFIG)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = parent['paths']['output_root']
	comparison['augmentations'] = parent['augmentations']

	assert comparison == parent
	resolved = resolve_barlow_twins_training_config(candidate)
	assert resolved['augmentations'] == EXPECTED_AUGMENTATIONS
	assert resolved['train']['epochs'] == 1
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 625
	)


def test_continuation_uses_the_base_checkpoint_and_fixed_budget() -> None:
	parent = load_config(
		P001_ROOT / '15_stage2/horizontal_trace_drop_p001_base1ep/01_continue_25ep.yaml'
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
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 15_625
	)


def test_embedding_uses_the_continuation_checkpoint() -> None:
	parent = load_config(
		P001_ROOT / '20_embeddings/01_extract_horizontal_trace_drop_p001_base1ep.yaml'
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
