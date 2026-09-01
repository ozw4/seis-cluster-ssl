"""Parity contracts for the F3 Local BT Gaussian-view screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)

EXPERIMENT_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/111_local_barlow_twins_gaussian_view_v1'
)
CONTROL_TRAINING_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1/02_full_100ep.yaml'
)
CANONICAL_CONTINUATION_CONFIG = Path(
	'experiments/f3/facies_benchmark_v1/'
	'110_lithology_mae_local_bt_five_way_v1/10_stage2/'
	'local_bt100/local_bt_continue/02_full_25ep.yaml'
)
REFERENCE_EXTRACTION_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v2/'
	'50_embeddings/05_extract_random.yaml'
)
HORIZONTAL_POLICY = 'horizontal_flip_gaussian_noise_v1'
IDENTITY_POLICY = 'identity_gaussian_noise_v1'
VARIANTS = (
	(
		'gaussian_noise_std005',
		HORIZONTAL_POLICY,
		0.05,
		EXPERIMENT_ROOT / '10_stage1/gaussian_noise_std005/01_screen_25ep.yaml',
		EXPERIMENT_ROOT / '15_stage2/gaussian_noise_std005/01_continue_25ep.yaml',
		EXPERIMENT_ROOT / '20_embeddings/01_extract_gaussian_noise_std005.yaml',
	),
	(
		'gaussian_noise_std010',
		HORIZONTAL_POLICY,
		0.10,
		EXPERIMENT_ROOT / '10_stage1/gaussian_noise_std010/01_screen_25ep.yaml',
		EXPERIMENT_ROOT / '15_stage2/gaussian_noise_std010/01_continue_25ep.yaml',
		EXPERIMENT_ROOT / '20_embeddings/02_extract_gaussian_noise_std010.yaml',
	),
	(
		'identity_gaussian_noise_std010',
		IDENTITY_POLICY,
		0.10,
		EXPERIMENT_ROOT
		/ '10_stage1/identity_gaussian_noise_std010/01_screen_25ep.yaml',
		EXPERIMENT_ROOT
		/ '15_stage2/identity_gaussian_noise_std010/01_continue_25ep.yaml',
		EXPERIMENT_ROOT
		/ '20_embeddings/04_extract_identity_gaussian_noise_std010.yaml',
	),
)
LEGACY_CONTROL_ID = 'local_barlow_twins_legacy_flip_25ep'
LEGACY_TRAINING_CONFIG = (
	EXPERIMENT_ROOT / '10_stage1/legacy_flip_25ep/01_matched_25ep.yaml'
)
LEGACY_CONTINUATION_CONFIG = (
	EXPERIMENT_ROOT / '15_stage2/legacy_flip_25ep/01_continue_25ep.yaml'
)
LEGACY_EXTRACTION_CONFIG = (
	EXPERIMENT_ROOT / '20_embeddings/03_extract_legacy_flip_25ep.yaml'
)


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)


@pytest.mark.parametrize(
	(
		'variant',
		'policy',
		'noise_std',
		'training_path',
		'_continuation_path',
		'_extraction_path',
	),
	VARIANTS,
)
def test_stage1_screen_changes_only_authorized_fields(
	variant: str,
	policy: str,
	noise_std: float,
	training_path: Path,
	_continuation_path: Path,
	_extraction_path: Path,
) -> None:
	del variant, policy, noise_std, _continuation_path, _extraction_path
	control = load_config(CONTROL_TRAINING_CONFIG)
	candidate = load_config(training_path)

	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = control['paths']['output_root']
	comparison['augmentations'] = control['augmentations']
	comparison['train']['epochs'] = control['train']['epochs']

	assert comparison == control
	assert candidate['train']['epochs'] == 25
	assert 'max_steps' not in candidate['train']


@pytest.mark.parametrize(
	(
		'variant',
		'policy',
		'noise_std',
		'training_path',
		'_continuation_path',
		'_extraction_path',
	),
	VARIANTS,
)
def test_stage1_screen_resolves_exact_policy_and_stable_output(
	variant: str,
	policy: str,
	noise_std: float,
	training_path: Path,
	_continuation_path: Path,
	_extraction_path: Path,
	tmp_path: Path,
) -> None:
	del _continuation_path, _extraction_path
	config = resolve_barlow_twins_training_config(load_config(training_path))
	expected_output = (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_gaussian_view_v1/stage1'
		/ variant
		/ 'full_25ep'
	)
	expected_augmentations = {
		'policy': policy,
		'gaussian_noise_std': noise_std,
	}
	if policy == HORIZONTAL_POLICY:
		expected_augmentations['horizontal_flip_probability'] = 0.5
	assert config['augmentations'] == expected_augmentations
	assert config['barlow_twins'] == {
		'method': 'local_barlow_twins_3d',
		'local_pairs_per_crop': 128,
		'projector_dim': 384,
		'redundancy_weight': 0.005,
		'normalization_eps': 0.0001,
	}
	assert Path(config['paths']['output_root']) == expected_output
	assert config['train']['epochs'] == 25
	assert (
		config['train']['epochs']
		* config['train']['samples_per_epoch']
		// config['train']['batch_size']
		== 15_625
	)


def test_legacy_control_changes_only_duration_and_output() -> None:
	control = load_config(CONTROL_TRAINING_CONFIG)
	matched = load_config(LEGACY_TRAINING_CONFIG)

	comparison = deepcopy(matched)
	comparison['paths']['output_root'] = control['paths']['output_root']
	comparison['train']['epochs'] = control['train']['epochs']

	assert comparison == control
	assert matched['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert matched['train']['epochs'] == 25
	assert 'max_steps' not in matched['train']


def test_legacy_control_resolves_to_exact_25_epoch_budget(tmp_path: Path) -> None:
	config = resolve_barlow_twins_training_config(load_config(LEGACY_TRAINING_CONFIG))

	assert config['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert Path(config['paths']['output_root']) == (
		tmp_path
		/ 'artifacts/pretraining/f3/facies_benchmark_v1'
		/ 'local_barlow_twins_gaussian_view_v1/stage1'
		/ 'legacy_flip_25ep/full_25ep'
	)
	assert config['train']['epochs'] == 25
	assert (
		config['train']['epochs']
		* config['train']['samples_per_epoch']
		// config['train']['batch_size']
		== 15_625
	)


@pytest.mark.parametrize(
	(
		'_variant',
		'_policy',
		'_noise_std',
		'base_path',
		'continuation_path',
		'_extraction_path',
	),
	VARIANTS,
)
def test_selectable_continuation_changes_only_authorized_fields(
	_variant: str,
	_policy: str,
	_noise_std: float,
	base_path: Path,
	continuation_path: Path,
	_extraction_path: Path,
) -> None:
	del _variant, _policy, _noise_std, _extraction_path
	canonical = load_config(CANONICAL_CONTINUATION_CONFIG)
	candidate = load_config(continuation_path)

	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = canonical['paths']['output_root']
	comparison['continuation']['init_checkpoint'] = canonical['continuation'][
		'init_checkpoint'
	]
	comparison['augmentations'] = canonical['augmentations']
	assert comparison == canonical

	base = resolve_barlow_twins_training_config(load_config(base_path))
	resolved = resolve_barlow_twins_training_config(candidate)
	reference = resolve_barlow_twins_training_config(canonical)
	assert resolved['continuation'] == {
		'init_checkpoint': str(Path(base['paths']['output_root']) / 'latest.pt'),
		'unfreeze_top_blocks': 1,
	}
	assert resolved['train'] == reference['train']
	for key in (
		'manifests',
		'data',
		'zero_mask',
		'model',
		'augmentations',
		'barlow_twins',
	):
		assert resolved[key] == (
			base[key] if key == 'augmentations' else reference[key]
		)
	assert resolved['train']['lr'] == 1e-5
	assert (
		resolved['train']['epochs']
		* resolved['train']['samples_per_epoch']
		// resolved['train']['batch_size']
		== 15_625
	)


def test_legacy_continuation_is_exact_canonical_top1_except_lineage_output() -> None:
	canonical = load_config(CANONICAL_CONTINUATION_CONFIG)
	candidate = load_config(LEGACY_CONTINUATION_CONFIG)
	comparison = deepcopy(candidate)
	comparison['paths']['output_root'] = canonical['paths']['output_root']
	comparison['continuation']['init_checkpoint'] = canonical['continuation'][
		'init_checkpoint'
	]
	assert comparison == canonical

	base = resolve_barlow_twins_training_config(load_config(LEGACY_TRAINING_CONFIG))
	resolved = resolve_barlow_twins_training_config(candidate)
	reference = resolve_barlow_twins_training_config(canonical)
	assert resolved['continuation'] == {
		'init_checkpoint': str(Path(base['paths']['output_root']) / 'latest.pt'),
		'unfreeze_top_blocks': 1,
	}
	assert resolved['augmentations'] == {'horizontal_flip_probability': 0.5}
	assert resolved['train'] == reference['train']


@pytest.mark.parametrize(
	(
		'variant',
		'_policy',
		'_noise',
		'_training_path',
		'continuation_path',
		'extraction_path',
	),
	VARIANTS,
)
def test_extraction_uses_matching_checkpoint_and_v2_overlap_x64_contract(
	variant: str,
	_policy: str,
	_noise: float,
	_training_path: Path,
	continuation_path: Path,
	extraction_path: Path,
	tmp_path: Path,
) -> None:
	del _policy, _noise, _training_path
	continuation = resolve_barlow_twins_training_config(load_config(continuation_path))
	candidate = load_config(extraction_path)
	reference = load_config(REFERENCE_EXTRACTION_CONFIG)

	comparison = deepcopy(candidate)
	comparison['embeddings'] = reference['embeddings']
	assert comparison == reference

	config = resolve_embedding_extraction_config(candidate)
	expected_checkpoint = Path(continuation['paths']['output_root']) / 'latest.pt'
	expected_output = (
		tmp_path
		/ 'artifacts/embeddings/f3/facies_benchmark_v2'
		/ 'local_barlow_twins_gaussian_view_v1'
		/ f'local_barlow_twins_{variant}'
		/ 'overlap_x64'
	)
	assert Path(config['embeddings']['checkpoint']) == expected_checkpoint
	assert Path(config['embeddings']['output_dir']) == expected_output
	assert expected_output.parent.name == f'local_barlow_twins_{variant}'
	assert config['manifests']['input'] == reference['manifests']['input']
	assert config['embedding'] == reference['embedding']


def test_legacy_extraction_uses_full_model_id_and_matching_checkpoint(
	tmp_path: Path,
) -> None:
	continuation = resolve_barlow_twins_training_config(
		load_config(LEGACY_CONTINUATION_CONFIG)
	)
	candidate = load_config(LEGACY_EXTRACTION_CONFIG)
	reference = load_config(REFERENCE_EXTRACTION_CONFIG)

	comparison = deepcopy(candidate)
	comparison['embeddings'] = reference['embeddings']
	assert comparison == reference

	config = resolve_embedding_extraction_config(candidate)
	expected_checkpoint = Path(continuation['paths']['output_root']) / 'latest.pt'
	expected_output = (
		tmp_path
		/ 'artifacts/embeddings/f3/facies_benchmark_v2'
		/ 'local_barlow_twins_gaussian_view_v1'
		/ LEGACY_CONTROL_ID
		/ 'overlap_x64'
	)
	assert Path(config['embeddings']['checkpoint']) == expected_checkpoint
	assert Path(config['embeddings']['output_dir']) == expected_output
	assert expected_output.parent.name == LEGACY_CONTROL_ID
	assert config['manifests']['input'] == reference['manifests']['input']
	assert config['embedding'] == reference['embedding']


def test_candidates_have_unique_base_final_and_embedding_artifacts() -> None:
	base_outputs: set[str] = set()
	final_outputs: set[str] = set()
	embedding_outputs: set[str] = set()
	stage2_paths: set[Path] = set()
	for _, _, _, base_path, continuation_path, extraction_path in VARIANTS:
		base = load_config(base_path)
		continuation = load_config(continuation_path)
		extraction = load_config(extraction_path)
		base_outputs.add(base['paths']['output_root'])
		final_outputs.add(continuation['paths']['output_root'])
		embedding_outputs.add(extraction['embeddings']['output_dir'])
		stage2_paths.add(continuation_path)
	base_outputs.add(load_config(LEGACY_TRAINING_CONFIG)['paths']['output_root'])
	final_outputs.add(load_config(LEGACY_CONTINUATION_CONFIG)['paths']['output_root'])
	stage2_paths.add(LEGACY_CONTINUATION_CONFIG)
	embedding_outputs.add(
		load_config(LEGACY_EXTRACTION_CONFIG)['embeddings']['output_dir']
	)

	assert len(base_outputs) == len(VARIANTS) + 1
	assert len(final_outputs) == len(VARIANTS) + 1
	assert base_outputs.isdisjoint(final_outputs)
	assert len(embedding_outputs) == len(VARIANTS) + 1
	assert set((EXPERIMENT_ROOT / '15_stage2').rglob('*.yaml')) == stage2_paths
	assert not list(EXPERIMENT_ROOT.rglob('*five_way*.yaml'))
