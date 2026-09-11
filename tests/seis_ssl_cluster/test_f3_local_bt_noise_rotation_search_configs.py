from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)

ROOT = Path(
	'experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1'
)
FROZEN_HMM_V1_COMPLETION_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/126_pretraining_comparison_completion_v1'
)
BASELINE_ROOT = Path('experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1')
ASCENT_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/122_local_bt_nuisance_region_ascent_v1'
)
VICREG_POC_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1'
)
CANONICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
SEARCH_NAMESPACE = 'local_bt_noise_rotation_search_v1'
FLIP = {'horizontal_flip_probability': 0.5}
ARM_CONTRACTS = {
	'gauss015': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.15,
		},
		[2, 2, 1],
	),
	'laplace_g005': (
		{
			'policy': 'horizontal_flip_laplace_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.05,
		},
		[2, 2, 1],
	),
	'laplace_g010': (
		{
			'policy': 'horizontal_flip_laplace_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 1],
	),
	'laplace_g020': (
		{
			'policy': 'horizontal_flip_laplace_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.2,
		},
		[2, 2, 1],
	),
	'colored_g010_z4': (
		{
			'policy': 'horizontal_flip_colored_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.1,
			'noise_z_smoothing': 4,
		},
		[2, 2, 1],
	),
	'colored_g020_z8': (
		{
			'policy': 'horizontal_flip_colored_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.2,
			'noise_z_smoothing': 8,
		},
		[2, 2, 1],
	),
	'gauss010_td002': (
		{
			'policy': 'horizontal_flip_gaussian_trace_drop_v1',
			**FLIP,
			'gaussian_noise_std': 0.1,
			'trace_drop_probability': 0.02,
		},
		[2, 2, 1],
	),
	'asym_g010': (
		{
			'policy': 'horizontal_flip_asymmetric_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 1],
	),
	'asym_g020': (
		{
			'policy': 'horizontal_flip_asymmetric_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.2,
		},
		[2, 2, 1],
	),
	'asym_g030': (
		{
			'policy': 'horizontal_flip_asymmetric_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.3,
		},
		[2, 2, 1],
	),
	'asym_g040': (
		{
			'policy': 'horizontal_flip_asymmetric_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.4,
		},
		[2, 2, 1],
	),
	'asym_g060': (
		{
			'policy': 'horizontal_flip_asymmetric_noise_v1',
			**FLIP,
			'gaussian_noise_std': 0.6,
		},
		[2, 2, 1],
	),
	'rot90_g010': (
		{'policy': 'xy_rot90_gaussian_noise_v1', 'gaussian_noise_std': 0.1},
		[2, 2, 1],
	),
	'rot90_g020': (
		{'policy': 'xy_rot90_gaussian_noise_v1', 'gaussian_noise_std': 0.2},
		[2, 2, 1],
	),
	'rot90_laplace_g010': (
		{'policy': 'xy_rot90_laplace_noise_v1', 'gaussian_noise_std': 0.1},
		[2, 2, 1],
	),
	'rot90_asym_g020': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.2},
		[2, 2, 1],
	),
	'rot90_asym_g040': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.4},
		[2, 2, 1],
	),
	'rot90_asym_g060': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.6},
		[2, 2, 1],
	),
	'rot90_asym_g080': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.8},
		[2, 2, 1],
	),
	'rot90_asym_g040_r111': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.4},
		[1, 1, 1],
	),
	'rot90_asym_g040_r222': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.4},
		[2, 2, 2],
	),
	'd4_g010': (
		{'policy': 'xy_d4_gaussian_noise_v1', 'gaussian_noise_std': 0.1},
		[2, 2, 1],
	),
}
ROTATION_POLICIES = frozenset(
	{
		'xy_rot90_gaussian_noise_v1',
		'xy_rot90_laplace_noise_v1',
		'xy_rot90_asymmetric_noise_v1',
		'xy_d4_gaussian_noise_v1',
	}
)
EPOCH_BUDGETS = {3: 1_875, 10: 6_250}
# The winning arm alone carries the epoch-scaling sweep; every point resumes
# from the next-lower checkpoint so the 3/6/10/20 series is one trajectory.
EPOCH_SWEEP_ARM = 'rot90_asym_g040'
EPOCH_SWEEP_BUDGETS = {6: 3_750, 20: 12_500}
EPOCH_SWEEP_RESUME_SOURCES = {6: 3, 10: 3, 20: 10}
README = ROOT / 'README.md'
SHARED_RUNBOOK = ROOT.parent / 'LOCAL_BT_CANDIDATE_RUNBOOK.md'
RESULT_REPORT_LINK = (
	'../../../../reports/f3/local_bt_rot90_asymmetric_noise_recipe.md'
)
EPOCH_REPORT_LINK = '../../../../reports/f3/local_bt_epoch_scaling_v1.md'


def arm_epochs(arm: str) -> dict[int, int]:
	if arm != EPOCH_SWEEP_ARM:
		return dict(EPOCH_BUDGETS)
	return {**EPOCH_BUDGETS, **EPOCH_SWEEP_BUDGETS}


def candidate_id(arm: str, epochs: int) -> str:
	return f'local_bt_nr_{arm}_{epochs}ep'


def pretrain_config(arm: str, epochs: int) -> Path:
	suffix = '_3ep.yaml' if epochs == 3 else f'_{epochs}ep_resume.yaml'
	return ROOT / '10_pretraining' / f'{arm}{suffix}'


def extraction_config(arm: str, epochs: int) -> Path:
	return ROOT / '20_embeddings' / f'{arm}_{epochs}ep.yaml'


def candidate_config(arm: str, epochs: int) -> Path:
	return ROOT / '30_downstream' / f'{arm}_{epochs}ep_medium_layout_001.yaml'


def all_configs() -> tuple[Path, ...]:
	paths: list[Path] = []
	for arm in ARM_CONTRACTS:
		for epochs in arm_epochs(arm):
			paths.append(pretrain_config(arm, epochs))
			paths.append(extraction_config(arm, epochs))
			paths.append(candidate_config(arm, epochs))
	return tuple(paths)


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	return root


def test_all_arm_configs_resolve_with_fixed_budgets_and_contracts(
	artifact_root: Path,
) -> None:
	for arm, (augmentations, window) in ARM_CONTRACTS.items():
		for epochs, budget in arm_epochs(arm).items():
			raw = load_config(pretrain_config(arm, epochs))
			assert 'continuation' not in raw, arm
			assert 'max_steps' not in raw['train'], arm

			full = resolve_barlow_twins_training_config(
				load_config(pretrain_config(arm, epochs))
			)
			train = full['train']

			assert full['stage'] == 'barlow_twins_training', arm
			assert train.get('max_steps') is None, arm
			assert train['epochs'] == epochs, arm
			assert (
				train['epochs'] * train['samples_per_epoch'] // train['batch_size']
				== budget
			), arm
			assert full['augmentations'] == augmentations, arm
			barlow_twins = full['barlow_twins']
			assert barlow_twins['positive_window_tokens'] == window, arm
			assert barlow_twins['local_pairs_per_crop'] == 128, arm
			assert barlow_twins['redundancy_weight'] == 0.005, arm
			assert barlow_twins['projector_dim'] == 384, arm
			assert Path(full['paths']['output_root']) == (
				artifact_root
				/ 'pretraining/f3/facies_benchmark_v1'
				/ SEARCH_NAMESPACE
				/ arm
				/ f'full_{epochs}ep'
			), arm


def test_arms_differ_from_canonical_only_in_declared_fields() -> None:
	baseline = load_config(BASELINE_ROOT / '02_full_100ep.yaml')
	for arm, (augmentations, window) in ARM_CONTRACTS.items():
		config = load_config(pretrain_config(arm, 3))
		assert config['augmentations'] == augmentations, arm

		comparison = deepcopy(config)
		comparison['paths']['output_root'] = baseline['paths']['output_root']
		comparison['train']['epochs'] = baseline['train']['epochs']
		comparison['augmentations'] = baseline['augmentations']
		assert comparison['barlow_twins']['positive_window_tokens'] == window, arm
		del comparison['barlow_twins']['positive_window_tokens']
		assert comparison == baseline, arm


def test_rotation_arms_drop_the_mirror_and_keep_only_rotation_plus_noise() -> None:
	for arm, (augmentations, _window) in ARM_CONTRACTS.items():
		if augmentations['policy'] not in ROTATION_POLICIES:
			continue
		assert 'horizontal_flip_probability' not in augmentations, arm
		assert set(augmentations) == {'policy', 'gaussian_noise_std'}, arm


def test_noise_axis_arms_differ_only_in_the_noise_declaration() -> None:
	reference = load_config(pretrain_config('rot90_asym_g040', 3))
	for arm in ('rot90_asym_g020', 'rot90_asym_g060', 'rot90_asym_g080'):
		config = load_config(pretrain_config(arm, 3))
		comparison = deepcopy(config)
		comparison['paths']['output_root'] = reference['paths']['output_root']
		comparison['augmentations']['gaussian_noise_std'] = (
			reference['augmentations']['gaussian_noise_std']
		)
		assert comparison == reference, arm


def test_window_axis_arms_differ_only_in_the_positive_window() -> None:
	reference = load_config(pretrain_config('rot90_asym_g040', 3))
	for arm, window in (
		('rot90_asym_g040_r111', [1, 1, 1]),
		('rot90_asym_g040_r222', [2, 2, 2]),
	):
		config = load_config(pretrain_config(arm, 3))
		assert config['barlow_twins']['positive_window_tokens'] == window, arm

		comparison = deepcopy(config)
		comparison['paths']['output_root'] = reference['paths']['output_root']
		comparison['barlow_twins']['positive_window_tokens'] = (
			reference['barlow_twins']['positive_window_tokens']
		)
		assert comparison == reference, arm


def test_asymmetric_arms_mirror_their_symmetric_twin_except_the_policy() -> None:
	symmetric = load_config(
		ASCENT_ROOT / '10_pretraining/gauss020_r221_3ep.yaml'
	)
	asymmetric = load_config(pretrain_config('asym_g020', 3))

	assert symmetric['augmentations']['policy'] == (
		'horizontal_flip_gaussian_noise_v1'
	)
	assert asymmetric['augmentations']['policy'] == (
		'horizontal_flip_asymmetric_noise_v1'
	)

	comparison = deepcopy(asymmetric)
	comparison['paths']['output_root'] = symmetric['paths']['output_root']
	comparison['augmentations']['policy'] = symmetric['augmentations']['policy']
	assert comparison == symmetric


def test_resume_configs_differ_from_3ep_only_in_output_root_and_epochs() -> None:
	for arm in ARM_CONTRACTS:
		base = load_config(pretrain_config(arm, 3))
		assert base['train']['epochs'] == 3, arm
		for epochs in arm_epochs(arm):
			if epochs == 3:
				continue
			extension = load_config(pretrain_config(arm, epochs))
			assert extension['train']['epochs'] == epochs, (arm, epochs)

			comparison = deepcopy(extension)
			comparison['paths']['output_root'] = base['paths']['output_root']
			comparison['train']['epochs'] = base['train']['epochs']
			assert comparison == base, (arm, epochs)


def test_epoch_sweep_resume_sources_form_one_ascending_chain(
	artifact_root: Path,
) -> None:
	budgets = arm_epochs(EPOCH_SWEEP_ARM)
	assert sorted(budgets) == [3, 6, 10, 20]
	for epochs, source in EPOCH_SWEEP_RESUME_SOURCES.items():
		assert source in budgets, epochs
		assert source < epochs, epochs
		source_root = resolve_barlow_twins_training_config(
			load_config(pretrain_config(EPOCH_SWEEP_ARM, source))
		)['paths']['output_root']
		assert Path(source_root) == (
			artifact_root
			/ 'pretraining/f3/facies_benchmark_v1'
			/ SEARCH_NAMESPACE
			/ EPOCH_SWEEP_ARM
			/ f'full_{source}ep'
		), epochs
		header = pretrain_config(EPOCH_SWEEP_ARM, epochs).read_text(
			encoding='utf-8'
		)
		assert f'full_{source}ep/latest.pt' in header, epochs


def test_extractions_reference_arm_checkpoints_and_v2_manifest(
	artifact_root: Path,
) -> None:
	for arm in ARM_CONTRACTS:
		for epochs in arm_epochs(arm):
			extract = resolve_embedding_extraction_config(
				load_config(extraction_config(arm, epochs))
			)
			full = resolve_barlow_twins_training_config(
				load_config(pretrain_config(arm, epochs))
			)

			assert extract['embeddings']['checkpoint'] == (
				f'{full["paths"]["output_root"]}/latest.pt'
			), arm
			assert '/facies_benchmark_v2/' in extract['manifests']['input']
			assert Path(extract['embeddings']['output_dir']) == (
				artifact_root
				/ 'embeddings/f3/facies_benchmark_v2'
				/ SEARCH_NAMESPACE
				/ candidate_id(arm, epochs)
				/ 'overlap_x64'
			), arm


def test_extractions_match_vicreg_poc_conditions_except_source_and_output(
) -> None:
	baseline = load_config(VICREG_POC_ROOT / '03_extract_v2_embeddings.yaml')
	for arm in ARM_CONTRACTS:
		for epochs in arm_epochs(arm):
			extract = load_config(extraction_config(arm, epochs))

			comparison = deepcopy(extract)
			comparison['embeddings'] = baseline['embeddings']
			assert comparison == baseline, arm
			assert set(extract['embeddings']) == {'checkpoint', 'output_dir'}


def test_candidates_resolve_against_canonical_v3_without_collisions(
	artifact_root: Path,
) -> None:
	for arm in ARM_CONTRACTS:
		for epochs in arm_epochs(arm):
			candidate = f3_lithology_candidate_config_from_mapping(
				load_config(candidate_config(arm, epochs))
			)
			canonical = load_f3_lithology_candidate_canonical_config(candidate)
			extract = resolve_embedding_extraction_config(
				load_config(extraction_config(arm, epochs))
			)
			full = resolve_barlow_twins_training_config(
				load_config(pretrain_config(arm, epochs))
			)

			assert candidate.canonical_config == (Path.cwd() / CANONICAL_CONFIG)
			assert candidate.candidate_id == candidate_id(arm, epochs), arm
			assert candidate.candidate_id not in canonical.model_ids
			assert candidate.checkpoint == Path(
				f'{full["paths"]["output_root"]}/latest.pt'
			), arm
			assert candidate.embeddings_dir == Path(
				extract['embeddings']['output_dir']
			), arm
			assert candidate.embeddings_dir.parent.name == candidate.candidate_id
			assert candidate.runs_root == (
				artifact_root
				/ 'f3_lithology_benchmark'
				/ SEARCH_NAMESPACE
				/ 'runs'
			)
			assert candidate.summary_root == (
				artifact_root
				/ 'f3_lithology_benchmark'
				/ SEARCH_NAMESPACE
				/ 'summary'
			)


def test_candidate_namespace_is_unused_by_other_experiments() -> None:
	tokens = [SEARCH_NAMESPACE]
	tokens.extend(
		candidate_id(arm, epochs)
		for arm in ARM_CONTRACTS
		for epochs in arm_epochs(arm)
	)
	for path in sorted(Path('experiments').rglob('*.yaml')):
		if path.is_relative_to(ROOT) or path.is_relative_to(
			FROZEN_HMM_V1_COMPLETION_ROOT
		):
			continue
		text = path.read_text(encoding='utf-8')
		for token in tokens:
			assert token not in text, (path, token)


def test_readme_references_shared_runbook_and_result_reports() -> None:
	text = README.read_text(encoding='utf-8')
	for path in all_configs():
		assert path.is_file(), path
	assert CANONICAL_CONFIG.is_file()
	assert (BASELINE_ROOT / '02_full_100ep.yaml').is_file()
	assert Path(
		'tests/seis_ssl_cluster/test_barlow_twins_region_positives.py'
	).is_file()
	assert SHARED_RUNBOOK.is_file()
	assert text.count('](../LOCAL_BT_CANDIDATE_RUNBOOK.md)') == 1
	assert (ROOT / RESULT_REPORT_LINK).is_file()
	assert text.count(f']({RESULT_REPORT_LINK})') == 1
	assert (ROOT / EPOCH_REPORT_LINK).is_file()
	assert text.count(f']({EPOCH_REPORT_LINK})') == 1
