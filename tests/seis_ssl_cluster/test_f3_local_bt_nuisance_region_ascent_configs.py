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
	'experiments/f3/facies_benchmark_v2/122_local_bt_nuisance_region_ascent_v1'
)
BASELINE_ROOT = Path('experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1')
VIEW_REGION_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/121_local_bt_view_region_search_v1'
)
VICREG_POC_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1'
)
CANONICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
SEARCH_NAMESPACE = 'local_bt_nuisance_region_ascent_v1'
ARM_CONTRACTS = {
	'gauss005_r221': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.05,
		},
		[2, 2, 1],
	),
	'gauss010': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		None,
	),
	'gauss010_r221_proj1536': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 1],
	),
	'gauss010_r221_lam00125': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 1],
	),
	'gauss010_r222': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 2],
	),
	'gauss020_r221': (
		{
			'policy': 'horizontal_flip_gaussian_noise_v1',
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.2,
		},
		[2, 2, 1],
	),
	'idgauss010_r221': (
		{
			'policy': 'identity_gaussian_noise_v1',
			'gaussian_noise_std': 0.1,
		},
		[2, 2, 1],
	),
	'idgauss020_r221': (
		{
			'policy': 'identity_gaussian_noise_v1',
			'gaussian_noise_std': 0.2,
		},
		[2, 2, 1],
	),
}
EPOCH_BUDGETS = {3: 1_875, 10: 6_250}
FEASIBILITY_CONFIGS = {
	'01_gpu_feasibility_1step_identity_region.yaml': 'idgauss010_r221',
}
README = ROOT / 'README.md'
SHARED_RUNBOOK = ROOT.parent / 'LOCAL_BT_CANDIDATE_RUNBOOK.md'
RESULT_REPORT_LINK = (
	'../../../../reports/f3/local_bt_random_parity_breakthrough.md'
)


def candidate_id(arm: str, epochs: int) -> str:
	return f'local_bt_na_{arm}_{epochs}ep'


def pretrain_config(arm: str, epochs: int) -> Path:
	suffix = '_3ep.yaml' if epochs == 3 else '_10ep_resume.yaml'
	return ROOT / '10_pretraining' / f'{arm}{suffix}'


def extraction_config(arm: str, epochs: int) -> Path:
	return ROOT / '20_embeddings' / f'{arm}_{epochs}ep.yaml'


def candidate_config(arm: str, epochs: int) -> Path:
	return ROOT / '30_downstream' / f'{arm}_{epochs}ep_medium_layout_001.yaml'


def all_configs() -> tuple[Path, ...]:
	paths = [ROOT / name for name in FEASIBILITY_CONFIGS]
	for arm in ARM_CONTRACTS:
		for epochs in EPOCH_BUDGETS:
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
		for epochs, budget in EPOCH_BUDGETS.items():
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
			if window is None:
				assert 'positive_window_tokens' not in barlow_twins, arm
			else:
				assert barlow_twins['positive_window_tokens'] == window, arm
			assert barlow_twins['local_pairs_per_crop'] == 128, arm
			assert barlow_twins['redundancy_weight'] == (
				0.00125 if arm == 'gauss010_r221_lam00125' else 0.005
			), arm
			assert barlow_twins['projector_dim'] == (
				1_536 if arm == 'gauss010_r221_proj1536' else 384
			), arm
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
		if window is not None:
			assert comparison['barlow_twins']['positive_window_tokens'] == window
			del comparison['barlow_twins']['positive_window_tokens']
		comparison['barlow_twins']['redundancy_weight'] = (
			baseline['barlow_twins']['redundancy_weight']
		)
		comparison['barlow_twins']['projector_dim'] = (
			baseline['barlow_twins']['projector_dim']
		)
		assert comparison == baseline, arm


def test_gauss010_control_matches_121_winner_except_root_and_window() -> None:
	control = load_config(pretrain_config('gauss010', 3))
	winner = load_config(
		VIEW_REGION_ROOT / '10_pretraining/gauss010_r221_3ep.yaml'
	)

	assert winner['barlow_twins']['positive_window_tokens'] == [2, 2, 1]
	assert 'positive_window_tokens' not in control['barlow_twins']

	comparison = deepcopy(winner)
	comparison['paths']['output_root'] = control['paths']['output_root']
	del comparison['barlow_twins']['positive_window_tokens']
	assert comparison == control


def test_identity_arms_drop_the_flip_and_keep_noise_as_only_view_difference(
) -> None:
	for arm, expected_std in (
		('idgauss010_r221', 0.1),
		('idgauss020_r221', 0.2),
	):
		config = load_config(pretrain_config(arm, 3))
		augmentations = config['augmentations']
		assert set(augmentations) == {'policy', 'gaussian_noise_std'}, arm
		assert augmentations['policy'] == 'identity_gaussian_noise_v1', arm
		assert augmentations['gaussian_noise_std'] == expected_std, arm
		assert 'horizontal_flip_probability' not in augmentations, arm

	strong = load_config(pretrain_config('idgauss020_r221', 3))
	weak = load_config(pretrain_config('idgauss010_r221', 3))
	comparison = deepcopy(strong)
	comparison['paths']['output_root'] = weak['paths']['output_root']
	comparison['augmentations']['gaussian_noise_std'] = (
		weak['augmentations']['gaussian_noise_std']
	)
	assert comparison == weak


def test_resume_configs_differ_from_3ep_only_in_output_root_and_epochs() -> None:
	for arm in ARM_CONTRACTS:
		extension = load_config(pretrain_config(arm, 10))
		base = load_config(pretrain_config(arm, 3))

		assert extension['train']['epochs'] == 10, arm
		assert base['train']['epochs'] == 3, arm

		comparison = deepcopy(extension)
		comparison['paths']['output_root'] = base['paths']['output_root']
		comparison['train']['epochs'] = base['train']['epochs']
		assert comparison == base, arm


def test_feasibility_configs_mirror_their_arm_with_1step_budget() -> None:
	baseline = load_config(BASELINE_ROOT / '01_gpu_feasibility_1step.yaml')
	for name, arm in FEASIBILITY_CONFIGS.items():
		poc = load_config(ROOT / name)
		augmentations, window = ARM_CONTRACTS[arm]

		assert poc['train']['max_steps'] == 1, name
		assert f'/{SEARCH_NAMESPACE}/' in poc['paths']['output_root'], name
		assert poc['augmentations'] == augmentations, name
		assert poc['barlow_twins']['positive_window_tokens'] == window, name

		comparison = deepcopy(poc)
		comparison['paths']['output_root'] = baseline['paths']['output_root']
		comparison['augmentations'] = baseline['augmentations']
		del comparison['barlow_twins']['positive_window_tokens']
		assert comparison == baseline, name


def test_extractions_reference_arm_checkpoints_and_v2_manifest(
	artifact_root: Path,
) -> None:
	for arm in ARM_CONTRACTS:
		for epochs in EPOCH_BUDGETS:
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
		for epochs in EPOCH_BUDGETS:
			extract = load_config(extraction_config(arm, epochs))

			comparison = deepcopy(extract)
			comparison['embeddings'] = baseline['embeddings']
			assert comparison == baseline, arm
			assert set(extract['embeddings']) == {'checkpoint', 'output_dir'}


def test_candidates_resolve_against_canonical_v3_without_collisions(
	artifact_root: Path,
) -> None:
	for arm in ARM_CONTRACTS:
		for epochs in EPOCH_BUDGETS:
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
		for epochs in EPOCH_BUDGETS
	)
	for path in sorted(Path('experiments').rglob('*.yaml')):
		if path.is_relative_to(ROOT):
			continue
		text = path.read_text(encoding='utf-8')
		for token in tokens:
			assert token not in text, (path, token)


def test_readme_references_shared_runbook_and_result_report() -> None:
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
