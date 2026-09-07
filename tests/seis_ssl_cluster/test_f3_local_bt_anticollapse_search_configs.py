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
	'experiments/f3/facies_benchmark_v2/119_local_bt_anticollapse_search_v1'
)
BASELINE_ROOT = Path('experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1')
VICREG_POC_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1'
)
CANONICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
SEARCH_NAMESPACE = 'local_bt_anticollapse_search_v1'
ARMS = ('legacy', 'lambda020', 'lambda080', 'proj1536_lambda020', 'd4td02')
EPOCH_BUDGETS = {3: 1_875, 10: 6_250}
LAMBDA_ARMS = {'lambda020': 0.02, 'lambda080': 0.08}
D4_AUGMENTATIONS = {
	'policy': 'xy_d4_trace_drop_v1',
	'reflection_probability': 0.5,
	'trace_drop_probability': 0.02,
}
README = ROOT / 'README.md'
SHARED_RUNBOOK = ROOT.parent / 'LOCAL_BT_CANDIDATE_RUNBOOK.md'
RESULT_REPORT_LINK = (
	'../../../../reports/f3/local_bt_anticollapse_search_v1_summary.md'
)


def candidate_id(arm: str, epochs: int) -> str:
	return f'local_bt_ac_{arm}_{epochs}ep'


def pretrain_config(arm: str, epochs: int) -> Path:
	suffix = '_3ep.yaml' if epochs == 3 else '_10ep_resume.yaml'
	return ROOT / '10_pretraining' / f'{arm}{suffix}'


def extraction_config(arm: str, epochs: int) -> Path:
	return ROOT / '20_embeddings' / f'{arm}_{epochs}ep.yaml'


def candidate_config(arm: str, epochs: int) -> Path:
	return ROOT / '30_downstream' / f'{arm}_{epochs}ep_medium_layout_001.yaml'


def all_configs() -> tuple[Path, ...]:
	paths = [ROOT / '01_gpu_feasibility_1step.yaml']
	for arm in ARMS:
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


def test_all_arm_configs_resolve_with_fixed_budgets(artifact_root: Path) -> None:
	for arm in ARMS:
		for epochs, budget in EPOCH_BUDGETS.items():
			raw = load_config(pretrain_config(arm, epochs))
			assert 'continuation' not in raw, arm
			assert 'max_steps' not in raw['train'], arm

			full = resolve_barlow_twins_training_config(
				load_config(pretrain_config(arm, epochs))
			)
			train = full['train']

			assert full['stage'] == 'barlow_twins_training', arm
			assert 'continuation' not in full, arm
			assert train.get('max_steps') is None, arm
			assert train['epochs'] == epochs, arm
			assert train['batch_size'] == 16, arm
			assert train['samples_per_epoch'] == 10_000, arm
			assert (
				train['epochs'] * train['samples_per_epoch'] // train['batch_size']
				== budget
			), arm
			assert Path(full['paths']['output_root']) == (
				artifact_root
				/ 'pretraining/f3/facies_benchmark_v1'
				/ SEARCH_NAMESPACE
				/ arm
				/ f'full_{epochs}ep'
			), arm


def test_legacy_3ep_differs_from_canonical_only_in_output_root_and_epochs(
) -> None:
	arm = load_config(pretrain_config('legacy', 3))
	baseline = load_config(BASELINE_ROOT / '02_full_100ep.yaml')

	assert arm['train']['epochs'] == 3
	assert baseline['train']['epochs'] == 100
	assert arm['paths']['output_root'] != baseline['paths']['output_root']

	comparison = deepcopy(arm)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	comparison['train']['epochs'] = baseline['train']['epochs']
	assert comparison == baseline


def test_lambda_arms_differ_from_legacy_only_in_output_root_and_weight() -> None:
	legacy = load_config(pretrain_config('legacy', 3))
	for arm, weight in LAMBDA_ARMS.items():
		config = load_config(pretrain_config(arm, 3))
		assert config['barlow_twins']['redundancy_weight'] == weight, arm
		assert legacy['barlow_twins']['redundancy_weight'] == 0.005

		comparison = deepcopy(config)
		comparison['paths']['output_root'] = legacy['paths']['output_root']
		comparison['barlow_twins']['redundancy_weight'] = (
			legacy['barlow_twins']['redundancy_weight']
		)
		assert comparison == legacy, arm


def test_proj_arm_differs_from_lambda020_only_in_output_root_and_projector(
) -> None:
	proj = load_config(pretrain_config('proj1536_lambda020', 3))
	lambda020 = load_config(pretrain_config('lambda020', 3))

	assert proj['barlow_twins']['projector_dim'] == 1_536
	assert lambda020['barlow_twins']['projector_dim'] == 384
	assert proj['barlow_twins']['redundancy_weight'] == 0.02

	comparison = deepcopy(proj)
	comparison['paths']['output_root'] = lambda020['paths']['output_root']
	comparison['barlow_twins']['projector_dim'] = (
		lambda020['barlow_twins']['projector_dim']
	)
	assert comparison == lambda020


def test_d4_arm_differs_from_legacy_only_in_output_root_and_augmentations(
) -> None:
	d4 = load_config(pretrain_config('d4td02', 3))
	legacy = load_config(pretrain_config('legacy', 3))

	assert d4['augmentations'] == D4_AUGMENTATIONS
	assert legacy['augmentations'] == {'horizontal_flip_probability': 0.5}

	comparison = deepcopy(d4)
	comparison['paths']['output_root'] = legacy['paths']['output_root']
	comparison['augmentations'] = legacy['augmentations']
	assert comparison == legacy


def test_resume_configs_differ_from_3ep_only_in_output_root_and_epochs() -> None:
	for arm in ARMS:
		extension = load_config(pretrain_config(arm, 10))
		base = load_config(pretrain_config(arm, 3))

		assert extension['train']['epochs'] == 10, arm
		assert base['train']['epochs'] == 3, arm
		assert extension['paths']['output_root'] != base['paths']['output_root']

		comparison = deepcopy(extension)
		comparison['paths']['output_root'] = base['paths']['output_root']
		comparison['train']['epochs'] = base['train']['epochs']
		assert comparison == base, arm


def test_feasibility_differs_from_baseline_only_in_output_namespace() -> None:
	poc = load_config(ROOT / '01_gpu_feasibility_1step.yaml')
	baseline = load_config(BASELINE_ROOT / '01_gpu_feasibility_1step.yaml')

	assert poc['paths']['output_root'] != baseline['paths']['output_root']
	assert f'/{SEARCH_NAMESPACE}/' in poc['paths']['output_root']
	assert poc['train']['max_steps'] == 1

	comparison = deepcopy(poc)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	assert comparison == baseline


def test_extractions_reference_arm_checkpoints_and_v2_manifest(
	artifact_root: Path,
) -> None:
	for arm in ARMS:
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
	for arm in ARMS:
		for epochs in EPOCH_BUDGETS:
			extract = load_config(extraction_config(arm, epochs))

			comparison = deepcopy(extract)
			comparison['embeddings'] = baseline['embeddings']
			assert comparison == baseline, arm
			assert set(extract['embeddings']) == {'checkpoint', 'output_dir'}


def test_candidates_resolve_against_canonical_v3_without_collisions(
	artifact_root: Path,
) -> None:
	for arm in ARMS:
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
		candidate_id(arm, epochs) for arm in ARMS for epochs in EPOCH_BUDGETS
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
	assert SHARED_RUNBOOK.is_file()
	assert text.count('](../LOCAL_BT_CANDIDATE_RUNBOOK.md)') == 1
	assert (ROOT / RESULT_REPORT_LINK).is_file()
	assert text.count(f']({RESULT_REPORT_LINK})') == 1
