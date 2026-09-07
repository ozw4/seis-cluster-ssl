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
	'experiments/f3/facies_benchmark_v2/120_local_bt_region_positive_poc_v1'
)
BASELINE_ROOT = Path('experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1')
VICREG_POC_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1'
)
ANTICOLLAPSE_ROOT = Path(
	'experiments/f3/facies_benchmark_v2/119_local_bt_anticollapse_search_v1'
)
CANONICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
POC_NAMESPACE = 'local_bt_region_positive_poc_v1'
ARM = 'region_w221'
POSITIVE_WINDOW_TOKENS = [2, 2, 1]
EPOCH_BUDGETS = {3: 1_875, 10: 6_250}
README = ROOT / 'README.md'
SHARED_RUNBOOK = ROOT.parent / 'LOCAL_BT_CANDIDATE_RUNBOOK.md'
RESULT_REPORT_LINK = (
	'../../../../reports/f3/local_bt_region_positive_poc_v1_summary.md'
)


def candidate_id(epochs: int) -> str:
	return f'local_bt_region_w221_{epochs}ep'


def pretrain_config(epochs: int) -> Path:
	suffix = '_3ep.yaml' if epochs == 3 else '_10ep_resume.yaml'
	return ROOT / '10_pretraining' / f'{ARM}{suffix}'


def extraction_config(epochs: int) -> Path:
	return ROOT / '20_embeddings' / f'{ARM}_{epochs}ep.yaml'


def candidate_config(epochs: int) -> Path:
	return ROOT / '30_downstream' / f'{ARM}_{epochs}ep_medium_layout_001.yaml'


def all_configs() -> tuple[Path, ...]:
	paths = [ROOT / '01_gpu_feasibility_1step.yaml']
	for epochs in EPOCH_BUDGETS:
		paths.append(pretrain_config(epochs))
		paths.append(extraction_config(epochs))
		paths.append(candidate_config(epochs))
	return tuple(paths)


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	return root


def test_region_configs_resolve_with_fixed_budgets_and_window(
	artifact_root: Path,
) -> None:
	for epochs, budget in EPOCH_BUDGETS.items():
		raw = load_config(pretrain_config(epochs))
		assert 'continuation' not in raw
		assert 'max_steps' not in raw['train']

		full = resolve_barlow_twins_training_config(
			load_config(pretrain_config(epochs))
		)
		train = full['train']

		assert full['stage'] == 'barlow_twins_training'
		assert 'continuation' not in full
		assert train.get('max_steps') is None
		assert train['epochs'] == epochs
		assert train['batch_size'] == 16
		assert train['samples_per_epoch'] == 10_000
		assert (
			train['epochs'] * train['samples_per_epoch'] // train['batch_size']
			== budget
		)
		assert full['barlow_twins']['positive_window_tokens'] == (
			POSITIVE_WINDOW_TOKENS
		)
		assert full['barlow_twins']['method'] == 'local_barlow_twins_3d'
		assert full['augmentations'] == {'horizontal_flip_probability': 0.5}
		assert Path(full['paths']['output_root']) == (
			artifact_root
			/ 'pretraining/f3/facies_benchmark_v1'
			/ POC_NAMESPACE
			/ ARM
			/ f'full_{epochs}ep'
		)


def test_region_3ep_differs_from_canonical_only_in_root_epochs_window() -> None:
	region = load_config(pretrain_config(3))
	baseline = load_config(BASELINE_ROOT / '02_full_100ep.yaml')

	assert region['train']['epochs'] == 3
	assert baseline['train']['epochs'] == 100
	assert region['paths']['output_root'] != baseline['paths']['output_root']
	assert region['barlow_twins']['positive_window_tokens'] == (
		POSITIVE_WINDOW_TOKENS
	)
	assert 'positive_window_tokens' not in baseline['barlow_twins']

	comparison = deepcopy(region)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	comparison['train']['epochs'] = baseline['train']['epochs']
	del comparison['barlow_twins']['positive_window_tokens']
	assert comparison == baseline


def test_region_3ep_matches_119_legacy_control_except_root_and_window() -> None:
	region = load_config(pretrain_config(3))
	legacy = load_config(ANTICOLLAPSE_ROOT / '10_pretraining/legacy_3ep.yaml')

	comparison = deepcopy(region)
	comparison['paths']['output_root'] = legacy['paths']['output_root']
	del comparison['barlow_twins']['positive_window_tokens']
	assert comparison == legacy


def test_resume_config_differs_from_3ep_only_in_output_root_and_epochs() -> None:
	extension = load_config(pretrain_config(10))
	base = load_config(pretrain_config(3))

	assert extension['train']['epochs'] == 10
	assert base['train']['epochs'] == 3
	assert extension['paths']['output_root'] != base['paths']['output_root']

	comparison = deepcopy(extension)
	comparison['paths']['output_root'] = base['paths']['output_root']
	comparison['train']['epochs'] = base['train']['epochs']
	assert comparison == base


def test_feasibility_differs_from_baseline_only_in_namespace_and_window() -> None:
	poc = load_config(ROOT / '01_gpu_feasibility_1step.yaml')
	baseline = load_config(BASELINE_ROOT / '01_gpu_feasibility_1step.yaml')

	assert poc['paths']['output_root'] != baseline['paths']['output_root']
	assert f'/{POC_NAMESPACE}/' in poc['paths']['output_root']
	assert poc['train']['max_steps'] == 1
	assert poc['barlow_twins']['positive_window_tokens'] == (
		POSITIVE_WINDOW_TOKENS
	)

	comparison = deepcopy(poc)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	del comparison['barlow_twins']['positive_window_tokens']
	assert comparison == baseline


def test_extractions_reference_region_checkpoints_and_v2_manifest(
	artifact_root: Path,
) -> None:
	for epochs in EPOCH_BUDGETS:
		extract = resolve_embedding_extraction_config(
			load_config(extraction_config(epochs))
		)
		full = resolve_barlow_twins_training_config(
			load_config(pretrain_config(epochs))
		)

		assert extract['embeddings']['checkpoint'] == (
			f'{full["paths"]["output_root"]}/latest.pt'
		)
		assert '/facies_benchmark_v2/' in extract['manifests']['input']
		assert Path(extract['embeddings']['output_dir']) == (
			artifact_root
			/ 'embeddings/f3/facies_benchmark_v2'
			/ POC_NAMESPACE
			/ candidate_id(epochs)
			/ 'overlap_x64'
		)


def test_extractions_match_vicreg_poc_conditions_except_source_and_output(
) -> None:
	baseline = load_config(VICREG_POC_ROOT / '03_extract_v2_embeddings.yaml')
	for epochs in EPOCH_BUDGETS:
		extract = load_config(extraction_config(epochs))

		comparison = deepcopy(extract)
		comparison['embeddings'] = baseline['embeddings']
		assert comparison == baseline
		assert set(extract['embeddings']) == {'checkpoint', 'output_dir'}


def test_candidates_resolve_against_canonical_v3_without_collisions(
	artifact_root: Path,
) -> None:
	for epochs in EPOCH_BUDGETS:
		candidate = f3_lithology_candidate_config_from_mapping(
			load_config(candidate_config(epochs))
		)
		canonical = load_f3_lithology_candidate_canonical_config(candidate)
		extract = resolve_embedding_extraction_config(
			load_config(extraction_config(epochs))
		)
		full = resolve_barlow_twins_training_config(
			load_config(pretrain_config(epochs))
		)

		assert candidate.canonical_config == (Path.cwd() / CANONICAL_CONFIG)
		assert candidate.candidate_id == candidate_id(epochs)
		assert candidate.candidate_id not in canonical.model_ids
		assert candidate.checkpoint == Path(
			f'{full["paths"]["output_root"]}/latest.pt'
		)
		assert candidate.embeddings_dir == Path(
			extract['embeddings']['output_dir']
		)
		assert candidate.embeddings_dir.parent.name == candidate.candidate_id
		assert candidate.runs_root == (
			artifact_root
			/ 'f3_lithology_benchmark'
			/ POC_NAMESPACE
			/ 'runs'
		)
		assert candidate.summary_root == (
			artifact_root
			/ 'f3_lithology_benchmark'
			/ POC_NAMESPACE
			/ 'summary'
		)


def test_candidate_namespace_is_unused_by_other_experiments() -> None:
	tokens = [POC_NAMESPACE]
	tokens.extend(candidate_id(epochs) for epochs in EPOCH_BUDGETS)
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
