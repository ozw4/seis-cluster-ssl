from __future__ import annotations

import re
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_vicreg_training_config,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)

ROOT = Path('experiments/f3/facies_benchmark_v2/117_local_vicreg_10ep_poc_v1')
BASELINE_ROOT = Path('experiments/f3/facies_benchmark_v2/115_local_vicreg_v1')
CANONICAL_CONFIG = Path(
	'experiments/f3/facies_benchmark_v2/'
	'110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)
POC_CONFIGS = (
	'01_gpu_feasibility_1step.yaml',
	'02_full_10ep.yaml',
	'03_extract_v2_embeddings.yaml',
	'04_candidate_medium_layout_001.yaml',
)
POC_NAMESPACE = 'local_vicreg_10ep_poc_v1'
CANDIDATE_ID = 'local_vicreg_10ep'
README = ROOT / 'README.md'
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/train_amp_vicreg.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/run_f3_lithology_candidate.py',
)


@pytest.fixture(autouse=True)
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(root))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	return root


def test_full_10ep_resolves_with_fresh_fixed_budget(artifact_root: Path) -> None:
	raw = load_config(ROOT / '02_full_10ep.yaml')
	assert 'continuation' not in raw
	assert 'max_steps' not in raw['train']

	full = resolve_vicreg_training_config(load_config(ROOT / '02_full_10ep.yaml'))
	train = full['train']

	assert full['stage'] == 'vicreg_training'
	assert 'continuation' not in full
	assert train.get('max_steps') is None
	assert train['epochs'] == 10
	assert train['batch_size'] == 16
	assert train['samples_per_epoch'] == 10_000
	assert train['epochs'] * train['samples_per_epoch'] // train['batch_size'] == (
		6_250
	)
	assert Path(full['paths']['output_root']) == (
		artifact_root
		/ 'pretraining/f3/facies_benchmark_v1/local_vicreg_10ep_poc_v1/full_10ep'
	)


def test_full_10ep_differs_from_100ep_baseline_only_in_output_root_and_epochs(
) -> None:
	poc = load_config(ROOT / '02_full_10ep.yaml')
	baseline = load_config(BASELINE_ROOT / '02_full_100ep.yaml')

	assert poc['train']['epochs'] == 10
	assert baseline['train']['epochs'] == 100
	assert poc['paths']['output_root'] != baseline['paths']['output_root']

	comparison = deepcopy(poc)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	comparison['train']['epochs'] = baseline['train']['epochs']
	assert comparison == baseline


def test_feasibility_differs_from_baseline_only_in_output_namespace() -> None:
	poc = load_config(ROOT / '01_gpu_feasibility_1step.yaml')
	baseline = load_config(BASELINE_ROOT / '01_gpu_feasibility_1step.yaml')
	full = load_config(ROOT / '02_full_10ep.yaml')

	assert poc['paths']['output_root'] != baseline['paths']['output_root']
	assert poc['paths']['output_root'] != full['paths']['output_root']
	assert f'/{POC_NAMESPACE}/' in poc['paths']['output_root']
	assert poc['train']['max_steps'] == 1

	comparison = deepcopy(poc)
	comparison['paths']['output_root'] = baseline['paths']['output_root']
	assert comparison == baseline


def test_extraction_references_10ep_checkpoint_and_v2_manifest(
	artifact_root: Path,
) -> None:
	extract = resolve_embedding_extraction_config(
		load_config(ROOT / '03_extract_v2_embeddings.yaml')
	)
	full = resolve_vicreg_training_config(load_config(ROOT / '02_full_10ep.yaml'))

	assert extract['embeddings']['checkpoint'] == (
		f'{full["paths"]["output_root"]}/latest.pt'
	)
	assert '/facies_benchmark_v2/' in extract['manifests']['input']
	assert Path(extract['embeddings']['output_dir']) == (
		artifact_root
		/ 'embeddings/f3/facies_benchmark_v2/local_vicreg_10ep_poc_v1'
		/ 'local_vicreg_10ep/overlap_x64'
	)


def test_extraction_matches_five_way_conditions_except_source_and_output() -> None:
	poc = load_config(ROOT / '03_extract_v2_embeddings.yaml')
	baseline = load_config(BASELINE_ROOT / '03_extract_v2_embeddings.yaml')

	comparison = deepcopy(poc)
	comparison['embeddings'] = baseline['embeddings']
	assert comparison == baseline
	assert set(poc['embeddings']) == {'checkpoint', 'output_dir'}


def test_candidate_resolves_against_canonical_v3_without_collisions(
	artifact_root: Path,
) -> None:
	candidate = f3_lithology_candidate_config_from_mapping(
		load_config(ROOT / '04_candidate_medium_layout_001.yaml')
	)
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	extract = resolve_embedding_extraction_config(
		load_config(ROOT / '03_extract_v2_embeddings.yaml')
	)
	full = resolve_vicreg_training_config(load_config(ROOT / '02_full_10ep.yaml'))

	assert candidate.canonical_config == (Path.cwd() / CANONICAL_CONFIG)
	assert candidate.candidate_id == CANDIDATE_ID
	assert candidate.candidate_id not in canonical.model_ids
	assert candidate.checkpoint == Path(f'{full["paths"]["output_root"]}/latest.pt')
	assert candidate.embeddings_dir == Path(extract['embeddings']['output_dir'])
	assert candidate.runs_root == (
		artifact_root / 'f3_lithology_benchmark/local_vicreg_10ep_poc_v1/runs'
	)
	assert candidate.summary_root == (
		artifact_root / 'f3_lithology_benchmark/local_vicreg_10ep_poc_v1/summary'
	)


def test_candidate_namespace_is_unused_by_other_experiments() -> None:
	for path in sorted(Path('experiments').rglob('*.yaml')):
		if path.is_relative_to(ROOT):
			continue
		text = path.read_text(encoding='utf-8')
		assert CANDIDATE_ID not in text, path
		assert POC_NAMESPACE not in text, path


def test_readme_references_resolvable_commands_and_example_configs() -> None:
	text = README.read_text(encoding='utf-8')
	for cli in RUNBOOK_CLIS:
		assert Path(cli).is_file()
		assert cli in text
	for name in POC_CONFIGS:
		assert (ROOT / name).is_file()
	for name in POC_CONFIGS[1:]:
		assert name in text
	assert CANONICAL_CONFIG.is_file()


def test_readme_shell_blocks_are_valid(tmp_path: Path) -> None:
	text = README.read_text(encoding='utf-8')
	blocks = re.findall(r'```bash\n(.*?)```', text, flags=re.DOTALL)
	assert blocks
	for index, block in enumerate(blocks):
		path = tmp_path / f'runbook_block_{index}.sh'
		path.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(path)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)
