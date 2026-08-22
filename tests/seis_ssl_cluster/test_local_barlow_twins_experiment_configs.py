from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
)

SURVEYS = ('f3', 'parihaka')
LOCAL_ROOTS = {
	'f3': Path('experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1'),
	'parihaka': Path(
		'experiments/parihaka/facies_benchmark_v1/'
		'21_ssl_hmm_continuation_v1/10_stage1/local_barlow_twins_v1'
	),
}
EMBEDDING_CONFIGS = {
	'f3': LOCAL_ROOTS['f3'] / '03_extract_embeddings.yaml',
	'parihaka': Path(
		'experiments/parihaka/facies_benchmark_v1/'
		'21_ssl_hmm_continuation_v1/20_hmm_targets/local_bt100/'
		'01_extract_embeddings.yaml'
	),
}
STANDARD_ROOTS = {
	survey: Path(
		f'experiments/{survey}/facies_benchmark_v1/'
		'21_ssl_hmm_continuation_v1/10_stage1/barlow_twins'
	)
	for survey in SURVEYS
}
TRAINING_CONFIGS = ('01_gpu_feasibility_1step.yaml', '02_full_100ep.yaml')
METADATA_FILENAMES = {
	'f3': 'f3_facies_benchmark.embedding_metadata.json',
}


@pytest.fixture(autouse=True)
def _artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(tmp_path / 'artifacts'),
	)


@pytest.mark.parametrize('survey', SURVEYS)
@pytest.mark.parametrize('filename', TRAINING_CONFIGS)
def test_local_training_configs_resolve_and_only_change_local_identity(
	survey: str,
	filename: str,
) -> None:
	local = load_config(LOCAL_ROOTS[survey] / filename)
	standard = load_config(STANDARD_ROOTS[survey] / filename)
	resolved = resolve_barlow_twins_training_config(local)

	assert resolved['barlow_twins']['method'] == 'local_barlow_twins_3d'
	assert resolved['barlow_twins']['local_pairs_per_crop'] == 128

	comparison = deepcopy(local)
	comparison['paths']['output_root'] = standard['paths']['output_root']
	comparison['barlow_twins'].pop('method')
	comparison['barlow_twins'].pop('local_pairs_per_crop')
	assert comparison == standard


@pytest.mark.parametrize('survey', SURVEYS)
def test_local_training_budgets_match_standard(survey: str) -> None:
	feasibility = resolve_barlow_twins_training_config(
		load_config(LOCAL_ROOTS[survey] / TRAINING_CONFIGS[0])
	)
	full = resolve_barlow_twins_training_config(
		load_config(LOCAL_ROOTS[survey] / TRAINING_CONFIGS[1])
	)
	standard_full = resolve_barlow_twins_training_config(
		load_config(STANDARD_ROOTS[survey] / TRAINING_CONFIGS[1])
	)

	assert feasibility['train']['epochs'] == 1
	assert feasibility['train']['max_steps'] == 1
	assert full['train']['epochs'] == standard_full['train']['epochs'] == 100
	assert (
		full['train']['samples_per_epoch']
		== standard_full['train']['samples_per_epoch']
		== 10_000
	)
	assert (
		full['train']['samples_per_epoch']
		* full['train']['epochs']
		// full['train']['batch_size']
		== 62_500
	)


def test_local_output_roots_do_not_collide() -> None:
	training_outputs: set[str] = set()
	extraction_outputs: set[str] = set()
	for survey in SURVEYS:
		full = resolve_barlow_twins_training_config(
			load_config(LOCAL_ROOTS[survey] / TRAINING_CONFIGS[1])
		)
		extraction = resolve_embedding_extraction_config(
			load_config(EMBEDDING_CONFIGS[survey])
		)
		checkpoint = extraction['embeddings']['checkpoint']
		assert checkpoint == f'{full["paths"]["output_root"]}/latest.pt'
		training_outputs.add(full['paths']['output_root'])
		extraction_outputs.add(extraction['embeddings']['output_dir'])

	assert len(training_outputs) == len(SURVEYS)
	assert len(extraction_outputs) == len(SURVEYS)


def test_f3_local_runbook_references_existing_commands_and_configs() -> None:
	survey = 'f3'
	readme = (LOCAL_ROOTS[survey] / 'README.md').read_text(encoding='utf-8')
	for script in (
		'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
		'proc/seis_ssl_cluster/extract_embeddings.py',
	):
		assert Path(script).is_file()
		assert script in readme
	for filename in (*TRAINING_CONFIGS, '03_extract_embeddings.yaml'):
		assert (LOCAL_ROOTS[survey] / filename).is_file()
		assert filename in readme
	assert METADATA_FILENAMES[survey] in readme


def test_local_feasibility_cli_dry_run_creates_no_artifacts(tmp_path: Path) -> None:
	artifact_root = tmp_path / 'dry-run-artifacts'
	environment = {**os.environ, 'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(artifact_root)}
	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
			'--config',
			str(LOCAL_ROOTS['f3'] / TRAINING_CONFIGS[0]),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
		env=environment,
	)

	assert 'barlow_twins.method: local_barlow_twins_3d' in result.stdout
	assert 'barlow_twins.local_pairs_per_crop: 128' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout
	assert not artifact_root.exists()
