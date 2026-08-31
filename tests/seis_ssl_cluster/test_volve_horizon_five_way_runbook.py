from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_five_way_config import FIVE_WAY_MODEL_IDS
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
	REPOSITORY_ROOT
	/ 'experiments/volve/horizon_benchmark_v1'
	/ '31_mae_local_bt_hmm_five_way_v1'
)
MAIN_CONFIG = EXPERIMENT_ROOT / '50_five_way.yaml'
LAUNCHER = EXPERIMENT_ROOT / 'run_five_way.sh'
README = EXPERIMENT_ROOT / 'README.md'
LEGACY_CONFIG = (
	REPOSITORY_ROOT
	/ 'experiments/volve/horizon_benchmark_v1'
	/ '30_mae_vs_random_frozen_v1/03_horizon_frozen.yaml'
)
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/prepare_volve_canonical_inputs.py',
	'proc/seis_ssl_cluster/train_amp_barlow_twins.py',
	'proc/seis_ssl_cluster/train_amp_mae.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/cluster_embeddings.py',
	'proc/seis_ssl_cluster/train_strat_hmm_pretext.py',
	'proc/seis_ssl_cluster/audit_volve_horizon_five_way_sources.py',
	'proc/seis_ssl_cluster/run_volve_horizon_five_way.py',
	'proc/seis_ssl_cluster/run_volve_horizon_five_way_suite.py',
	'proc/seis_ssl_cluster/summarize_volve_horizon_five_way.py',
)


@pytest.fixture
def configured_environment(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_VOLVE_ROOT', str(tmp_path / 'volve'))
	return artifact_root


def test_main_config_preserves_the_legacy_downstream_contract(
	configured_environment: Path,
) -> None:
	del configured_environment
	five_way = load_config(MAIN_CONFIG)
	legacy = load_config(LEGACY_CONFIG)

	assert tuple(five_way['models']) == FIVE_WAY_MODEL_IDS
	for section in ('dataset', 'inputs', 'decoder', 'tiles', 'train'):
		assert five_way[section] == legacy[section]
	assert five_way['outputs']['runs_root'] != legacy['outputs']['runs_root']
	assert five_way['outputs']['summary_root'] != legacy['outputs']['runs_root']
	assert (
		Path(five_way['outputs']['runs_root']).parent
		== Path(five_way['outputs']['summary_root']).parent
	)


def test_launcher_uses_python_planner_for_exactly_75_jobs_without_artifacts(
	tmp_path: Path,
) -> None:
	environment = os.environ.copy()
	environment['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'] = str(tmp_path / 'artifacts')
	environment['SEIS_SSL_CLUSTER_VOLVE_ROOT'] = str(tmp_path / 'volve')
	environment['DRY_RUN'] = '1'
	completed = subprocess.run(  # noqa: S603
		['/usr/bin/bash', str(LAUNCHER)],
		cwd=tmp_path,
		env=environment,
		check=True,
		capture_output=True,
		text=True,
	)
	lines = [
		line
		for line in completed.stdout.splitlines()
		if line.startswith('model=')
	]
	assert len(lines) == 75
	actual = [
		(
			_option(line, 'model'),
			_option(line, 'layout'),
			_option(line, 'size'),
		)
		for line in lines
	]
	expected = [
		(model, layout, size)
		for model in FIVE_WAY_MODEL_IDS
		for layout in LAYOUT_IDS
		for size in DATA_SIZE_PREFIX
	]
	assert actual == expected
	assert len(set(actual)) == 75
	assert all(_option(line, 'action') == 'fresh' for line in lines)
	assert 'no artifact preflight or files written' in completed.stdout
	assert 'summarize_volve_horizon_five_way.py' not in LAUNCHER.read_text(
		encoding='utf-8'
	)


def test_launcher_and_readme_shell_are_valid_bash(tmp_path: Path) -> None:
	subprocess.run(  # noqa: S603
		['/usr/bin/bash', '-n', str(LAUNCHER)],
		check=True,
		capture_output=True,
		text=True,
	)
	blocks = re.findall(
		r'```bash\n(.*?)```', README.read_text(encoding='utf-8'), flags=re.DOTALL
	)
	assert blocks
	for index, block in enumerate(blocks):
		script = tmp_path / f'block_{index}.sh'
		script.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['/usr/bin/bash', '-n', str(script)],
			check=True,
			capture_output=True,
			text=True,
		)


def test_runbook_documents_complete_execution_and_recovery_contract() -> None:
	text = README.read_text(encoding='utf-8')
	for model_id in FIVE_WAY_MODEL_IDS:
		assert f'- `{model_id}`' in text
	for cli in RUNBOOK_CLIS:
		assert cli in text
		assert (REPOSITORY_ROOT / cli).is_file()
	for reference in re.findall(r'\$EXP/([^"\s]+\.(?:yaml|sh))', text):
		assert (EXPERIMENT_ROOT / reference).is_file(), reference

	assert '**75 jobs**' in text
	assert '--max-steps 1' in text
	assert '--resume "$RUN_DIR/latest.pt"' in text
	assert 'cellの完了判定fileは`metrics.json`' in text
	assert 'DRY_RUN=1 bash "$EXP/run_five_way.sh"' in text
	assert 'bash "$EXP/run_five_way.sh" --continue' in text
	assert '--check-only' in text
	assert 'complete_jobs: 75' in text
	assert 'comparison.csv' in text
	assert 'Definition of Done' in text


def test_new_experiment_inventory_excludes_forbidden_augmentation_token() -> None:
	forbidden = 'trace' + '_' + 'drop'
	for path in EXPERIMENT_ROOT.rglob('*'):
		if path.is_file():
			assert forbidden not in path.read_text(encoding='utf-8'), path


def _option(line: str, name: str) -> str:
	prefix = f'{name}='
	return next(
		part.removeprefix(prefix)
		for part in line.split()
		if part.startswith(prefix)
	)
