"""Non-destructive command planning, restart, locking and failure behavior."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.hmm import multi_source_driver as driver
from seis_ssl_cluster.hmm.multi_source_receipts import CANDIDATE_IDS

WORKSPACE = Path(__file__).resolve().parents[2]
ROOTS = tuple(sorted(WORKSPACE.glob('experiments/*/*/*hmm_v2_k6810_multi_source_v1')))


def _args(**values: object) -> argparse.Namespace:
	return argparse.Namespace(
		**dict(
			{
				'first': 'targets',
				'last': 'summary',
				'candidate': None,
				'layout': None,
				'size': None,
				'resume': False,
			},
			**values,
		)
	)


@pytest.mark.parametrize('root', ROOTS, ids=lambda p: p.parts[-3])
def test_exact_command_matrix_and_audit_order(root: Path) -> None:
	plan = driver.command_plan(root, _args())
	survey = yaml.safe_load((root / 'execution.yaml').read_text())['survey']
	count = 2 if survey == 'f3' else 3
	assert len(plan) == count * 23 + 1
	assert sum(stage == 'full' for stage, _ in plan) == count
	assert sum(stage == 'downstream' for stage, _ in plan) == count * 15
	assert sum(stage == 'audit' for stage, _ in plan) == count
	assert 'replay' not in driver.STAGES
	assert all('replay' not in ' '.join(command) for _, command in plan)
	assert sum('--freeze-k6-receipt' in command for _, command in plan) == count
	if survey == 'f3':
		assert all(CANDIDATE_IDS['mae'] not in ' '.join(command) for _, command in plan)
	last_audit = max(i for i, (stage, _) in enumerate(plan) if stage == 'audit')
	assert last_audit < next(
		i for i, (stage, _) in enumerate(plan) if stage == 'embeddings'
	)
	assert [
		stage for stage, _ in driver.command_plan(root, _args(first='summary'))
	] == ['audit'] * count + ['summary']


@pytest.mark.parametrize('root', ROOTS, ids=lambda p: p.parts[-3])
def test_only_new_heads_are_generated_and_historical_k6_is_frozen(root: Path) -> None:
	definition = yaml.safe_load((root / 'execution.yaml').read_text())
	plan = driver.command_plan(root, _args(first='manifests', last='manifests'))
	assert len(plan) == len(definition['arms']) * 2
	for index, arm in enumerate(definition['arms'].values()):
		targets = root / '10_targets' / arm['target_family']
		assert {p.name for p in targets.iterdir()} == {
			'01_cluster_hmm_k8k10.yaml',
			'03_export_pseudo_targets.sh',
		}
		cluster = yaml.safe_load((targets / '01_cluster_hmm_k8k10.yaml').read_text())
		assert cluster['clustering']['k_values'] == [8, 10]
		exports = (targets / '03_export_pseudo_targets.sh').read_text().splitlines()
		commands = [line for line in exports if line.startswith('python ')]
		assert len(commands) == 2
		assert '--k 8 ' in commands[0]
		assert '--k 10 ' in commands[1]
		freeze, build = plan[index * 2][1], plan[index * 2 + 1][1]
		assert '--freeze-k6-receipt' in freeze
		assert '--freeze-k6-receipt' not in build
		assert (
			freeze[freeze.index('--frozen-k6-receipt') + 1]
			== (build[build.index('--frozen-k6-receipt') + 1])
		)
		assert freeze.count('--head-root') == 1
		assert freeze[freeze.index('--head-root') + 1].startswith('6=')
		assert build.count('--head-root') == 3
		assert '--only-missing' in freeze
		assert '--only-missing' in build


def test_reuse_candidate_never_has_write_commands() -> None:
	with pytest.raises(ValueError, match='reuse'):
		driver.command_plan(
			ROOTS[0], _args(first='full', last='full', candidate=CANDIDATE_IDS['mae'])
		)


@pytest.mark.parametrize('stage', ['full', 'downstream'])
def test_resume_owns_exact_latest(
	stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts with spaces')
	)
	cid = CANDIDATE_IDS['random']
	args = _args(
		first=stage,
		last=stage,
		candidate=cid,
		resume=True,
		layout='layout_002' if stage == 'downstream' else None,
		size='small' if stage == 'downstream' else None,
	)
	plan = driver.command_plan(ROOTS[0], args)
	command = plan[-1][1]
	latest = command[command.index('--resume') + 1]
	assert latest.startswith(str(tmp_path / 'artifacts with spaces') + '/')
	assert (
		f'/{cid}/full_25ep/latest.pt' in latest
		if stage == 'full'
		else f'/model={cid}/layout=layout_002/size=small/decoder/latest.pt' in latest
	)


def test_failure_stops_after_first_command(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'driver',
			'--experiment',
			str(ROOTS[0]),
			'--execute',
			'--from',
			'audit',
			'--to',
			'audit',
		],
	)
	calls = []

	def fail(_stage: str, _command: list[str], **_kwargs: object) -> None:
		calls.append(True)
		raise subprocess.CalledProcessError(7, ['stub'])

	monkeypatch.setattr(driver, 'run_command', fail)
	with pytest.raises(subprocess.CalledProcessError):
		driver.main()
	assert calls == [True]


def test_survey_lock_blocks_second_executor(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	logs = tmp_path / 'hmm_v2/k6810_multi_source_evaluation_v1/f3/logs'
	logs.mkdir(parents=True)
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'driver',
			'--experiment',
			str(ROOTS[0]),
			'--execute',
			'--from',
			'audit',
			'--to',
			'audit',
		],
	)
	with (logs / 'driver.lock').open('a') as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		with pytest.raises(BlockingIOError):
			driver.main()
	assert list(logs.iterdir()) == [logs / 'driver.lock']


def test_dry_run_makes_no_logs_or_artifacts(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	artifact = tmp_path / 'not created'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact))
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'driver',
			'--experiment',
			str(ROOTS[0]),
			'--dry-run',
			'--from',
			'audit',
			'--to',
			'audit',
		],
	)
	commands = []
	monkeypatch.setattr(
		driver.subprocess, 'run', lambda command, **_kwargs: commands.append(command)
	)
	driver.main()
	assert len(commands) == 2
	assert all(command[-1] == '--dry-run' for command in commands)
	assert not artifact.exists()


def test_subprocess_receives_spaces_as_single_argument(tmp_path: Path) -> None:
	log = tmp_path / 'log with spaces.txt'
	driver.run_command(
		'audit',
		[sys.executable, '-c', 'import sys; print(sys.argv[1])', 'path with spaces'],
		dry_run=False,
		log=log,
	)
	assert log.read_text().splitlines()[-1] == 'path with spaces'


def test_existing_clustering_and_partial_embeddings_fail_without_write(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	output = tmp_path / 'existing'
	output.mkdir()
	sentinel = output / 'partial'
	sentinel.write_text('keep')
	config = tmp_path / 'config.yaml'
	config.write_text(
		yaml.safe_dump(
			{
				'clustering': {'output_dir': str(output)},
				'embeddings': {'output_dir': str(output)},
			}
		)
	)

	def fail(_path: Path) -> None:
		raise ValueError('incomplete embeddings')

	monkeypatch.setattr(driver, 'audit_existing_embeddings', fail)
	for stage, error in [('targets', FileExistsError), ('embeddings', ValueError)]:
		with pytest.raises(error):
			driver.run_command(
				stage, ['stub', '--config', str(config)], dry_run=False, log=None
			)
	assert sentinel.read_text() == 'keep'
