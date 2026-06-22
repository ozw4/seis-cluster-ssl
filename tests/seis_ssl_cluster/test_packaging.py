from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = 'seis-cluster-ssl'
PACKAGE_VERSION = '0.1.0'
INSTALL_EXTRAS = '.[dev,cluster,visualization]'
REPRESENTATIVE_PROC_DRY_RUNS = (
	PROJECT_ROOT / 'proc' / 'seis_ssl_cluster' / 'train_amp_mae.py',
	PROJECT_ROOT / 'proc' / 'seis_ssl_cluster' / 'cluster_embeddings.py',
	PROJECT_ROOT / 'proc' / 'seis_ssl_cluster' / 'visualize_clusters.py',
)
METADATA_SCRIPT = f"""
from importlib.metadata import requires, version

print(version({PACKAGE_NAME!r}))
for requirement in sorted(requires({PACKAGE_NAME!r}) or ()):
    print(requirement)
"""


def test_editable_install_with_extras_imports_outside_source_tree(
	tmp_path: Path,
) -> None:
	venv_dir = tmp_path / 'venv'
	outside_dir = tmp_path / 'outside'
	outside_dir.mkdir()
	venv.EnvBuilder(with_pip=False, system_site_packages=True).create(venv_dir)
	python = _venv_python(venv_dir)
	env = _subprocess_env(tmp_path)

	_run(
		[python, '-m', 'pip', 'install', '-e', INSTALL_EXTRAS],
		cwd=PROJECT_ROOT,
		env=env,
	)

	import_result = _run(
		[
			python,
			'-I',
			'-c',
			(
				'import seis_ssl_cluster; '
				'print(seis_ssl_cluster.__file__); '
				'print(seis_ssl_cluster.__version__)'
			),
		],
		cwd=outside_dir,
		env=env,
	)
	import_lines = import_result.stdout.strip().splitlines()
	assert import_lines[0].endswith('__init__.py')
	assert import_lines[1] == PACKAGE_VERSION

	metadata_result = _run(
		[python, '-I', '-c', METADATA_SCRIPT],
		cwd=outside_dir,
		env=env,
	)
	metadata_lines = metadata_result.stdout.strip().splitlines()
	assert metadata_lines[0] == PACKAGE_VERSION
	requirements = metadata_lines[1:]
	for package in ('numpy', 'PyYAML', 'torch'):
		_assert_declares_requirement(requirements, package)
	for package in ('pytest', 'ruff'):
		_assert_declares_requirement(requirements, package, extra='dev')
	for package in ('scikit-learn', 'joblib'):
		_assert_declares_requirement(requirements, package, extra='cluster')
	_assert_declares_requirement(requirements, 'matplotlib', extra='visualization')

	for script_path in REPRESENTATIVE_PROC_DRY_RUNS:
		result = _run(
			[python, '-I', script_path, '--dry-run'],
			cwd=outside_dir,
			env=env,
		)
		assert 'execution: dry-run' in result.stdout


def _venv_python(venv_dir: Path) -> Path:
	if os.name == 'nt':
		return venv_dir / 'Scripts' / 'python.exe'
	return venv_dir / 'bin' / 'python'


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
	env = dict(os.environ)
	env.pop('PYTHONHOME', None)
	env.pop('PYTHONPATH', None)
	env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
	env['PIP_NO_INPUT'] = '1'
	env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
	return env


def _run(
	command: list[object],
	*,
	cwd: Path,
	env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
	result = subprocess.run(  # noqa: S603
		[str(part) for part in command],
		check=False,
		capture_output=True,
		text=True,
		cwd=cwd,
		env=env,
	)
	assert result.returncode == 0, (
		f'command failed: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}'
	)
	return result


def _assert_declares_requirement(
	requirements: list[str],
	package: str,
	*,
	extra: str | None = None,
) -> None:
	matches = [
		requirement
		for requirement in requirements
		if requirement.lower().startswith(package.lower())
	]
	assert matches, f'missing requirement for {package}'
	if extra is None:
		assert any('extra ==' not in requirement for requirement in matches)
		return
	assert any(
		f'extra == "{extra}"' in requirement or f"extra == '{extra}'" in requirement
		for requirement in matches
	)
