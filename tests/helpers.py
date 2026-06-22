from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_python_proc(
	script_path: Path,
	*args: object,
	extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
	"""Run a proc script with the repo src tree importable."""
	env = dict(os.environ)
	extra_pythonpath = None if extra_env is None else extra_env.get('PYTHONPATH')
	pythonpath_parts = []
	if extra_pythonpath:
		pythonpath_parts.append(extra_pythonpath)
	pythonpath_parts.extend([str(Path('src').resolve()), str(Path.cwd())])
	if env.get('PYTHONPATH') and env['PYTHONPATH'] != extra_pythonpath:
		pythonpath_parts.append(env['PYTHONPATH'])
	env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
	if extra_env is not None:
		env.update(
			{key: value for key, value in extra_env.items() if key != 'PYTHONPATH'},
		)
	return subprocess.run(  # noqa: S603
		[sys.executable, str(script_path), *(str(arg) for arg in args)],
		check=False,
		capture_output=True,
		text=True,
		env=env,
	)
