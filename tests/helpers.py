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
	pythonpath_parts = [str(Path('src').resolve()), str(Path.cwd())]
	if env.get('PYTHONPATH'):
		pythonpath_parts.append(env['PYTHONPATH'])
	env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
	if extra_env is not None:
		env.update(extra_env)
	return subprocess.run(  # noqa: S603
		[sys.executable, str(script_path), *(str(arg) for arg in args)],
		check=False,
		capture_output=True,
		text=True,
		env=env,
	)
