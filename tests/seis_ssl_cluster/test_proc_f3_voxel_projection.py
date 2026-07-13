from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from tests.seis_ssl_cluster.test_config_f3_lithology_voxel_projection import (
	projection_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
	REPO_ROOT
	/ 'proc'
	/ 'seis_ssl_cluster'
	/ 'project_f3_lithology_tokens_to_voxels.py'
)


def test_cli_dry_run_does_not_write_and_prints_source_summary(
	tmp_path: Path,
) -> None:
	config = projection_config(tmp_path)
	config_path = _write_config(tmp_path, config)
	output_dir = Path(config['voxel_projection']['output_dir'])  # type: ignore[index]

	completed = _run_cli(config_path, '--dry-run')

	assert completed.returncode == 0, completed.stderr
	assert not output_dir.exists()
	assert 'stage: project_f3_lithology_tokens_to_voxels' in completed.stdout
	assert 'source.token_grid_shape_xyz: (2, 1, 1)' in completed.stdout
	assert 'source.patch_size_xyz: (2, 2, 2)' in completed.stdout
	assert 'source.volume_shape_xyz: (3, 2, 2)' in completed.stdout
	assert 'voxel_projection.mode: nearest' in completed.stdout
	assert 'execution: dry-run' in completed.stdout


def test_cli_synthetic_end_to_end_writes_outputs_and_counts(tmp_path: Path) -> None:
	config = projection_config(tmp_path)
	config['voxel_projection']['write_probabilities'] = True  # type: ignore[index]
	config_path = _write_config(tmp_path, config)
	output_dir = Path(config['voxel_projection']['output_dir'])  # type: ignore[index]

	completed = _run_cli(config_path)

	assert completed.returncode == 0, completed.stderr
	assert np.load(output_dir / 'f3_voxel_predictions.npy').shape == (3, 2, 2)
	assert (output_dir / 'f3_voxel_probabilities.npy').is_file()
	assert (output_dir / 'prediction_metadata.json').is_file()
	assert 'valid_voxel_count: 12' in completed.stdout
	assert 'invalid_voxel_count: 0' in completed.stdout
	assert 'execution: complete' in completed.stdout


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
	path = tmp_path / 'projection.yaml'
	path.write_text(yaml.safe_dump(config), encoding='utf-8')
	return path


def _run_cli(config_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', ''))
	)
	return subprocess.run(  # noqa: S603
		[sys.executable, str(SCRIPT), '--config', str(config_path), *extra_args],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=False,
		timeout=30,
	)
