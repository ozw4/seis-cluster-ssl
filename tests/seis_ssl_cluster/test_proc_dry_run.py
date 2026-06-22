from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers import run_python_proc

PROC_SCRIPTS = (
	Path('proc/seis_ssl_cluster/build_nopims_manifests.py'),
	Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
	Path('proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py'),
	Path('proc/seis_ssl_cluster/train_amp_mae.py'),
	Path('proc/seis_ssl_cluster/extract_embeddings.py'),
	Path('proc/seis_ssl_cluster/cluster_embeddings.py'),
	Path('proc/seis_ssl_cluster/visualize_clusters.py'),
)


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_help_exits_zero(script_path: Path) -> None:
	result = run_python_proc(script_path, '--help')

	assert result.returncode == 0, result.stderr
	assert '--config' in result.stdout
	assert '--dry-run' in result.stdout


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_dry_run_exits_zero_and_prints_summary(
	script_path: Path,
) -> None:
	result = run_python_proc(script_path, '--dry-run')

	assert result.returncode == 0, result.stderr
	assert 'stage:' in result.stdout
	assert 'data.input_channels: 1' in result.stdout
	assert 'data.target_channels: 1' in result.stdout
	assert 'data.use_context: false' in result.stdout
	if script_path == Path('proc/seis_ssl_cluster/build_nopims_manifests.py'):
		assert 'manifest scan: skipped' in result.stdout
	elif script_path == Path(
		'proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py',
	):
		assert 'normalization_stats.compute: skipped' in result.stdout
	elif script_path == Path(
		'proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py',
	):
		assert 'normalization_qc.compute: skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/train_amp_mae.py'):
		assert 'execution: dry-run; training skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/extract_embeddings.py'):
		assert 'execution: dry-run; extraction skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/cluster_embeddings.py'):
		assert 'execution: dry-run; clustering skipped' in result.stdout
	else:
		assert 'execution: dry-run; visualization skipped' in result.stdout


def test_cluster_embeddings_dry_run_does_not_import_optional_cluster_stack(
	tmp_path: Path,
) -> None:
	sitecustomize = tmp_path / 'sitecustomize.py'
	sitecustomize.write_text(
		"""
import builtins

_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in {'joblib', 'sklearn'}:
        raise ModuleNotFoundError(name)
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import
""",
		encoding='utf-8',
	)
	pythonpath = str(tmp_path)
	existing_pythonpath = os.environ.get('PYTHONPATH')
	if existing_pythonpath:
		pythonpath = f'{pythonpath}{os.pathsep}{existing_pythonpath}'

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/cluster_embeddings.py'),
		'--dry-run',
		extra_env={'PYTHONPATH': pythonpath},
	)

	assert result.returncode == 0, result.stderr
	assert 'execution: dry-run; clustering skipped' in result.stdout


def test_proc_script_rejects_legacy_attribute_config(tmp_path: Path) -> None:
	config_path = tmp_path / 'legacy.yaml'
	config_path.write_text(
		"""
stage: train_amp_mae
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  input_channels: 1
  target_channels: 1
  use_context: false
  local_crop_size: [128, 128, 128]
attributes:
  names: [amplitude_norm]
model:
  name: amp_mae3d
  in_channels: 1
  out_channels: 1
  patch_size: [8, 8, 8]
masking:
  spatial_mask_ratio: 0.75
  spatial_mask_mode: block
  block_size_tokens: [2, 2, 2]
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  amp: false
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode != 0
	assert 'attributes.names' in result.stderr
	assert 'amplitude-only MVP' in result.stderr
	assert 'stage:' not in result.stdout


def test_proc_script_rejects_nondivisible_geometry(tmp_path: Path) -> None:
	config_path = tmp_path / 'nondivisible.yaml'
	config_path.write_text(
		"""
stage: train_amp_mae
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  input_channels: 1
  target_channels: 1
  use_context: false
  local_crop_size: [128, 128, 128]
model:
  name: amp_mae3d
  in_channels: 1
  out_channels: 1
  patch_size: [8, 7, 8]
masking:
  spatial_mask_ratio: 0.75
  spatial_mask_mode: block
  block_size_tokens: [2, 2, 2]
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  amp: false
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode != 0
	assert 'local_crop_size' in result.stderr
	assert 'patch_size' in result.stderr
	assert 'stage:' not in result.stdout


def test_build_nopims_manifests_rejects_non_bare_output_name(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'escaped_output_name.yaml'
	config_path.write_text(
		"""
stage: build_nopims_manifests
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
manifest:
  input_path_list: /external/NOPIMS/inputs/train_npy_paths.txt
  output_dir: /external/artifacts/registry/manifests
  output_name: ../nopims_amplitude_manifests.json
  normalization_stats_dir: /external/artifacts/registry/normalization_stats
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  input_channels: 1
  target_channels: 1
  use_context: false
  local_crop_size: [128, 128, 128]
model:
  name: amp_mae3d
  in_channels: 1
  out_channels: 1
  patch_size: [8, 8, 8]
masking:
  spatial_mask_ratio: 0.75
  spatial_mask_mode: block
  block_size_tokens: [2, 2, 2]
train:
  batch_size: 4
  samples_per_epoch: 10000
  epochs: 100
  num_workers: 8
  amp: false
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/build_nopims_manifests.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode != 0
	assert 'manifest.output_name must be a bare filename' in result.stderr
	assert 'stage:' not in result.stdout
