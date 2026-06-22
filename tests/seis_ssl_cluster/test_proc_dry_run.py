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
DRY_RUN_FORBIDDEN_KEYS = {
	Path('proc/seis_ssl_cluster/build_nopims_manifests.py'): (
		'data.input_channels:',
		'embedding.window_size:',
		'clustering.k_values:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'): (
		'data.input_channels:',
		'manifest.input_path_list:',
		'embedding.window_size:',
		'clustering.k_values:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py'): (
		'data.input_channels:',
		'normalization.max_samples:',
		'embedding.window_size:',
		'clustering.k_values:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/train_amp_mae.py'): (
		'data.input_channels:',
		'embedding.window_size:',
		'clustering.k_values:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/extract_embeddings.py'): (
		'stage:',
		'paths.artifact_root:',
		'data.input_channels:',
		'zero_mask.enabled:',
		'model.encoder_depth:',
		'masking.spatial_mask_ratio:',
		'loss.gradient_weight:',
		'train.lr:',
		'clustering.k_values:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/cluster_embeddings.py'): (
		'stage:',
		'data.input_channels:',
		'zero_mask.enabled:',
		'model.encoder_depth:',
		'masking.spatial_mask_ratio:',
		'loss.gradient_weight:',
		'train.lr:',
		'embedding.window_size:',
		'visualization.modes:',
	),
	Path('proc/seis_ssl_cluster/visualize_clusters.py'): (
		'stage:',
		'data.input_channels:',
		'zero_mask.enabled:',
		'model.encoder_depth:',
		'masking.spatial_mask_ratio:',
		'loss.gradient_weight:',
		'train.lr:',
		'embedding.window_size:',
		'clustering.k_values:',
	),
}


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
	for key in DRY_RUN_FORBIDDEN_KEYS[script_path]:
		assert key not in result.stdout
	if script_path == Path('proc/seis_ssl_cluster/extract_embeddings.py'):
		assert 'stage:' not in result.stdout
		assert 'paths.artifact_root:' not in result.stdout
	elif script_path in {
		Path('proc/seis_ssl_cluster/cluster_embeddings.py'),
		Path('proc/seis_ssl_cluster/visualize_clusters.py'),
	}:
		assert 'stage:' not in result.stdout
		assert 'paths.artifact_root:' in result.stdout
	else:
		assert 'stage:' in result.stdout
		assert 'paths.artifact_root:' in result.stdout
	if script_path == Path('proc/seis_ssl_cluster/build_nopims_manifests.py'):
		assert 'manifest.input_path_list:' in result.stdout
		assert 'model.encoder_depth:' not in result.stdout
		assert 'manifest scan: skipped' in result.stdout
	elif script_path == Path(
		'proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py',
	):
		assert 'normalization.max_samples:' in result.stdout
		assert 'normalization_stats.compute: skipped' in result.stdout
	elif script_path == Path(
		'proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py',
	):
		assert 'qc.output_json:' in result.stdout
		assert 'normalization_qc.compute: skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/train_amp_mae.py'):
		assert 'model.encoder_depth:' in result.stdout
		assert 'loss.gradient_weight:' in result.stdout
		assert 'execution: dry-run; training skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/extract_embeddings.py'):
		assert 'manifests.input:' in result.stdout
		assert 'embeddings.checkpoint:' in result.stdout
		assert 'embeddings.output_dir:' in result.stdout
		assert 'embedding.window_size:' in result.stdout
		assert 'embedding.overlap:' in result.stdout
		assert 'embedding.output_dtype:' in result.stdout
		assert 'embedding.batch_size:' in result.stdout
		assert 'embedding.min_token_valid_fraction:' in result.stdout
		assert 'loss.gradient_weight:' not in result.stdout
		assert 'masking.spatial_mask_ratio:' not in result.stdout
		assert 'execution: dry-run; extraction skipped' in result.stdout
	elif script_path == Path('proc/seis_ssl_cluster/cluster_embeddings.py'):
		_assert_cluster_dry_run_summary(result.stdout)
		assert 'execution: dry-run; clustering skipped' in result.stdout
	else:
		_assert_cluster_visualization_dry_run_summary(result.stdout)
		assert 'execution: dry-run; visualization skipped' in result.stdout


def _assert_cluster_dry_run_summary(stdout: str) -> None:
	for key in (
		'embeddings.input_dir:',
		'clustering.output_dir:',
		'clustering.embedding_normalization:',
		'clustering.pca.enabled:',
		'clustering.pca.n_components:',
		'clustering.pca.whiten:',
		'clustering.sample_tokens:',
		'clustering.method:',
		'clustering.k_values:',
		'clustering.minibatch_size:',
		'clustering.seed:',
	):
		assert key in stdout
	assert 'model.encoder_depth:' not in stdout
	assert 'loss.gradient_weight:' not in stdout


def _assert_cluster_visualization_dry_run_summary(stdout: str) -> None:
	for key in (
		'clustering.input_dir:',
		'visualization.output_dir:',
		'visualization.survey_ids:',
		'visualization.modes:',
		'visualization.slice_coordinate_space:',
		'visualization.xy_slices:',
		'visualization.xz_slices:',
		'visualization.reconstruct_voxel:',
		'visualization.allow_all_surveys_for_voxel_reconstruction:',
		'visualization.skip_existing_voxel_labels:',
		'visualization.max_voxel_output_gib:',
		'visualization.allow_large_voxel_output:',
	):
		assert key in stdout
	assert 'model.encoder_depth:' not in stdout
	assert 'loss.gradient_weight:' not in stdout


def test_extract_embeddings_dry_run_prints_device_override() -> None:
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/extract_embeddings.py'),
		'--dry-run',
		'--device',
		'cpu',
	)

	assert result.returncode == 0, result.stderr
	assert 'device_override: cpu' in result.stdout


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


def test_train_amp_mae_cli_overrides_are_resolved_before_dry_run(
	tmp_path: Path,
) -> None:
	output_root = tmp_path / 'override-run'

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--dry-run',
		'--device',
		'cpu',
		'--max-steps',
		'1',
		'--output-root',
		output_root,
	)

	assert result.returncode == 0, result.stderr
	assert f'paths.output_root: {output_root}' in result.stdout
	assert 'train.device: cpu' in result.stdout


def test_train_amp_mae_cli_overrides_are_validated_after_apply() -> None:
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--dry-run',
		'--max-steps',
		'-1',
	)

	assert result.returncode != 0
	assert 'train.max_steps' in result.stderr
	assert 'stage:' not in result.stdout


def test_proc_script_rejects_legacy_attribute_config(tmp_path: Path) -> None:
	config_path = tmp_path / 'legacy.yaml'
	config_path.write_text(
		"""
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
manifests:
  train: /external/artifacts/registry/manifests/train.json
attributes:
  names: [amplitude_norm]
data:
  local_crop_size: [128, 128, 128]
model:
  patch_size: [8, 8, 8]
masking:
  spatial_mask_ratio: 0.75
  block_size_tokens: [2, 2, 2]
loss: {}
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
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
  output_root: /external/artifacts/runs/nondivisible
manifests:
  train: /external/artifacts/registry/manifests/train.json
  train_path_list: /external/artifacts/registry/splits/train_npy_paths.txt
data:
  local_crop_size: [128, 128, 128]
model:
  patch_size: [8, 7, 8]
  encoder_dim: 384
  encoder_depth: 8
  encoder_heads: 6
  decoder_dim: 256
  decoder_depth: 4
  decoder_heads: 4
masking:
  spatial_mask_ratio: 0.75
  block_size_tokens: [2, 2, 2]
loss: {}
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
paths:
  nopims_root: /external/NOPIMS
  artifact_root: /external/artifacts
manifest:
  input_path_list: /external/NOPIMS/inputs/train_npy_paths.txt
  output_dir: /external/artifacts/registry/manifests
  output_name: ../nopims_amplitude_manifests.json
  normalization_stats_dir: /external/artifacts/registry/normalization_stats
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
