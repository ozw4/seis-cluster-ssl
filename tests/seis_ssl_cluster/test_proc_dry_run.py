from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from proc.seis_ssl_cluster import run_parihaka_channel_decoder as channel_cli
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from tests.helpers import run_python_proc

pytestmark = pytest.mark.integration

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

_CHANNEL_MODEL_IDS = (
	'mae',
	'barlow_twins',
	'mae_hmm_k6',
	'barlow_twins_hmm_k6',
)


def _write_channel_cli_fixture(
	tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path]]:
	labels_path = tmp_path / 'labels.npy'
	labels_metadata_path = tmp_path / 'labels_metadata.json'
	labels = np.ones((8, 8, 8), dtype=np.int8)
	labels[:, :, ::2] = 5
	np.save(labels_path, labels, allow_pickle=False)
	labels_metadata_path.write_text(
		json.dumps(
			{
				'schema_version': 2,
				'artifact_type': 'parihaka_channel_labels',
				'output_labels': str(labels_path),
				'prepared_label_identity': {
					'labels_sha256': file_sha256(labels_path),
					'source_npz_path': '/data/parihaka_labels.npz',
					'source_key': 'labels',
					'shape': [8, 8, 8],
					'dtype': 'int8',
					'class_definition': {
						'positive_class_id': 5,
						'negative_class_ids': [1, 2, 3, 4, 6],
					},
				},
			}
		),
		encoding='utf-8',
	)
	common_metadata = {
		'survey_id': 'parihaka',
		'source_amplitude_path': '/data/parihaka_amplitude.npy',
		'patch_size': [8, 8, 8],
		'token_grid_shape': [1, 1, 1],
		'volume_shape_xyz': [8, 8, 8],
		'embedding_dim': 384,
		'model_geometry': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
		'window_size': [8, 8, 8],
		'overlap': [0, 0, 0],
		'output_dtype': 'float16',
		'min_token_valid_fraction': 0.5,
		'normalization_stats_path': '/data/parihaka_stats.json',
		'preprocessing': {'mode': 'same'},
		'zero_mask': {'enabled': True},
		'precision': {'device_type': 'cpu', 'autocast': False},
	}
	checkpoints: dict[str, Path] = {}
	embedding_models: dict[str, dict[str, str]] = {}
	for model_id in _CHANNEL_MODEL_IDS:
		checkpoint = tmp_path / 'checkpoints' / f'{model_id}.pt'
		checkpoint.parent.mkdir(parents=True, exist_ok=True)
		checkpoint.write_bytes(f'checkpoint:{model_id}'.encode())
		checkpoints[model_id] = checkpoint
		embedding_dir = tmp_path / 'embeddings' / model_id
		embedding_dir.mkdir(parents=True)
		paths = output_paths(embedding_dir, 'parihaka')
		np.save(
			paths.embeddings,
			np.zeros((1, 1, 1, 384), dtype=np.float16),
			allow_pickle=False,
		)
		np.save(
			paths.valid_tokens,
			np.ones((1, 1, 1), dtype=np.bool_),
			allow_pickle=False,
		)
		metadata = {
			**common_metadata,
			'checkpoint_path': str(checkpoint),
			'checkpoint_sha256': file_sha256(checkpoint),
			'pretraining_objective': {'name': model_id},
		}
		if '_hmm_' in model_id:
			metadata['stratigraphy_pretext'] = {
				'method': 'hmm',
				'cluster_count': 6,
			}
		paths.metadata.write_text(json.dumps(metadata), encoding='utf-8')
		embedding_models[model_id] = {
			'dir': str(embedding_dir),
			'checkpoint': str(checkpoint),
		}
	config_path = tmp_path / 'channel.yaml'
	config_path.write_text(
		yaml.safe_dump(
			{
				'dataset': {'survey_id': 'parihaka'},
				'inputs': {
					'labels_npy': str(labels_path),
					'labels_metadata_json': str(labels_metadata_path),
				},
				'embeddings': {'models': embedding_models},
				'outputs': {'runs_root': str(tmp_path / 'runs')},
				'decoder': {
					'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
					'embedding_dim': 384,
					'class_count': 2,
					'hidden_channels': [128, 64, 32],
					'upsample_factors': [[2, 2, 2]] * 3,
					'upsample_mode': 'nearest',
					'normalization': 'voxelwise_layer_norm',
				},
				'train': {
					'epochs': 50,
					'batch_size': 1,
					'learning_rate': 0.001,
					'weight_decay': 0.0001,
					'class_weight': 'balanced',
					'sampling_mode': 'all_tiles_once',
					'seed': 42000,
					'amp': False,
					'gradient_clip_norm': 1.0,
				},
				'tiles': {
					'core_size_tokens': [8, 8, 8],
					'context_halo_tokens': [1, 1, 1],
				},
			}
		),
		encoding='utf-8',
	)
	training_lines = (0, 1, 2, 3, 6, 7)
	layout_path = tmp_path / 'layouts.yaml'
	layout_path.write_text(
		yaml.safe_dump(
			{
				'training_selection': {
					'semantics': (
						'stable_hash_partial_section_token_footprints_v1'
					),
					'allowed_relative_error': 0.05,
					'target_train_voxel_counts': {
						'small': 104,
						'medium': 192,
						'large': 320,
					},
				},
				'validation': {'inline': [4], 'crossline': [4]},
				'layouts': {
					f'layout_{index:03d}': {
						'inline': [
							training_lines[
								(index + offset) % len(training_lines)
							]
							for offset in range(4)
						],
						'crossline': [
							training_lines[
								(index + offset) % len(training_lines)
							]
							for offset in range(4)
						],
					}
					for index in range(5)
				},
			}
		),
		encoding='utf-8',
	)
	return config_path, layout_path, checkpoints


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_help_exits_zero(script_path: Path) -> None:
	result = run_python_proc(script_path, '--help')

	assert result.returncode == 0, result.stderr
	assert '--config' in result.stdout
	assert '--dry-run' in result.stdout


@pytest.mark.parametrize(
	'model_id',
	[*_CHANNEL_MODEL_IDS, 'pretrained', 'random'],
)
def test_parihaka_channel_decoder_cli_parser_accepts_dynamic_model_ids(
	model_id: str,
) -> None:
	args = channel_cli.build_parser().parse_args(
		[
			'--model',
			model_id,
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--layout-config',
			'layouts.yaml',
		]
	)
	assert args.model == model_id


@pytest.mark.parametrize(
	('model_id', 'pretext_present'),
	[('mae', False), ('mae_hmm_k6', True)],
)
def test_parihaka_channel_decoder_cli_generic_dry_run_prints_compact_source_identity(
	tmp_path: Path,
	model_id: str,
	*,
	pretext_present: bool,
) -> None:
	config_path, layout_path, checkpoints = _write_channel_cli_fixture(tmp_path)
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'),
		'--config',
		config_path,
		'--model',
		model_id,
		'--layout',
		'layout_000',
		'--size',
		'small',
		'--layout-config',
		layout_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert f'model: {model_id}' in result.stdout
	assert 'available_models:' in result.stdout
	for available_model in _CHANNEL_MODEL_IDS:
		assert available_model in result.stdout
	assert f'checkpoint_path: {checkpoints[model_id]}' in result.stdout
	assert (
		f'checkpoint_sha256: {file_sha256(checkpoints[model_id])}'
		in result.stdout
	)
	assert f"pretraining_objective: {{'name': '{model_id}'}}" in result.stdout
	assert (
		f'stratigraphy_pretext_present: {pretext_present}' in result.stdout
	)
	decoder_sha_line = next(
		line
		for line in result.stdout.splitlines()
		if line.startswith('decoder_initial_state_sha256: ')
	)
	decoder_sha256 = decoder_sha_line.removeprefix(
		'decoder_initial_state_sha256: '
	)
	assert len(decoder_sha256) == 64
	assert set(decoder_sha256) <= set('0123456789abcdef')
	assert 'source_amplitude_path' not in result.stdout
	assert 'normalization_stats_path' not in result.stdout
	assert 'execution: dry-run; no files written' in result.stdout
	assert not (tmp_path / 'runs').exists()


def test_parihaka_channel_decoder_cli_rejects_unknown_model_with_available_models(
	tmp_path: Path,
) -> None:
	config_path, layout_path, _ = _write_channel_cli_fixture(tmp_path)
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/run_parihaka_channel_decoder.py'),
		'--config',
		config_path,
		'--model',
		'unknown',
		'--layout',
		'layout_000',
		'--size',
		'small',
		'--layout-config',
		layout_path,
		'--dry-run',
	)

	assert result.returncode != 0
	assert 'available models' in result.stderr
	for model_id in _CHANNEL_MODEL_IDS:
		assert model_id in result.stderr


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
		assert 'continuation.init_checkpoint:' not in result.stdout
		assert 'continuation.unfreeze_top_blocks:' not in result.stdout
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


def test_train_amp_mae_cli_overrides_are_resolved_before_dry_run() -> None:
	output_root = Path(
		'/workspace/artifacts/seis_ssl_cluster/pretraining/override-dry-run',
	)

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


def test_train_amp_mae_dry_run_rejects_invalid_mae_debug_config(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'invalid_mae_debug.yaml'
	config_path.write_text(
		Path('proc/configs/seis_ssl_cluster/train_amp_mae.yaml')
		.read_text(encoding='utf-8')
		.replace('every_steps: 1000', 'every_steps: 0'),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode != 0
	assert 'visualization.mae_debug.every_steps' in result.stderr
	assert 'stage:' not in result.stdout


def test_train_amp_mae_dry_run_prints_enabled_mae_debug_summary(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'enabled_mae_debug.yaml'
	config_path.write_text(
		Path('proc/configs/seis_ssl_cluster/train_amp_mae.yaml')
		.read_text(encoding='utf-8')
		.replace('enabled: false', 'enabled: true')
		.replace('every_steps: 1000', 'every_steps: 25'),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'visualization.mae_debug.enabled: true' in result.stdout
	assert (
		'visualization.mae_debug.output_dir: '
		'/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/amp_mae_v1/full_100ep/'
		'visualizations/mae_debug'
	) in result.stdout
	assert 'visualization.mae_debug.every_steps: 25' in result.stdout
	assert 'visualization.mae_debug.every_epochs: null' in result.stdout
	assert 'visualization.mae_debug.panel_width:' not in result.stdout


def test_train_amp_mae_dry_run_prints_continuation_summary(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'continuation.yaml'
	config_path.write_text(
		Path('proc/configs/seis_ssl_cluster/train_amp_mae.yaml').read_text(
			encoding='utf-8',
		)
		+ """
continuation:
  init_checkpoint: ${MAE_CONTINUATION_CHECKPOINT}
  unfreeze_top_blocks: 1
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--config',
		config_path,
		'--dry-run',
		extra_env={
			'MAE_CONTINUATION_CHECKPOINT': '/checkpoints/mae/latest.pt',
		},
	)

	assert result.returncode == 0, result.stderr
	assert (
		'continuation.init_checkpoint: /checkpoints/mae/latest.pt'
		in result.stdout
	)
	assert 'continuation.unfreeze_top_blocks: 1' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout


def test_train_amp_barlow_twins_dry_run_prints_continuation_summary(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'barlow_twins_continuation.yaml'
	config_path.write_text(
		Path('proc/configs/seis_ssl_cluster/train_amp_barlow_twins.yaml').read_text(
			encoding='utf-8',
		)
		+ """
continuation:
  init_checkpoint: /checkpoints/barlow_twins/latest.pt
  unfreeze_top_blocks: 1
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_barlow_twins.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert (
		'continuation.init_checkpoint: /checkpoints/barlow_twins/latest.pt'
		in result.stdout
	)
	assert 'continuation.unfreeze_top_blocks: 1' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout


def test_stage1_barlow_twins_dry_run_omits_continuation_summary(
	tmp_path: Path,
) -> None:
	config_path = Path(
		'experiments/parihaka/facies_benchmark_v1/'
		'21_ssl_hmm_continuation_v1/10_stage1/barlow_twins/'
		'01_gpu_feasibility_1step.yaml'
	)

	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_barlow_twins.py'),
		'--config',
		config_path,
		'--dry-run',
		extra_env={
			'SEIS_SSL_CLUSTER_ARTIFACT_ROOT': str(tmp_path / 'artifacts'),
		},
	)

	assert result.returncode == 0, result.stderr
	assert 'continuation.init_checkpoint:' not in result.stdout
	assert 'continuation.unfreeze_top_blocks:' not in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout


@pytest.mark.parametrize(
	'output_root',
	[
		'relative/run',
	],
)
def test_train_amp_mae_cli_output_root_override_must_be_absolute(
	output_root: str,
) -> None:
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--dry-run',
		'--output-root',
		output_root,
	)

	assert result.returncode != 0
	assert 'paths.output_root' in result.stderr
	assert 'stage:' not in result.stdout


def test_train_amp_mae_cli_accepts_explicit_output_root_outside_artifacts() -> None:
	result = run_python_proc(
		Path('proc/seis_ssl_cluster/train_amp_mae.py'),
		'--dry-run',
		'--output-root',
		'/external/untracked-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'paths.output_root: /external/untracked-run' in result.stdout
	assert 'execution: dry-run; training skipped' in result.stdout


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
loss:
  reconstruction: huber
  huber_delta: 1.0
  gradient_weight: 0.05
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
loss:
  reconstruction: huber
  huber_delta: 1.0
  gradient_weight: 0.05
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
