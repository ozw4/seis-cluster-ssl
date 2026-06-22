from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster.visualize_clusters import run_cluster_visualization
from seis_ssl_cluster.clustering import run_embedding_clustering
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	NormalizationStatsQcThresholds,
	compute_normalization_stats,
	filter_manifests_by_stats_qc,
	load_npy_path_list,
	read_manifest_json,
	scan_nopims_amplitude_manifests_from_path_list,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.training import load_checkpoint, run_mae_pretraining

pytest.importorskip('matplotlib')


def test_synthetic_amplitude_mvp_flow(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	artifact_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	path_list = _write_synthetic_nopims_inputs(nopims_root)
	manifest_path, survey_ids = _build_manifest_and_stats(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		path_list=path_list,
	)
	clean_manifest_path, clean_path_list = _filter_manifest(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		manifest_path=manifest_path,
		path_list=path_list,
	)
	checkpoint_path = _train_tiny_mae(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		clean_manifest_path=clean_manifest_path,
		clean_path_list=clean_path_list,
	)
	embedding_dir = _extract_embeddings(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		clean_manifest_path=clean_manifest_path,
		checkpoint_path=checkpoint_path,
		survey_ids=survey_ids,
	)
	cluster_dir = _cluster_embeddings(artifact_root, embedding_dir)
	visualization_dir = _visualize_clusters(
		artifact_root,
		cluster_dir,
		survey_ids=survey_ids,
	)
	_assert_end_to_end_outputs(
		cluster_dir=cluster_dir,
		visualization_dir=visualization_dir,
		survey_id=survey_ids[0],
	)


def _build_manifest_and_stats(
	*,
	nopims_root: Path,
	artifact_root: Path,
	path_list: Path,
) -> tuple[Path, list[str]]:
	manifest_path = (
		artifact_root
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'nopims_amplitude_manifests.json'
	)
	stats_dir = (
		artifact_root
		/ 'registry'
		/ 'normalization_stats'
		/ 'nopims'
		/ 'pretrain_v1'
	)

	manifest_result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root=nopims_root,
		input_path_list=path_list,
		normalization_stats_dir=stats_dir,
	)
	survey_ids = [manifest.survey_id for manifest in manifest_result.manifests]
	manifest_path.parent.mkdir(parents=True)
	write_manifest_json(manifest_result.manifests, manifest_path)
	for manifest in manifest_result.manifests:
		stats = compute_normalization_stats(
			manifest.amplitude.path,
			survey_id=manifest.survey_id,
			grid_order=manifest.amplitude.grid_order,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			max_samples=None,
			seed=7,
		)
		write_normalization_stats(stats, manifest.amplitude.normalization_stats_path)
	return manifest_path, survey_ids


def _filter_manifest(
	*,
	nopims_root: Path,
	artifact_root: Path,
	manifest_path: Path,
	path_list: Path,
) -> tuple[Path, Path]:
	clean_manifest_path = (
		artifact_root
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'nopims_amplitude_manifests.json'
	)
	clean_path_list = (
		artifact_root
		/ 'registry'
		/ 'splits'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'train_npy_paths.txt'
	)
	qc_result = filter_manifests_by_stats_qc(
		read_manifest_json(manifest_path),
		load_npy_path_list(path_list),
		nopims_root=nopims_root,
		thresholds=NormalizationStatsQcThresholds(),
	)
	assert qc_result.excluded_surveys == ()
	clean_manifest_path.parent.mkdir(parents=True)
	write_manifest_json(qc_result.clean_manifests, clean_manifest_path)
	clean_path_list.parent.mkdir(parents=True, exist_ok=True)
	clean_path_list.write_text(
		''.join(f'{entry}\n' for entry in qc_result.clean_path_entries),
		encoding='utf-8',
	)
	return clean_manifest_path, clean_path_list


def _train_tiny_mae(
	*,
	nopims_root: Path,
	artifact_root: Path,
	clean_manifest_path: Path,
	clean_path_list: Path,
) -> Path:
	train_config = _base_config(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		stage='train_amp_mae',
	)
	train_config['paths']['output_root'] = str(
		artifact_root / 'runs' / 'synthetic_smoke',
	)
	train_config['manifests'] = {
		'train': str(clean_manifest_path),
		'train_path_list': str(clean_path_list),
	}
	train_config['train']['samples_per_epoch'] = 1
	train_config['train']['max_steps'] = 1
	checkpoint_path = run_mae_pretraining(train_config)
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert checkpoint['global_step'] == 1
	run_root = Path(train_config['paths']['output_root'])
	assert (run_root / 'resolved_config.json').is_file()
	assert (run_root / 'inputs' / clean_path_list.name).is_file()
	return checkpoint_path


def _extract_embeddings(
	*,
	nopims_root: Path,
	artifact_root: Path,
	clean_manifest_path: Path,
	checkpoint_path: Path,
	survey_ids: list[str],
) -> Path:
	embedding_config = _base_config(
		nopims_root=nopims_root,
		artifact_root=artifact_root,
		stage='extract_embeddings',
	)
	embedding_config['manifests'] = {'input': str(clean_manifest_path)}
	embedding_config['embeddings'] = {
		'checkpoint': str(checkpoint_path),
		'output_dir': str(artifact_root / 'runs' / 'embeddings' / 'synthetic'),
	}
	embedding_config['embedding'] = {
		'window_size': [4, 4, 4],
		'overlap': [2, 2, 2],
		'output_dtype': 'float32',
		'batch_size': 2,
		'min_token_valid_fraction': 0.5,
	}
	embedding_results = run_embedding_extraction(embedding_config, device='cpu')
	assert [result.survey_id for result in embedding_results] == survey_ids
	assert all(result.embeddings_path.is_file() for result in embedding_results)
	output_dir = embedding_config['embeddings']['output_dir']
	assert isinstance(output_dir, str)
	return Path(output_dir)


def _cluster_embeddings(artifact_root: Path, embedding_dir: Path) -> Path:
	cluster_dir = artifact_root / 'runs' / 'clusters' / 'synthetic'
	cluster_result = run_embedding_clustering(
		{
			'embeddings': {'input_dir': str(embedding_dir)},
			'clustering': {
				'output_dir': str(cluster_dir),
				'embedding_normalization': 'l2',
				'pca': {'enabled': True, 'n_components': 2, 'whiten': False},
				'sample_tokens': 32,
				'method': 'minibatch_kmeans',
				'k_values': [2],
				'minibatch_size': 4,
				'prediction_batch_size': 8,
				'seed': 11,
			},
		},
	)
	assert cluster_result.results[0].k == 2
	assert cluster_result.results[0].label_results[0].labels_path.is_file()
	return cluster_dir


def _visualize_clusters(
	artifact_root: Path,
	cluster_dir: Path,
	*,
	survey_ids: list[str],
) -> Path:
	visualization_dir = artifact_root / 'runs' / 'figures' / 'synthetic'
	visualization_result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(cluster_dir)},
			'visualization': {
				'output_dir': str(visualization_dir),
				'survey_ids': survey_ids,
				'modes': ['token', 'voxel'],
				'reconstruct_voxel': True,
				'xy_slices': [1],
				'xz_slices': [1],
				'summaries': {'enabled': True},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)
	assert visualization_result == {
		'png_count': 8,
		'voxel_count': 2,
		'summary_count': 1,
	}
	return visualization_dir


def _assert_end_to_end_outputs(
	*,
	cluster_dir: Path,
	visualization_dir: Path,
	survey_id: str,
) -> None:
	token_labels = np.load(
		cluster_dir / 'labels' / 'k2' / f'{survey_id}.cluster_labels_token.npy',
	)
	voxel_labels = np.load(
		cluster_dir / 'labels' / 'k2' / f'{survey_id}.cluster_labels_voxel.npy',
	)
	assert token_labels.ndim == 3
	assert voxel_labels.shape == (8, 8, 8)
	assert (
		visualization_dir / 'token' / f'{survey_id}_k2_xy_z1.png'
	).is_file()
	assert (
		visualization_dir / 'voxel' / f'{survey_id}_k2_xz_y1.png'
	).is_file()

	metadata = json.loads(
		(
			cluster_dir
			/ 'models'
			/ 'k2'
			/ 'clustering_metadata.json'
		).read_text(encoding='utf-8'),
	)
	assert metadata['method'] == 'minibatch_kmeans'
	assert metadata['sample']['count'] > 0


def _write_synthetic_nopims_inputs(nopims_root: Path) -> Path:
	entries = []
	for index, survey_id in enumerate(('survey_a', 'survey_b')):
		survey_dir = nopims_root / survey_id
		survey_dir.mkdir(parents=True)
		values = np.linspace(-1.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(
			8,
			8,
			8,
		)
		values = values + np.float32(index)
		np.save(survey_dir / 'amplitude.npy', values)
		entries.append(f'{survey_id}/amplitude.npy')
	path_list = nopims_root / 'inputs' / 'train_npy_paths.txt'
	path_list.parent.mkdir(parents=True)
	path_list.write_text(''.join(f'{entry}\n' for entry in entries), encoding='utf-8')
	return path_list


def _base_config(
	*,
	nopims_root: Path,
	artifact_root: Path,
	stage: str,
) -> dict[str, object]:
	return {
		'stage': stage,
		'paths': {
			'nopims_root': str(nopims_root),
			'artifact_root': str(artifact_root),
		},
		'data': {
			'grid_order': list(GRID_ORDER_XYZ),
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
		},
		'model': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': 1,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'loss': {
			'reconstruction': 'huber',
			'huber_delta': 1.0,
			'gradient_weight': 0.0,
			'valid_mask_mode': 'voxel',
		},
		'zero_mask': {'enabled': False},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 2,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 7,
			'grad_clip_norm': 1.0,
		},
	}
