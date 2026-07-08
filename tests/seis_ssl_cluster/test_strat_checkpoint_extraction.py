from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy.prototypes import OrderedPrototypeHead
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.strat_hmm_checkpoint import save_strat_hmm_checkpoint

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def test_strat_checkpoint_extracts_student_embeddings_and_metadata(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path, strat=True)
	loaded_keys: list[tuple[str, ...]] = []
	original_load_state_dict = AmplitudeMAE3D.load_state_dict

	def wrapped_load_state_dict(
		self: AmplitudeMAE3D,
		state_dict: dict[str, torch.Tensor],
		*args: object,
		**kwargs: object,
	) -> object:
		loaded_keys.append(tuple(state_dict))
		return original_load_state_dict(self, state_dict, *args, **kwargs)

	monkeypatch.setattr(AmplitudeMAE3D, 'load_state_dict', wrapped_load_state_dict)

	result = run_embedding_extraction(config, device='cpu')[0]

	embeddings = np.load(result.embeddings_path)
	assert embeddings.shape == (2, 2, 2, 12)
	assert loaded_keys
	assert not any(
		key == 'prototypes' or key.startswith('projection.')
		for key in loaded_keys[0]
	)

	checkpoint = load_checkpoint(config['embeddings']['checkpoint'], map_location='cpu')
	assert 'prototypes' in checkpoint['stratigraphy_state_dict']
	assert 'prototypes' not in checkpoint['model_state_dict']

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['stratigraphy_pretext'] == {
		'method': 'strat_hmm_pretext',
		'base_objective': 'amp_mae3d',
		'head_num_prototypes': 3,
		'unfreeze_top_blocks': 1,
		'distillation_weight': 0.1,
		'pseudo_target_input_dir': str(tmp_path / 'pseudo_targets'),
	}


def test_standard_mae_checkpoint_extracts_without_strat_metadata(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path, strat=False)

	result = run_embedding_extraction(config, device='cpu')[0]

	embeddings = np.load(result.embeddings_path)
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert embeddings.shape == (2, 2, 2, 12)
	assert 'stratigraphy_pretext' not in metadata
	assert metadata['pretraining_objective'] == {
		'reconstruction': 'huber',
		'gradient_weight': 0.0,
		'visible_reconstruction_weight': 0.0,
		'huber_delta': 1.0,
		'target_normalization': {'mode': 'none'},
	}


def _write_fixture(tmp_path: Path, *, strat: bool) -> dict[str, object]:
	manifest_path, path_list = _write_manifest_fixture(tmp_path)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	mae_config = _mae_config(
		tmp_path,
		manifest_path=manifest_path,
		path_list=path_list,
	)
	checkpoint_path = tmp_path / ('strat.pt' if strat else 'mae.pt')
	if strat:
		_write_strat_checkpoint(
			checkpoint_path,
			student=model,
			mae_config=mae_config,
			stratigraphy_config=_stratigraphy_config(
				tmp_path,
				manifest_path=manifest_path,
				path_list=path_list,
				teacher_checkpoint=tmp_path / 'teacher.pt',
			),
		)
	else:
		torch.save(
			{'model_state_dict': model.state_dict(), 'config': mae_config},
			checkpoint_path,
		)
	return {
		'paths': {'artifact_root': str(tmp_path / 'artifacts')},
		'manifests': {'input': str(manifest_path)},
		'embeddings': {
			'checkpoint': str(checkpoint_path),
			'output_dir': str(tmp_path / 'embeddings'),
		},
		'embedding': {
			'window_size': [4, 4, 4],
			'overlap': [0, 0, 0],
			'output_dtype': 'float16',
			'batch_size': 1,
			'min_token_valid_fraction': 0.0,
		},
	}


def _write_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
	survey_root = tmp_path / 'survey'
	survey_root.mkdir()
	volume_path = survey_root / 'amplitude.npy'
	volume = np.linspace(-1.0, 1.0, num=4 * 4 * 4, dtype=np.float32).reshape(
		4,
		4,
		4,
	)
	np.save(volume_path, volume)
	stats_path = survey_root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1.0,
			clip_high=1.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json(
		[
			SurveyManifest(
				survey_id='survey',
				root=survey_root,
				amplitude=AmplitudeVolumeRecord(
					survey_id='survey',
					path=volume_path,
					shape_xyz=(4, 4, 4),
					dtype='float32',
					grid_order=GRID_ORDER_XYZ,
					normalization_stats_path=stats_path,
				),
			),
		],
		manifest_path,
	)
	path_list = tmp_path / 'train_path_list.txt'
	path_list.write_text(f'{volume_path}\n', encoding='utf-8')
	return manifest_path, path_list


def _write_strat_checkpoint(
	path: Path,
	*,
	student: AmplitudeMAE3D,
	mae_config: dict[str, object],
	stratigraphy_config: dict[str, object],
) -> None:
	head = OrderedPrototypeHead(
		feature_dim=student.encoder_dim,
		num_prototypes=3,
		projection_dim=6,
		temperature=0.5,
	)
	optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3)
	save_strat_hmm_checkpoint(
		path,
		student=student,
		head=head,
		optimizer=optimizer,
		epoch=1,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		metrics={'loss': 1.0},
		global_step=1,
		checkpoint_kind='epoch',
		batch_index=None,
	)


def _mae_config(
	tmp_path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
) -> dict[str, object]:
	return deepcopy(
		{
			'stage': 'train_amp_mae',
			'paths': {'output_root': str(tmp_path / 'mae_run')},
			'manifests': {
				'train': str(manifest_path),
				'train_path_list': str(path_list),
			},
			'data': {
				'grid_order': list(GRID_ORDER_XYZ),
				'volume_format': 'npy_memmap',
				'input_channels': 1,
				'target_channels': 1,
				'use_context': False,
				'local_crop_size': [4, 4, 4],
				'min_valid_fraction': 0.0,
				'max_resample_attempts': 2,
				'amplitude_agc': {'enabled': False},
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
				'visible_reconstruction_weight': 0.0,
				'target_normalization': {'mode': 'none'},
				'valid_mask_mode': 'voxel',
			},
			'train': {
				'batch_size': 1,
				'samples_per_epoch': 1,
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
			'zero_mask': {
				'enabled': False,
				'zero_atol': 0.0,
				'z_sample_influence_radius': 0,
				'xy_trace_influence_radius': 0,
			},
		},
	)


def _stratigraphy_config(
	tmp_path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
	teacher_checkpoint: Path,
) -> dict[str, object]:
	pseudo_target_root = tmp_path / 'pseudo_targets'
	pseudo_target_root.mkdir()
	return {
		'stage': 'train_strat_hmm_pretext',
		'paths': {'output_root': str(tmp_path / 'strat_run')},
		'manifests': {
			'train': str(manifest_path),
			'train_path_list': str(path_list),
		},
		'data': {
			'local_crop_size': [4, 4, 4],
			'min_valid_fraction': 0.0,
			'max_resample_attempts': 2,
		},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 12,
			'encoder_depth': 1,
			'encoder_heads': 3,
			'decoder_dim': 12,
			'decoder_depth': 1,
			'decoder_heads': 3,
		},
		'pseudo_targets': {
			'input_dir': str(pseudo_target_root),
			'k': 3,
			'min_confidence': 0.0,
		},
		'teacher': {'checkpoint': str(teacher_checkpoint)},
		'student': {'unfreeze_top_blocks': 1},
		'head': {
			'num_prototypes': 3,
			'projection_dim': 6,
			'temperature': 0.5,
			'normalize': True,
		},
		'loss': {
			'prototype_weight': 1.0,
			'usage_weight': 0.01,
			'entropy_floor': None,
			'distillation_weight': 0.1,
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-3,
			'encoder_lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 11,
			'grad_clip_norm': 1.0,
			'allow_overwrite_output': False,
		},
		'zero_mask': {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 0,
		},
	}
