from __future__ import annotations

import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

import seis_ssl_cluster.stratigraphy.pseudo_target_builder as builder_module
from proc.seis_ssl_cluster import build_strat_hmm_pseudo_targets as cli
from seis_ssl_cluster.config.strat_hmm_pseudo_targets import (
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	NpyMemmapVolumeStore,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy.prototypes import OrderedPrototypeHead
from seis_ssl_cluster.stratigraphy.pseudo_target_builder import (
	build_strat_hmm_pseudo_target_results,
	build_strat_hmm_pseudo_targets,
)
from seis_ssl_cluster.stratigraphy.targets import (
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	validate_pseudo_target_arrays,
)
from tests.seis_ssl_cluster.helpers_window_preprocessing import (
	PATCH_SIZE_XYZ,
	read_fixture_crop,
	write_window_preprocessing_fixture,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


def test_tiny_survey_builds_pseudo_target_files_on_cpu(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)

	results = build_strat_hmm_pseudo_target_results(config, device='cpu')

	assert len(results) == 1
	result = results[0]
	assert result.survey_id == 'survey-a'
	assert result.valid_token_count == 12
	assert result.labels_path.is_file()
	assert result.confidence_path.is_file()
	assert result.valid_tokens_path.is_file()
	assert result.boundary_weight_path.is_file()
	assert result.metadata_path.is_file()


def test_builder_read_window_matches_shared_preprocessing(tmp_path: Path) -> None:
	fixture = write_window_preprocessing_fixture(tmp_path)
	expected = read_fixture_crop(fixture, min_token_valid_fraction=0.5)

	window, x, token_valid_mask = builder_module._read_window(  # noqa: SLF001
		fixture.window,
		manifest=fixture.manifest,
		amplitude_path=fixture.amplitude_path,
		stats=fixture.stats,
		store=NpyMemmapVolumeStore(),
		settings=SimpleNamespace(
			zero_mask=fixture.zero_mask,
			normalized_clip_abs=fixture.normalized_clip_abs,
			amplitude_agc=fixture.amplitude_agc,
			min_token_valid_fraction=0.5,
		),
		patch_size_xyz=PATCH_SIZE_XYZ,
	)

	assert window == fixture.window
	np.testing.assert_allclose(x, expected.x, rtol=1.0e-6)
	np.testing.assert_array_equal(token_valid_mask, expected.token_valid_mask)
	np.testing.assert_array_equal(x[0][~expected.local_valid_mask], 0.0)


def test_builder_outputs_validate_as_pseudo_target_arrays(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)
	build_strat_hmm_pseudo_targets(config, device='cpu')

	paths = pseudo_target_paths(
		config['outputs']['pseudo_target_root'],
		k=3,
		survey_id='survey-a',
	)
	arrays = load_pseudo_target_arrays(paths)

	validate_pseudo_target_arrays(
		arrays.labels,
		arrays.confidence,
		arrays.valid_tokens,
		boundary_weight=arrays.boundary_weight,
		k=3,
		survey_id='survey-a',
	)


def test_hmm_k_mismatch_with_checkpoint_head_is_rejected(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)
	config['hmm']['k'] = 2

	with pytest.raises(ValueError, match=r'hmm\.k'):
		build_strat_hmm_pseudo_targets(config, device='cpu')


def test_existing_outputs_are_rejected_unless_overwrite_is_true(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path)
	first = build_strat_hmm_pseudo_target_results(config, device='cpu')[0]

	with pytest.raises(ValueError, match='existing pseudo-target output'):
		build_strat_hmm_pseudo_targets(config, device='cpu')

	skipped = build_strat_hmm_pseudo_target_results(
		config,
		device='cpu',
		skip_existing=True,
	)[0]
	assert skipped.skipped is True
	assert skipped.metadata_path == first.metadata_path
	assert skipped.valid_token_count == first.valid_token_count

	results = build_strat_hmm_pseudo_target_results(
		config,
		device='cpu',
		overwrite=True,
	)
	assert results[0].metadata_path == first.metadata_path


def test_skip_existing_rejects_metadata_mismatch(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)
	build_strat_hmm_pseudo_target_results(config, device='cpu')
	config['hmm']['boundary_weighting']['alpha'] = 0.5

	with pytest.raises(ValueError, match='metadata does not match'):
		build_strat_hmm_pseudo_target_results(
			config,
			device='cpu',
			skip_existing=True,
		)


def test_partial_existing_outputs_are_rejected_without_overwrite(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path)
	first = build_strat_hmm_pseudo_target_results(config, device='cpu')[0]
	first.boundary_weight_path.unlink()

	with pytest.raises(ValueError, match='incomplete existing pseudo-target output'):
		build_strat_hmm_pseudo_target_results(config, device='cpu')

	assert first.labels_path.is_file()
	assert not first.boundary_weight_path.exists()


def test_edge_margins_reduce_valid_token_count(tmp_path: Path) -> None:
	config = _resolved_config(tmp_path)
	config['hmm']['edge_margin_tokens'] = [0, 0, 1]

	result = build_strat_hmm_pseudo_target_results(config, device='cpu')[0]

	assert result.valid_token_count == 4
	arrays = load_pseudo_target_arrays(
		pseudo_target_paths(
			config['outputs']['pseudo_target_root'],
			k=3,
			survey_id='survey-a',
		),
	)
	assert int(np.count_nonzero(arrays.valid_tokens)) == 4


def test_metadata_records_checkpoint_provenance_and_hmm_settings(
	tmp_path: Path,
) -> None:
	config = _resolved_config(tmp_path)
	build_strat_hmm_pseudo_targets(config, device='cpu')

	metadata = load_pseudo_target_metadata(
		pseudo_target_paths(
			config['outputs']['pseudo_target_root'],
			k=3,
			survey_id='survey-a',
		),
	)
	source = metadata['source']
	assert isinstance(source, dict)
	assert source['checkpoint_path'] == config['checkpoint']['path']
	assert source['checkpoint_sha256']
	assert source['checkpoint_training_stage'] == 'train_strat_hmm_pretext'
	assert source['head_config']['num_prototypes'] == 3
	assert source['hmm']['transition']['max_jump'] == 1
	assert source['hmm']['boundary_weighting'] == {'alpha': 0.0, 'tau': 1.0}
	assert source['hmm']['path_prior'] is None
	assert source['decode']['boundary_weight_summary']['transition_boundary_count'] >= 0
	assert source['inference']['window_size'] == [4, 4, 4]
	assert source['valid_summary']['decoded_valid_token_count'] == 12
	assert source['confidence_summary']['min'] >= 0.0


def test_cli_non_dry_run_executes_one_survey_tiny_config(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	config = _config(tmp_path)
	config_path = tmp_path / 'config.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
	monkeypatch.setattr(
		sys,
		'argv',
		[
			'build_strat_hmm_pseudo_targets.py',
			'--config',
			str(config_path),
			'--device',
			'cpu',
		],
	)

	cli.main()

	stdout = capsys.readouterr().out
	assert 'pseudo_target:' in stdout
	paths = pseudo_target_paths(
		config['outputs']['pseudo_target_root'],
		k=3,
		survey_id='survey-a',
	)
	assert paths.metadata.is_file()


def _resolved_config(tmp_path: Path) -> dict[str, object]:
	return resolve_strat_hmm_pseudo_target_config(_config(tmp_path))


def _config(tmp_path: Path) -> dict[str, object]:
	paths = _write_fixture(tmp_path)
	return {
		'paths': {'artifact_root': str(paths['artifact_root'])},
		'manifests': {'train': str(paths['manifest'])},
		'checkpoint': {'path': str(paths['checkpoint'])},
		'model': {'patch_size': [2, 2, 2]},
		'inference': {
			'window_size': [4, 4, 4],
			'overlap': [2, 2, 2],
			'batch_size': 2,
			'output_dtype': 'float32',
			'min_token_valid_fraction': 1.0,
			'device': 'auto',
		},
		'hmm': {
			'k': 3,
			'edge_margin_tokens': [0, 0, 0],
			'transition': {
				'same_cost': 0.0,
				'advance_cost': 0.1,
				'jump_cost': 0.5,
				'reverse_cost': 2.0,
				'forbid_reverse': True,
				'max_jump': 1,
			},
			'path_prior': {'enabled': False},
		},
		'outputs': {
			'pseudo_target_root': str(
				paths['artifact_root'] / 'pseudo_targets' / 'refresh',
			),
			'overwrite': False,
			'skip_existing': False,
		},
	}


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
	artifact_root = tmp_path / 'artifacts'
	survey_root = tmp_path / 'survey-a'
	survey_root.mkdir(exist_ok=True)
	volume_path = survey_root / 'amplitude.npy'
	volume = np.linspace(-1.0, 1.0, 4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6)
	np.save(volume_path, volume)
	stats_path = survey_root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey-a',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-2.0,
			clip_high=2.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='survey-a',
		root=survey_root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey-a',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	path_list = tmp_path / 'train_paths.txt'
	path_list.write_text(f'{volume_path}\n', encoding='utf-8')
	checkpoint_path = tmp_path / 'strat.pt'
	_write_checkpoint(
		checkpoint_path,
		manifest_path=manifest_path,
		path_list=path_list,
		tmp_path=tmp_path,
	)
	return {
		'artifact_root': artifact_root,
		'checkpoint': checkpoint_path,
		'manifest': manifest_path,
	}


def _write_checkpoint(
	checkpoint_path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
	tmp_path: Path,
) -> None:
	torch.manual_seed(3)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=8,
		encoder_depth=1,
		encoder_heads=2,
		decoder_dim=8,
		decoder_depth=1,
		decoder_heads=2,
	)
	head = OrderedPrototypeHead(
		feature_dim=8,
		num_prototypes=3,
		temperature=0.5,
	)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'stratigraphy_state_dict': head.state_dict(),
			'config': _mae_config(
				tmp_path,
				manifest_path=manifest_path,
				path_list=path_list,
			),
			'stratigraphy_config': {
				'stage': 'train_strat_hmm_pretext',
				'head': {
					'num_prototypes': 3,
					'temperature': 0.5,
				},
			},
			'training_state': {'stage': 'train_strat_hmm_pretext'},
		},
		checkpoint_path,
	)


def _mae_config(
	tmp_path: Path,
	*,
	manifest_path: Path,
	path_list: Path,
) -> Mapping[str, object]:
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
			},
			'model': {
				'name': 'amp_mae3d',
				'in_channels': 1,
				'out_channels': 1,
				'patch_size': [2, 2, 2],
				'encoder_dim': 8,
				'encoder_depth': 1,
				'encoder_heads': 2,
				'decoder_dim': 8,
				'decoder_depth': 1,
				'decoder_heads': 2,
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
				'seed': 3,
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
