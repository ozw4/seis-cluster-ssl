from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import seis_ssl_cluster.training.joint_embedding_common as joint_embedding_common_module
from seis_ssl_cluster.config import resolve_barlow_twins_training_config
from seis_ssl_cluster.config.schema import (
	BARLOW_TWINS_PRETRAINING_METHOD,
	LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
)
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudePretrainDataset,
	AmplitudeVolumeRecord,
	LocalBarlowTwinsPretrainDataset,
	OverlappingLocalBarlowTwinsPretrainDataset,
	RegionLocalBarlowTwinsPretrainDataset,
	SurveyManifest,
	SurveyNormalizationStats,
	XyRotationLocalBarlowTwinsPretrainDataset,
	ZeroMaskConfig,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.models.barlow_twins import BarlowTwins3D
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.mae.patching import patchify_3d
from seis_ssl_cluster.training.barlow_twins import run_barlow_twins_pretraining
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	load_barlow_twins_checkpoint,
)

if TYPE_CHECKING:
	from pathlib import Path


def _manifest(
	tmp_path: Path,
	volume: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
) -> SurveyManifest:
	volume_path = tmp_path / 'survey' / 'amplitude.npy'
	volume_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(volume_path, volume.astype(np.float32, copy=False))
	stats_path = tmp_path / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-10_000.0,
			clip_high=10_000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	valid_mask_path = None
	if valid_mask is not None:
		valid_mask_path = tmp_path / 'survey' / 'valid.npy'
		np.save(valid_mask_path, valid_mask.astype(bool, copy=False))
	return SurveyManifest(
		survey_id='survey',
		root=tmp_path,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
			valid_mask_path=valid_mask_path,
		),
	)


def _patch_id_volume(
	token_grid_shape_xyz: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	patch_ids = np.arange(
		1,
		int(np.prod(token_grid_shape_xyz)) + 1,
		dtype=np.float32,
	).reshape(token_grid_shape_xyz)
	volume = patch_ids
	for axis, repeat in enumerate(patch_size_xyz):
		volume = volume.repeat(repeat, axis=axis)
	return volume


def _base_dataset(  # noqa: PLR0913
	tmp_path: Path,
	volume: np.ndarray,
	*,
	patch_size_xyz: tuple[int, int, int],
	valid_mask: np.ndarray | None = None,
	seed: int = 19,
	samples_per_epoch: int = 8,
	min_valid_token_count: int = 0,
) -> AmplitudePretrainDataset:
	return AmplitudePretrainDataset(
		[_manifest(tmp_path, volume, valid_mask=valid_mask)],
		local_crop_size_xyz=volume.shape,
		patch_size_xyz=patch_size_xyz,
		emit_spatial_mask=False,
		seed=seed,
		samples_per_epoch=samples_per_epoch,
		zero_mask=ZeroMaskConfig(enabled=False),
		min_valid_token_count=min_valid_token_count,
	)


def test_region_window_111_matches_legacy_local_samples(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)
	legacy = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'legacy',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
	)
	region = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'region',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		positive_window_tokens=(1, 1, 1),
	)

	for index in range(len(legacy)):
		legacy_sample = legacy[index]
		region_sample = region[index]
		np.testing.assert_array_equal(
			region_sample['view_a'],
			legacy_sample['view_a'],
		)
		np.testing.assert_array_equal(
			region_sample['view_b'],
			legacy_sample['view_b'],
		)
		np.testing.assert_array_equal(
			region_sample['horizontal_flip_state_a'],
			legacy_sample['horizontal_flip_state_a'],
		)
		np.testing.assert_array_equal(
			region_sample['horizontal_flip_state_b'],
			legacy_sample['horizontal_flip_state_b'],
		)
		for key in ('local_pair_indices_a', 'local_pair_indices_b'):
			assert region_sample[key].shape == (4, 1)
			np.testing.assert_array_equal(
				region_sample[key][:, 0],
				legacy_sample[key],
			)


def test_region_member_indices_select_same_physical_blocks(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	token_grid_shape_xyz = (2, 3, 2)
	positive_window_tokens = (2, 2, 1)
	volume = _patch_id_volume(token_grid_shape_xyz, patch_size_xyz)
	dataset = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		positive_window_tokens=positive_window_tokens,
	)

	for index in range(len(dataset)):
		sample = dataset[index]
		indices_a = torch.as_tensor(sample['local_pair_indices_a'])
		indices_b = torch.as_tensor(sample['local_pair_indices_b'])
		assert indices_a.shape == (4, 4)
		assert indices_b.shape == (4, 4)
		patches_a = patchify_3d(
			torch.as_tensor(sample['view_a']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		patches_b = patchify_3d(
			torch.as_tensor(sample['view_b']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		ids_a = patches_a[:, 0, 0].round().to(torch.int64)
		ids_b = patches_b[:, 0, 0].round().to(torch.int64)
		block_ids_a = ids_a[indices_a.reshape(-1)].reshape(4, 4)
		block_ids_b = ids_b[indices_b.reshape(-1)].reshape(4, 4)
		np.testing.assert_array_equal(
			np.sort(block_ids_a.numpy(), axis=1),
			np.sort(block_ids_b.numpy(), axis=1),
		)
		for block in block_ids_a.numpy():
			coordinates = np.asarray(
				np.unravel_index(block - 1, token_grid_shape_xyz, order='C'),
			)
			assert len(set(block.tolist())) == 4
			spans = coordinates.max(axis=1) - coordinates.min(axis=1)
			np.testing.assert_array_equal(
				spans,
				np.asarray(positive_window_tokens) - 1,
			)


def test_region_anchor_erosion_excludes_blocks_touching_invalid_tokens(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	token_grid_shape_xyz = (2, 3, 2)
	positive_window_tokens = (1, 2, 1)
	volume = _patch_id_volume(token_grid_shape_xyz, patch_size_xyz)
	patch_ids = np.arange(1, 13, dtype=np.float32).reshape(token_grid_shape_xyz)
	valid_mask = np.ones_like(volume, dtype=bool)
	invalid_token_xyz = (1, 1, 0)
	valid_mask[
		tuple(
			slice(token * patch, (token + 1) * patch)
			for token, patch in zip(
				invalid_token_xyz,
				patch_size_xyz,
				strict=True,
			)
		)
	] = False
	invalid_patch_id = int(patch_ids[invalid_token_xyz])
	# Anchors along y must avoid windows containing the invalid token.
	eligible_anchor_ids = {
		int(patch_ids[x, y, z])
		for x in range(2)
		for y in range(2)
		for z in range(2)
		if not (x == 1 and z == 0 and y in (0, 1))
	}
	dataset = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			valid_mask=valid_mask,
			min_valid_token_count=6,
		),
		local_pairs_per_crop=6,
		positive_window_tokens=positive_window_tokens,
	)

	sample = dataset[0]
	indices_a = np.asarray(sample['local_pair_indices_a'])
	assert indices_a.shape == (6, 2)
	flip_state_a = np.asarray(sample['horizontal_flip_state_a'])
	view_ids = patch_ids
	if bool(flip_state_a[0]):
		view_ids = view_ids[::-1, :, :]
	if bool(flip_state_a[1]):
		view_ids = view_ids[:, ::-1, :]
	member_ids = view_ids.ravel(order='C')[indices_a.reshape(-1)]
	assert invalid_patch_id not in set(member_ids.astype(np.int64).tolist())
	anchor_ids = {
		int(min(block))
		for block in view_ids.ravel(order='C')[indices_a].astype(np.int64)
	}
	assert anchor_ids == eligible_anchor_ids


def test_region_rejects_insufficient_anchors_and_oversized_window(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)
	dataset = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=12,
		),
		local_pairs_per_crop=12,
		positive_window_tokens=(2, 2, 1),
	)
	with pytest.raises(ValueError, match='fewer fully valid block anchors'):
		dataset[0]

	oversized = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'oversized',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=2,
		),
		local_pairs_per_crop=2,
		positive_window_tokens=(3, 1, 1),
	)
	with pytest.raises(ValueError, match='must fit within the crop token grid'):
		oversized[0]


def test_region_samples_are_deterministic_per_seed_epoch_index(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)

	def build() -> RegionLocalBarlowTwinsPretrainDataset:
		return RegionLocalBarlowTwinsPretrainDataset(
			_base_dataset(
				tmp_path / 'determinism',
				volume,
				patch_size_xyz=patch_size_xyz,
				min_valid_token_count=4,
			),
			local_pairs_per_crop=4,
			positive_window_tokens=(2, 1, 1),
		)

	first = build()
	second = build()
	first.set_epoch(3)
	second.set_epoch(3)
	sample_a = first[1]
	sample_b = second[1]
	np.testing.assert_array_equal(sample_a['view_a'], sample_b['view_a'])
	np.testing.assert_array_equal(
		sample_a['local_pair_indices_a'],
		sample_b['local_pair_indices_a'],
	)
	second.set_epoch(4)
	changed = second[1]
	assert not (
		np.array_equal(sample_a['view_a'], changed['view_a'])
		and np.array_equal(
			sample_a['local_pair_indices_a'],
			changed['local_pair_indices_a'],
		)
	)


def _tiny_model() -> BarlowTwins3D:
	torch.manual_seed(11)
	backbone = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=4,
		encoder_depth=1,
		encoder_heads=1,
		decoder_dim=4,
		decoder_depth=1,
		decoder_heads=1,
	)
	return BarlowTwins3D(backbone, projector_dim=4)


def test_forward_local_region_indices_average_member_tokens() -> None:
	model = _tiny_model()
	model.eval()
	torch.manual_seed(23)
	view_a = torch.randn(2, 1, 4, 4, 4)
	view_b = torch.randn(2, 1, 4, 4, 4)
	valid_mask = torch.ones(2, 4, 4, 4, dtype=torch.bool)
	member_indices_a = torch.tensor(
		[[[0, 1], [2, 3], [4, 5]], [[1, 3], [0, 4], [6, 7]]],
		dtype=torch.int64,
	)
	member_indices_b = torch.tensor(
		[[[1, 0], [3, 2], [5, 4]], [[3, 1], [4, 0], [7, 6]]],
		dtype=torch.int64,
	)

	with torch.no_grad():
		outputs = model.forward_local(
			view_a,
			view_b,
			valid_mask_a=valid_mask,
			valid_mask_b=valid_mask,
			local_pair_indices_a=member_indices_a,
			local_pair_indices_b=member_indices_b,
		)
		tokens_a = model.backbone.encode_tokens(view_a, valid_mask=valid_mask)
		tokens = tokens_a['tokens']
		assert isinstance(tokens, torch.Tensor)
		manual_regions = []
		for batch_index in range(2):
			gathered = tokens[batch_index][member_indices_a[batch_index]]
			manual_regions.append(gathered.mean(dim=1))
		manual_projection = model.projector(
			torch.stack(manual_regions).flatten(0, 1)
		)

	assert outputs['z_a'].shape == (6, 4)
	assert outputs['z_b'].shape == (6, 4)
	torch.testing.assert_close(outputs['z_a'], manual_projection)


def test_forward_local_region_window_one_matches_token_indices() -> None:
	model = _tiny_model()
	model.eval()
	torch.manual_seed(29)
	view_a = torch.randn(2, 1, 4, 4, 4)
	view_b = torch.randn(2, 1, 4, 4, 4)
	valid_mask = torch.ones(2, 4, 4, 4, dtype=torch.bool)
	token_indices = torch.tensor([[0, 3, 5], [7, 2, 1]], dtype=torch.int64)

	with torch.no_grad():
		token_outputs = model.forward_local(
			view_a,
			view_b,
			valid_mask_a=valid_mask,
			valid_mask_b=valid_mask,
			local_pair_indices_a=token_indices,
			local_pair_indices_b=token_indices,
		)
		region_outputs = model.forward_local(
			view_a,
			view_b,
			valid_mask_a=valid_mask,
			valid_mask_b=valid_mask,
			local_pair_indices_a=token_indices.unsqueeze(-1),
			local_pair_indices_b=token_indices.unsqueeze(-1),
		)

	torch.testing.assert_close(region_outputs['z_a'], token_outputs['z_a'])
	torch.testing.assert_close(region_outputs['z_b'], token_outputs['z_b'])


def test_forward_local_rejects_mismatched_member_index_shapes() -> None:
	model = _tiny_model()
	view = torch.randn(1, 1, 4, 4, 4)
	valid_mask = torch.ones(1, 4, 4, 4, dtype=torch.bool)
	with pytest.raises(ValueError, match='matching'):
		model.forward_local(
			view,
			view,
			valid_mask_a=valid_mask,
			valid_mask_b=valid_mask,
			local_pair_indices_a=torch.zeros(1, 2, 2, dtype=torch.int64),
			local_pair_indices_b=torch.zeros(1, 2, dtype=torch.int64),
		)
	with pytest.raises(ValueError, match=r'\[B, K\] or \[B, K, W\]'):
		model.forward_local(
			view,
			view,
			valid_mask_a=valid_mask,
			valid_mask_b=valid_mask,
			local_pair_indices_a=torch.zeros(1, 2, 2, 1, dtype=torch.int64),
			local_pair_indices_b=torch.zeros(1, 2, 2, 1, dtype=torch.int64),
		)


def _write_synthetic_manifest(root: Path) -> Path:
	root.mkdir(parents=True, exist_ok=True)
	volume_path = root / 'amplitude.npy'
	volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
	np.save(volume_path, volume)
	stats_path = root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='tiny',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='tiny',
		root=root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='tiny',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = root / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	return manifest_path


def _tiny_region_config(
	tmp_path: Path,
	*,
	output_name: str = 'region-run',
	positive_window_tokens: list[int] | None = None,
	method: str = LOCAL_BARLOW_TWINS_PRETRAINING_METHOD,
	local_pairs_per_crop: int = 2,
) -> dict[str, object]:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	path_list = tmp_path / 'train_npy_paths.txt'
	path_list.write_text(
		f'{tmp_path / "survey" / "amplitude.npy"}\n',
		encoding='utf-8',
	)
	barlow_twins: dict[str, object] = {
		'method': method,
		'projector_dim': 4,
		'positive_window_tokens': (
			[2, 2, 1] if positive_window_tokens is None else positive_window_tokens
		),
	}
	if method == LOCAL_BARLOW_TWINS_PRETRAINING_METHOD:
		barlow_twins['local_pairs_per_crop'] = local_pairs_per_crop
	return {
		'paths': {
			'artifact_root': str(tmp_path / 'artifacts'),
			'output_root': str(tmp_path / 'artifacts' / output_name),
		},
		'manifests': {
			'train': str(manifest_path),
			'train_path_list': str(path_list),
		},
		'data': {'local_crop_size': [4, 4, 4]},
		'zero_mask': {'enabled': False},
		'model': {
			'patch_size': [2, 2, 2],
			'encoder_dim': 4,
			'encoder_depth': 1,
			'encoder_heads': 1,
			'decoder_dim': 4,
			'decoder_depth': 1,
			'decoder_heads': 1,
		},
		'barlow_twins': barlow_twins,
		'train': {
			'batch_size': 2,
			'samples_per_epoch': 2,
			'epochs': 1,
			'num_workers': 0,
			'shuffle': False,
			'lr': 1.0e-3,
			'weight_decay': 0.0,
			'amp': False,
			'device': 'cpu',
			'seed': 7,
			'grad_clip_norm': 1.0,
			'max_steps': 1,
		},
	}


def test_config_accepts_and_keeps_positive_window_tokens(
	tmp_path: Path,
) -> None:
	resolved = resolve_barlow_twins_training_config(
		_tiny_region_config(tmp_path)
	)
	assert resolved['barlow_twins']['positive_window_tokens'] == [2, 2, 1]


def test_config_rejects_positive_window_for_global_method(
	tmp_path: Path,
) -> None:
	config = _tiny_region_config(
		tmp_path,
		method=BARLOW_TWINS_PRETRAINING_METHOD,
	)
	with pytest.raises(ValueError, match='positive_window_tokens is only allowed'):
		resolve_barlow_twins_training_config(config)


def test_config_allows_positive_window_with_nuisance_view_policies(
	tmp_path: Path,
) -> None:
	config = _tiny_region_config(tmp_path)
	config['augmentations'] = {
		'policy': 'horizontal_flip_trace_drop_v1',
		'horizontal_flip_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	resolved = resolve_barlow_twins_training_config(config)
	assert resolved['barlow_twins']['positive_window_tokens'] == [2, 2, 1]

	token_dropout = _tiny_region_config(tmp_path, output_name='region-tokdrop')
	token_dropout['augmentations'] = {
		'policy': 'horizontal_flip_token_dropout_v1',
		'horizontal_flip_probability': 0.5,
		'token_dropout_fraction': 0.3,
	}
	resolved = resolve_barlow_twins_training_config(token_dropout)
	assert resolved['augmentations']['token_dropout_fraction'] == 0.3

	smooth_gain = _tiny_region_config(tmp_path, output_name='region-sgain')
	smooth_gain['augmentations'] = {
		'policy': 'horizontal_flip_smooth_gain_v1',
		'horizontal_flip_probability': 0.5,
		'gain_jitter': 0.2,
	}
	resolved = resolve_barlow_twins_training_config(smooth_gain)
	assert resolved['augmentations']['gain_jitter'] == 0.2


def test_config_rejects_positive_window_with_unsupported_policies(
	tmp_path: Path,
) -> None:
	d4 = _tiny_region_config(tmp_path)
	d4['augmentations'] = {
		'policy': 'xy_d4_trace_drop_v1',
		'reflection_probability': 0.5,
		'trace_drop_probability': 0.02,
	}
	with pytest.raises(ValueError, match='is not supported for'):
		resolve_barlow_twins_training_config(d4)

	overlap_zshift = _tiny_region_config(tmp_path, output_name='region-overlap-z')
	overlap_zshift['augmentations'] = {
		'policy': 'overlapping_subcrop_xy_v1',
		'horizontal_flip_probability': 0.5,
		'max_subcrop_shift_tokens': [1, 1, 0],
	}
	overlap_zshift['barlow_twins']['positive_window_tokens'] = [2, 2, 1]
	with pytest.raises(
		ValueError,
		match='must fit within the minimum overlapping token grid',
	):
		resolve_barlow_twins_training_config(overlap_zshift)


def test_config_rejects_oversized_window_and_insufficient_anchors(
	tmp_path: Path,
) -> None:
	oversized = _tiny_region_config(
		tmp_path,
		positive_window_tokens=[3, 1, 1],
	)
	with pytest.raises(
		ValueError,
		match='must fit within the minimum overlapping token grid',
	):
		resolve_barlow_twins_training_config(oversized)

	crowded = _tiny_region_config(
		tmp_path,
		positive_window_tokens=[2, 2, 1],
		local_pairs_per_crop=4,
	)
	with pytest.raises(ValueError, match='positive-window anchor count'):
		resolve_barlow_twins_training_config(crowded)


def test_region_one_step_dispatches_region_dataset_and_saves_config(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	original_dataset = (
		joint_embedding_common_module.RegionLocalBarlowTwinsPretrainDataset
	)
	dataset_calls: list[dict[str, object]] = []

	def record_dataset(
		base_dataset: object,
		**kwargs: object,
	) -> object:
		dataset_calls.append(kwargs)
		return original_dataset(base_dataset, **kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(
		joint_embedding_common_module,
		'RegionLocalBarlowTwinsPretrainDataset',
		record_dataset,
	)
	config = resolve_barlow_twins_training_config(
		_tiny_region_config(tmp_path, output_name='region-one-step')
	)

	checkpoint_path = run_barlow_twins_pretraining(config)
	payload = load_barlow_twins_checkpoint(checkpoint_path, map_location='cpu')

	assert dataset_calls == [
		{
			'local_pairs_per_crop': 2,
			'positive_window_tokens': (2, 2, 1),
			'horizontal_flip_probability': 0.5,
		}
	]
	assert payload['config']['barlow_twins']['positive_window_tokens'] == [
		2,
		2,
		1,
	]
	assert payload['global_step'] == 1
	assert all(np.isfinite(value) for value in payload['metrics'].values())


def test_region_resume_rejects_changed_positive_window(
	tmp_path: Path,
) -> None:
	config = _tiny_region_config(tmp_path, output_name='region-resume')
	config['train']['max_steps'] = None  # type: ignore[index]
	resolved = resolve_barlow_twins_training_config(config)
	checkpoint_path = run_barlow_twins_pretraining(resolved)

	changed = _tiny_region_config(
		tmp_path,
		output_name='region-resume-changed',
		positive_window_tokens=[1, 2, 1],
	)
	changed['train']['max_steps'] = None  # type: ignore[index]
	changed['train']['epochs'] = 2  # type: ignore[index]
	resolved_changed = resolve_barlow_twins_training_config(changed)
	with pytest.raises(ValueError, match="section 'barlow_twins' does not match"):
		run_barlow_twins_pretraining(
			resolved_changed,
			resume=checkpoint_path,
		)


def test_token_dropout_zeroes_whole_tokens_only_in_view_b(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = (
		_patch_id_volume((2, 3, 2), patch_size_xyz)
		+ np.arange(4 * 6 * 4, dtype=np.float32).reshape(4, 6, 4) * 1e-3
	)
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
		token_dropout_fraction=0.3,
	)

	sample = dataset[0]
	view_a = np.asarray(sample['view_a'])
	view_b = np.asarray(sample['view_b'])
	assert not np.any(view_a == 0.0)
	blocks = view_b.reshape(1, 2, 2, 3, 2, 2, 2)
	token_is_zero = np.all(blocks == 0.0, axis=(0, 2, 4, 6))
	token_has_zero = np.any(blocks == 0.0, axis=(0, 2, 4, 6))
	assert int(token_is_zero.sum()) == round(0.3 * 12)
	np.testing.assert_array_equal(token_is_zero, token_has_zero)
	np.testing.assert_array_equal(
		np.asarray(sample['valid_mask_b']),
		np.ones_like(volume, dtype=bool),
	)


def test_token_dropout_zero_fraction_preserves_sample_bytes(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)

	def build(fraction: float) -> LocalBarlowTwinsPretrainDataset:
		return LocalBarlowTwinsPretrainDataset(
			_base_dataset(
				tmp_path / f'fraction-{fraction}',
				volume,
				patch_size_xyz=patch_size_xyz,
				min_valid_token_count=4,
			),
			local_pairs_per_crop=4,
			token_dropout_fraction=fraction,
		)

	baseline = build(0.0)[1]
	legacy = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'legacy',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
	)[1]
	np.testing.assert_array_equal(
		np.asarray(baseline['view_b']),
		np.asarray(legacy['view_b']),
	)
	np.testing.assert_array_equal(
		np.asarray(baseline['local_pair_indices_b']),
		np.asarray(legacy['local_pair_indices_b']),
	)


def test_smooth_gain_is_bounded_smooth_and_view_independent(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = np.ones((4, 6, 4), dtype=np.float32)
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
		smooth_gain_jitter=0.2,
	)

	sample = dataset[0]
	view_a = np.asarray(sample['view_a'])[0]
	view_b = np.asarray(sample['view_b'])[0]
	for view in (view_a, view_b):
		assert np.all(view >= 1.0 - 0.2 - 1e-6)
		assert np.all(view <= 1.0 + 0.2 + 1e-6)
		np.testing.assert_allclose(
			view,
			np.broadcast_to(view[:, :, :1], view.shape),
			rtol=0.0,
			atol=1e-6,
		)
		xy_field = view[:, :, 0]
		x_steps = np.abs(np.diff(xy_field, axis=0))
		y_steps = np.abs(np.diff(xy_field, axis=1))
		assert float(x_steps.max()) <= 2.0 * 0.2 * 2.0 / (4 - 1) + 1e-6
		assert float(y_steps.max()) <= 2.0 * 0.2 * 2.0 / (6 - 1) + 1e-6
	assert not np.allclose(view_a, view_b)


def test_region_dataset_applies_token_dropout_after_member_sampling(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)
	dataset = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		positive_window_tokens=(2, 1, 1),
		token_dropout_fraction=0.3,
	)

	sample = dataset[0]
	assert np.asarray(sample['local_pair_indices_a']).shape == (4, 2)
	view_b = np.asarray(sample['view_b'])
	blocks = view_b.reshape(1, 2, 2, 3, 2, 2, 2)
	token_is_zero = np.all(blocks == 0.0, axis=(0, 2, 4, 6))
	assert int(token_is_zero.sum()) == round(0.3 * 12)


def test_overlap_region_members_pair_same_parent_blocks(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	parent_token_shape = (5, 4, 2)
	positive_window_tokens = (2, 2, 1)
	parent = _patch_id_volume(parent_token_shape, patch_size_xyz)
	base = _base_dataset(
		tmp_path,
		parent,
		patch_size_xyz=patch_size_xyz,
		min_valid_token_count=0,
		samples_per_epoch=8,
	)
	dataset = OverlappingLocalBarlowTwinsPretrainDataset(
		base,
		view_crop_size_xyz=(8, 6, 4),
		local_pairs_per_crop=2,
		max_subcrop_shift_tokens=(1, 1, 0),
		positive_window_tokens=positive_window_tokens,
	)

	for index in range(len(dataset)):
		sample = dataset[index]
		indices_a = np.asarray(sample['local_pair_indices_a'])
		indices_b = np.asarray(sample['local_pair_indices_b'])
		assert indices_a.shape == (2, 4)
		assert indices_b.shape == (2, 4)
		patches_a = patchify_3d(
			torch.as_tensor(sample['view_a']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		patches_b = patchify_3d(
			torch.as_tensor(sample['view_b']).unsqueeze(0),
			patch_size_xyz,
		)[0]
		ids_a = patches_a[:, 0, 0].round().to(torch.int64).numpy()
		ids_b = patches_b[:, 0, 0].round().to(torch.int64).numpy()
		block_ids_a = ids_a[indices_a.reshape(-1)].reshape(2, 4)
		block_ids_b = ids_b[indices_b.reshape(-1)].reshape(2, 4)
		np.testing.assert_array_equal(
			np.sort(block_ids_a, axis=1),
			np.sort(block_ids_b, axis=1),
		)
		for block in block_ids_a:
			coordinates = np.asarray(
				np.unravel_index(block - 1, parent_token_shape, order='C'),
			)
			assert len(set(block.tolist())) == 4
			spans = coordinates.max(axis=1) - coordinates.min(axis=1)
			np.testing.assert_array_equal(
				spans,
				np.asarray(positive_window_tokens) - 1,
			)


def test_identity_gaussian_region_pairs_use_unflipped_views(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)
	dataset = RegionLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		positive_window_tokens=(2, 2, 1),
		horizontal_flip_probability=0.0,
		gaussian_noise_std=0.1,
		require_distinct_horizontal_views=False,
	)

	sample = dataset[0]
	np.testing.assert_array_equal(
		np.asarray(sample['horizontal_flip_state_a']),
		np.zeros(2, dtype=bool),
	)
	np.testing.assert_array_equal(
		np.asarray(sample['horizontal_flip_state_b']),
		np.zeros(2, dtype=bool),
	)
	np.testing.assert_array_equal(
		np.asarray(sample['local_pair_indices_a']),
		np.asarray(sample['local_pair_indices_b']),
	)
	assert not np.array_equal(
		np.asarray(sample['view_a']),
		np.asarray(sample['view_b']),
	)


def test_config_and_trainer_accept_identity_gaussian_with_region(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _tiny_region_config(tmp_path, output_name='identity-region')
	config['augmentations'] = {
		'policy': 'identity_gaussian_noise_v1',
		'gaussian_noise_std': 0.1,
	}
	resolved = resolve_barlow_twins_training_config(config)
	assert resolved['barlow_twins']['positive_window_tokens'] == [2, 2, 1]

	original_dataset = (
		joint_embedding_common_module.RegionLocalBarlowTwinsPretrainDataset
	)
	dataset_calls: list[dict[str, object]] = []

	def record_dataset(
		base_dataset: object,
		**kwargs: object,
	) -> object:
		dataset_calls.append(kwargs)
		return original_dataset(base_dataset, **kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(
		joint_embedding_common_module,
		'RegionLocalBarlowTwinsPretrainDataset',
		record_dataset,
	)
	checkpoint_path = run_barlow_twins_pretraining(resolved)
	payload = load_barlow_twins_checkpoint(checkpoint_path, map_location='cpu')

	assert dataset_calls == [
		{
			'local_pairs_per_crop': 2,
			'positive_window_tokens': (2, 2, 1),
			'horizontal_flip_probability': 0.0,
			'gaussian_noise_std': 0.1,
			'require_distinct_horizontal_views': False,
		}
	]
	assert payload['config']['augmentations'] == {
		'policy': 'identity_gaussian_noise_v1',
		'gaussian_noise_std': 0.1,
	}
	assert all(np.isfinite(value) for value in payload['metrics'].values())


def test_laplace_noise_is_heavier_tailed_at_matched_std(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = np.zeros((4, 6, 4), dtype=np.float32)
	kinds = {}
	for kind in ('gaussian', 'laplace'):
		dataset = LocalBarlowTwinsPretrainDataset(
			_base_dataset(
				tmp_path / kind,
				volume,
				patch_size_xyz=patch_size_xyz,
				min_valid_token_count=4,
			),
			local_pairs_per_crop=4,
			horizontal_flip_probability=0.0,
			gaussian_noise_std=1.0,
			noise_kind=kind,
		)
		samples = np.concatenate(
			[np.asarray(dataset[i]['view_a']).ravel() for i in range(len(dataset))]
		)
		kinds[kind] = samples
	for kind, samples in kinds.items():
		assert abs(float(samples.std()) - 1.0) < 0.15, kind
		assert abs(float(samples.mean())) < 0.1, kind
	# Excess kurtosis: gaussian ~0, laplace ~3.
	def kurtosis(values: np.ndarray) -> float:
		centered = values - values.mean()
		return float((centered**4).mean() / (centered**2).mean() ** 2 - 3.0)

	assert kurtosis(kinds['laplace']) > kurtosis(kinds['gaussian']) + 1.0


def test_colored_noise_is_z_correlated_and_keeps_unit_std(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = np.zeros((4, 6, 16), dtype=np.float32)
	dataset = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
		gaussian_noise_std=1.0,
		noise_z_smoothing=4,
	)
	samples = np.stack(
		[np.asarray(dataset[i]['view_a'])[0] for i in range(len(dataset))]
	)
	assert abs(float(samples.std()) - 1.0) < 0.25
	# Neighbouring z samples of band-limited noise are positively correlated.
	first, second = samples[..., :-1].ravel(), samples[..., 1:].ravel()
	corr = float(np.corrcoef(first, second)[0, 1])
	assert corr > 0.4, corr


def test_asymmetric_noise_leaves_the_first_view_clean(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	volume = _patch_id_volume((2, 3, 2), patch_size_xyz)
	clean = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'clean',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
	)
	asym = LocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path / 'asym',
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		horizontal_flip_probability=0.0,
		gaussian_noise_std=0.5,
		noise_second_view_only=True,
	)
	sample_clean, sample_asym = clean[0], asym[0]
	np.testing.assert_array_equal(
		np.asarray(sample_asym['view_a']),
		np.asarray(sample_clean['view_a']),
	)
	assert not np.array_equal(
		np.asarray(sample_asym['view_b']),
		np.asarray(sample_clean['view_b']),
	)


def test_rot90_views_pair_the_same_physical_tokens(tmp_path: Path) -> None:
	patch_size_xyz = (2, 2, 2)
	token_grid_shape_xyz = (2, 2, 2)
	volume = _patch_id_volume(token_grid_shape_xyz, patch_size_xyz)
	dataset = XyRotationLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		rotation_only=True,
	)
	for index in range(len(dataset)):
		sample = dataset[index]
		assert int(sample['xy_transform_id_a']) != int(sample['xy_transform_id_b'])
		assert int(sample['xy_transform_id_a']) < 4
		assert int(sample['xy_transform_id_b']) < 4
		patches_a = patchify_3d(
			torch.as_tensor(sample['view_a']).unsqueeze(0), patch_size_xyz
		)[0]
		patches_b = patchify_3d(
			torch.as_tensor(sample['view_b']).unsqueeze(0), patch_size_xyz
		)[0]
		ids_a = patches_a[:, 0, 0].round().to(torch.int64)
		ids_b = patches_b[:, 0, 0].round().to(torch.int64)
		sel_a = ids_a[torch.as_tensor(sample['local_pair_indices_a'])]
		sel_b = ids_b[torch.as_tensor(sample['local_pair_indices_b'])]
		torch.testing.assert_close(sel_a, sel_b)


def test_rot90_region_blocks_pair_and_reject_anisotropic_windows(
	tmp_path: Path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	token_grid_shape_xyz = (4, 4, 2)
	volume = _patch_id_volume(token_grid_shape_xyz, patch_size_xyz)
	dataset = XyRotationLocalBarlowTwinsPretrainDataset(
		_base_dataset(
			tmp_path,
			volume,
			patch_size_xyz=patch_size_xyz,
			min_valid_token_count=4,
		),
		local_pairs_per_crop=4,
		rotation_only=True,
		positive_window_tokens=(2, 2, 1),
	)
	sample = dataset[0]
	indices_a = torch.as_tensor(sample['local_pair_indices_a'])
	indices_b = torch.as_tensor(sample['local_pair_indices_b'])
	assert indices_a.shape == (4, 4)
	patches_a = patchify_3d(
		torch.as_tensor(sample['view_a']).unsqueeze(0), patch_size_xyz
	)[0]
	patches_b = patchify_3d(
		torch.as_tensor(sample['view_b']).unsqueeze(0), patch_size_xyz
	)[0]
	ids_a = patches_a[:, 0, 0].round().to(torch.int64).numpy()
	ids_b = patches_b[:, 0, 0].round().to(torch.int64).numpy()
	block_a = ids_a[indices_a.reshape(-1)].reshape(4, 4)
	block_b = ids_b[indices_b.reshape(-1)].reshape(4, 4)
	np.testing.assert_array_equal(
		np.sort(block_a, axis=1), np.sort(block_b, axis=1)
	)

	with pytest.raises(ValueError, match='X and Y sizes must be equal'):
		XyRotationLocalBarlowTwinsPretrainDataset(
			_base_dataset(
				tmp_path / 'aniso',
				volume,
				patch_size_xyz=patch_size_xyz,
				min_valid_token_count=4,
			),
			local_pairs_per_crop=4,
			positive_window_tokens=(2, 1, 1),
		)


def test_new_policies_resolve_and_dispatch(tmp_path: Path) -> None:
	policies = {
		'horizontal_flip_colored_noise_v1': {
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
			'noise_z_smoothing': 4,
		},
		'horizontal_flip_laplace_noise_v1': {
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		'horizontal_flip_asymmetric_noise_v1': {
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
		},
		'horizontal_flip_gaussian_trace_drop_v1': {
			'horizontal_flip_probability': 0.5,
			'gaussian_noise_std': 0.1,
			'trace_drop_probability': 0.02,
		},
		'xy_rot90_gaussian_noise_v1': {'gaussian_noise_std': 0.1},
		'xy_d4_gaussian_noise_v1': {'gaussian_noise_std': 0.1},
	}
	for policy, options in policies.items():
		config = _tiny_region_config(tmp_path, output_name=f'poc-{policy}')
		config['augmentations'] = {'policy': policy, **options}
		resolved = resolve_barlow_twins_training_config(config)
		assert resolved['augmentations']['policy'] == policy
		assert resolved['barlow_twins']['positive_window_tokens'] == [2, 2, 1]
		dataset = joint_embedding_common_module.build_local_joint_embedding_dataset(
			_base_dataset(
				tmp_path / f'ds-{policy}',
				_patch_id_volume((4, 4, 2), (2, 2, 2)),
				patch_size_xyz=(2, 2, 2),
				min_valid_token_count=4,
			),
			local_pairs_per_crop=4,
			augmentations=resolved['augmentations'],
			positive_window_tokens=(2, 2, 1),
		)
		sample = dataset[0]
		assert np.asarray(sample['local_pair_indices_a']).shape == (4, 4)
		assert np.all(np.isfinite(np.asarray(sample['view_a'])))
