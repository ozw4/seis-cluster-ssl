from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

import seis_ssl_cluster.data.amplitude_crop_dataset as crop_dataset_module
import seis_ssl_cluster.data.window_preprocessing as preprocessing_module
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeAgcConfig,
	AmplitudeVolumeRecord,
	CropRequest,
	NopimsAmplitudeCropDataset,
	NopimsAmplitudePretrainDataset,
	NoTargetProvider,
	SurveyManifest,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	write_normalization_stats,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, MutableMapping, Sequence

	from seis_ssl_cluster.data import TargetProviderContext


class _FakeTargetProvider:
	def __init__(
		self,
		*,
		acceptable: bool = True,
		reject_attempts: int = 0,
		validation_error: str | None = None,
	) -> None:
		self.acceptable = acceptable
		self.reject_attempts = reject_attempts
		self.validation_error = validation_error
		self.validated = False
		self.context: TargetProviderContext | None = None
		self.add_targets_calls = 0
		self.acceptability_calls = 0

	def validate_manifests(
		self,
		manifests: Sequence[SurveyManifest],
		*,
		local_crop_size_xyz: tuple[int, int, int],
		patch_size_xyz: tuple[int, int, int],
		token_grid_shape_xyz: tuple[int, int, int],
	) -> None:
		self.validated = True
		if self.validation_error is not None:
			raise ValueError(self.validation_error)
		assert len(manifests) == 1
		assert local_crop_size_xyz == (4, 4, 4)
		assert patch_size_xyz == (2, 2, 2)
		assert token_grid_shape_xyz == (2, 2, 2)

	def add_targets(
		self,
		sample: MutableMapping[str, object],
		context: TargetProviderContext,
	) -> None:
		self.context = context
		self.add_targets_calls += 1
		sample['fake_target'] = np.full(
			context.token_size_xyz,
			self.add_targets_calls,
			dtype=np.int64,
		)

	def sample_is_acceptable(self, sample: Mapping[str, object]) -> bool:
		assert 'fake_target' in sample
		self.acceptability_calls += 1
		return self.acceptable and self.acceptability_calls > self.reject_attempts

	def rejection_message(
		self,
		*,
		survey_id: str,
		max_resample_attempts: int,
		last_valid_fraction: float,
	) -> str:
		return (
			f'provider rejected {survey_id} after {max_resample_attempts} attempts '
			f'at {last_valid_fraction:.6f}.'
		)


def test_no_target_provider_returns_only_base_sample_fields(tmp_path: Path) -> None:
	dataset = _dataset(
		tmp_path,
		np.ones((8, 8, 8), dtype=np.float32),
		target_provider=NoTargetProvider(),
	)

	sample = dataset[0]

	assert set(sample) == {'x', 'local_valid_mask', 'coords'}
	assert sample['x'].shape == (1, 4, 4, 4)
	assert sample['x'].dtype == np.float32
	assert sample['local_valid_mask'].shape == (4, 4, 4)
	assert sample['local_valid_mask'].dtype == np.bool_
	assert set(sample['coords']) == {
		'survey_id',
		'local_start_xyz',
		'local_size_xyz',
	}
	for key in (
		'strat_labels',
		'strat_confidence',
		'strat_boundary_weight',
		'strat_valid_mask',
		'_token_valid_mask',
	):
		assert key not in sample


def test_crop_start_is_patch_aligned(tmp_path: Path) -> None:
	dataset = _dataset(tmp_path, np.ones((10, 12, 14), dtype=np.float32))

	start_xyz = dataset[0]['coords']['local_start_xyz']

	assert all(axis % 2 == 0 for axis in start_xyz)


def test_set_epoch_changes_sampling_deterministically(tmp_path: Path) -> None:
	dataset = _dataset(tmp_path, np.ones((20, 20, 20), dtype=np.float32))

	dataset.set_epoch(3)
	first = dataset[0]['coords']['local_start_xyz']
	dataset.set_epoch(4)
	changed = dataset[0]['coords']['local_start_xyz']
	dataset.set_epoch(3)
	repeated = dataset[0]['coords']['local_start_xyz']

	assert changed != first
	assert repeated == first


def test_min_valid_fraction_retries_then_raises(tmp_path: Path) -> None:
	dataset = _dataset(
		tmp_path,
		np.zeros((4, 4, 4), dtype=np.float32),
		min_valid_fraction=1.0,
		max_resample_attempts=2,
	)

	with pytest.raises(
		ValueError,
		match=r'min_valid_fraction=1\.000000.*last local valid fraction was 0\.000000',
	):
		dataset[0]


def test_crop_dataset_skips_expensive_preprocessing_for_rejected_crop(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	volume = np.ones((8, 4, 4), dtype=np.float32)
	volume[:4] = 0.0
	dataset = _dataset(
		tmp_path,
		volume,
		zero_mask=ZeroMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=0,
		),
		min_valid_fraction=1.0,
		max_resample_attempts=2,
		amplitude_agc=AmplitudeAgcConfig(
			enabled=True,
			mode='trace_rms_z',
			window_z=3,
			eps=1.0e-6,
			clip_abs=2.0,
		),
	)
	sampler = Mock(
		side_effect=[
			CropRequest('survey', (0, 0, 0), (4, 4, 4)),
			CropRequest('survey', (4, 0, 0), (4, 4, 4)),
		],
	)
	normalize = Mock(
		wraps=preprocessing_module._normalize_amplitude_inplace,  # noqa: SLF001
	)
	agc = Mock(wraps=preprocessing_module.apply_configured_agc)
	monkeypatch.setattr(
		crop_dataset_module,
		'sample_random_token_aligned_local_crop',
		sampler,
	)
	monkeypatch.setattr(preprocessing_module, '_normalize_amplitude_inplace', normalize)
	monkeypatch.setattr(preprocessing_module, 'apply_configured_agc', agc)

	sample = dataset[0]

	assert sample['coords']['local_start_xyz'] == (4, 0, 0)
	assert sampler.call_count == 2
	assert normalize.call_count == 1
	assert agc.call_count == 1


def test_crop_and_pretrain_datasets_share_amplitude_preprocessing(
	tmp_path: Path,
) -> None:
	volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape((4, 4, 4))
	manifest = _manifest(tmp_path, volume=volume)
	shared = {
		'local_crop_size_xyz': (4, 4, 4),
		'patch_size_xyz': (2, 2, 2),
		'zero_mask': ZeroMaskConfig(enabled=False),
		'normalized_clip_abs': 3.0,
	}
	crop_dataset = NopimsAmplitudeCropDataset([manifest], **shared)
	pretrain_dataset = NopimsAmplitudePretrainDataset([manifest], **shared)

	crop_sample = crop_dataset[0]
	pretrain_sample = pretrain_dataset[0]

	np.testing.assert_array_equal(crop_sample['x'], pretrain_sample['x'])
	np.testing.assert_array_equal(
		crop_sample['local_valid_mask'],
		pretrain_sample['local_valid_mask'],
	)
	assert crop_sample['coords'] == pretrain_sample['coords']


def test_pretrain_dataset_uses_manifest_source_valid_mask(tmp_path: Path) -> None:
	volume = np.ones((4, 4, 4), dtype=np.float32)
	volume[1, 2, :] = np.nan
	valid_mask = np.ones((4, 4), dtype=bool)
	valid_mask[1, 2] = False
	manifest = _manifest(tmp_path, volume=volume, valid_mask=valid_mask)
	dataset = NopimsAmplitudePretrainDataset(
		[manifest],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	assert not sample['local_valid_mask'][1, 2, :].any()
	np.testing.assert_array_equal(sample['x'][0, 1, 2, :], 0.0)
	assert np.isfinite(sample['x']).all()


def test_dataset_rejects_missing_manifest_source_valid_mask(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path)
	manifest = SurveyManifest(
		survey_id=manifest.survey_id,
		root=manifest.root,
		amplitude=AmplitudeVolumeRecord(
			manifest.amplitude.survey_id,
			manifest.amplitude.path,
			manifest.amplitude.shape_xyz,
			manifest.amplitude.dtype,
			manifest.amplitude.grid_order,
			manifest.amplitude.normalization_stats_path,
			Path('missing.npy'),
		),
	)

	with pytest.raises(FileNotFoundError, match='source-valid mask file'):
		NopimsAmplitudeCropDataset(
			[manifest],
			local_crop_size_xyz=(4, 4, 4),
			patch_size_xyz=(2, 2, 2),
		)


def test_missing_amplitude_file_raises(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path, write_amplitude=False, write_stats=True)

	with pytest.raises(FileNotFoundError, match='amplitude file does not exist'):
		NopimsAmplitudeCropDataset(
			[manifest],
			local_crop_size_xyz=(4, 4, 4),
			patch_size_xyz=(2, 2, 2),
		)


def test_missing_normalization_stats_raises(tmp_path: Path) -> None:
	manifest = _manifest(tmp_path, write_amplitude=True, write_stats=False)

	with pytest.raises(FileNotFoundError, match='normalization stats file'):
		NopimsAmplitudeCropDataset(
			[manifest],
			local_crop_size_xyz=(4, 4, 4),
			patch_size_xyz=(2, 2, 2),
		)


def test_target_provider_validation_is_called(tmp_path: Path) -> None:
	provider = _FakeTargetProvider()

	_dataset(
		tmp_path,
		np.ones((4, 4, 4), dtype=np.float32),
		target_provider=provider,
	)

	assert provider.validated


def test_target_provider_validation_error_fails_dataset_init(tmp_path: Path) -> None:
	provider = _FakeTargetProvider(validation_error='invalid provider manifests')

	with pytest.raises(ValueError, match='invalid provider manifests'):
		_dataset(
			tmp_path,
			np.ones((4, 4, 4), dtype=np.float32),
			target_provider=provider,
		)

	assert provider.validated


def test_target_provider_rejection_resamples_until_accepted(tmp_path: Path) -> None:
	provider = _FakeTargetProvider(reject_attempts=1)
	dataset = _dataset(
		tmp_path,
		np.ones((8, 8, 8), dtype=np.float32),
		target_provider=provider,
		max_resample_attempts=2,
	)

	sample = dataset[0]

	assert provider.add_targets_calls == 2
	assert provider.acceptability_calls == 2
	np.testing.assert_array_equal(sample['fake_target'], 2)


def test_target_provider_fields_are_returned_without_private_mask(
	tmp_path: Path,
) -> None:
	volume = np.ones((4, 4, 4), dtype=np.float32)
	volume[0, 0, :] = 0.0
	provider = _FakeTargetProvider()
	dataset = _dataset(tmp_path, volume, target_provider=provider)

	sample = dataset[0]

	assert 'fake_target' in sample
	assert '_token_valid_mask' not in sample
	assert provider.context is not None
	assert provider.context.token_valid_mask.shape == (2, 2, 2)
	assert not provider.context.token_valid_mask[0, 0, 0]


def _dataset(
	tmp_path: Path,
	volume: np.ndarray,
	**kwargs: object,
) -> NopimsAmplitudeCropDataset:
	manifest = _manifest(tmp_path, volume=volume)
	return NopimsAmplitudeCropDataset(
		[manifest],
		local_crop_size_xyz=(4, 4, 4),
		patch_size_xyz=(2, 2, 2),
		**kwargs,
	)


def _manifest(
	tmp_path: Path,
	*,
	volume: np.ndarray | None = None,
	write_amplitude: bool = True,
	write_stats: bool = True,
	valid_mask: np.ndarray | None = None,
) -> SurveyManifest:
	root = tmp_path / 'survey'
	root.mkdir()
	amplitude_path = root / 'amplitude.npy'
	if volume is None:
		volume = np.ones((4, 4, 4), dtype=np.float32)
	if write_amplitude:
		np.save(amplitude_path, volume)
	valid_mask_path = None
	if valid_mask is not None:
		valid_mask_path = Path('valid_mask.npy')
		np.save(root / valid_mask_path, valid_mask)
	stats_path = root / 'stats.json'
	if write_stats:
		write_normalization_stats(
			SurveyNormalizationStats(
				survey_id='survey',
				source_path=amplitude_path,
				grid_order=GRID_ORDER_XYZ,
				clip_low_percentile=0.0,
				clip_high_percentile=100.0,
				clip_low=-10.0,
				clip_high=10.0,
				median=0.0,
				iqr=1.0,
			),
			stats_path,
		)
	return SurveyManifest(
		survey_id='survey',
		root=root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey',
			path=amplitude_path,
			shape_xyz=tuple(volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
			valid_mask_path=valid_mask_path,
		),
	)
