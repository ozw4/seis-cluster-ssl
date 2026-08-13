# ruff: noqa: TC003

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import seis_ssl_cluster.parihaka.channel_end_to_end as end_to_end
from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	write_normalization_stats,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.parihaka.channel_data import SectionLines
from seis_ssl_cluster.parihaka.channel_end_to_end import (
	ChannelAmplitudeTileDataset,
	ChannelEndToEndModel,
	resolve_channel_reference_artifact,
)


def _raw_fixture(tmp_path: Path) -> tuple[Path, Path]:
	amplitude_path = tmp_path / 'amplitude.npy'
	labels_path = tmp_path / 'labels.npy'
	stats_path = tmp_path / 'stats.json'
	np.save(amplitude_path, np.ones((64, 64, 64), dtype=np.float32))
	labels = np.ones((64, 64, 64), dtype=np.int8)
	labels[..., ::2] = 5
	np.save(labels_path, labels)
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='parihaka',
			source_path=amplitude_path,
			grid_order=('x', 'y', 'z'),
			clip_low_percentile=1.0,
			clip_high_percentile=99.0,
			clip_low=-2.0,
			clip_high=2.0,
			median=0.0,
			iqr=1.0,
		),
		stats_path,
	)
	artifact_dir = tmp_path / 'reference'
	artifact_dir.mkdir()
	np.save(
		artifact_dir / 'parihaka.valid_tokens.npy',
		np.ones((8, 8, 8), dtype=np.bool_),
	)
	(artifact_dir / 'parihaka.metadata.json').write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'normalization_stats_path': str(stats_path),
				'volume_shape_xyz': [64, 64, 64],
				'patch_size': [8, 8, 8],
				'token_grid_shape': [8, 8, 8],
				'min_token_valid_fraction': 1.0,
				'preprocessing': {
					'normalized_clip_abs': 8.0,
					'amplitude_agc': {'enabled': False},
					'finite_check_mode': 'strict',
				},
				'zero_mask': {'enabled': False},
			}
		),
		encoding='utf-8',
	)
	return artifact_dir, labels_path


def _dataset(tmp_path: Path) -> ChannelAmplitudeTileDataset:
	artifact_dir, labels_path = _raw_fixture(tmp_path)
	return ChannelAmplitudeTileDataset(
		reference=resolve_channel_reference_artifact(artifact_dir),
		labels_path=labels_path,
		lines=SectionLines((0,), (0,)),
		validation=SectionLines((62,), (62,)),
		test=SectionLines((63,), (63,)),
		split='train',
	)


def test_raw_amplitude_uses_shared_preprocessing_and_matches_reference(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = _dataset(tmp_path)
	called = 0
	original = end_to_end.read_amplitude_crop

	def wrapped(**kwargs: object) -> object:
		nonlocal called
		called += 1
		return original(**kwargs)  # type: ignore[arg-type]

	monkeypatch.setattr(end_to_end, 'read_amplitude_crop', wrapped)
	item = dataset[0]
	assert called == 1
	assert item['amplitude'].dtype == torch.float32
	assert item['amplitude'].shape == (1, 80, 80, 80)
	assert item['token_valid_mask'].shape == (10, 10, 10)
	assert item['labels'].shape == (80, 80, 80)
	assert item['core_mask'][8:72, 8:72, 8:72].all()
	assert int(item['supervision_mask'].sum()) == dataset.records[0].supervised_voxels


def test_runtime_token_valid_mismatch_is_rejected(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	dataset = _dataset(tmp_path)
	original = end_to_end.read_amplitude_crop

	def mismatched(**kwargs: object) -> object:
		prepared = original(**kwargs)  # type: ignore[arg-type]
		mask = prepared.token_valid_mask.copy()
		mask[1, 1, 1] = False
		return replace(prepared, token_valid_mask=mask)

	monkeypatch.setattr(end_to_end, 'read_amplitude_crop', mismatched)
	with pytest.raises(ValueError, match='runtime token-valid mask'):
		dataset[0]


def test_model_forward_gradient_path_and_parameter_partition() -> None:
	mae = AmplitudeMAE3D(
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
		runtime_check_mode='strict',
	)
	decoder = VoxelDecoder3D(
		embedding_dim=12,
		class_count=2,
		hidden_channels=(8,),
		upsample_factors=((2, 2, 2),),
		patch_size_xyz=(2, 2, 2),
	)
	model = ChannelEndToEndModel(mae, decoder).float()
	amplitude = torch.randn(1, 1, 8, 8, 8)
	valid = torch.ones(1, 4, 4, 4, dtype=torch.bool)
	logits = model(amplitude, valid)
	assert logits.shape == (1, 2, 8, 8, 8)
	logits.square().mean().backward()
	assert mae.patch_projection.weight.grad is not None
	assert any(parameter.grad is not None for parameter in mae.encoder.parameters())
	assert any(parameter.grad is not None for parameter in decoder.parameters())
	assert mae.mask_token.grad is None
	assert all(parameter.grad is None for parameter in mae.decoder.parameters())
	assert all(parameter.grad is None for parameter in mae.prediction_head.parameters())
	assert all(
		parameter.grad is None for parameter in mae.encoder_to_decoder.parameters()
	)
	encoder_ids = {id(parameter) for parameter in model.encoder_parameters()}
	decoder_ids = {id(parameter) for parameter in model.decoder_parameters()}
	assert encoder_ids
	assert decoder_ids
	assert encoder_ids.isdisjoint(decoder_ids)
