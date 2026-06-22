from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import seis_ssl_cluster.embedding.extractor as extractor_module
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


def test_embedding_extraction_writes_deterministic_nondivisible_outputs(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)

	first = run_embedding_extraction(config, device='cpu')
	embeddings_path = first[0].embeddings_path
	valid_tokens_path = first[0].valid_tokens_path
	metadata_path = first[0].metadata_path
	first_embeddings = np.load(embeddings_path)
	first_valid = np.load(valid_tokens_path)

	second = run_embedding_extraction(config, device='cpu')
	second_embeddings = np.load(second[0].embeddings_path)
	second_valid = np.load(second[0].valid_tokens_path)

	assert first[0].skipped is False
	assert second[0].skipped is False
	assert first_embeddings.shape == (3, 3, 4, 12)
	assert first_embeddings.dtype == np.float16
	assert first_valid.shape == (3, 3, 4)
	assert first_valid.dtype == np.bool_
	assert first_valid.any()
	np.testing.assert_array_equal(first_embeddings, second_embeddings)
	np.testing.assert_array_equal(first_valid, second_valid)

	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	assert metadata['source_amplitude_path'].endswith('amplitude.npy')
	assert metadata['volume_shape_xyz'] == [5, 6, 7]
	assert metadata['checkpoint_path'].endswith('mae.pt')
	assert metadata['checkpoint_sha256']
	assert metadata['patch_size'] == [2, 2, 2]
	assert metadata['token_grid_shape'] == [3, 3, 4]
	assert metadata['window_size'] == [4, 4, 4]
	assert metadata['overlap'] == [2, 2, 2]
	assert metadata['output_dtype'] == 'float16'
	assert metadata['min_token_valid_fraction'] == 0.5


def test_embedding_extraction_uses_checkpoint_floating_dtype(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path, checkpoint_dtype=torch.bfloat16)
	observed_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
	original_encode_tokens = AmplitudeMAE3D.encode_tokens

	def wrapped_encode_tokens(
		self: AmplitudeMAE3D,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int] | None]:
		observed_dtypes.append((next(self.parameters()).dtype, x.dtype))
		return original_encode_tokens(self, x, valid_mask=valid_mask)

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', wrapped_encode_tokens)

	run_embedding_extraction(config, device='cpu')

	assert observed_dtypes
	assert set(observed_dtypes) == {(torch.bfloat16, torch.bfloat16)}


def test_embedding_extraction_skip_existing_uses_matching_metadata(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	run_embedding_extraction(config, device='cpu')

	result = run_embedding_extraction(config, skip_existing=True, device='cpu')

	assert result[0].skipped is True


def test_embedding_extraction_skip_existing_restarts_incomplete_final_outputs(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	first = run_embedding_extraction(config, device='cpu')[0]
	first.metadata_path.unlink()

	result = run_embedding_extraction(config, skip_existing=True, device='cpu')

	assert result[0].skipped is False
	assert result[0].embeddings_path.is_file()
	assert result[0].valid_tokens_path.is_file()
	assert result[0].metadata_path.is_file()


def test_embedding_extraction_rejects_complete_output_metadata_mismatch(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	first = run_embedding_extraction(config, device='cpu')[0]
	metadata = json.loads(first.metadata_path.read_text(encoding='utf-8'))
	metadata['output_dtype'] = 'float32'
	first.metadata_path.write_text(
		json.dumps(metadata, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)

	with pytest.raises(ValueError, match='metadata does not match'):
		run_embedding_extraction(config, skip_existing=True, device='cpu')


def test_embedding_extraction_hashes_checkpoint_once_for_multiple_surveys(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path, survey_count=2)
	hash_calls: list[Path] = []

	def fake_file_sha256(path: str | Path) -> str:
		hash_calls.append(Path(path))
		return 'cached-checkpoint-digest'

	monkeypatch.setattr(extractor_module, 'file_sha256', fake_file_sha256)

	results = run_embedding_extraction(config, device='cpu')

	assert [result.survey_id for result in results] == ['survey-a', 'survey-b']
	assert hash_calls == [Path(config['embeddings']['checkpoint'])]
	for result in results:
		metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
		assert metadata['checkpoint_sha256'] == 'cached-checkpoint-digest'


def test_embedding_extraction_uses_checkpoint_zero_mask_settings(
	tmp_path: Path,
) -> None:
	zero_mask = {
		'enabled': True,
		'zero_atol': 0.0,
		'z_sample_influence_radius': 0,
		'xy_trace_influence_radius': 1,
	}
	config = _write_fixture(
		tmp_path,
		checkpoint_zero_mask=zero_mask,
		include_extraction_zero_mask=False,
	)
	config['embedding']['min_token_valid_fraction'] = 1.0

	result = run_embedding_extraction(config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['zero_mask'] == zero_mask
	valid_tokens = np.load(result.valid_tokens_path)
	assert not valid_tokens[0, 0, :].any()
	assert valid_tokens[1, 1, 1]


def test_embedding_extraction_rejects_conflicting_zero_mask_override(
	tmp_path: Path,
) -> None:
	config = _write_fixture(
		tmp_path,
		checkpoint_zero_mask={
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 1,
		},
	)
	config['zero_mask'] = {'enabled': False}

	with pytest.raises(ValueError, match='zero_mask override must match checkpoint'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_rejects_integer_output_dtype(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['output_dtype'] = 'int16'

	with pytest.raises(TypeError, match='floating-point NumPy dtype'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_metadata_records_full_model_geometry(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)

	result = run_embedding_extraction(config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['model_geometry'] == {
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
	}


def _write_fixture(
	tmp_path: Path,
	*,
	checkpoint_dtype: torch.dtype = torch.float32,
	checkpoint_zero_mask: dict[str, object] | None = None,
	include_extraction_zero_mask: bool = True,
	survey_count: int = 1,
) -> dict[str, object]:
	if checkpoint_zero_mask is None:
		checkpoint_zero_mask = {'enabled': False}
	manifests = []
	for survey_index in range(survey_count):
		survey_id = f'survey-{chr(ord("a") + survey_index)}'
		survey_root = tmp_path / survey_id
		survey_root.mkdir()
		volume_path = survey_root / 'amplitude.npy'
		volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
		volume[0, 0, :] = 0.0
		np.save(volume_path, volume)
		stats_path = survey_root / 'stats.json'
		write_normalization_stats(
			SurveyNormalizationStats(
				survey_id=survey_id,
				source_path=volume_path,
				grid_order=GRID_ORDER_XYZ,
				clip_low_percentile=0.0,
				clip_high_percentile=100.0,
				clip_low=-1000.0,
				clip_high=1000.0,
				median=0.0,
				iqr=100.0,
			),
			stats_path,
		)
		manifests.append(
			SurveyManifest(
				survey_id=survey_id,
				root=survey_root,
				amplitude=AmplitudeVolumeRecord(
					survey_id=survey_id,
					path=volume_path,
					shape_xyz=tuple(int(axis) for axis in volume.shape),
					dtype='float32',
					grid_order=GRID_ORDER_XYZ,
					normalization_stats_path=stats_path,
				),
			),
		)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json(manifests, manifest_path)
	checkpoint_path = tmp_path / 'mae.pt'
	model_config = {
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
	}
	torch.manual_seed(7)
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
	model.to(dtype=checkpoint_dtype)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'config': {
				'stage': 'train_amp_mae',
				'model': model_config,
				'zero_mask': checkpoint_zero_mask,
			},
		},
		checkpoint_path,
	)
	config: dict[str, object] = {
		'stage': 'extract_embeddings',
		'paths': {
			'nopims_root': str(tmp_path),
			'artifact_root': str(tmp_path / 'artifacts'),
		},
		'manifests': {'input': str(manifest_path)},
		'embeddings': {
			'checkpoint': str(checkpoint_path),
			'output_dir': str(tmp_path / 'embeddings'),
		},
		'embedding': {
			'window_size': [4, 4, 4],
			'overlap': [2, 2, 2],
			'output_dtype': 'float16',
			'batch_size': 2,
			'min_token_valid_fraction': 0.5,
		},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
		},
		'model': model_config,
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 1,
			'amp': False,
			'device': 'cpu',
		},
	}
	if include_extraction_zero_mask:
		config['zero_mask'] = dict(checkpoint_zero_mask)
	return config
