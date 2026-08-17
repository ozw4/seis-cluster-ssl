from __future__ import annotations

import json
import math
import os
import threading
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import seis_ssl_cluster.data.survey_preprocessing_cache as cache_module
import seis_ssl_cluster.embedding.extractor as extractor_module
from seis_ssl_cluster.config import resolve_barlow_twins_training_config
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudePreprocessSettings,
	AmplitudeVolumeRecord,
	CropRequest,
	NpyMemmapVolumeStore,
	SurveyManifest,
	SurveyNormalizationStats,
	load_normalization_stats,
	read_amplitude_crop,
	read_manifest_json,
	read_prepared_survey_amplitude_crop,
	resolve_manifest_path,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.data.window_preprocessing import (
	reduce_valid_mask_to_tokens as shared_reduce_valid_mask_to_tokens,
)
from seis_ssl_cluster.embedding import (
	EmbeddingMerger,
	build_model_from_checkpoint_payload,
	extract_embeddings_from_loaded_model,
	iter_sliding_windows,
	run_embedding_extraction,
	token_grid_shape_xyz,
)
from seis_ssl_cluster.embedding import (
	reduce_valid_mask_to_tokens as package_reduce_valid_mask_to_tokens,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.parihaka.channel_end_to_end import ChannelEndToEndModel
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	PRETRAINING_METHOD as BARLOW_TWINS_PRETRAINING_METHOD,
)
from seis_ssl_cluster.training.barlow_twins_checkpoint import (
	save_barlow_twins_checkpoint,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint
from tests.seis_ssl_cluster.helpers_window_preprocessing import (
	PATCH_SIZE_XYZ,
	read_fixture_crop,
	write_window_preprocessing_fixture,
)

if TYPE_CHECKING:
	from collections.abc import Callable


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
	assert metadata['precision'] == {
		'amp_requested': False,
		'amp_dtype_requested': 'auto',
		'resolved_dtype': 'float32',
		'amp_enabled': False,
	}
	assert metadata['min_token_valid_fraction'] == 0.5
	assert metadata['preprocessing']['amplitude_agc'] == {'enabled': False}
	assert metadata['amplitude_agc'] == {'enabled': False}


def test_barlow_checkpoint_uses_existing_full_volume_extraction_contract(
	tmp_path: Path,
) -> None:
	mae_root = tmp_path / 'mae'
	barlow_root = tmp_path / 'barlow'
	mae_root.mkdir()
	barlow_root.mkdir()
	mae_config = _write_fixture(mae_root)
	barlow_config = _write_fixture(barlow_root)
	_make_fixture_checkpoint_barlow(barlow_config)

	mae_result = run_embedding_extraction(mae_config, device='cpu')[0]
	barlow_result = run_embedding_extraction(barlow_config, device='cpu')[0]
	mae_embeddings = np.load(mae_result.embeddings_path)
	barlow_embeddings = np.load(barlow_result.embeddings_path)
	barlow_valid = np.load(barlow_result.valid_tokens_path)
	metadata = json.loads(barlow_result.metadata_path.read_text(encoding='utf-8'))

	assert barlow_embeddings.shape == mae_embeddings.shape == (3, 3, 4, 12)
	assert barlow_valid.shape == (3, 3, 4)
	assert metadata['pretraining_method'] == BARLOW_TWINS_PRETRAINING_METHOD
	assert metadata['pretraining_objective'] == {
		'method': BARLOW_TWINS_PRETRAINING_METHOD,
		'projector_dim': 8,
		'redundancy_weight': 0.005,
		'normalization_eps': 1.0e-4,
	}
	assert metadata['model_geometry'] == json.loads(
		mae_result.metadata_path.read_text(encoding='utf-8')
	)['model_geometry']


def test_barlow_checkpoint_loads_exact_encoder_without_projector_or_wrapper(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	_make_fixture_checkpoint_barlow(config)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	mae = build_model_from_checkpoint_payload(payload)
	for key, expected in payload['model_state_dict'].items():
		assert torch.equal(mae.state_dict()[key], expected)
	assert not any(
		key.startswith(('backbone.', 'projector.')) for key in mae.state_dict()
	)
	assert set(payload['projector_state_dict']).isdisjoint(mae.state_dict())

	voxel_decoder = VoxelDecoder3D(
		embedding_dim=12,
		class_count=2,
		hidden_channels=(8,),
		upsample_factors=((2, 2, 2),),
		patch_size_xyz=(2, 2, 2),
	)
	downstream = ChannelEndToEndModel(mae, voxel_decoder)
	encoder_ids = {id(parameter) for parameter in downstream.encoder_parameters()}
	assert encoder_ids == {
		id(parameter)
		for name, parameter in mae.named_parameters()
		if name.startswith(('patch_projection.', 'encoder.'))
	}
	assert all(
		id(parameter) not in encoder_ids for parameter in mae.decoder.parameters()
	)
	for parameter in downstream.encoder_parameters():
		parameter.requires_grad_(requires_grad=False)
	assert all(
		not parameter.requires_grad for parameter in downstream.encoder_parameters()
	)
	for parameter in downstream.encoder_parameters():
		parameter.requires_grad_(requires_grad=True)
	assert all(parameter.requires_grad for parameter in downstream.encoder_parameters())


def test_barlow_checkpoint_rejects_method_identity_drift(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	_make_fixture_checkpoint_barlow(config)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	payload['pretraining_method'] = 'amp_mae3d'

	with pytest.raises(ValueError, match='pretraining_method'):
		build_model_from_checkpoint_payload(payload)


def test_barlow_checkpoint_consumer_keeps_method_config_strict(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	_make_fixture_checkpoint_barlow(config)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	payload['config']['loss'] = {'reconstruction': 'mse'}

	with pytest.raises(ValueError, match='unsupported top-level'):
		build_model_from_checkpoint_payload(payload)


def test_embedding_extraction_uses_manifest_source_valid_mask(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['preprocessing_cache'] = {'mode': 'memory'}
	manifest_path = Path(config['manifests']['input'])
	manifest = read_manifest_json(manifest_path)[0]
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	volume = np.load(amplitude_path)
	volume[0:2, 0:2, :] = np.nan
	np.save(amplitude_path, volume)
	valid_mask = np.ones(volume.shape[:2], dtype=bool)
	valid_mask[0:2, 0:2] = False
	valid_mask_path = manifest.root / 'valid_mask.npy'
	np.save(valid_mask_path, valid_mask)
	masked_manifest = replace(
		manifest,
		amplitude=replace(
			manifest.amplitude,
			valid_mask_path=Path('valid_mask.npy'),
		),
	)
	write_manifest_json([masked_manifest], manifest_path)

	result = run_embedding_extraction(config, device='cpu')[0]
	embeddings = np.load(result.embeddings_path)
	valid_tokens = np.load(result.valid_tokens_path)
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))

	assert np.isfinite(embeddings).all()
	assert not valid_tokens[0, 0, :].any()
	assert metadata['source_valid_mask_path'] == str(valid_mask_path)
	assert metadata['preprocessing_cache']['requested_mode'] == 'memory'
	assert metadata['preprocessing_cache']['effective_mode'] == 'off'
	assert 'window-local memmap reads' in metadata['preprocessing_cache'][
		'fallback_reason'
	]


def test_loaded_model_extraction_matches_checkpoint_and_publishes_descriptor(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	checkpoint_config = payload['config']
	assert isinstance(checkpoint_config, dict)
	student = extractor_module.build_model_from_config(checkpoint_config)
	student.load_state_dict(payload['model_state_dict'])
	standard = run_embedding_extraction(config, device='cpu')[0]
	refresh_root = tmp_path / 'refresh_generation'
	state_hash = extractor_module.file_sha256(checkpoint_path)
	student.train()
	student.encoder.layers[0].eval()
	training_before = {
		name: module.training for name, module in student.named_modules()
	}
	state_before = {
		name: value.detach().clone()
		for name, value in student.state_dict().items()
	}
	cpu_rng_before = torch.random.get_rng_state()

	def fail_checkpoint_load(*_args: object, **_kwargs: object) -> None:
		raise AssertionError('loaded-model refresh must not load a checkpoint')

	monkeypatch.setattr(extractor_module, 'load_checkpoint', fail_checkpoint_load)

	refresh = extract_embeddings_from_loaded_model(
		student,
		config,
		refresh_root,
		state_hash,
		checkpoint_config=checkpoint_config,
		device='cpu',
	)

	np.testing.assert_array_equal(
		np.load(refresh[0].embeddings_path),
		np.load(standard.embeddings_path),
	)
	np.testing.assert_array_equal(
		np.load(refresh[0].valid_tokens_path),
		np.load(standard.valid_tokens_path),
	)
	assert json.loads(refresh[0].metadata_path.read_text()) == json.loads(
		standard.metadata_path.read_text(),
	)
	descriptor = json.loads(
		(refresh_root / 'refresh_extraction_descriptor.json').read_text(),
	)
	assert descriptor['source_student_state_sha256'] == state_hash
	assert (
		descriptor['embedding_semantics']
		== 'current_student_unmasked_eval_full_survey_v1'
	)
	assert descriptor['completion_status'] == 'COMPLETE'
	assert {
		name: module.training for name, module in student.named_modules()
	} == training_before
	assert all(
		torch.equal(value, student.state_dict()[name])
		for name, value in state_before.items()
	)
	assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)


def test_loaded_model_extraction_restores_mode_state_and_rng_on_failure(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	checkpoint_config = payload['config']
	assert isinstance(checkpoint_config, dict)
	student = extractor_module.build_model_from_config(checkpoint_config)
	student.load_state_dict(payload['model_state_dict'])
	student.train()
	student.encoder.layers[0].eval()
	training_before = {
		name: module.training for name, module in student.named_modules()
	}
	state_before = {
		name: value.detach().clone()
		for name, value in student.state_dict().items()
	}
	cpu_rng_before = torch.random.get_rng_state()

	def fail(*_args: object, **_kwargs: object) -> None:
		raise RuntimeError('injected loaded extraction failure')

	monkeypatch.setattr(extractor_module, '_process_prepared_batch', fail)
	with pytest.raises(RuntimeError, match='injected loaded extraction failure'):
		extract_embeddings_from_loaded_model(
			student,
			config,
			tmp_path / 'refresh_generation',
			'a' * 64,
			checkpoint_config=checkpoint_config,
			device='cpu',
		)

	assert not (tmp_path / 'refresh_generation').exists()
	assert {
		name: module.training for name, module in student.named_modules()
	} == training_before
	assert all(
		torch.equal(value, student.state_dict()[name])
		for name, value in state_before.items()
	)
	assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)


def test_loaded_model_refresh_reuse_rejects_foreign_and_partial_outputs(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	checkpoint_path = Path(config['embeddings']['checkpoint'])  # type: ignore[index]
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	checkpoint_config = payload['config']
	assert isinstance(checkpoint_config, dict)
	student = extractor_module.build_model_from_config(checkpoint_config)
	student.load_state_dict(payload['model_state_dict'])
	refresh_root = tmp_path / 'refresh_generation'
	extract_embeddings_from_loaded_model(
		student,
		config,
		refresh_root,
		'a' * 64,
		checkpoint_config=checkpoint_config,
		device='cpu',
	)

	reused = extract_embeddings_from_loaded_model(
		student,
		config,
		refresh_root,
		'a' * 64,
		checkpoint_config=checkpoint_config,
		reuse=True,
		device='cpu',
	)
	assert reused[0].skipped is True

	with pytest.raises(ValueError, match='complete validation'):
		extract_embeddings_from_loaded_model(
			student,
			config,
			refresh_root,
			'b' * 64,
			checkpoint_config=checkpoint_config,
			reuse=True,
			device='cpu',
		)

	(refresh_root / 'refresh_extraction_descriptor.json').unlink()
	with pytest.raises(ValueError, match='incomplete or does not match'):
		extract_embeddings_from_loaded_model(
			student,
			config,
			refresh_root,
			'a' * 64,
			checkpoint_config=checkpoint_config,
			reuse=True,
			device='cpu',
		)


@pytest.mark.parametrize('mode', ['memory', 'memmap'])
@pytest.mark.parametrize('zero_mask_enabled', [False, True])
def test_survey_preprocessing_cache_matches_legacy_windows(
	tmp_path: Path,
	mode: str,
	zero_mask_enabled: bool,  # noqa: FBT001
) -> None:
	zero_mask = {
		'enabled': zero_mask_enabled,
		'zero_atol': 0.0,
		'z_sample_influence_radius': 1,
		'xy_trace_influence_radius': 1,
	}
	agc = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-3,
		'clip_abs': 5.0,
	}
	reference_root = tmp_path / 'reference'
	reference_root.mkdir()
	reference_config = _write_fixture(
		reference_root,
		checkpoint_zero_mask=zero_mask,
		checkpoint_amplitude_agc=agc,
	)
	reference = run_embedding_extraction(reference_config, device='cpu')[0]

	actual_root = tmp_path / 'actual'
	actual_root.mkdir()
	actual_config = _write_fixture(
		actual_root,
		checkpoint_zero_mask=zero_mask,
		checkpoint_amplitude_agc=agc,
	)
	actual_config['embedding']['preprocessing_cache'] = {
		'mode': mode,
		'chunk_size_x': 2,
		'reuse': True,
		'cleanup': False,
	}
	actual = run_embedding_extraction(actual_config, device='cpu')[0]

	np.testing.assert_allclose(
		np.load(actual.embeddings_path),
		np.load(reference.embeddings_path),
		rtol=1.0e-3,
		atol=1.0e-3,
	)
	np.testing.assert_array_equal(
		np.load(actual.valid_tokens_path),
		np.load(reference.valid_tokens_path),
	)
	metadata = json.loads(actual.metadata_path.read_text(encoding='utf-8'))
	assert metadata['preprocessing_cache']['requested_mode'] == mode
	assert metadata['preprocessing_cache']['effective_mode'] == mode
	assert len(metadata['preprocessing_cache']['fingerprint']) == 64


@pytest.mark.parametrize('zero_mask_enabled', [False, True])
@pytest.mark.parametrize('agc_enabled', [False, True])
@pytest.mark.parametrize('zero_atol', [0.0, 1.0])
def test_cached_window_amplitude_and_masks_match_legacy_exactly(
	tmp_path: Path,
	zero_mask_enabled: bool,  # noqa: FBT001
	agc_enabled: bool,  # noqa: FBT001
	zero_atol: float,
) -> None:
	zero_mask = {
		'enabled': zero_mask_enabled,
		'zero_atol': zero_atol,
		'z_sample_influence_radius': 1,
		'xy_trace_influence_radius': 1,
	}
	agc = (
		{
			'enabled': True,
			'mode': 'trace_rms_z',
			'window_z': 3,
			'eps': 1.0e-3,
			'clip_abs': 5.0,
		}
		if agc_enabled
		else {'enabled': False}
	)
	config = _write_fixture(
		tmp_path,
		checkpoint_zero_mask=zero_mask,
		checkpoint_amplitude_agc=agc,
	)
	config['embedding']['preprocessing_cache'] = {
		'mode': 'memory',
		'chunk_size_x': 2,
	}
	manifest = read_manifest_json(Path(config['manifests']['input']))[0]
	payload = load_checkpoint(
		Path(config['embeddings']['checkpoint']),
		map_location='cpu',
	)
	settings = extractor_module.extraction_settings_from_config(
		config,
		checkpoint_config=payload['config'],
	)
	preprocess = extractor_module._amplitude_preprocess_settings(settings)  # noqa: SLF001
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats = load_normalization_stats(
		resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		),
	)
	store = NpyMemmapVolumeStore()
	plan = cache_module.plan_survey_preprocessing_cache(
		amplitude_path=amplitude_path,
		stats=stats,
		preprocess_settings=preprocess,
		cache_settings=settings.preprocessing_cache,
		default_cache_root=tmp_path / 'cache',
	)
	prepared_survey = cache_module.prepare_survey_preprocessing_cache(
		plan=plan,
		amplitude_path=amplitude_path,
		stats=stats,
		preprocess_settings=preprocess,
		cache_settings=settings.preprocessing_cache,
		store=store,
	)
	assert prepared_survey is not None
	windows = list(
		iter_sliding_windows(
			manifest.amplitude.shape_xyz,
			window_size_xyz=settings.window_size_xyz,
			overlap_xyz=settings.overlap_xyz,
			patch_size_xyz=(2, 2, 2),
		),
	)
	assert any(
		any(start + size > shape for start, size, shape in zip(
			window.start_xyz,
			window.size_xyz,
			manifest.amplitude.shape_xyz,
			strict=True,
		))
		for window in windows
	)
	for window in windows:
		request = CropRequest(
			survey_id=manifest.survey_id,
			start_xyz=window.start_xyz,
			size_xyz=window.size_xyz,
		)
		legacy = read_amplitude_crop(
			request=request,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			patch_size_xyz=(2, 2, 2),
			settings=preprocess,
		)
		cached = read_prepared_survey_amplitude_crop(
			request=request,
			normalized_amplitude=prepared_survey.normalized_amplitude,
			zero_like_mask=prepared_survey.zero_like_mask,
			patch_size_xyz=(2, 2, 2),
			settings=preprocess,
		)
		np.testing.assert_allclose(cached.x, legacy.x, rtol=1.0e-6, atol=1.0e-6)
		np.testing.assert_array_equal(
			cached.local_valid_mask,
			legacy.local_valid_mask,
		)
		np.testing.assert_array_equal(
			cached.token_valid_mask,
			legacy.token_valid_mask,
		)


def test_memmap_preprocessing_cache_normalizes_by_chunk_and_reuses(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['preprocessing_cache'] = {
		'mode': 'memmap',
		'chunk_size_x': 2,
		'reuse': True,
		'cleanup': False,
	}
	calls = 0
	original = cache_module._normalize_amplitude_inplace  # noqa: SLF001

	def count_normalization(*args: object, **kwargs: object) -> np.ndarray:
		nonlocal calls
		calls += 1
		return original(*args, **kwargs)

	monkeypatch.setattr(
		'seis_ssl_cluster.data.survey_preprocessing_cache._normalize_amplitude_inplace',
		count_normalization,
	)
	run_embedding_extraction(config, device='cpu')
	assert calls == 3
	run_embedding_extraction(config, device='cpu')
	assert calls == 3

	config['embedding']['preprocessing_cache']['chunk_size_x'] = 4
	run_embedding_extraction(config, device='cpu')
	assert calls == 3

	cache_root = tmp_path / 'embeddings' / '.preprocessing_cache'
	cache_path = next(cache_root.iterdir())
	(cache_path / 'metadata.json').unlink()
	run_embedding_extraction(config, device='cpu')
	assert calls == 5

	config['embedding']['preprocessing_cache']['cleanup'] = True
	run_embedding_extraction(config, device='cpu')
	assert not cache_path.exists()


def test_memmap_preprocessing_cache_removes_interrupted_build(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['preprocessing_cache'] = {
		'mode': 'memmap',
		'cleanup': True,
	}
	manifest = read_manifest_json(Path(config['manifests']['input']))[0]
	payload = load_checkpoint(
		Path(config['embeddings']['checkpoint']),
		map_location='cpu',
	)
	settings = extractor_module.extraction_settings_from_config(
		config,
		checkpoint_config=payload['config'],
	)
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats = load_normalization_stats(
		resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		),
	)
	plan = cache_module.plan_survey_preprocessing_cache(
		amplitude_path=amplitude_path,
		stats=stats,
		preprocess_settings=extractor_module._amplitude_preprocess_settings(  # noqa: SLF001
			settings,
		),
		cache_settings=settings.preprocessing_cache,
		default_cache_root=tmp_path / 'embeddings' / '.preprocessing_cache',
	)
	assert plan.cache_root is not None
	assert plan.fingerprint is not None
	staging = plan.cache_root / f'.{plan.fingerprint}.building-interrupted'
	staging.mkdir(parents=True)
	(staging / 'partial.npy').write_bytes(b'partial')

	run_embedding_extraction(config, device='cpu')

	assert not staging.exists()
	assert not (plan.cache_root / plan.fingerprint).exists()


def test_preprocessing_cache_fingerprint_tracks_source_and_config(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['preprocessing_cache'] = {'mode': 'memory'}
	manifest = read_manifest_json(Path(config['manifests']['input']))[0]
	payload = load_checkpoint(
		Path(config['embeddings']['checkpoint']),
		map_location='cpu',
	)
	settings = extractor_module.extraction_settings_from_config(
		config,
		checkpoint_config=payload['config'],
	)
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats = load_normalization_stats(
		resolve_manifest_path(
			manifest,
			manifest.amplitude.normalization_stats_path,
		),
	)
	preprocess = extractor_module._amplitude_preprocess_settings(settings)  # noqa: SLF001

	def plan(
		current_stats: SurveyNormalizationStats,
		current_preprocess: AmplitudePreprocessSettings = preprocess,
	) -> str:
		value = cache_module.plan_survey_preprocessing_cache(
			amplitude_path=amplitude_path,
			stats=current_stats,
			preprocess_settings=current_preprocess,
			cache_settings=settings.preprocessing_cache,
			default_cache_root=tmp_path / 'cache',
		)
		assert value.fingerprint is not None
		return value.fingerprint

	initial = plan(stats)
	changed_stats = plan(replace(stats, median=stats.median + 1.0))
	changed_finite_check_mode = plan(
		stats,
		replace(preprocess, finite_check_mode='off'),
	)
	original_stat = amplitude_path.stat()
	volume = np.load(amplitude_path)
	volume[1, 1, 1] += 1.0
	replacement_path = amplitude_path.with_name('replacement.npy')
	np.save(replacement_path, volume)
	assert replacement_path.stat().st_size == original_stat.st_size
	os.utime(
		replacement_path,
		ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
	)
	replacement_path.replace(amplitude_path)
	assert amplitude_path.stat().st_size == original_stat.st_size
	assert amplitude_path.stat().st_mtime_ns == original_stat.st_mtime_ns
	changed_source = plan(stats)

	assert changed_stats != initial
	assert changed_finite_check_mode != initial
	assert changed_source != initial


def test_extraction_settings_resolve_average_chunk_size(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['average_chunk_size_x'] = 3
	payload = load_checkpoint(
		Path(config['embeddings']['checkpoint']),
		map_location='cpu',
	)

	settings = extractor_module.extraction_settings_from_config(
		config,
		checkpoint_config=payload['config'],
	)

	assert settings.average_chunk_size_x == 3


@pytest.mark.parametrize(
	('batch_size', 'prefetch_queue_depth'),
	[(1, 0), (2, 1), (5, 3)],
)
def test_batched_prefetch_matches_synchronous_reference(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	batch_size: int,
	prefetch_queue_depth: int,
) -> None:
	reference_root = tmp_path / 'reference'
	reference_root.mkdir()
	reference_config = _write_fixture(reference_root)
	reference_config['embedding']['batch_size'] = 1
	reference_config['embedding']['prefetch_queue_depth'] = 0
	reference = run_embedding_extraction(reference_config, device='cpu')[0]
	reference_embeddings = np.load(reference.embeddings_path)
	reference_valid = np.load(reference.valid_tokens_path)

	actual_root = tmp_path / 'actual'
	actual_root.mkdir()
	config = _write_fixture(actual_root)
	config['embedding']['batch_size'] = batch_size
	config['embedding']['prefetch_queue_depth'] = prefetch_queue_depth
	encode_batch_sizes: list[int] = []
	original_encode_tokens = AmplitudeMAE3D.encode_tokens

	def wrapped_encode_tokens(
		self: AmplitudeMAE3D,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int] | None]:
		encode_batch_sizes.append(int(x.shape[0]))
		return original_encode_tokens(self, x, valid_mask=valid_mask)

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', wrapped_encode_tokens)
	actual = run_embedding_extraction(config, device='cpu')[0]

	np.testing.assert_array_equal(np.load(actual.embeddings_path), reference_embeddings)
	np.testing.assert_array_equal(np.load(actual.valid_tokens_path), reference_valid)
	assert sum(encode_batch_sizes) == 12
	assert len(encode_batch_sizes) == math.ceil(12 / batch_size)
	assert encode_batch_sizes[-1] == (12 % batch_size or batch_size)


@pytest.mark.parametrize('prefetch_queue_depth', [0, 2])
def test_batched_prefetch_skips_all_invalid_windows(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	prefetch_queue_depth: int,
) -> None:
	config = _write_fixture(
		tmp_path,
		checkpoint_zero_mask={
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 0,
		},
	)
	manifest = read_manifest_json(Path(config['manifests']['input']))[0]
	np.save(manifest.amplitude.path, np.zeros(manifest.amplitude.shape_xyz, np.float32))
	config['embedding']['min_token_valid_fraction'] = 1.0
	config['embedding']['prefetch_queue_depth'] = prefetch_queue_depth
	encode_calls = 0

	def unexpected_encode(*_args: object, **_kwargs: object) -> None:
		nonlocal encode_calls
		encode_calls += 1

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', unexpected_encode)
	result = run_embedding_extraction(config, device='cpu')[0]

	assert encode_calls == 0
	assert not np.load(result.valid_tokens_path).any()
	assert not np.load(result.embeddings_path).any()


def test_prefetch_producer_exception_stops_worker(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['prefetch_queue_depth'] = 1

	def failing_read(*_args: object, **_kwargs: object) -> None:
		msg = 'injected producer failure'
		raise RuntimeError(msg)

	monkeypatch.setattr(extractor_module, '_read_window', failing_read)
	with pytest.raises(RuntimeError, match='injected producer failure'):
		run_embedding_extraction(config, device='cpu')

	assert all(
		thread.name != 'embedding-prefetch' for thread in threading.enumerate()
	)


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_prefetch_consumer_exception_stops_worker(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	error_type: type[BaseException],
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['prefetch_queue_depth'] = 2

	def failing_encode(*_args: object, **_kwargs: object) -> None:
		msg = 'injected consumer failure'
		raise error_type(msg)

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', failing_encode)
	with pytest.raises(error_type, match='injected consumer failure'):
		run_embedding_extraction(config, device='cpu')

	assert all(
		thread.name != 'embedding-prefetch' for thread in threading.enumerate()
	)


@pytest.mark.parametrize('prefetch_queue_depth', [0, 1])
def test_prefetch_pipeline_records_stages_and_uses_inference_mode(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	prefetch_queue_depth: int,
) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['prefetch_queue_depth'] = prefetch_queue_depth
	config['embedding']['stage_timing'] = True
	inference_modes: list[bool] = []
	original_encode_tokens = AmplitudeMAE3D.encode_tokens

	def wrapped_encode_tokens(
		self: AmplitudeMAE3D,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int] | None]:
		inference_modes.append(torch.is_inference_mode_enabled())
		return original_encode_tokens(self, x, valid_mask=valid_mask)

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', wrapped_encode_tokens)
	run_embedding_extraction(config, device='cpu')

	assert inference_modes
	assert all(inference_modes)
	timings = json.loads(
		(tmp_path / 'embeddings' / 'stage_timings.json').read_text(
			encoding='utf-8',
		),
	)
	expected_stages = {
		'd2h',
		'encode',
		'h2d',
		'merge_write',
		'read_preprocess',
	}
	if prefetch_queue_depth > 0:
		expected_stages.add('queue_wait')
	assert set(timings['stages']) == expected_stages


def test_embedding_auto_amp_queries_configured_indexed_cuda_device(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	queried_devices: list[torch.device] = []

	def cuda_device(device: torch.device) -> nullcontext[None]:
		queried_devices.append(device)
		return nullcontext()

	monkeypatch.setattr(torch.cuda, 'device', cuda_device)
	monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
	settings = SimpleNamespace(amp=True, amp_dtype='auto')

	dtype = extractor_module._resolve_autocast_dtype(  # noqa: SLF001
		settings,
		device=torch.device('cuda:1'),
	)

	assert dtype == torch.bfloat16
	assert queried_devices == [torch.device('cuda:1')]


def test_reduce_valid_mask_to_tokens_legacy_import_path_is_shared() -> None:
	assert (
		extractor_module.reduce_valid_mask_to_tokens
		is shared_reduce_valid_mask_to_tokens
	)
	assert package_reduce_valid_mask_to_tokens is shared_reduce_valid_mask_to_tokens


def test_extractor_read_window_matches_shared_preprocessing(
	tmp_path: Path,
) -> None:
	fixture = write_window_preprocessing_fixture(tmp_path)
	expected = read_fixture_crop(fixture, min_token_valid_fraction=0.5)

	window, x, token_valid_mask = extractor_module._read_window(  # noqa: SLF001
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
			finite_check_mode='strict',
		),
		patch_size_xyz=PATCH_SIZE_XYZ,
	)

	assert window == fixture.window
	np.testing.assert_allclose(x, expected.x, rtol=1.0e-6)
	np.testing.assert_array_equal(token_valid_mask, expected.token_valid_mask)
	np.testing.assert_array_equal(x[0][~expected.local_valid_mask], 0.0)


def test_embedding_extraction_valid_tokens_match_shared_preprocessing(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)

	result = run_embedding_extraction(config, device='cpu')[0]

	actual = np.load(result.valid_tokens_path)
	expected = _expected_valid_tokens_from_shared_preprocessing(config)
	np.testing.assert_array_equal(actual, expected)


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


@pytest.mark.parametrize(
	('key', 'value'),
	[('amp', True), ('amp_dtype', 'float16')],
)
def test_embedding_extraction_skip_existing_rejects_precision_change(
	tmp_path: Path,
	key: str,
	value: object,
) -> None:
	config = _write_fixture(tmp_path)
	run_embedding_extraction(config, device='cpu')
	config['embedding'][key] = value

	with pytest.raises(ValueError, match='metadata does not match'):
		run_embedding_extraction(config, skip_existing=True, device='cpu')


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
	)
	config['embedding']['min_token_valid_fraction'] = 1.0

	result = run_embedding_extraction(config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['zero_mask'] == zero_mask
	valid_tokens = np.load(result.valid_tokens_path)
	assert not valid_tokens[0, 0, :].any()
	assert valid_tokens[1, 1, 1]


def test_embedding_extraction_uses_checkpoint_amplitude_agc_settings(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	amplitude_agc = {
		'enabled': True,
		'mode': 'trace_rms_z',
		'window_z': 3,
		'eps': 1.0e-6,
		'clip_abs': 0.75,
	}
	config = _write_fixture(tmp_path, checkpoint_amplitude_agc=amplitude_agc)
	captured_inputs: list[np.ndarray] = []
	original_encode_tokens = AmplitudeMAE3D.encode_tokens

	def wrapped_encode_tokens(
		self: AmplitudeMAE3D,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int] | None]:
		captured_inputs.append(x.detach().to(dtype=torch.float32).cpu().numpy())
		return original_encode_tokens(self, x, valid_mask=valid_mask)

	monkeypatch.setattr(AmplitudeMAE3D, 'encode_tokens', wrapped_encode_tokens)

	result = run_embedding_extraction(config, device='cpu')[0]

	assert captured_inputs
	assert np.max(np.abs(captured_inputs[0])) <= amplitude_agc['clip_abs'] + 1.0e-6
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['amplitude_agc'] == amplitude_agc
	assert metadata['preprocessing'] == {
		'normalized_clip_abs': None,
		'amplitude_agc': amplitude_agc,
		'finite_check_mode': 'strict',
	}


def test_embedding_extraction_defaults_legacy_finite_check_mode_to_strict(
	tmp_path: Path,
) -> None:
	def remove_finite_check_mode(checkpoint_config: dict[str, object]) -> None:
		data = checkpoint_config['data']
		assert isinstance(data, dict)
		data.pop('finite_check_mode')

	config = _write_fixture(
		tmp_path,
		checkpoint_config_modifier=remove_finite_check_mode,
	)

	result = run_embedding_extraction(config, device='cpu')[0]
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))

	assert metadata['finite_check_mode'] == 'strict'
	assert metadata['preprocessing']['finite_check_mode'] == 'strict'


def test_embedding_extraction_uses_explicit_checkpoint_config_override(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	payload = load_checkpoint(config['embeddings']['checkpoint'], map_location='cpu')
	serialized_config = payload['config']
	assert isinstance(serialized_config, dict)
	checkpoint_config_override = deepcopy(serialized_config)
	override_data = checkpoint_config_override['data']
	assert isinstance(override_data, dict)
	override_data['finite_check_mode'] = 'off'

	result = run_embedding_extraction(
		config,
		device='cpu',
		checkpoint_config_override=checkpoint_config_override,
	)[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['finite_check_mode'] == 'off'
	assert metadata['preprocessing']['finite_check_mode'] == 'off'
	serialized_data = serialized_config['data']
	assert isinstance(serialized_data, dict)
	assert serialized_data['finite_check_mode'] == 'strict'


def test_embedding_extraction_rejects_checkpoint_data_zero_mask_only(
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
		checkpoint_config_modifier=_move_zero_mask_to_data,
	)
	config['embedding']['min_token_valid_fraction'] = 1.0

	with pytest.raises(ValueError, match=r'missing resolved section.*zero_mask'):
		run_embedding_extraction(config, device='cpu')


@pytest.mark.parametrize(
	('mutate_checkpoint_config', 'error'),
	[
		(lambda checkpoint_config: checkpoint_config.pop('data'), 'data'),
		(
			lambda checkpoint_config: checkpoint_config['loss'].pop(
				'valid_mask_mode',
			),
			'loss.*valid_mask_mode',
		),
		(
			lambda checkpoint_config: checkpoint_config['loss'].pop(
				'reconstruction',
			),
			'loss.*reconstruction',
		),
		(
			lambda checkpoint_config: checkpoint_config['manifests'].pop(
				'train_path_list',
			),
			'manifests.*train_path_list',
		),
	],
)
def test_embedding_extraction_rejects_incomplete_checkpoint_resolved_config(
	tmp_path: Path,
	mutate_checkpoint_config: Callable[[dict[str, object]], object],
	error: str,
) -> None:
	config = _write_fixture(
		tmp_path,
		checkpoint_config_modifier=lambda checkpoint_config: mutate_checkpoint_config(
			checkpoint_config,
		),
	)

	with pytest.raises(ValueError, match=error):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_accepts_mse_checkpoint_without_huber_delta(
	tmp_path: Path,
) -> None:
	def use_mse(checkpoint_config: dict[str, object]) -> None:
		loss = checkpoint_config['loss']
		assert isinstance(loss, dict)
		loss['reconstruction'] = 'mse'
		loss.pop('huber_delta')

	config = _write_fixture(tmp_path, checkpoint_config_modifier=use_mse)

	result = run_embedding_extraction(config, device='cpu')[0]

	assert result.metadata_path.is_file()


def test_embedding_extraction_rejects_mse_checkpoint_huber_delta(
	tmp_path: Path,
) -> None:
	def use_mse_with_huber_delta(checkpoint_config: dict[str, object]) -> None:
		loss = checkpoint_config['loss']
		assert isinstance(loss, dict)
		loss['reconstruction'] = 'mse'

	config = _write_fixture(
		tmp_path,
		checkpoint_config_modifier=use_mse_with_huber_delta,
	)

	with pytest.raises(ValueError, match=r'loss\.huber_delta.*huber'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_rejects_extraction_zero_mask_section(
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

	with pytest.raises(ValueError, match=r'checkpoint-owned.*zero_mask'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_rejects_integer_output_dtype(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['output_dtype'] = 'int16'

	with pytest.raises(ValueError, match='float16 or float32'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_requires_explicit_embedding_section(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	del config['embedding']

	with pytest.raises(TypeError, match='embedding must be a mapping'):
		run_embedding_extraction(config, device='cpu')


def test_embedding_extraction_allows_zero_overlap(tmp_path: Path) -> None:
	config = _write_fixture(tmp_path)
	config['embedding']['overlap'] = [0, 0, 0]

	result = run_embedding_extraction(config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['overlap'] == [0, 0, 0]


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


def test_embedding_extraction_accepts_patch_zscore_checkpoint_metadata(
	tmp_path: Path,
) -> None:
	def use_patch_zscore(checkpoint_config: dict[str, object]) -> None:
		loss = checkpoint_config['loss']
		assert isinstance(loss, dict)
		loss['reconstruction'] = 'mse'
		loss.pop('huber_delta')
		loss['gradient_weight'] = 0.0
		loss['target_normalization'] = {
			'mode': 'patch_zscore',
			'eps': 1.0e-6,
			'min_std': 0.05,
		}

	config = _write_fixture(tmp_path, checkpoint_config_modifier=use_patch_zscore)
	result = run_embedding_extraction(config, device='cpu')[0]
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))

	assert metadata['pretraining_objective'] == {
		'reconstruction': 'mse',
		'gradient_weight': 0.0,
		'target_normalization': {
			'mode': 'patch_zscore',
			'eps': 1.0e-6,
			'min_std': 0.05,
		},
	}


def _write_fixture(  # noqa: PLR0913
	tmp_path: Path,
	*,
	checkpoint_dtype: torch.dtype = torch.float32,
	checkpoint_zero_mask: dict[str, object] | None = None,
	checkpoint_amplitude_agc: dict[str, object] | None = None,
	checkpoint_config_modifier: Callable[[dict[str, object]], object] | None = None,
	survey_count: int = 1,
) -> dict[str, object]:
	if checkpoint_zero_mask is None:
		checkpoint_zero_mask = {
			'enabled': False,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 16,
			'xy_trace_influence_radius': 1,
		}
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
	path_list = tmp_path / 'train_npy_paths.txt'
	path_list.write_text(
		'\n'.join(str(manifest.amplitude.path) for manifest in manifests) + '\n',
		encoding='utf-8',
	)
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
	checkpoint_config: dict[str, object] = {
		'stage': 'train_amp_mae',
		'paths': {'output_root': str(tmp_path / 'run')},
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
			'min_valid_fraction': 0.1,
			'max_resample_attempts': 16,
			'finite_check_mode': 'strict',
		},
		'model': model_config,
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'loss': {
			'reconstruction': 'huber',
			'huber_delta': 1.0,
			'gradient_weight': 0.05,
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
	}
	if checkpoint_amplitude_agc is not None:
		checkpoint_config['data']['amplitude_agc'] = checkpoint_amplitude_agc
	checkpoint_config['zero_mask'] = checkpoint_zero_mask
	if checkpoint_config_modifier is not None:
		checkpoint_config_modifier(checkpoint_config)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'config': checkpoint_config,
		},
		checkpoint_path,
	)
	config: dict[str, object] = {
		'paths': {
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
	}
	return config


def _make_fixture_checkpoint_barlow(config: dict[str, object]) -> None:
	embeddings = config['embeddings']
	assert isinstance(embeddings, dict)
	checkpoint_path = Path(embeddings['checkpoint'])
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	mae_config = payload['config']
	assert isinstance(mae_config, dict)
	model_config = mae_config['model']
	assert isinstance(model_config, dict)
	manifests = mae_config['manifests']
	assert isinstance(manifests, dict)
	zero_mask = mae_config['zero_mask']
	assert isinstance(zero_mask, dict)
	barlow_config = resolve_barlow_twins_training_config(
		{
			'paths': {
				'artifact_root': str(checkpoint_path.parent / 'artifacts'),
				'output_root': str(checkpoint_path.parent / 'artifacts' / 'barlow'),
			},
			'manifests': dict(manifests),
			'data': {'local_crop_size': [4, 4, 4]},
			'zero_mask': dict(zero_mask),
			'model': {
				key: model_config[key]
				for key in (
					'patch_size',
					'encoder_dim',
					'encoder_depth',
					'encoder_heads',
					'decoder_dim',
					'decoder_depth',
					'decoder_heads',
				)
			},
			'barlow_twins': {'projector_dim': 8},
			'train': {
				'batch_size': 2,
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
	)
	mae = build_model_from_checkpoint_payload(payload)
	projector = torch.nn.Linear(mae.encoder_dim, 8)
	optimizer = torch.optim.AdamW(
		[
			*mae.patch_projection.parameters(),
			*mae.encoder.parameters(),
			*projector.parameters(),
		],
		lr=1.0e-4,
	)
	save_barlow_twins_checkpoint(
		checkpoint_path,
		backbone=mae,
		projector=projector,
		optimizer=optimizer,
		epoch=1,
		global_step=1,
		config=barlow_config,
		metrics={'train_loss': 0.5},
		amp_enabled=False,
		scaler=None,
		scaler_required=False,
		dataset_epoch=1,
		completed_epoch=True,
	)


def _expected_valid_tokens_from_shared_preprocessing(
	config: dict[str, object],
) -> np.ndarray:
	manifests_config = config['manifests']
	embeddings_config = config['embeddings']
	assert isinstance(manifests_config, dict)
	assert isinstance(embeddings_config, dict)
	manifest_path = Path(manifests_config['input'])
	manifest = read_manifest_json(manifest_path)[0]
	payload = load_checkpoint(
		Path(embeddings_config['checkpoint']),
		map_location='cpu',
	)
	checkpoint_config = payload['config']
	assert isinstance(checkpoint_config, dict)
	settings = extractor_module.extraction_settings_from_config(
		config,
		checkpoint_config=checkpoint_config,
	)
	model_config = checkpoint_config['model']
	assert isinstance(model_config, dict)
	patch_size = tuple(model_config['patch_size'])
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats_path = resolve_manifest_path(
		manifest,
		manifest.amplitude.normalization_stats_path,
	)
	stats = load_normalization_stats(stats_path)
	merger = EmbeddingMerger(
		token_grid_shape_xyz=token_grid_shape_xyz(
			manifest.amplitude.shape_xyz,
			patch_size,
		),
		embedding_dim=1,
	)
	preprocess_settings = AmplitudePreprocessSettings(
		zero_mask=settings.zero_mask,
		normalized_clip_abs=settings.normalized_clip_abs,
		amplitude_agc=settings.amplitude_agc,
		min_token_valid_fraction=settings.min_token_valid_fraction,
		finite_check_mode=settings.finite_check_mode,
	)
	store = NpyMemmapVolumeStore()
	for window in iter_sliding_windows(
		manifest.amplitude.shape_xyz,
		window_size_xyz=settings.window_size_xyz,
		overlap_xyz=settings.overlap_xyz,
		patch_size_xyz=patch_size,
	):
		prepared = read_amplitude_crop(
			request=CropRequest(
				survey_id=manifest.survey_id,
				start_xyz=window.start_xyz,
				size_xyz=window.size_xyz,
			),
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			patch_size_xyz=patch_size,
			settings=preprocess_settings,
		)
		merger.add_window(
			window,
			patch_size_xyz=patch_size,
			token_embeddings=np.ones((*prepared.token_valid_mask.shape, 1)),
			token_valid_mask=prepared.token_valid_mask,
		)
	return merger.finalize(output_dtype=np.float32)[1]


def _move_zero_mask_to_data(checkpoint_config: dict[str, object]) -> None:
	zero_mask = checkpoint_config.pop('zero_mask')
	data = checkpoint_config['data']
	assert isinstance(data, dict)
	data['zero_mask'] = zero_mask
