from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	NopimsAmplitudeCropDataset,
	ZeroMaskConfig,
	load_normalization_stats,
	read_manifest_json,
)
from seis_ssl_cluster.volve import (
	VOLVE_CANONICAL_RELATIVE_ROOT,
	VolveCanonicalIdentity,
	VolveCanonicalInputConfig,
	prepare_volve_canonical_inputs,
	resolve_volve_canonical_input_config,
)


def test_prepare_registers_read_only_canonical_inputs(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	before = _directory_snapshot(config.paths.canonical_root)

	result = prepare_volve_canonical_inputs(config)

	assert result.action == 'WROTE'
	assert _directory_snapshot(config.paths.canonical_root) == before
	assert {path.name for path in config.paths.output_dir.iterdir()} == {
		'volve_amplitude_manifest.json',
		'volve_npy_paths.txt',
		'volve.normalization_stats.json',
		'volve_canonical_input_metadata.json',
	}
	manifest = read_manifest_json(config.paths.manifest_path)[0]
	assert manifest.root == config.paths.canonical_root.resolve()
	assert manifest.amplitude.path == Path('amplitude.npy')
	assert manifest.amplitude.valid_mask_path == Path('valid_trace_mask.npy')
	assert manifest.amplitude.normalization_stats_path == (
		config.paths.normalization_stats_path
	)
	assert config.paths.path_list_path.read_text(encoding='utf-8') == (
		f'{config.paths.canonical_root.resolve() / "amplitude.npy"}\n'
	)

	stats = load_normalization_stats(config.paths.normalization_stats_path)
	assert stats.clip_low_percentile == 1.0
	assert stats.clip_high_percentile == 99.0
	assert stats.clip_low == -4.0
	assert stats.clip_high == 5.0
	assert stats.median == 0.25
	assert stats.iqr == 2.0
	assert stats.eps == 1.0e-6
	metadata = _read_json(config.paths.metadata_path)
	assert metadata['normalization_provenance']['sample_policy'] == (
		'synthetic deterministic sample'
	)
	assert len(metadata['normalization_provenance']['source_stats_sha256']) == 64
	assert metadata['provenance']['amplitude']['path'] == str(
		config.paths.canonical_root.resolve() / 'amplitude.npy'
	)
	assert set(metadata['provenance']['public_inputs']) == {
		'amplitude.npy',
		'valid_trace_mask.npy',
		'inline_values.npy',
		'crossline_values.npy',
		'time_ms.npy',
		'canonical_volume_manifest.json',
		'normalization_stats.json',
		'trace_parity.json',
	}
	assert 'canonical_root' not in metadata['scientific_identity']


def test_dry_run_validates_without_creating_outputs(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	before = _directory_snapshot(config.volve_root)

	result = prepare_volve_canonical_inputs(config, dry_run=True)

	assert result.action == 'DRY_RUN'
	assert not config.paths.output_dir.exists()
	assert _directory_snapshot(config.volve_root) == before


def test_registered_manifest_reads_finite_crop_with_missing_trace_masked(
	tmp_path: Path,
) -> None:
	config = _synthetic_config(tmp_path)
	prepare_volve_canonical_inputs(config)
	manifest = read_manifest_json(config.paths.manifest_path)[0]
	dataset = NopimsAmplitudeCropDataset(
		[manifest],
		local_crop_size_xyz=config.identity.shape_xyz,
		patch_size_xyz=(1, 1, 1),
		zero_mask=ZeroMaskConfig(enabled=False),
	)

	sample = dataset[0]

	assert np.isfinite(sample['x']).all()
	assert not sample['local_valid_mask'][1, 2, :].any()
	np.testing.assert_array_equal(sample['x'][0, 1, 2, :], 0.0)


def test_only_missing_reuses_only_complete_matching_outputs(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	first = prepare_volve_canonical_inputs(config)

	reused = prepare_volve_canonical_inputs(config, only_missing=True)

	assert first.action == 'WROTE'
	assert reused.action == 'REUSE'
	assert reused.scientific_identity_sha256 == first.scientific_identity_sha256

	metadata = _read_json(config.paths.metadata_path)
	metadata['scientific_identity']['dtype'] = 'float64'
	_write_json(config.paths.metadata_path, metadata)
	with pytest.raises(ValueError, match='metadata identity'):
		prepare_volve_canonical_inputs(config, only_missing=True)


def test_only_missing_rejects_incomplete_outputs(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	config.paths.output_dir.mkdir(parents=True)
	config.paths.manifest_path.write_text('[]\n', encoding='utf-8')

	with pytest.raises(FileExistsError, match='incomplete'):
		prepare_volve_canonical_inputs(config, only_missing=True)


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('schema_version', 2),
		('status', 'FAIL'),
		('dataset_id', 'another_dataset'),
	],
)
def test_rejects_canonical_manifest_identity_mismatch(
	tmp_path: Path,
	field: str,
	value: object,
) -> None:
	config = _synthetic_config(tmp_path)
	path = config.paths.canonical_root / 'canonical_volume_manifest.json'
	payload = _read_json(path)
	payload[field] = value
	_write_json(path, payload)

	with pytest.raises(ValueError, match=field):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_amplitude_hash_mismatch(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	path = config.paths.canonical_root / 'amplitude.npy'
	array = np.load(path)
	array[0, 0, 0] += 1.0
	np.save(path, array)

	with pytest.raises(ValueError, match='SHA-256 mismatch'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_amplitude_shape_mismatch(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	path = config.paths.canonical_root / 'amplitude.npy'
	np.save(path, np.ones((2, 3, 5), dtype=np.float32))

	with pytest.raises(ValueError, match='amplitude shape'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_amplitude_dtype_mismatch(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	path = config.paths.canonical_root / 'amplitude.npy'
	np.save(path, np.ones(config.identity.shape_xyz, dtype=np.float64))

	with pytest.raises(TypeError, match='amplitude dtype'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_valid_mask_shape_and_dtype(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	mask_path = config.paths.canonical_root / 'valid_trace_mask.npy'
	np.save(mask_path, np.ones((2, 2), dtype=np.uint8))

	with pytest.raises(ValueError, match='valid mask shape'):
		prepare_volve_canonical_inputs(config, dry_run=True)

	np.save(mask_path, np.ones(config.identity.shape_xyz[:2], dtype=np.uint8))
	with pytest.raises(TypeError, match='valid mask dtype'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_time_axis_mismatch(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	time_path = config.paths.canonical_root / 'time_ms.npy'
	np.save(time_path, np.array([4.0, 8.0, 13.0, 16.0], dtype=np.float32))

	with pytest.raises(ValueError, match='time axis'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_nonfinite_valid_trace(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	amplitude_path = config.paths.canonical_root / 'amplitude.npy'
	amplitude = np.load(amplitude_path)
	amplitude[0, 0, 1] = np.nan
	config = _refresh_amplitude_identity(config, amplitude)

	with pytest.raises(ValueError, match='valid trace'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_rejects_non_nan_invalid_trace(tmp_path: Path) -> None:
	config = _synthetic_config(tmp_path)
	amplitude_path = config.paths.canonical_root / 'amplitude.npy'
	amplitude = np.load(amplitude_path)
	amplitude[1, 2, :] = 0.0
	config = _refresh_amplitude_identity(config, amplitude)

	with pytest.raises(ValueError, match='invalid traces'):
		prepare_volve_canonical_inputs(config, dry_run=True)


def test_config_resolver_and_proc_entrypoint_contract(tmp_path: Path) -> None:
	config = resolve_volve_canonical_input_config(
		{
			'paths': {
				'volve_root': str(tmp_path / 'public'),
				'artifact_root': str(tmp_path / 'artifacts'),
			},
		},
	)
	module = importlib.import_module(
		'proc.seis_ssl_cluster.prepare_volve_canonical_inputs'
	)
	help_text = module.build_parser().format_help()

	assert config.paths.canonical_root == (
		tmp_path / 'public' / VOLVE_CANONICAL_RELATIVE_ROOT
	)
	assert '--dry-run' in help_text
	assert '--only-missing' in help_text
	assert callable(module.main)


def _synthetic_config(tmp_path: Path) -> VolveCanonicalInputConfig:
	volve_root = tmp_path / 'public' / 'volve'
	artifact_root = tmp_path / 'artifacts'
	canonical_root = volve_root / VOLVE_CANONICAL_RELATIVE_ROOT
	canonical_root.mkdir(parents=True)
	shape_xyz = (2, 3, 4)
	amplitude = np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(shape_xyz)
	valid_mask = np.ones(shape_xyz[:2], dtype=bool)
	valid_mask[1, 2] = False
	amplitude[~valid_mask] = np.nan
	amplitude_path = canonical_root / 'amplitude.npy'
	np.save(amplitude_path, amplitude)
	np.save(canonical_root / 'valid_trace_mask.npy', valid_mask)
	np.save(canonical_root / 'inline_values.npy', np.array([100, 101]))
	np.save(canonical_root / 'crossline_values.npy', np.array([200, 201, 202]))
	np.save(
		canonical_root / 'time_ms.npy',
		np.array([4.0, 8.0, 12.0, 16.0], dtype=np.float32),
	)
	stats = {
		'schema_version': 1,
		'status': 'PASS',
		'deterministic_sample': {
			'policy': 'synthetic deterministic sample',
			'value_quantiles': {
				'p1': -4.0,
				'p25': -1.0,
				'p50': 0.25,
				'p75': 1.0,
				'p99': 5.0,
			},
		},
	}
	stats_path = canonical_root / 'normalization_stats.json'
	_write_json(stats_path, stats)
	parity = {
		'schema_version': 1,
		'status': 'PASS',
		'all_exact': True,
		'full_volume_checks': {
			'valid_finite_voxel_count': 5 * shape_xyz[2],
			'missing_nan_voxel_count': shape_xyz[2],
		},
	}
	parity_path = canonical_root / 'trace_parity.json'
	_write_json(parity_path, parity)
	amplitude_hash = _sha256(amplitude_path)
	identity = VolveCanonicalIdentity(
		dataset_id='synthetic_volve',
		survey_id='synthetic_survey',
		shape_xyz=shape_xyz,
		dtype='float32',
		valid_trace_count=5,
		missing_trace_count=1,
		source_segy_sha256='a' * 64,
		amplitude_sha256=amplitude_hash,
		amplitude_size_bytes=amplitude_path.stat().st_size,
	)
	canonical_manifest = {
		'schema_version': 1,
		'status': 'PASS',
		'dataset_id': identity.dataset_id,
		'shape': list(shape_xyz),
		'axis_order': ['inline', 'crossline', 'twt'],
		'dtype': 'float32',
		'geometry': {
			'valid_trace_count': 5,
			'missing_trace_count': 1,
		},
		'sources': {'segy': {'sha256': identity.source_segy_sha256}},
		'artifacts': {
			'amplitude.npy': _artifact_record(amplitude_path),
			'valid_trace_mask.npy': _artifact_record(
				canonical_root / 'valid_trace_mask.npy'
			),
			'inline_values.npy': _artifact_record(
				canonical_root / 'inline_values.npy'
			),
			'crossline_values.npy': _artifact_record(
				canonical_root / 'crossline_values.npy'
			),
			'time_ms.npy': _artifact_record(canonical_root / 'time_ms.npy'),
			'normalization_stats.json': _artifact_record(stats_path),
			'trace_parity.json': _artifact_record(parity_path),
		},
	}
	_write_json(canonical_root / 'canonical_volume_manifest.json', canonical_manifest)
	return VolveCanonicalInputConfig(
		volve_root=volve_root.resolve(),
		artifact_root=artifact_root.resolve(),
		identity=identity,
	)


def _refresh_amplitude_identity(
	config: VolveCanonicalInputConfig,
	amplitude: np.ndarray,
) -> VolveCanonicalInputConfig:
	path = config.paths.canonical_root / 'amplitude.npy'
	np.save(path, amplitude)
	digest = _sha256(path)
	manifest_path = config.paths.canonical_root / 'canonical_volume_manifest.json'
	manifest = _read_json(manifest_path)
	manifest['artifacts']['amplitude.npy'] = _artifact_record(path)
	_write_json(manifest_path, manifest)
	return replace(
		config,
		identity=replace(
			config.identity,
			amplitude_sha256=digest,
			amplitude_size_bytes=path.stat().st_size,
		),
	)


def _artifact_record(path: Path) -> dict[str, object]:
	return {
		'relative_path': path.name,
		'size_bytes': path.stat().st_size,
		'sha256': _sha256(path),
	}


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
	return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _directory_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path.relative_to(root)): _sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}
