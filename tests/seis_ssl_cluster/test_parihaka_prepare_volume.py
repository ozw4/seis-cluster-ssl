# ruff: noqa: CPY001

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import seis_ssl_cluster.parihaka.prepare_volume as prepare_module
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.data import load_normalization_stats, read_manifest_json
from seis_ssl_cluster.parihaka import (
	PARIHAKA_AMPLITUDE_NPY_NAME,
	PARIHAKA_MANIFEST_NAME,
	PARIHAKA_METADATA_NAME,
	PARIHAKA_NORMALIZATION_STATS_NAME,
	PARIHAKA_PATH_LIST_NAME,
	ArrayStatistics,
	ParihakaPrepareConversionConfig,
	ParihakaPrepareDatasetConfig,
	ParihakaPrepareInputPaths,
	ParihakaPrepareNormalizationConfig,
	ParihakaPrepareOutputPaths,
	ParihakaPrepareRootPaths,
	ParihakaPrepareSourceConfig,
	ParihakaPrepareVolumeConfig,
	inspect_parihaka_preparation,
	parihaka_prepare_volume_config_from_mapping,
	prepare_parihaka_volume,
)

if TYPE_CHECKING:
	from collections.abc import Callable

PRODUCTION_CONFIG = Path(
	'experiments/parihaka/facies_benchmark_v1/10_prepare/'
	'01_prepare_parihaka_volume.yaml',
)


def test_prepare_parihaka_volume_converts_every_coordinate_and_writes_contract(
	tmp_path: Path,
) -> None:
	source = np.asfortranarray(
		(np.arange(5 * 3 * 4, dtype=np.float32) - np.float32(19.5)).reshape(5, 3, 4),
	)
	config = _fixture_config(tmp_path, source=source, chunk_size_z=2)

	result = prepare_parihaka_volume(config)
	actual = np.load(result.amplitude_npy, mmap_mode='r', allow_pickle=False)
	expected = source.transpose(1, 2, 0)

	assert isinstance(actual, np.memmap)
	assert actual.flags.c_contiguous
	assert not actual.flags.f_contiguous
	assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
	assert result.shape_xyz == (3, 4, 5)
	assert result.dtype == 'float32'
	assert result.order == 'C'
	assert result.source_statistics == result.output_statistics

	manifests = read_manifest_json(result.manifest)
	assert len(manifests) == 1
	assert manifests[0].survey_id == 'parihaka'
	assert manifests[0].amplitude.path == result.amplitude_npy
	assert manifests[0].amplitude.shape_xyz == (3, 4, 5)
	assert manifests[0].amplitude.grid_order == ('x', 'y', 'z')
	assert manifests[0].amplitude.normalization_stats_path == result.normalization_stats
	assert result.path_list.read_text(encoding='utf-8') == f'{result.amplitude_npy}\n'
	stats = load_normalization_stats(result.normalization_stats)
	assert stats.source_path == result.amplitude_npy
	assert stats.survey_id == 'parihaka'

	metadata = json.loads(result.metadata.read_text(encoding='utf-8'))
	assert metadata['artifact_type'] == 'parihaka_amplitude_preparation'
	assert metadata['schema_version'] == 1
	assert metadata['status'] == 'complete'
	assert metadata['conversion']['verification'] == 'full_chunkwise_bitwise'
	assert metadata['conversion']['chunk_size_z'] == 2
	assert metadata['source']['npz_sha256'] == _sha256(config.inputs.amplitude_npz)
	assert metadata['outputs']['amplitude_npy']['sha256'] == _sha256(
		result.amplitude_npy,
	)
	for key, path in (
		('manifest', result.manifest),
		('path_list', result.path_list),
		('normalization_stats', result.normalization_stats),
	):
		assert metadata['outputs'][key]['path'] == str(path)
		assert metadata['outputs'][key]['sha256'] == _sha256(path)
	assert not list(config.outputs.data_dir.glob('.parihaka_prepare_staging-*'))


def test_parihaka_dry_run_reads_header_without_creating_outputs(tmp_path: Path) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source, chunk_size_z=2)

	inspection = inspect_parihaka_preparation(config)

	assert inspection.shape_zxy == source.shape
	assert inspection.dtype == 'float32'
	assert inspection.fortran_order is True
	assert not config.outputs.data_dir.exists()


@pytest.mark.parametrize(
	('members', 'match'),
	[
		([], 'keys must be exactly'),
		([('other.npy', 'valid')], 'keys must be exactly'),
		([('data.npy', 'valid'), ('other.npy', 'valid')], 'keys must be exactly'),
		([('data.npy', 'valid'), ('data.npy', 'valid')], 'duplicate'),
		([('../data.npy', 'valid')], 'unsafe'),
		([('/data.npy', 'valid')], 'unsafe'),
	],
)
def test_parihaka_source_rejects_invalid_zip_inventory(
	tmp_path: Path,
	members: list[tuple[str, str]],
	match: str,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	payload = _npy_bytes(source)
	with zipfile.ZipFile(config.inputs.amplitude_npz, 'w') as archive:
		for name, _kind in members:
			archive.writestr(name, payload)

	with pytest.raises(ValueError, match=match):
		inspect_parihaka_preparation(config)


@pytest.mark.parametrize(
	('mutator', 'match'),
	[
		(lambda array: np.asfortranarray(array[:, :, :3]), 'shape'),
		(lambda array: np.asfortranarray(array.astype(np.float64)), 'dtype'),
		(np.ascontiguousarray, 'fortran_order'),
		(lambda array: np.asfortranarray(array.astype(object)), 'object dtype'),
	],
)
def test_parihaka_source_rejects_wrong_npy_header(
	tmp_path: Path,
	mutator: Callable[[np.ndarray], np.ndarray],
	match: str,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	_write_npz(config.inputs.amplitude_npz, mutator(source))

	with pytest.raises((TypeError, ValueError), match=match):
		inspect_parihaka_preparation(config)


@pytest.mark.parametrize('failure', ['nonfinite', 'statistics'])
def test_parihaka_source_rejects_value_contract_failures(
	tmp_path: Path,
	failure: str,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	changed = source.copy(order='F')
	changed[0, 0, 0] = np.nan if failure == 'nonfinite' else np.float32(999.0)
	_write_npz(config.inputs.amplitude_npz, changed)

	with pytest.raises(ValueError, match='source amplitude'):
		prepare_parihaka_volume(config)

	assert not config.outputs.metadata.exists()
	assert not list(config.outputs.data_dir.glob('.parihaka_prepare_staging-*'))


def test_parihaka_source_rejects_crc_error_and_truncated_archive(
	tmp_path: Path,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	_corrupt_stored_member(config.inputs.amplitude_npz)

	with pytest.raises(ValueError, match='CRC'):
		prepare_parihaka_volume(config)
	assert not config.outputs.metadata.exists()

	config = _fixture_config(tmp_path / 'truncated', source=source)
	archive_bytes = config.inputs.amplitude_npz.read_bytes()
	config.inputs.amplitude_npz.write_bytes(archive_bytes[:-12])
	with pytest.raises(ValueError, match='invalid or truncated'):
		inspect_parihaka_preparation(config)


@pytest.mark.parametrize('chunk_size', [0, -1, 4])
def test_parihaka_rejects_invalid_chunk_size(tmp_path: Path, chunk_size: int) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	config = replace(
		config,
		conversion=replace(config.conversion, chunk_size_z=chunk_size),
	)

	with pytest.raises(ValueError, match='chunk_size_z'):
		inspect_parihaka_preparation(config)


def test_parihaka_config_is_closed_and_rejects_label_fields(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	raw = _production_mapping(tmp_path, monkeypatch)
	raw['inputs']['label_npz'] = '/must/not/be/read.npz'
	with pytest.raises(ValueError, match='label_npz'):
		parihaka_prepare_volume_config_from_mapping(raw)

	raw = _production_mapping(tmp_path, monkeypatch)
	raw['source']['label_key'] = 'labels'
	with pytest.raises(ValueError, match='label_key'):
		parihaka_prepare_volume_config_from_mapping(raw)

	raw = _production_mapping(tmp_path, monkeypatch)
	raw['unexpected'] = True
	with pytest.raises(ValueError, match='unexpected'):
		parihaka_prepare_volume_config_from_mapping(raw)


def test_parihaka_rejects_output_roots_and_partial_or_stale_state(
	tmp_path: Path,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	overlap = replace(
		config,
		paths=replace(config.paths, artifact_root=config.paths.parihaka_root),
	)
	with pytest.raises(ValueError, match='disjoint'):
		inspect_parihaka_preparation(overlap)

	outside = replace(
		config,
		outputs=replace(config.outputs, amplitude_npy=tmp_path / 'outside.npy'),
	)
	with pytest.raises(ValueError, match='artifact_root'):
		inspect_parihaka_preparation(outside)

	config.outputs.data_dir.mkdir(parents=True)
	config.outputs.amplitude_npy.touch()
	with pytest.raises(FileExistsError, match='partial final'):
		inspect_parihaka_preparation(config)
	config.outputs.amplitude_npy.unlink()
	(config.outputs.data_dir / '.parihaka_prepare_staging-abandoned').mkdir()
	with pytest.raises(FileExistsError, match='stale'):
		inspect_parihaka_preparation(config)


def test_parihaka_existing_outputs_fail_closed_and_detect_hash_drift(
	tmp_path: Path,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	prepare_parihaka_volume(config)

	with pytest.raises(FileExistsError, match='--overwrite'):
		inspect_parihaka_preparation(config)
	config.outputs.path_list.write_text('/drift\n', encoding='utf-8')
	with pytest.raises(ValueError, match='hash/path drift'):
		inspect_parihaka_preparation(config, overwrite=True)


def test_parihaka_failure_cleans_staging_and_overwrite_preserves_old_complete(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	prepare_parihaka_volume(config)
	old_hashes = {path: _sha256(path) for path in config.outputs.files()}

	def fail_verification(*_args: object, **_kwargs: object) -> ArrayStatistics:
		raise RuntimeError('injected verification failure')

	monkeypatch.setattr(prepare_module, '_verify_destination', fail_verification)
	with pytest.raises(RuntimeError, match='injected'):
		prepare_parihaka_volume(config, overwrite=True)

	assert {path: _sha256(path) for path in config.outputs.files()} == old_hashes
	assert not list(config.outputs.data_dir.glob('.parihaka_prepare_staging-*'))


def test_parihaka_completion_metadata_is_created_after_full_verification(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	config = _fixture_config(tmp_path, source=source)
	original = prepare_module._verify_destination  # noqa: SLF001

	def observe_verification(*args: object, **kwargs: object) -> ArrayStatistics:
		destination = Path(args[1])
		assert not (destination.parent / PARIHAKA_METADATA_NAME).exists()
		return original(*args, **kwargs)

	monkeypatch.setattr(prepare_module, '_verify_destination', observe_verification)
	prepare_parihaka_volume(config)
	assert config.outputs.metadata.exists()


def _fixture_config(
	tmp_path: Path,
	*,
	source: np.ndarray,
	chunk_size_z: int = 2,
) -> ParihakaPrepareVolumeConfig:
	raw_root = tmp_path / 'raw'
	artifact_root = tmp_path / 'artifacts'
	data_dir = artifact_root / 'data' / 'parihaka' / 'facies_benchmark_v1'
	source_path = raw_root / 'fixture_data.npz'
	raw_root.mkdir(parents=True)
	_write_npz(source_path, source)
	statistics = _statistics(source)
	return ParihakaPrepareVolumeConfig(
		paths=ParihakaPrepareRootPaths(
			artifact_root=artifact_root.resolve(),
			parihaka_root=raw_root.resolve(),
		),
		inputs=ParihakaPrepareInputPaths(amplitude_npz=source_path.resolve()),
		outputs=ParihakaPrepareOutputPaths(
			data_dir=data_dir.resolve(),
			amplitude_npy=(data_dir / PARIHAKA_AMPLITUDE_NPY_NAME).resolve(),
			manifest=(data_dir / PARIHAKA_MANIFEST_NAME).resolve(),
			path_list=(data_dir / PARIHAKA_PATH_LIST_NAME).resolve(),
			normalization_stats=(
				data_dir / PARIHAKA_NORMALIZATION_STATS_NAME
			).resolve(),
			metadata=(data_dir / PARIHAKA_METADATA_NAME).resolve(),
		),
		dataset=ParihakaPrepareDatasetConfig(
			name='parihaka',
			version='facies_benchmark_v1',
			survey_id='parihaka',
		),
		source=ParihakaPrepareSourceConfig(
			direct_distributor='Mendeley Data',
			dataset_title='fixture',
			contributor='fixture',
			version=1,
			doi='fixture',
			upstream='AIcrowd',
			displayed_license='CC BY 4.0',
			archive_name='fixture.zip',
			upstream_member='fixture.npz',
			local_filename=source_path.name,
			local_modification='filename only',
			aicrowd_byte_identity='unverified',
			redistribution_transformation='unverified',
			acquisition_date=None,
			acquisition_by=None,
			array_key='data',
			member_name='data.npy',
			shape_zxy=source.shape,
			dtype='float32',
			fortran_order=True,
			expected_statistics=statistics,
		),
		conversion=ParihakaPrepareConversionConfig(
			transpose_axes=(1, 2, 0),
			chunk_size_z=chunk_size_z,
		),
		normalization=ParihakaPrepareNormalizationConfig(
			clip_low_percentile=0.5,
			clip_high_percentile=99.5,
			eps=1.0e-6,
			max_samples=1_000_000,
			seed=42,
		),
	)


def _production_mapping(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
	monkeypatch.setenv('PARIHAKA_DATA_ROOT', str((tmp_path / 'raw').resolve()))
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str((tmp_path / 'artifacts').resolve()),
	)
	return load_config(PRODUCTION_CONFIG)


def _write_npz(path: Path, array: np.ndarray) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, data=array)


def _npy_bytes(array: np.ndarray) -> bytes:
	buffer = BytesIO()
	np.save(buffer, array, allow_pickle=True)
	return buffer.getvalue()


def _statistics(array: np.ndarray) -> ArrayStatistics:
	values = np.asarray(array, dtype=np.float64)
	finite = values[np.isfinite(values)]
	return ArrayStatistics(
		element_count=int(values.size),
		finite_count=int(finite.size),
		nonfinite_count=int(values.size - finite.size),
		minimum=float(np.min(finite)),
		maximum=float(np.max(finite)),
		mean=float(np.mean(finite, dtype=np.float64)),
		population_std=float(np.std(finite, dtype=np.float64)),
	)


def _sha256(path: Path) -> str:
	return sha256(path.read_bytes()).hexdigest()


def _corrupt_stored_member(path: Path) -> None:
	source = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(3, 2, 4))
	with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
		archive.writestr('data.npy', _npy_bytes(source))
	with zipfile.ZipFile(path) as archive:
		info = archive.getinfo('data.npy')
		name_length = len(info.filename.encode())
		extra_length = len(info.extra)
		data_offset = (
			info.header_offset
			+ struct.calcsize('<IHHHHHIIIHH')
			+ name_length
			+ extra_length
		)
	payload = bytearray(path.read_bytes())
	payload[data_offset + info.file_size - 1] ^= 0xFF
	path.write_bytes(payload)
