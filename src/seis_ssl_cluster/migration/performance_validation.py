# ruff: noqa: C901, E501, PERF401, PLR0911, PLR0912, PLR0915, PTH105, S603, S607, TC001, TRY300, TRY301
"""Real-data compatibility validation for post-performance-change migrations.

The helpers in this module deliberately treat historical science artifacts as
read-only inputs.  Every mutable result is staged below a caller supplied
``migration_root`` and receives a completion manifest before it can be reused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import joblib
import numpy as np
import torch

from seis_ssl_cluster.clustering.kmeans import run_embedding_clustering
from seis_ssl_cluster.config import (
	load_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
)
from seis_ssl_cluster.config.performance_migration_validation import (
	PerformanceMigrationValidationConfig,
)
from seis_ssl_cluster.data.normalization import load_normalization_stats
from seis_ssl_cluster.data.schema import CropRequest, read_manifest_json
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	read_amplitude_crop,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.embedding.extractor import (
	_amplitude_agc_from_config,
	_zero_mask_from_config,
	build_model_from_config,
)
from seis_ssl_cluster.f3.lithology.metrics import (
	compute_lithology_metrics,
	write_confusion_matrix_csv,
)
from seis_ssl_cluster.f3.lithology.token_dataset import (
	load_f3_lithology_token_dataset,
)
from seis_ssl_cluster.f3.lithology.tokens import read_f3_lithology_class_info
from seis_ssl_cluster.runtime_checks import RuntimeChecks
from seis_ssl_cluster.stratigraphy import OrderedPrototypeHead
from seis_ssl_cluster.training.checkpoint import load_checkpoint

MigrationDecision = Literal[
	'PASS_REUSE_EXISTING',
	'PASS_WITH_NUMERIC_DRIFT',
	'REEXTRACT_REQUIRED',
	'REBUILD_M1_REQUIRED',
	'BLOCKED_NUMERIC_CONTRACT',
]

ARTIFACT_TYPE = 'performance_migration_validation'
SCHEMA_VERSION = 1
HISTORICAL_FINITE_CHECK_EVIDENCE = (
	'legacy checkpoint config omitted data.finite_check_mode; extraction code at '
	'the recorded producer commit performed no finite validation, therefore the '
	'audited compatibility reconstruction is off'
)
_EMBEDDING_FILE_NAME = 'f3_facies_benchmark.embeddings.npy'
_VALID_FILE_NAME = 'f3_facies_benchmark.valid_tokens.npy'
_EMBEDDING_METADATA_NAME = 'f3_facies_benchmark.embedding_metadata.json'


def build_input_inventory(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Inventory every historical input without mutating it."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'preflight'
	json_path = output_dir / 'input_inventory.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	output_dir.mkdir(parents=True, exist_ok=True)
	inputs = _inventory_input_records(config)
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'preflight_inventory',
		'generated_at_utc': _utc_now(),
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'git': _git_metadata(config.historical_baseline_sha),
		'environment': _environment_metadata(),
		'paths': {
			'artifact_root': str(config.artifact_root),
			'migration_root': str(config.migration_root),
			'publish_root': str(config.publish_root),
		},
		'changed_components_since_historical_baseline': _changed_components(
			config.historical_baseline_sha,
		),
		'inputs': inputs,
		'missing_input_count': sum(
			1 for item in inputs if item['status'] == 'MISSING'
		),
		'legacy_contract_reconstruction': {
			'm1_finite_check_mode': config.compatibility[
				'm1_historical_finite_check_mode'
			],
			'evidence_commit': config.compatibility[
				'historical_finite_check_evidence_commit'
			],
			'evidence_path': config.compatibility[
				'historical_finite_check_evidence_path'
			],
			'evidence': HISTORICAL_FINITE_CHECK_EVIDENCE,
		},
	}
	_write_json_atomic(json_path, payload)
	_write_text_atomic(output_dir / 'input_inventory.md', _render_inventory_markdown(payload))
	return payload


def checkpoint_compatibility_smoke(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Load all fixed checkpoints and run one common deterministic CPU crop."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'checkpoint_smoke'
	json_path = output_dir / 'checkpoint_smoke.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	output_dir.mkdir(parents=True, exist_ok=True)
	try:
		manifests = read_manifest_json(config.f3['amplitude_manifest'])
		if len(manifests) != 1:
			raise ValueError('checkpoint smoke requires exactly one F3 amplitude manifest')
		manifest = manifests[0]
		payloads = {
			role: load_checkpoint(path, map_location='cpu')
			for role, path in config.checkpoints.items()
		}
		crop = _select_common_smoke_crop(manifest, payloads)
		runs = {
			role: _smoke_one_checkpoint(
				role=role,
				checkpoint_path=config.checkpoints[role],
				payload=payload,
				crop=crop,
				manifest=manifest,
			)
			for role, payload in payloads.items()
		}
		result: dict[str, object] = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'stage': 'checkpoint_smoke',
			'status': 'PASS',
			'current_git_sha': config.current_git_sha,
			'historical_baseline_sha': config.historical_baseline_sha,
			'device': 'cpu',
			'amp': False,
			'batch_size': 1,
			'runtime_check_mode': 'strict',
			'crop': crop,
			'checkpoints': runs,
		}
	except BaseException as error:  # noqa: BLE001
		result = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'stage': 'checkpoint_smoke',
			'status': 'BLOCKED_NUMERIC_CONTRACT',
			'current_git_sha': config.current_git_sha,
			'historical_baseline_sha': config.historical_baseline_sha,
			'error_type': type(error).__name__,
			'error': str(error),
			'stack_trace': traceback.format_exc(),
		}
	_write_json_atomic(json_path, result)
	_write_text_atomic(output_dir / 'checkpoint_smoke.md', _render_checkpoint_markdown(result))
	if result['status'] != 'PASS':
		raise RuntimeError(
			'BLOCKED_NUMERIC_CONTRACT: checkpoint compatibility smoke failed; '
			f'see {json_path}',
		)
	return result


def _select_common_smoke_crop(
	manifest: object,
	payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
	"""Find one unpadded crop valid for every fixed checkpoint contract."""
	from seis_ssl_cluster.data.schema import SurveyManifest  # noqa: PLC0415

	if not isinstance(manifest, SurveyManifest):
		raise TypeError('F3 manifest must be a SurveyManifest')
	amplitude_path = manifest.amplitude.path
	if not amplitude_path.is_absolute():
		amplitude_path = manifest.root / amplitude_path
	stats = load_normalization_stats(manifest.amplitude.normalization_stats_path)
	store = NpyMemmapVolumeStore()
	shape = manifest.amplitude.shape_xyz
	candidates = (
		(0, 0, 0),
		(128, 128, 64),
		(256, 256, 64),
		(384, 512, 64),
		(448, 640, 96),
	)
	for start_xyz in candidates:
		if any(start + 128 > axis for start, axis in zip(start_xyz, shape, strict=True)):
			continue
		request = CropRequest(
			survey_id=manifest.survey_id,
			start_xyz=start_xyz,
			size_xyz=(128, 128, 128),
		)
		prepared_by_role: dict[str, object] = {}
		for role, payload in payloads.items():
			checkpoint_config = _checkpoint_config(payload)
			settings = _preprocess_settings(checkpoint_config, finite_check_mode='strict')
			prepared = read_amplitude_crop(
				request=request,
				amplitude_path=amplitude_path,
				stats=stats,
				store=store,
				patch_size_xyz=tuple(checkpoint_config['model']['patch_size']),  # type: ignore[index]
				settings=settings,
			)
			if not bool(np.asarray(prepared.token_valid_mask).any()):
				break
			prepared_by_role[role] = prepared
		else:
			first = next(iter(prepared_by_role.values()))
			return {
				'survey_id': manifest.survey_id,
				'start_xyz': list(start_xyz),
				'size_xyz': [128, 128, 128],
				'padding_required': False,
				'valid_fraction': float(np.mean(first.local_valid_mask)),  # type: ignore[union-attr]
				'token_valid_count': int(np.count_nonzero(first.token_valid_mask)),  # type: ignore[union-attr]
				'source_amplitude_path': str(amplitude_path),
				'source_amplitude_sha256': _sha256_path(amplitude_path),
				'normalization_stats_path': str(
					manifest.amplitude.normalization_stats_path,
				),
				'normalization_stats_sha256': _sha256_path(
					manifest.amplitude.normalization_stats_path,
				),
			}
	raise RuntimeError('no fixed 128^3 F3 crop met the shared valid-mask contract')


def _smoke_one_checkpoint(
	*,
	role: str,
	checkpoint_path: Path,
	payload: Mapping[str, object],
	crop: Mapping[str, object],
	manifest: object,
) -> dict[str, object]:
	"""Strictly load one checkpoint and encode the shared deterministic crop."""
	from seis_ssl_cluster.data.schema import SurveyManifest  # noqa: PLC0415

	if not isinstance(manifest, SurveyManifest):
		raise TypeError('F3 manifest must be a SurveyManifest')
	checkpoint_config = _checkpoint_config(payload)
	state_dict = _model_state_dict(payload)
	model = build_model_from_config(checkpoint_config)
	_set_runtime_check_mode_strict(model)
	load_result = model.load_state_dict(state_dict, strict=False)
	if load_result.missing_keys or load_result.unexpected_keys:
		raise ValueError(
			f'{role} state_dict mismatch: missing={load_result.missing_keys!r} '
			f'unexpected={load_result.unexpected_keys!r}',
		)
	model.eval()
	start_xyz = tuple(int(value) for value in cast('Sequence[object]', crop['start_xyz']))
	request = CropRequest(
		survey_id=manifest.survey_id,
		start_xyz=cast('tuple[int, int, int]', start_xyz),
		size_xyz=(128, 128, 128),
	)
	amplitude_path = manifest.amplitude.path
	if not amplitude_path.is_absolute():
		amplitude_path = manifest.root / amplitude_path
	prepared = read_amplitude_crop(
		request=request,
		amplitude_path=amplitude_path,
		stats=load_normalization_stats(manifest.amplitude.normalization_stats_path),
		store=NpyMemmapVolumeStore(),
		patch_size_xyz=tuple(checkpoint_config['model']['patch_size']),  # type: ignore[index]
		settings=_preprocess_settings(checkpoint_config, finite_check_mode='strict'),
	)
	x = torch.from_numpy(prepared.x[np.newaxis, ...]).to(dtype=torch.float32)
	valid = torch.from_numpy(prepared.token_valid_mask[np.newaxis, ...])
	with torch.no_grad():
		output = model.encode_tokens(x, valid_mask=valid)
	tokens = cast('torch.Tensor', output['tokens']).detach().cpu().to(torch.float32)
	if not bool(torch.isfinite(tokens).all()):
		raise ValueError(f'{role} encoder output contains non-finite values')
	params = [parameter.detach().cpu() for parameter in model.parameters()]
	if not all(bool(torch.isfinite(parameter).all()) for parameter in params):
		raise ValueError(f'{role} checkpoint contains non-finite model parameters')
	stratigraphy = _stratigraphy_checkpoint_identity(
		payload,
		feature_dim=model.encoder_dim,
	)
	return {
		'checkpoint_path': str(checkpoint_path),
		'checkpoint_sha256': _sha256_path(checkpoint_path),
		'checkpoint_stage': checkpoint_config.get('stage'),
		'checkpoint_schema_version': payload.get('schema_version', 1),
		'resolved_config': _jsonable(checkpoint_config),
		'model_geometry': _model_geometry(checkpoint_config),
		'state_dict_keys': sorted(str(key) for key in state_dict),
		'state_dict_parameter_shapes': {
			str(key): list(value.shape)
			for key, value in state_dict.items()
			if isinstance(value, torch.Tensor)
		},
		'state_dict_floating_dtypes': sorted(
			{str(value.dtype) for value in state_dict.values() if value.is_floating_point()},
		),
		'missing_keys': list(load_result.missing_keys),
		'unexpected_keys': list(load_result.unexpected_keys),
		'all_parameters_finite': True,
		'stratigraphy_pretext_identity': stratigraphy,
		'input': {
			'shape': list(x.shape),
			'dtype': str(x.dtype),
			'sha256_f32_c_order': _sha256_array(x.numpy(), dtype=np.dtype('<f4')),
			'voxel_valid_mask_sha256': _sha256_array(prepared.local_valid_mask),
			'token_valid_mask_sha256': _sha256_array(prepared.token_valid_mask),
			'token_valid_count': int(np.count_nonzero(prepared.token_valid_mask)),
		},
		'encoder_output': {
			'shape': list(tokens.shape),
			'dtype': str(tokens.dtype),
			'serialize_format': 'little_endian_float32_c_order',
			'sha256': _sha256_array(tokens.numpy(), dtype=np.dtype('<f4')),
			'mean': float(tokens.mean()),
			'std': float(tokens.std(unbiased=False)),
			'min': float(tokens.min()),
			'max': float(tokens.max()),
			'finite_count': int(tokens.numel()),
			'non_finite_count': 0,
		},
	}


def run_m1_embedding_extraction(  # noqa: PLR0913
	config: PerformanceMigrationValidationConfig,
	*,
	embedding_config_path: Path,
	mode: Literal['cache_off', 'cache_memmap'],
	device: str | None,
	dry_run: bool = False,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Extract M1 embeddings through the current code under an audited override.

	The only reconstructed checkpoint-owned setting is the historically absent
	finite-check mode.  It is deliberately applied in-memory after a narrow diff
	check rather than written into either the checkpoint or active YAML.
	"""
	_assert_live_git_sha(config)
	if mode not in {'cache_off', 'cache_memmap'}:
		raise ValueError(f'unsupported migration embedding mode: {mode!r}')
	raw = load_config(embedding_config_path)
	resolved = resolve_embedding_extraction_config(raw)
	final_dir = config.migration_root / 'embeddings' / f'm1_{mode}' / 'overlap_x16'
	_expected_migration_embedding_config(
		resolved,
		checkpoint=config.checkpoints['m1'],
		output_dir=final_dir,
		mode=mode,
	)
	if dry_run:
		return {
			'stage': f'm1_embedding_{mode}',
			'status': 'DRY_RUN',
			'config_path': str(embedding_config_path),
			'final_output_dir': str(final_dir),
			'legacy_finite_check_reconstruction': _legacy_finite_contract(config),
		}
	if only_missing and _embedding_complete_and_valid(
		final_dir,
		config=config,
		mode=mode,
	):
		return _load_json(final_dir / 'migration_completion.json')
	_prepare_output_for_regeneration(final_dir, only_missing=only_missing)
	staging_dir = final_dir.parent / f'.{final_dir.name}.staging-{uuid4().hex}'
	try:
		staging_config = deepcopy(resolved)
		cast('dict[str, object]', staging_config['embeddings'])['output_dir'] = str(
			staging_dir,
		)
		cache = cast('dict[str, object]', staging_config['embedding']).get(
			'preprocessing_cache',
		)
		if isinstance(cache, dict) and cache.get('mode') == 'memmap':
			cache['directory'] = str(staging_dir / '.preprocessing_cache')
		payload = load_checkpoint(config.checkpoints['m1'], map_location='cpu')
		serialized = _checkpoint_config(payload)
		override = _legacy_m1_checkpoint_config_override(config, serialized)
		results = run_embedding_extraction(
			staging_config,
			skip_existing=False,
			device=device,
			checkpoint_config_override=override,
		)
		if len(results) != 1 or results[0].skipped:
			raise RuntimeError('migration embedding extraction did not write exactly one survey')
		completion = _embedding_completion_payload(
			config=config,
			mode=mode,
			output_dir=staging_dir,
			config_path=embedding_config_path,
			resolved_config=staging_config,
			override=override,
		)
		_write_json_atomic(staging_dir / 'migration_completion.json', completion)
		if not _embedding_complete_and_valid(
			staging_dir,
			config=config,
			mode=mode,
		):
			raise RuntimeError('staged M1 embedding artifact did not validate')
		final_dir.parent.mkdir(parents=True, exist_ok=True)
		os.replace(staging_dir, final_dir)
		return _load_json(final_dir / 'migration_completion.json')
	except BaseException:
		if staging_dir.exists():
			_quarantine_path(staging_dir, reason='embedding_stage_failure')
		raise


def compare_embedding_artifacts(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Compare historical/cache-off/memmap M1 arrays and metadata."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'embedding_parity'
	json_path = output_dir / 'embedding_parity.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	paths = {
		'A_historical': config.historical_embeddings['m1'],
		'B_current_cache_off': config.migration_root
		/ 'embeddings'
		/ 'm1_cache_off'
		/ 'overlap_x16',
		'C_current_memmap_cache': config.migration_root
		/ 'embeddings'
		/ 'm1_cache_memmap'
		/ 'overlap_x16',
	}
	for label, path in paths.items():
		if not _embedding_files_complete(path):
			raise FileNotFoundError(f'{label} embedding artifact is incomplete: {path}')
	output_dir.mkdir(parents=True, exist_ok=True)
	artifacts = {label: _load_embedding_artifact(path) for label, path in paths.items()}
	pairs = (
		('A_historical', 'B_current_cache_off'),
		('A_historical', 'C_current_memmap_cache'),
		('B_current_cache_off', 'C_current_memmap_cache'),
	)
	comparisons = {
		f'{left}_vs_{right}': _compare_one_embedding_pair(
			artifacts[left],
			artifacts[right],
			left_name=left,
			right_name=right,
		)
		for left, right in pairs
	}
	axis_rows, channel_rows = _embedding_diagnostic_rows(comparisons)
	_write_csv_atomic(
		output_dir / 'embedding_parity_by_axis.csv',
		axis_rows,
		fieldnames=('pair', 'axis', 'index', 'valid_element_count', 'mean_abs_error', 'max_abs_error'),
	)
	_write_csv_atomic(
		output_dir / 'embedding_parity_by_channel.csv',
		channel_rows,
		fieldnames=('pair', 'channel', 'mean_abs_error', 'max_abs_error'),
	)
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'embedding_parity',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'artifacts': {
			label: _embedding_identity(artifact)
			for label, artifact in artifacts.items()
		},
		'comparisons': comparisons,
		'legacy_finite_check_contract': _legacy_finite_contract(config),
	}
	_write_json_atomic(json_path, payload)
	_write_text_atomic(output_dir / 'embedding_parity.md', _render_embedding_markdown(payload))
	return payload


def _expected_migration_embedding_config(
	resolved: Mapping[str, object],
	*,
	checkpoint: Path,
	output_dir: Path,
	mode: str,
) -> None:
	embeddings = _mapping(resolved, 'embeddings')
	embedding = _mapping(resolved, 'embedding')
	if Path(cast('str', embeddings['checkpoint'])) != checkpoint:
		raise ValueError('migration embedding config checkpoint does not equal canonical M1')
	if Path(cast('str', embeddings['output_dir'])) != output_dir:
		raise ValueError('migration embedding config output_dir is not the isolated migration path')
	if embedding.get('window_size') != [128, 128, 128]:
		raise ValueError('migration embedding window_size must be [128, 128, 128]')
	if embedding.get('overlap') != [112, 64, 64]:
		raise ValueError('migration embedding overlap must be [112, 64, 64]')
	if embedding.get('output_dtype') != 'float16':
		raise ValueError('migration embedding output_dtype must be float16')
	if embedding.get('batch_size') != 1:
		raise ValueError('migration embedding batch_size must be 1')
	if embedding.get('prefetch_queue_depth', 0) != 0:
		raise ValueError('migration embedding prefetch_queue_depth must be 0')
	if embedding.get('amp', False) is not False:
		raise ValueError('migration embedding amp must be false')
	if embedding.get('stage_timing') is not True:
		raise ValueError('migration embedding stage_timing must be true')
	if float(cast('float', embedding.get('min_token_valid_fraction'))) != 0.5:
		raise ValueError('migration embedding min_token_valid_fraction must be 0.5')
	cache = _mapping(embedding, 'preprocessing_cache')
	expected_cache_mode = 'off' if mode == 'cache_off' else 'memmap'
	if cache.get('mode') != expected_cache_mode:
		raise ValueError(
			f'migration embedding {mode} needs preprocessing_cache.mode={expected_cache_mode!r}',
		)
	if mode == 'cache_memmap':
		if cache.get('chunk_size_x') != 16 or cache.get('reuse') is not True:
			raise ValueError('memmap migration config must use chunk_size_x=16 and reuse=true')
		if cache.get('cleanup') is not False:
			raise ValueError('memmap migration config must preserve its prepared cache')


def _load_embedding_artifact(root: Path) -> dict[str, object]:
	"""Open one embedding triplet read-only and retain its concrete identity."""
	embedding_path = root / _EMBEDDING_FILE_NAME
	valid_path = root / _VALID_FILE_NAME
	metadata_path = root / _EMBEDDING_METADATA_NAME
	return {
		'root': root,
		'embeddings_path': embedding_path,
		'valid_tokens_path': valid_path,
		'metadata_path': metadata_path,
		'embeddings': np.load(embedding_path, mmap_mode='r'),
		'valid_tokens': np.load(valid_path, mmap_mode='r'),
		'metadata': _load_json(metadata_path),
	}


def _embedding_identity(artifact: Mapping[str, object]) -> dict[str, object]:
	embeddings = np.asarray(artifact['embeddings'])
	valid = np.asarray(artifact['valid_tokens'])
	metadata = _mapping(artifact, 'metadata')
	return {
		'root': str(artifact['root']),
		'embeddings_path': str(artifact['embeddings_path']),
		'embeddings_sha256': _sha256_path(cast('Path', artifact['embeddings_path'])),
		'embeddings_shape': list(embeddings.shape),
		'embeddings_dtype': str(embeddings.dtype),
		'valid_tokens_path': str(artifact['valid_tokens_path']),
		'valid_tokens_sha256': _sha256_path(cast('Path', artifact['valid_tokens_path'])),
		'valid_tokens_shape': list(valid.shape),
		'valid_tokens_dtype': str(valid.dtype),
		'valid_token_count': int(np.count_nonzero(valid)),
		'invalid_token_count': int(valid.size - np.count_nonzero(valid)),
		'metadata_path': str(artifact['metadata_path']),
		'metadata_sha256': _sha256_path(cast('Path', artifact['metadata_path'])),
		'metadata_scientific_identity': _embedding_scientific_metadata(metadata),
		'metadata_runtime_identity': _embedding_runtime_metadata(metadata),
	}


def _compare_one_embedding_pair(
	left: Mapping[str, object],
	right: Mapping[str, object],
	*,
	left_name: str,
	right_name: str,
) -> dict[str, object]:
	"""Calculate exact and valid-token-only diagnostics for one embedding pair."""
	left_embeddings = np.asarray(left['embeddings'])
	right_embeddings = np.asarray(right['embeddings'])
	left_valid = np.asarray(left['valid_tokens'])
	right_valid = np.asarray(right['valid_tokens'])
	structural = {
		'survey_id_equal': _mapping(left, 'metadata').get('survey_id')
		== _mapping(right, 'metadata').get('survey_id'),
		'embedding_shape_equal': left_embeddings.shape == right_embeddings.shape,
		'embedding_dtype_equal': left_embeddings.dtype == right_embeddings.dtype,
		'valid_token_shape_equal': left_valid.shape == right_valid.shape,
		'valid_token_dtype_equal': left_valid.dtype == right_valid.dtype,
	}
	if not all(structural.values()):
		return {
			'status': 'BLOCKED_NUMERIC_CONTRACT',
			'left': left_name,
			'right': right_name,
			'structural_identity': structural,
			'reason': 'embedding structural identity mismatch',
		}
	valid_equal = bool(np.array_equal(left_valid, right_valid))
	if not valid_equal:
		return {
			'status': 'BLOCKED_NUMERIC_CONTRACT',
			'left': left_name,
			'right': right_name,
			'structural_identity': structural,
			'valid_token_mask_exact': False,
			'valid_token_mismatch_count': int(np.count_nonzero(left_valid != right_valid)),
			'reason': 'valid-token mask mismatch',
		}
	metadata_diff = _compare_embedding_metadata(
		_mapping(left, 'metadata'),
		_mapping(right, 'metadata'),
	)
	exact = bool(np.array_equal(left_embeddings, right_embeddings))
	valid_mask = left_valid.astype(bool, copy=False)
	left_values = np.asarray(left_embeddings[valid_mask], dtype=np.float32)
	right_values = np.asarray(right_embeddings[valid_mask], dtype=np.float32)
	if left_values.ndim != 2 or right_values.shape != left_values.shape:
		raise RuntimeError('valid embedding values must be matching 2D matrices')
	if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
		return {
			'status': 'BLOCKED_NUMERIC_CONTRACT',
			'left': left_name,
			'right': right_name,
			'structural_identity': structural,
			'valid_token_mask_exact': True,
			'non_finite': _embedding_non_finite_diagnostics(
				left_embeddings,
				right_embeddings,
				valid_mask,
			),
			'reason': 'non-finite valid embedding values',
		}
	abs_error = np.abs(left_values - right_values)
	different = left_values != right_values
	stable_denominator = np.maximum(
		np.maximum(np.abs(left_values), np.abs(right_values)),
		np.float32(1.0e-7),
	)
	stable_relative = abs_error / stable_denominator
	cosine = _per_token_cosine(left_values, right_values)
	by_axis = _embedding_axis_diagnostics(
		left_embeddings,
		right_embeddings,
		valid_mask,
	)
	by_channel = {
		'mean_abs_error': [float(value) for value in abs_error.mean(axis=0)],
		'max_abs_error': [float(value) for value in abs_error.max(axis=0)],
	}
	edge_mask = _embedding_edge_mask(valid_mask.shape)
	spatial = {
		'edge': _error_summary_for_token_mask(
			left_embeddings,
			right_embeddings,
			valid_mask & edge_mask,
		),
		'interior': _error_summary_for_token_mask(
			left_embeddings,
			right_embeddings,
			valid_mask & ~edge_mask,
		),
	}
	status = (
		'EXACT'
		if exact and metadata_diff['scientific_identity_status'] == 'EXACT'
		else 'NUMERIC_DRIFT'
	)
	return {
		'status': status,
		'left': left_name,
		'right': right_name,
		'structural_identity': structural,
		'valid_token_mask_exact': True,
		'invalid_token_locations_exact': True,
		'embedding_array_equal': exact,
		'metadata_diff': metadata_diff,
		'valid_element_count': int(left_values.size),
		'different_element_count': int(np.count_nonzero(different)),
		'different_element_fraction': float(np.mean(different)),
		'absolute_error': _distribution_summary(abs_error),
		'stable_relative_error': _distribution_summary(stable_relative),
		'per_token_cosine_similarity': _distribution_summary(cosine),
		'by_axis': by_axis,
		'by_channel': by_channel,
		'edge_vs_interior': spatial,
		'non_finite': _embedding_non_finite_diagnostics(
			left_embeddings,
			right_embeddings,
			valid_mask,
		),
	}


def _embedding_axis_diagnostics(
	left: np.ndarray,
	right: np.ndarray,
	valid: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
	result: dict[str, list[dict[str, object]]] = {}
	for axis, name in enumerate(('x', 'y', 'z')):
		rows: list[dict[str, object]] = []
		for index in range(valid.shape[axis]):
			slices = [slice(None)] * 3
			slices[axis] = index
			mask = valid[tuple(slices)]
			if not bool(mask.any()):
				rows.append(
					{
						'index': index,
						'valid_element_count': 0,
						'mean_abs_error': 0.0,
						'max_abs_error': 0.0,
					},
				)
				continue
			left_values = np.asarray(left[tuple(slices)][mask], dtype=np.float32)
			right_values = np.asarray(right[tuple(slices)][mask], dtype=np.float32)
			error = np.abs(left_values - right_values)
			rows.append(
				{
					'index': index,
					'valid_element_count': int(error.size),
					'mean_abs_error': float(error.mean()),
					'max_abs_error': float(error.max()),
				},
			)
		result[name] = rows
	return result


def _embedding_diagnostic_rows(
	comparisons: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	axis_rows: list[dict[str, object]] = []
	channel_rows: list[dict[str, object]] = []
	for pair, raw in comparisons.items():
		comparison = cast('Mapping[str, object]', raw)
		by_axis = comparison.get('by_axis')
		if isinstance(by_axis, Mapping):
			for axis, rows in by_axis.items():
				for row in cast('Sequence[Mapping[str, object]]', rows):
					axis_rows.append({'pair': pair, 'axis': axis, **dict(row)})
		by_channel = comparison.get('by_channel')
		if isinstance(by_channel, Mapping):
			means = cast('Sequence[object]', by_channel['mean_abs_error'])
			maxes = cast('Sequence[object]', by_channel['max_abs_error'])
			for channel, (mean, maximum) in enumerate(zip(means, maxes, strict=True)):
				channel_rows.append(
					{
						'pair': pair,
						'channel': channel,
						'mean_abs_error': mean,
						'max_abs_error': maximum,
					},
				)
	return axis_rows, channel_rows


def compare_probe_predictions(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Apply the existing M1 scaler/probe to A/B/C embeddings without fitting."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'probe_parity'
	json_path = output_dir / 'probe_parity.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	output_dir.mkdir(parents=True, exist_ok=True)
	embedding_paths = {
		'A_historical': config.historical_embeddings['m1'],
		'B_current_cache_off': config.migration_root
		/ 'embeddings'
		/ 'm1_cache_off'
		/ 'overlap_x16',
		'C_current_memmap_cache': config.migration_root
		/ 'embeddings'
		/ 'm1_cache_memmap'
		/ 'overlap_x16',
	}
	artifacts = {name: _load_embedding_artifact(path) for name, path in embedding_paths.items()}
	validations = load_f3_lithology_token_dataset(config.m1_probe['validation_tokens'])
	coordinates = np.asarray(validations.token_xyz, dtype=np.int64)
	labels = np.asarray(validations.labels, dtype=np.int64)
	if coordinates.shape != (labels.size, 3):
		raise ValueError('validation token coordinate contract is invalid')
	classes = read_f3_lithology_class_info(config.f3['class_info'])
	scaler = joblib.load(config.m1_probe['scaler'])
	probe = joblib.load(config.m1_probe['probe'])
	if not np.array_equal(np.asarray(probe.classes_, dtype=np.int64), np.arange(6)):
		raise ValueError('historical linear probe class order must be [0, 1, 2, 3, 4, 5]')
	historical_grid = np.load(config.m1_probe['predictions'], mmap_mode='r')
	historical_valid_grid = np.load(config.m1_probe['valid_grid'], mmap_mode='r')
	if not np.array_equal(historical_valid_grid, artifacts['A_historical']['valid_tokens']):
		raise RuntimeError('historical prediction valid-token grid differs from embedding valid grid')
	predictions: dict[str, np.ndarray] = {}
	decision_values: dict[str, np.ndarray] = {}
	feature_identities: dict[str, object] = {}
	for name, artifact in artifacts.items():
		valid_grid = np.asarray(artifact['valid_tokens'])
		if not bool(valid_grid[tuple(coordinates.T)].all()):
			raise RuntimeError(f'{name} is missing a validation coordinate in valid-token mask')
		features = np.asarray(artifact['embeddings'])[tuple(coordinates.T)].astype(
			np.float32,
			copy=False,
		)
		scaled = np.asarray(scaler.transform(features), dtype=np.float64)
		decision = np.asarray(probe.decision_function(scaled), dtype=np.float64)
		prediction = np.asarray(probe.predict(scaled), dtype=np.int64)
		predictions[name] = prediction
		decision_values[name] = decision
		feature_identities[name] = {
			'raw_feature_shape': list(features.shape),
			'raw_feature_dtype': str(features.dtype),
			'raw_feature_sha256_f32': _sha256_array(features, dtype=np.dtype('<f4')),
			'scaled_feature_shape': list(scaled.shape),
			'scaled_feature_dtype': str(scaled.dtype),
			'scaled_feature_sha256_f64': _sha256_array(scaled, dtype=np.dtype('<f8')),
			'decision_shape': list(decision.shape),
			'decision_sha256_f64': _sha256_array(decision, dtype=np.dtype('<f8')),
			'prediction_sha256_i64': _sha256_array(prediction, dtype=np.dtype('<i8')),
		}
	historical_row_predictions = np.asarray(historical_grid[tuple(coordinates.T)], dtype=np.int64)
	if not np.array_equal(predictions['A_historical'], historical_row_predictions):
		raise RuntimeError('direct historical probe output differs from historical prediction artifact')
	metrics = {
		name: compute_lithology_metrics(labels, prediction, classes)
		for name, prediction in predictions.items()
	}
	for name, item in metrics.items():
		write_confusion_matrix_csv(output_dir / f'confusion_{name[0]}.csv', item, classes)
	pairs = (
		('A_historical', 'B_current_cache_off'),
		('A_historical', 'C_current_memmap_cache'),
		('B_current_cache_off', 'C_current_memmap_cache'),
	)
	parity: dict[str, object] = {}
	mismatch_rows: list[dict[str, object]] = []
	for left, right in pairs:
		pair_name = f'{left}_vs_{right}'
		mismatch = predictions[left] != predictions[right]
		metric_equal = _probe_metrics_exact(metrics[left], metrics[right])
		parity[pair_name] = {
			'validation_coordinates_exact': True,
			'true_labels_exact': True,
			'prediction_exact': bool(not mismatch.any()),
			'prediction_mismatch_count': int(np.count_nonzero(mismatch)),
			'confusion_matrix_exact': metrics[left]['confusion_matrix']
			== metrics[right]['confusion_matrix'],
			'primary_metrics_exact': metric_equal,
			'status': (
				'NUMERICALLY_EQUIVALENT'
				if not mismatch.any()
				and metrics[left]['confusion_matrix'] == metrics[right]['confusion_matrix']
				and metric_equal
				else 'DRIFT'
			),
		}
		if mismatch.any():
			for index in np.flatnonzero(mismatch):
				mismatch_rows.append(
					{
						'pair': pair_name,
						'x': int(coordinates[index, 0]),
						'y': int(coordinates[index, 1]),
						'z': int(coordinates[index, 2]),
						'true_class': int(labels[index]),
						'left_prediction': int(predictions[left][index]),
						'right_prediction': int(predictions[right][index]),
						'left_decision_margin': _decision_margin(decision_values[left][index]),
						'right_decision_margin': _decision_margin(decision_values[right][index]),
						'embedding_error_max_abs': _row_embedding_error(
							artifacts[left], artifacts[right], coordinates[index]
						),
					}
				)
	_write_csv_atomic(
		output_dir / 'prediction_mismatches.csv',
		mismatch_rows,
		fieldnames=(
			'pair',
			'x',
			'y',
			'z',
			'true_class',
			'left_prediction',
			'right_prediction',
			'left_decision_margin',
			'right_decision_margin',
			'embedding_error_max_abs',
		),
	)
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'probe_parity',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'probe_path': str(config.m1_probe['probe']),
		'probe_sha256': _sha256_path(config.m1_probe['probe']),
		'scaler_path': str(config.m1_probe['scaler']),
		'scaler_sha256': _sha256_path(config.m1_probe['scaler']),
		'validation_coordinates_sha256': _sha256_array(coordinates, dtype=np.dtype('<i8')),
		'true_labels_sha256': _sha256_array(labels, dtype=np.dtype('<i8')),
		'validation_row_count': int(labels.size),
		'class_order': [int(item.class_id) for item in classes],
		'features': feature_identities,
		'metrics': metrics,
		'parity': parity,
		'prediction_mismatch_rows': len(mismatch_rows),
	}
	_write_json_atomic(json_path, payload)
	_write_text_atomic(output_dir / 'probe_parity.md', _render_probe_markdown(payload))
	return payload


def reconstruct_historical_hmm_config(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Reconstruct K=6 replay science fields from recorded artifact metadata."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'clustering'
	config_path = output_dir / 'historical_k6_scientific_config.json'
	manifest_path = output_dir / 'historical_k6_source_manifest.json'
	if only_missing and _is_complete_file(config_path) and _is_complete_file(manifest_path):
		return _load_json(config_path)
	metadata = _load_json(config.hmm['clustering_metadata'])
	hmm = _mapping(metadata, 'stratigraphic_hmm')
	reconstructed: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'historical_k6_scientific_config_reconstruction',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'source_metadata_path': str(config.hmm['clustering_metadata']),
		'source_metadata_sha256': _sha256_path(config.hmm['clustering_metadata']),
		'objective_history': 'NOT_RECORDED_IN_HISTORICAL_ARTIFACT',
		'embeddings': {'input_dir': str(config.historical_embeddings['mae'])},
		'clustering': {
			'embedding_normalization': metadata['normalization'],
			'residualization': _scientific_residualization(_mapping(metadata, 'residualization')),
			'pca': _scientific_pca(_mapping(metadata, 'pca')),
			'sample_tokens': metadata['sample']['requested_count'],  # type: ignore[index]
			'method': metadata['method'],
			'k_values': [int(metadata['k'])],
			'minibatch_size': metadata['minibatch_size'],
			'prediction_batch_size': metadata['prediction_batch_size'],
			'seed': metadata['random_seed'],
			'stratigraphic_hmm': {
				'emission_source': hmm['emission_source'],
				'iterations': hmm['iterations'],
				'z_axis': hmm['z_axis'],
				'z_direction': hmm['z_direction'],
				'edge_margin_tokens': hmm['edge_margin_tokens'],
				'transition': hmm['transition'],
				'init': hmm['init'],
				'update': hmm['update'],
				'path_prior': _scientific_path_prior(_mapping(hmm, 'path_prior')),
			},
		},
	}
	source_manifest = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'historical_k6_source_manifest',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'inputs': {
			key: _path_record(path, dependent_stages=('hmm_replay', 'hmm_parity'))
			for key, path in config.hmm.items()
		},
		'mae_embedding_input': _path_record(
			config.historical_embeddings['mae'],
			dependent_stages=('hmm_replay',),
		),
		'actual_sample_count': _mapping(metadata, 'sample')['count'],
		'historical_objective_history': 'NOT_RECORDED_IN_HISTORICAL_ARTIFACT',
	}
	_write_json_atomic(config_path, reconstructed)
	_write_json_atomic(manifest_path, source_manifest)
	return reconstructed


def run_m1_hmm_replay(
	config: PerformanceMigrationValidationConfig,
	*,
	hmm_config_path: Path,
	dry_run: bool = False,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Replay the historical MAE-embedding K=6 HMM with current code."""
	_assert_live_git_sha(config)
	historical = reconstruct_historical_hmm_config(config, only_missing=True)
	raw = load_config(hmm_config_path)
	resolved = resolve_clustering_config(raw)
	final_dir = config.migration_root / 'clustering' / 'm1_k6_current_replay'
	if Path(cast('str', _mapping(resolved, 'clustering')['output_dir'])) != final_dir:
		raise ValueError('HMM replay config output_dir is not the isolated migration path')
	if _scientific_clustering_view(resolved) != _scientific_clustering_view(historical):
		diff = _mapping_diff(
			_scientific_clustering_view(historical),
			_scientific_clustering_view(resolved),
		)
		raise ValueError(f'HMM replay scientific config drift: {json.dumps(diff, sort_keys=True)}')
	if dry_run:
		return {
			'stage': 'm1_k6_hmm_replay',
			'status': 'DRY_RUN',
			'config_path': str(hmm_config_path),
			'final_output_dir': str(final_dir),
			'scientific_config_exact': True,
		}
	if only_missing and _hmm_complete_and_valid(final_dir, config=config):
		return _load_json(final_dir / 'migration_completion.json')
	_prepare_output_for_regeneration(final_dir, only_missing=only_missing)
	staging_dir = final_dir.parent / f'.{final_dir.name}.staging-{uuid4().hex}'
	try:
		staging = deepcopy(resolved)
		clustering = cast('dict[str, object]', staging['clustering'])
		clustering['output_dir'] = str(staging_dir)
		hmm = cast('dict[str, object]', clustering['stratigraphic_hmm'])
		cache = cast('dict[str, object]', hmm['prepared_feature_cache'])
		cache['directory'] = str(staging_dir / 'prepared_feature_cache')
		result = run_embedding_clustering(staging)
		if len(result.results) != 1 or result.results[0].k != 6:
			raise RuntimeError('HMM replay did not produce exactly K=6')
		completion = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'stage': 'm1_k6_hmm_replay',
			'completion_status': 'COMPLETE',
			'current_git_sha': config.current_git_sha,
			'historical_baseline_sha': config.historical_baseline_sha,
			'config_path': str(hmm_config_path),
			'config_sha256': _sha256_path(hmm_config_path),
			'scientific_config_exact': True,
			'files': _hmm_output_hashes(staging_dir),
		}
		_write_json_atomic(staging_dir / 'migration_completion.json', completion)
		if not _hmm_complete_and_valid(staging_dir, config=config):
			raise RuntimeError('staged HMM replay failed completion validation')
		final_dir.parent.mkdir(parents=True, exist_ok=True)
		os.replace(staging_dir, final_dir)
		return _load_json(final_dir / 'migration_completion.json')
	except BaseException:
		if staging_dir.exists():
			_quarantine_path(staging_dir, reason='hmm_replay_stage_failure')
		raise


def compare_hmm_replay(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Check ordered K=6 labels and numeric artifacts without permutation fixes."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'clustering'
	json_path = output_dir / 'hmm_parity.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	current_root = output_dir / 'm1_k6_current_replay'
	if not _hmm_complete_and_valid(current_root, config=config):
		raise FileNotFoundError(f'current HMM replay incomplete: {current_root}')
	historical_labels = np.load(config.hmm['labels'], mmap_mode='r')
	current_labels_path = current_root / 'labels' / 'k6' / 'f3_facies_benchmark.cluster_labels_token.npy'
	current_labels = np.load(current_labels_path, mmap_mode='r')
	historical_metadata = _load_json(config.hmm['clustering_metadata'])
	current_metadata_path = current_root / 'models' / 'k6' / 'clustering_metadata.json'
	current_metadata = _load_json(current_metadata_path)
	historical_centers = np.load(config.hmm['centers'], mmap_mode='r')
	current_centers_path = current_root / 'models' / 'k6' / 'cluster_centers.npy'
	current_centers = np.load(current_centers_path, mmap_mode='r')
	shape_equal = historical_labels.shape == current_labels.shape
	dtype_equal = historical_labels.dtype == current_labels.dtype
	if not shape_equal:
		raise RuntimeError('BLOCKED_NUMERIC_CONTRACT: HMM label shape mismatch')
	historical_valid = historical_labels >= 0
	current_valid = current_labels >= 0
	valid_equal = bool(np.array_equal(historical_valid, current_valid))
	labels_equal = bool(np.array_equal(historical_labels, current_labels))
	mismatch = historical_labels != current_labels
	diagnostics = _hmm_label_mismatch_diagnostics(
		historical_labels,
		current_labels,
		mismatch,
	)
	_write_csv_atomic(
		output_dir / 'hmm_label_mismatch.csv',
		diagnostics['mismatch_rows'],
		fieldnames=('x', 'y', 'z', 'historical_label', 'current_label', 'edge_or_interior', 'boundary_or_interior'),
	)
	_write_csv_atomic(
		output_dir / 'hmm_transition_diagnostics.csv',
		diagnostics['transition_rows'],
		fieldnames=('metric', 'historical', 'current', 'equal'),
	)
	center_numeric = _numeric_array_comparison(
		np.asarray(historical_centers),
		np.asarray(current_centers),
		rtol=1.0e-6,
		atol=1.0e-6,
	)
	historical_model = joblib.load(config.hmm['hmm_model'])
	current_model = joblib.load(current_root / 'models' / 'k6' / 'hmm_model.joblib')
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'hmm_parity',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'labels': {
			'historical_path': str(config.hmm['labels']),
			'historical_sha256': _sha256_path(config.hmm['labels']),
			'current_path': str(current_labels_path),
			'current_sha256': _sha256_path(current_labels_path),
			'shape_equal': shape_equal,
			'dtype_equal': dtype_equal,
			'valid_token_mask_exact': valid_equal,
			'invalid_label_locations_exact': valid_equal,
			'decoded_labels_exact': labels_equal,
			'mismatch_token_count': int(np.count_nonzero(mismatch)),
			'mismatch_token_fraction': float(np.mean(mismatch)),
		},
		'ordered_diagnostics': {
			'historical': _mapping(historical_metadata, 'ordered_diagnostics').get('aggregate'),
			'current': _mapping(current_metadata, 'ordered_diagnostics').get('aggregate'),
			'equal': _mapping(historical_metadata, 'ordered_diagnostics').get('aggregate')
			== _mapping(current_metadata, 'ordered_diagnostics').get('aggregate'),
		},
		'centers': {
			'historical_path': str(config.hmm['centers']),
			'current_path': str(current_centers_path),
			'comparison': center_numeric,
			'tolerance': {'rtol': 1.0e-6, 'atol': 1.0e-6},
		},
		'iteration_summaries': _iteration_summary_comparison(
			historical_model,
			current_model,
		),
		'prepared_feature_summary': {
			'historical': _mapping(historical_metadata, 'stratigraphic_hmm').get(
				'prepared_feature_cache',
			),
			'current': _mapping(current_metadata, 'stratigraphic_hmm').get(
				'prepared_feature_cache',
			),
		},
		'label_mismatch_diagnostics': {
			key: value
			for key, value in diagnostics.items()
			if key not in {'mismatch_rows', 'transition_rows'}
		},
		'status': 'EXACT' if labels_equal and valid_equal else 'DRIFT',
	}
	_write_json_atomic(json_path, payload)
	_write_text_atomic(output_dir / 'hmm_parity.md', _render_hmm_markdown(payload))
	return payload


def export_legacy_m1_pseudo_targets(
	config: PerformanceMigrationValidationConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Export current K=6 labels in the historical M1 schema-v1 form.

	The historical M1 target has no boundary-weight field.  This compatibility
	exporter intentionally does not call the newer schema-v2 exporter because it
	would introduce a boundary artifact that did not participate in M1 science.
	"""
	_assert_live_git_sha(config)
	final_root = config.migration_root / 'pseudo_targets' / 'm1_k6_current_replay'
	if dry_run:
		label_path = (
			config.migration_root
			/ 'clustering'
			/ 'm1_k6_current_replay'
			/ 'labels'
			/ 'k6'
			/ 'f3_facies_benchmark.cluster_labels_token.npy'
		)
		if not label_path.is_file():
			raise FileNotFoundError(f'current K=6 label needed for pseudo export: {label_path}')
		return {
			'stage': 'm1_k6_pseudo_target_export',
			'status': 'DRY_RUN',
			'output_root': str(final_root),
			'schema_version': 1,
			'write_boundary_weight': False,
		}
	if only_missing and _pseudo_complete_and_valid(final_root, config=config):
		return _load_json(final_root / 'migration_completion.json')
	_prepare_output_for_regeneration(final_root, only_missing=only_missing)
	staging_root = final_root.parent / f'.{final_root.name}.staging-{uuid4().hex}'
	try:
		label_path = (
			config.migration_root
			/ 'clustering'
			/ 'm1_k6_current_replay'
			/ 'labels'
			/ 'k6'
			/ 'f3_facies_benchmark.cluster_labels_token.npy'
		)
		label_metadata = label_path.with_name(
			'f3_facies_benchmark.cluster_label_metadata.json',
		)
		labels = np.asarray(np.load(label_path), dtype=np.int32)
		valid = labels >= 0
		confidence = np.zeros(labels.shape, dtype=np.float32)
		confidence[valid] = np.float32(1.0)
		output_dir = staging_root / 'k6'
		output_dir.mkdir(parents=True, exist_ok=False)
		prefix = 'f3_facies_benchmark'
		labels_output = output_dir / f'{prefix}.hmm_labels_token.npy'
		confidence_output = output_dir / f'{prefix}.hmm_confidence_token.npy'
		valid_output = output_dir / f'{prefix}.valid_tokens.npy'
		metadata_output = output_dir / f'{prefix}.pseudo_target_metadata.json'
		np.save(labels_output, labels)
		np.save(confidence_output, confidence)
		np.save(valid_output, valid.astype(np.bool_, copy=False))
		metadata = {
			'artifact_type': 'strat_hmm_pseudo_target',
			'schema_version': 1,
			'survey_id': prefix,
			'k': 6,
			'token_grid_shape': list(labels.shape),
			'valid_token_count': int(np.count_nonzero(valid)),
			'invalid_token_count': int(valid.size - np.count_nonzero(valid)),
			'label_counts': {
				str(index): int(value)
				for index, value in enumerate(np.bincount(labels[valid], minlength=6))
			},
			'source': {
				'export_confidence': 1.0,
				'source_clustering_output_dir': str(label_path.parents[3]),
				'source_label_path': str(label_path),
				'source_metadata_path': str(label_metadata),
				'source_metadata_sha256': _sha256_path(label_metadata),
				'source_method': 'stratigraphic_hmm_kmeans',
			},
			'migration': {
				'current_git_sha': config.current_git_sha,
				'historical_baseline_sha': config.historical_baseline_sha,
				'legacy_schema_contract': 'v1_no_boundary_weight',
			},
		}
		_write_json_atomic(metadata_output, metadata)
		completion = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'stage': 'm1_k6_pseudo_target_export',
			'completion_status': 'COMPLETE',
			'current_git_sha': config.current_git_sha,
			'historical_baseline_sha': config.historical_baseline_sha,
			'legacy_schema_version': 1,
			'boundary_weight_written': False,
			'files': {
				'labels': _sha256_path(labels_output),
				'confidence': _sha256_path(confidence_output),
				'valid_tokens': _sha256_path(valid_output),
				'metadata': _sha256_path(metadata_output),
			},
		}
		_write_json_atomic(staging_root / 'migration_completion.json', completion)
		if not _pseudo_complete_and_valid(staging_root, config=config):
			raise RuntimeError('staged pseudo-target artifact failed completion validation')
		final_root.parent.mkdir(parents=True, exist_ok=True)
		os.replace(staging_root, final_root)
		return _load_json(final_root / 'migration_completion.json')
	except BaseException:
		if staging_root.exists():
			_quarantine_path(staging_root, reason='pseudo_target_stage_failure')
		raise


def compare_pseudo_targets(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Compare historical and current schema-v1 pseudo-target artifacts."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'pseudo_targets'
	json_path = output_dir / 'pseudo_target_parity.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	current_root = output_dir / 'm1_k6_current_replay' / 'k6'
	if not _pseudo_complete_and_valid(current_root.parent, config=config):
		raise FileNotFoundError(f'current pseudo target artifact incomplete: {current_root}')
	prefix = 'f3_facies_benchmark'
	current_paths = {
		'labels': current_root / f'{prefix}.hmm_labels_token.npy',
		'confidence': current_root / f'{prefix}.hmm_confidence_token.npy',
		'valid_tokens': current_root / f'{prefix}.valid_tokens.npy',
		'metadata': current_root / f'{prefix}.pseudo_target_metadata.json',
	}
	historical_labels = np.load(config.pseudo_targets['labels'], mmap_mode='r')
	historical_confidence = np.load(config.pseudo_targets['confidence'], mmap_mode='r')
	historical_valid = np.load(config.pseudo_targets['valid_tokens'], mmap_mode='r')
	current_labels = np.load(current_paths['labels'], mmap_mode='r')
	current_confidence = np.load(current_paths['confidence'], mmap_mode='r')
	current_valid = np.load(current_paths['valid_tokens'], mmap_mode='r')
	labels_exact = bool(np.array_equal(historical_labels, current_labels))
	valid_exact = bool(np.array_equal(historical_valid, current_valid))
	confidence_exact = bool(np.array_equal(historical_confidence, current_confidence))
	confidence_comparison = _numeric_array_comparison(
		np.asarray(historical_confidence),
		np.asarray(current_confidence),
		rtol=1.0e-6,
		atol=1.0e-6,
	)
	min_confidence = 0.0
	threshold_crossing = int(
		np.count_nonzero(
			(np.asarray(historical_valid, dtype=bool))
			& ((historical_confidence >= min_confidence) != (current_confidence >= min_confidence))
		),
	)
	historical_metadata = _load_json(config.pseudo_targets['metadata'])
	current_metadata = _load_json(current_paths['metadata'])
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'pseudo_target_parity',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'labels': {
			'exact': labels_exact,
			'historical_sha256': _sha256_path(config.pseudo_targets['labels']),
			'current_sha256': _sha256_path(current_paths['labels']),
			'shape_equal': historical_labels.shape == current_labels.shape,
			'dtype_equal': historical_labels.dtype == current_labels.dtype,
		},
		'valid_tokens': {
			'exact': valid_exact,
			'historical_sha256': _sha256_path(config.pseudo_targets['valid_tokens']),
			'current_sha256': _sha256_path(current_paths['valid_tokens']),
		},
		'confidence': {
			'exact': confidence_exact,
			'comparison': confidence_comparison,
			'min_confidence_threshold': min_confidence,
			'threshold_crossing_count': threshold_crossing,
		},
		'schema': {
			'historical_version': historical_metadata.get('schema_version'),
			'current_version': current_metadata.get('schema_version'),
			'historical_boundary_weight_present': _historical_boundary_weight_present(
				config.pseudo_targets['historical_root'],
			),
			'current_boundary_weight_present': any(
				current_root.glob('*.hmm_boundary_weight_token.npy'),
			),
			'artifact_type_equal': historical_metadata.get('artifact_type')
			== current_metadata.get('artifact_type'),
			'k_equal': historical_metadata.get('k') == current_metadata.get('k'),
			'survey_id_equal': historical_metadata.get('survey_id')
			== current_metadata.get('survey_id'),
		},
		'metadata_scientific_fields': _pseudo_scientific_metadata_comparison(
			historical_metadata,
			current_metadata,
		),
		'status': (
			'EXACT'
			if labels_exact and valid_exact and confidence_exact and threshold_crossing == 0
			else 'DRIFT'
		),
	}
	_write_json_atomic(json_path, payload)
	_write_text_atomic(output_dir / 'pseudo_target_parity.md', _render_pseudo_markdown(payload))
	return payload


def run_performance_benchmark(
	config: PerformanceMigrationValidationConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> dict[str, object]:
	"""Run the repository's synthetic performance smoke and fixed benchmark."""
	_assert_live_git_sha(config)
	output_dir = config.migration_root / 'benchmark'
	json_path = output_dir / 'benchmark_manifest.json'
	if only_missing and _is_complete_file(json_path):
		return _load_json(json_path)
	command_root = Path(__file__).resolve().parents[3]
	tool = command_root / 'tools' / 'benchmark_seis_ssl_cluster_performance.py'
	if dry_run:
		return {
			'stage': 'benchmark',
			'status': 'DRY_RUN',
			'smoke_command': [sys.executable, str(tool), '--smoke'],
			'full_command': _benchmark_command(config, tool, output_dir),
		}
	output_dir.mkdir(parents=True, exist_ok=True)
	smoke = _run_command([sys.executable, str(tool), '--smoke'], cwd=command_root)
	if smoke['returncode'] != 0:
		payload = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'stage': 'benchmark',
			'status': 'FAILED',
			'smoke': smoke,
		}
		_write_json_atomic(json_path, payload)
		raise RuntimeError('synthetic benchmark smoke failed')
	full_command = _benchmark_command(config, tool, output_dir)
	environment = os.environ.copy()
	environment.update(
		{
			'OMP_NUM_THREADS': str(config.benchmark['threads']),
			'MKL_NUM_THREADS': str(config.benchmark['threads']),
			'PYTORCH_NUM_THREADS': str(config.benchmark['threads']),
		},
	)
	full = _run_command(full_command, cwd=command_root, environment=environment)
	if full['returncode'] != 0:
		raise RuntimeError('full synthetic benchmark failed')
	benchmark = _load_json(output_dir / 'current.json')
	payload = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'benchmark',
		'status': 'PASS',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'smoke': smoke,
		'full': full,
		'benchmark': benchmark,
		'comparison_status': 'NOT_COMPARABLE',
		'reason': 'no historical-compatible benchmark JSON was supplied',
	}
	_write_json_atomic(json_path, payload)
	return payload


def summarize_performance_migration(
	config: PerformanceMigrationValidationConfig,
	*,
	only_missing: bool = False,
	publish: bool = True,
	dry_run: bool = False,
) -> dict[str, object]:
	"""Aggregate independent stage reports, decide scope, and publish only light files."""
	_assert_live_git_sha(config)
	reports = config.migration_root / 'reports'
	summary_path = reports / 'performance_migration_summary.json'
	if dry_run:
		return {
			'stage': 'publish' if publish else 'summarize',
			'status': 'DRY_RUN',
			'reports_root': str(reports),
			'publish_root': str(config.publish_root),
			'publish_enabled': publish,
		}
	if only_missing and _is_complete_file(summary_path):
		return _load_json(summary_path)
	reports.mkdir(parents=True, exist_ok=True)
	# This invocation rewrites report sources. Preserve any previous lightweight
	# publish before its source-hash contract can become stale.
	if config.publish_root.exists():
		_quarantine_publish_output(config, config.publish_root)
	stage_paths = {
		'preflight': config.migration_root / 'preflight' / 'input_inventory.json',
		'checkpoint_smoke': config.migration_root
		/ 'checkpoint_smoke'
		/ 'checkpoint_smoke.json',
		'embedding_parity': config.migration_root
		/ 'embedding_parity'
		/ 'embedding_parity.json',
		'probe_parity': config.migration_root / 'probe_parity' / 'probe_parity.json',
		'historical_hmm_config': config.migration_root
		/ 'clustering'
		/ 'historical_k6_scientific_config.json',
		'hmm_parity': config.migration_root / 'clustering' / 'hmm_parity.json',
		'pseudo_target_parity': config.migration_root
		/ 'pseudo_targets'
		/ 'pseudo_target_parity.json',
		'benchmark': config.migration_root / 'benchmark' / 'benchmark_manifest.json',
	}
	stages = {
		name: (_load_json(path) if path.is_file() else {'status': 'MISSING', 'path': str(path)})
		for name, path in stage_paths.items()
	}
	decision = _decide_migration(stages)
	manifest = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'performance_migration_manifest',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'stage_artifacts': {
			name: {
				'path': str(path),
				'exists': path.is_file(),
				'sha256': _sha256_path(path) if path.is_file() else None,
			}
			for name, path in stage_paths.items()
		},
		'quarantine_paths': _quarantine_paths(config.migration_root),
	}
	decision_payload = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': 'migration_decision',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		**decision,
	}
	summary: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'status': 'COMPLETE' if decision['complete'] else decision['status'],
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'input_inventory': stages['preflight'],
		'checkpoint_status': stages['checkpoint_smoke'],
		'embedding_abc_status': stages['embedding_parity'],
		'probe_parity': stages['probe_parity'],
		'historical_hmm_config': stages['historical_hmm_config'],
		'hmm_parity': stages['hmm_parity'],
		'pseudo_target_parity': stages['pseudo_target_parity'],
		'benchmark_status': stages['benchmark'],
		'migration_decision': decision_payload,
		'required_rerun_scope': decision['required_rerun_scope'],
		'multi_head_baseline_policy': decision['multi_head_baseline_policy'],
		'atomic_write_provenance': (
			'Producer runtime configuration may retain its temporary staging path; '
			'the committed artifact directory and completion manifest identify the '
			'final location. These are path-only provenance fields, not scientific identity.'
		),
		'unresolved_issues': decision['reasons'],
	}
	_write_json_atomic(reports / 'performance_migration_manifest.json', manifest)
	_write_json_atomic(reports / 'performance_migration_decision.json', decision_payload)
	_write_json_atomic(summary_path, summary)
	_write_text_atomic(
		reports / 'performance_migration_summary.md',
		_render_summary_markdown(summary),
	)
	_write_text_atomic(
		reports / 'performance_migration_handoff.md',
		_render_handoff_markdown(summary, manifest),
	)
	if publish and decision['complete']:
		publish_manifest = _publish_migration_reports(config, reports)
		_write_json_atomic(reports / 'publish_manifest.json', publish_manifest)
		return {**summary, 'publish_manifest': publish_manifest}
	return summary


def _decide_migration(stages: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
	"""Implement the documented ordered decision rules without significance claims."""
	reasons: list[str] = []
	missing = [name for name, stage in stages.items() if stage.get('status') == 'MISSING']
	checkpoint = stages['checkpoint_smoke']
	preflight = stages['preflight']
	embedding = stages['embedding_parity']
	probe = stages['probe_parity']
	hmm = stages['hmm_parity']
	pseudo = stages['pseudo_target_parity']
	if missing:
		reasons.append(f'missing required stage reports: {missing!r}')
		return _decision_payload(
			'BLOCKED_NUMERIC_CONTRACT',
			reasons,
			complete=False,
		)
	if checkpoint.get('status') != 'PASS':
		reasons.append('checkpoint compatibility smoke did not pass')
		return _decision_payload('BLOCKED_NUMERIC_CONTRACT', reasons, complete=False)
	if int(preflight.get('missing_input_count', 0)) != 0:
		reasons.append('preflight inventory has missing fixed inputs')
		return _decision_payload('BLOCKED_NUMERIC_CONTRACT', reasons, complete=False)
	if stages['benchmark'].get('status') != 'PASS':
		reasons.append('synthetic benchmark smoke or full benchmark did not pass')
		return _decision_payload('BLOCKED_NUMERIC_CONTRACT', reasons, complete=False)
	for comparison in cast('Mapping[str, object]', embedding.get('comparisons', {})).values():
		if cast('Mapping[str, object]', comparison).get('status') == 'BLOCKED_NUMERIC_CONTRACT':
			reasons.append('embedding valid-mask or structural contract failed')
			return _decision_payload('BLOCKED_NUMERIC_CONTRACT', reasons, complete=False)
	labels = _mapping(hmm, 'labels')
	if labels.get('valid_token_mask_exact') is not True or _mapping(
		pseudo,
		'valid_tokens',
	).get('exact') is not True:
		reasons.append('HMM or pseudo-target valid-token mask corruption detected')
		return _decision_payload('BLOCKED_NUMERIC_CONTRACT', reasons, complete=False)
	if not bool(labels.get('decoded_labels_exact')) or not bool(
		_mapping(pseudo, 'labels').get('exact')
	) or int(
		_mapping(pseudo, 'confidence').get('threshold_crossing_count', 0)
	) != 0:
		reasons.append('K=6 HMM or pseudo-target decoded target contract drifted')
		return _decision_payload('REBUILD_M1_REQUIRED', reasons, complete=True)
	probe_pairs = _mapping(probe, 'parity')
	if any(
		cast('Mapping[str, object]', item).get('prediction_exact') is not True
		for item in probe_pairs.values()
	):
		reasons.append('existing linear-probe predictions changed for an embedding variant')
		return _decision_payload('REEXTRACT_REQUIRED', reasons, complete=True)
	drift = any(
		cast('Mapping[str, object]', comparison).get('status') != 'EXACT'
		for comparison in _mapping(embedding, 'comparisons').values()
	)
	centers = _mapping(hmm, 'centers').get('comparison')
	if isinstance(centers, Mapping) and centers.get('allclose') is not True:
		drift = True
	if not bool(_mapping(pseudo, 'confidence').get('exact')):
		drift = True
	if drift:
		reasons.append('small numeric artifact drift with downstream labels and predictions exact')
		return _decision_payload('PASS_WITH_NUMERIC_DRIFT', reasons, complete=True)
	reasons.append('all required historical/current parity gates are exact')
	return _decision_payload('PASS_REUSE_EXISTING', reasons, complete=True)


def _decision_payload(
	status: MigrationDecision,
	reasons: list[str],
	*,
	complete: bool,
) -> dict[str, object]:
	policy = {
		'PASS_REUSE_EXISTING': (
			'historical M1 can be reused as the multi-head baseline; retain a '
			'no-consistency guardrail for future multi-head comparisons.'
		),
		'PASS_WITH_NUMERIC_DRIFT': (
			'train a current-code single-head K=6 control under the same conditions '
			'before comparing multi-head K=6/8/10.'
		),
		'REEXTRACT_REQUIRED': (
			'regenerate versioned current-code MAE/M1/M2-A embeddings and required '
			'downstream baselines; retain historical results unchanged.'
		),
		'REBUILD_M1_REQUIRED': (
			'build current K=6 targets and train a current-code single-head M1 '
			'before any multi-head baseline comparison.'
		),
		'BLOCKED_NUMERIC_CONTRACT': 'do not proceed to multi-head experiments.',
	}[status]
	run_scope = {
		'PASS_REUSE_EXISTING': 'none; existing M3-V and M3-V-LB artifacts remain reusable',
		'PASS_WITH_NUMERIC_DRIFT': 'no historical rerun; add a future current-code K=6 control',
		'REEXTRACT_REQUIRED': 'embedding and downstream baselines only; no pretraining/HMM rerun',
		'REBUILD_M1_REQUIRED': 'current K=6 HMM/pseudo-target and single-head M1 rebuild',
		'BLOCKED_NUMERIC_CONTRACT': 'repair numeric contract then resume migration validation',
	}[status]
	return {
		'status': status,
		'complete': complete,
		'priority_order': [
			'BLOCKED_NUMERIC_CONTRACT',
			'REBUILD_M1_REQUIRED',
			'REEXTRACT_REQUIRED',
			'PASS_WITH_NUMERIC_DRIFT',
			'PASS_REUSE_EXISTING',
		],
		'reasons': reasons,
		'required_rerun_scope': run_scope,
		'multi_head_baseline_policy': policy,
	}


def _publish_migration_reports(
	config: PerformanceMigrationValidationConfig,
	reports: Path,
) -> dict[str, object]:
	"""Atomically publish a whitelisted small report set and verify all hashes."""
	selected = [
		reports / 'performance_migration_summary.md',
		reports / 'performance_migration_summary.json',
		reports / 'performance_migration_manifest.json',
		reports / 'performance_migration_decision.json',
		reports / 'performance_migration_handoff.md',
		config.migration_root / 'preflight' / 'input_inventory.md',
		config.migration_root / 'embedding_parity' / 'embedding_parity.md',
		config.migration_root / 'embedding_parity' / 'embedding_parity.json',
		config.migration_root / 'probe_parity' / 'probe_parity.md',
		config.migration_root / 'probe_parity' / 'probe_parity.json',
		config.migration_root / 'clustering' / 'hmm_parity.md',
		config.migration_root / 'clustering' / 'hmm_parity.json',
		config.migration_root / 'pseudo_targets' / 'pseudo_target_parity.md',
		config.migration_root / 'pseudo_targets' / 'pseudo_target_parity.json',
		config.migration_root / 'benchmark' / 'current.md',
		config.migration_root / 'benchmark' / 'current.json',
	]
	if any(not path.is_file() for path in selected):
		missing = [str(path) for path in selected if not path.is_file()]
		raise FileNotFoundError(f'cannot publish incomplete migration reports: {missing!r}')
	root = config.publish_root
	if root.exists():
		manifest_path = root / 'publish_manifest.json'
		try:
			manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
			if _validate_publish_manifest(root, manifest):
				return manifest
		except (OSError, TypeError, ValueError):
			pass
		_quarantine_publish_output(config, root)
	staging = root.parent / f'.{root.name}.staging-{uuid4().hex}'
	try:
		staging.mkdir(parents=True, exist_ok=False)
		items: list[dict[str, object]] = []
		standard_items: list[dict[str, object]] = []
		for source in selected:
			if source.suffix not in {'.md', '.json', '.csv', '.png'}:
				raise ValueError(f'publish raw artifact exclusion rejected: {source}')
			destination = staging / source.name
			shutil.copy2(source, destination)
			items.append(
				{
					'source_artifact_path': str(source),
					'source_sha256': _sha256_path(source),
					'relative_path': destination.name,
					'published_sha256': _sha256_path(destination),
					'byte_size': destination.stat().st_size,
				},
			)
			standard_items.append(
				{
					'source': str(source),
					'target': destination.name,
					'size_bytes': destination.stat().st_size,
					'sha256': _sha256_path(destination),
				},
			)
		manifest = {
			'artifact_type': ARTIFACT_TYPE,
			'schema_version': SCHEMA_VERSION,
			'current_git_sha': config.current_git_sha,
			'historical_baseline_sha': config.historical_baseline_sha,
			'created_at_utc': _utc_now(),
			'source_artifact_root': str(config.migration_root),
			'output_dir': str(root),
			'items': standard_items,
			'skipped_optional_items': [],
			'warnings': [],
			'files': items,
		}
		_write_json_atomic(staging / 'publish_manifest.json', manifest)
		if not _validate_publish_manifest(staging, manifest):
			raise RuntimeError('staged publish manifest validation failed')
		root.parent.mkdir(parents=True, exist_ok=True)
		os.replace(staging, root)
		return manifest
	except BaseException:
		if staging.exists():
			_quarantine_path(staging, reason='publish_stage_failure')
		raise


def _inventory_input_records(
	config: PerformanceMigrationValidationConfig,
) -> list[dict[str, object]]:
	"""Return deterministic file-level preflight records for all fixed inputs."""
	records: list[tuple[str, Path, tuple[str, ...], Path | None]] = []
	for role, path in sorted(config.checkpoints.items()):
		records.append((f'{role}_checkpoint', path, ('checkpoint_smoke',), None))
	for role, root in sorted(config.historical_embeddings.items()):
		records.extend(
			[
				(
					f'{role}_historical_embedding',
					root / _EMBEDDING_FILE_NAME,
					('embedding_parity',),
					root / _EMBEDDING_METADATA_NAME,
				),
				(
					f'{role}_historical_valid_tokens',
					root / _VALID_FILE_NAME,
					('embedding_parity',),
					root / _EMBEDDING_METADATA_NAME,
				),
				(
					f'{role}_historical_embedding_metadata',
					root / _EMBEDDING_METADATA_NAME,
					('embedding_parity',),
					None,
				),
			],
		)
	for key, path in sorted(config.m1_probe.items()):
		records.append((f'm1_probe_{key}', path, ('probe_parity',), None))
	for key, path in sorted(config.hmm.items()):
		records.append((f'historical_hmm_{key}', path, ('hmm_replay', 'hmm_parity'), None))
	for key, path in sorted(config.pseudo_targets.items()):
		records.append((f'historical_pseudo_target_{key}', path, ('pseudo_target_parity',), None))
	for key, path in sorted(config.f3.items()):
		records.append((f'f3_{key}', path, ('checkpoint_smoke',), None))
	return [
		_path_record(path, logical_role=role, dependent_stages=stages, metadata_path=metadata)
		for role, path, stages, metadata in records
	]


def _path_record(
	path: Path,
	*,
	logical_role: str | None = None,
	dependent_stages: tuple[str, ...],
	metadata_path: Path | None = None,
) -> dict[str, object]:
	"""Represent a file or directory without making assumptions for missing data."""
	record: dict[str, object] = {
		'logical_role': logical_role,
		'path': str(path),
		'exists': path.exists(),
		'dependent_stages': list(dependent_stages),
		'status': 'PRESENT' if path.exists() else 'MISSING',
	}
	if not path.exists():
		return record
	stat = path.stat()
	record.update(
		{
			'file_type': 'directory' if path.is_dir() else 'file',
			'byte_size': _path_size(path),
		'mtime_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
			'sha256': _sha256_path(path),
			'metadata_path': None if metadata_path is None else str(metadata_path),
			'metadata_sha256': (
				_sha256_path(metadata_path)
				if metadata_path is not None and metadata_path.is_file()
				else None
			),
		}
	)
	metadata = _try_metadata_for_path(path, metadata_path)
	if metadata is not None:
		record['schema_version'] = metadata.get('schema_version')
		record['producer_git_sha'] = _producer_git_sha(metadata)
	else:
		record['schema_version'] = None
		record['producer_git_sha'] = _run_metadata_git_sha(path)
	return record


def _try_metadata_for_path(
	path: Path,
	metadata_path: Path | None,
) -> Mapping[str, object] | None:
	candidates = [metadata_path] if metadata_path is not None else []
	if path.suffix == '.json':
		candidates.append(path)
	for candidate in candidates:
		if candidate is not None and candidate.is_file():
			try:
				payload = _load_json(candidate)
			except (OSError, ValueError, TypeError):
				continue
			return payload
	return None


def _producer_git_sha(metadata: Mapping[str, object]) -> str | None:
	for key in ('git_sha', 'git_commit', 'producer_git_sha', 'commit'):
		value = metadata.get(key)
		if isinstance(value, str) and value:
			return value
	producer = metadata.get('producer')
	if isinstance(producer, Mapping):
		return _producer_git_sha(cast('Mapping[str, object]', producer))
	return None


def _run_metadata_git_sha(path: Path) -> str | None:
	for parent in (path.parent, *path.parents):
		candidate = parent / 'run_metadata.json'
		if candidate.is_file():
			try:
				return _producer_git_sha(_load_json(candidate))
			except (OSError, ValueError, TypeError):
				return None
	return None


def _environment_metadata() -> dict[str, object]:
	"""Collect portability context without making it part of scientific identity."""
	cuda_available = torch.cuda.is_available()
	device_properties = torch.cuda.get_device_properties(0) if cuda_available else None
	return {
		'python_version': sys.version,
		'numpy_version': np.__version__,
		'torch_version': torch.__version__,
		'cuda_available': cuda_available,
		'cuda_version': torch.version.cuda,
		'gpu_name': None if device_properties is None else device_properties.name,
		'bf16_supported': bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
		'fp16_supported': cuda_available,
		'hostname': socket.gethostname(),
		'platform': platform.platform(),
	}


def _git_metadata(historical_baseline_sha: str) -> dict[str, object]:
	return {
		'head': _git_output(('rev-parse', 'HEAD')).strip(),
		'dirty_status': _git_output(('status', '--short')).splitlines(),
		'historical_baseline_sha': historical_baseline_sha,
		'baseline_is_ancestor': _git_returncode(
			('merge-base', '--is-ancestor', historical_baseline_sha, 'HEAD'),
		)
		== 0,
		'log': _git_output(('log', '-10', '--oneline')).splitlines(),
	}


def _changed_components(historical_baseline_sha: str) -> dict[str, object]:
	files = _git_output(('diff', '--name-only', f'{historical_baseline_sha}..HEAD')).splitlines()
	prefixes = {
		'embedding': ('src/seis_ssl_cluster/embedding/',),
		'clustering': ('src/seis_ssl_cluster/clustering/',),
		'stratigraphy': ('src/seis_ssl_cluster/stratigraphy/',),
		'pseudo_target': ('src/seis_ssl_cluster/stratigraphy/',),
		'checkpoint_schema': ('src/seis_ssl_cluster/training/checkpoint',),
		'artifact_writer': (
			'src/seis_ssl_cluster/embedding/writer',
			'src/seis_ssl_cluster/clustering/writer',
		),
	}
	return {
		'changed_files': files,
		'components': {
			name: any(file.startswith(prefix) for file in files for prefix in values)
			for name, values in prefixes.items()
		},
	}


def _assert_live_git_sha(config: PerformanceMigrationValidationConfig) -> None:
	actual = _git_output(('rev-parse', 'HEAD')).strip()
	if actual != config.current_git_sha:
		raise RuntimeError(
			'active git SHA differs from migration config: '
			f'expected={config.current_git_sha} actual={actual}',
		)


def _git_output(arguments: tuple[str, ...]) -> str:
	return subprocess.run(
		['git', *arguments],
		cwd=Path(__file__).resolve().parents[3],
		check=True,
		capture_output=True,
		text=True,
	).stdout


def _git_returncode(arguments: tuple[str, ...]) -> int:
	return subprocess.run(
		['git', *arguments],
		cwd=Path(__file__).resolve().parents[3],
		check=False,
		capture_output=True,
		text=True,
	).returncode


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	value = payload.get('config')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint is missing a resolved config mapping')
	return cast('Mapping[str, object]', value)


def _model_state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	value = payload.get('model_state_dict')
	if not isinstance(value, Mapping):
		raise TypeError('checkpoint is missing model_state_dict')
	state = cast('Mapping[str, torch.Tensor]', value)
	if not state or not all(isinstance(item, torch.Tensor) for item in state.values()):
		raise TypeError('checkpoint model_state_dict must contain tensors only')
	return state


def _preprocess_settings(
	checkpoint_config: Mapping[str, object],
	*,
	finite_check_mode: str,
) -> AmplitudePreprocessSettings:
	if finite_check_mode not in {'strict', 'output_only', 'off'}:
		raise ValueError(f'unsupported finite check mode: {finite_check_mode!r}')
	data = _mapping(checkpoint_config, 'data')
	return AmplitudePreprocessSettings(
		zero_mask=_zero_mask_from_config(checkpoint_config),
		normalized_clip_abs=(
			None
			if data.get('normalized_clip_abs') is None
			else float(cast('float', data['normalized_clip_abs']))
		),
		amplitude_agc=_amplitude_agc_from_config(checkpoint_config),
		min_token_valid_fraction=float(cast('float', data['min_valid_fraction'])),
		finite_check_mode=cast('Any', finite_check_mode),
	)


def _model_geometry(checkpoint_config: Mapping[str, object]) -> dict[str, object]:
	model = _mapping(checkpoint_config, 'model')
	return {
		'patch_size': list(cast('Sequence[object]', model['patch_size'])),
		'encoder_dim': model['encoder_dim'],
		'encoder_depth': model['encoder_depth'],
		'encoder_heads': model['encoder_heads'],
		'decoder_dim': model['decoder_dim'],
		'decoder_depth': model['decoder_depth'],
		'decoder_heads': model['decoder_heads'],
	}


def _stratigraphy_checkpoint_identity(
	payload: Mapping[str, object],
	*,
	feature_dim: int,
) -> dict[str, object] | None:
	config = payload.get('stratigraphy_config')
	state = payload.get('stratigraphy_state_dict')
	if config is None and state is None:
		return None
	if not isinstance(config, Mapping) or not isinstance(state, Mapping):
		raise TypeError('stratigraphy checkpoint identity is incomplete')
	head_config = _mapping(cast('Mapping[str, object]', config), 'head')
	head = OrderedPrototypeHead(
		feature_dim=feature_dim,
		num_prototypes=int(head_config['num_prototypes']),
		projection_dim=(
			None
			if head_config.get('projection_dim') is None
			else int(cast('int', head_config['projection_dim']))
		),
		temperature=float(cast('float', head_config.get('temperature', 0.1))),
		normalize=bool(head_config.get('normalize', True)),
	)
	load_result = head.load_state_dict(cast('Mapping[str, torch.Tensor]', state), strict=False)
	if load_result.missing_keys or load_result.unexpected_keys:
		raise ValueError(
			'stratigraphy state_dict mismatch: '
			f'missing={load_result.missing_keys!r} '
			f'unexpected={load_result.unexpected_keys!r}',
		)
	return {
		'config': _jsonable(config),
		'state_dict_keys': sorted(str(key) for key in state),
		'state_dict_shapes': {
			str(key): list(value.shape)
			for key, value in state.items()
			if isinstance(value, torch.Tensor)
		},
		'all_finite': all(
			bool(torch.isfinite(value).all())
			for value in state.values()
			if isinstance(value, torch.Tensor) and value.is_floating_point()
		),
		'missing_keys': list(load_result.missing_keys),
		'unexpected_keys': list(load_result.unexpected_keys),
	}


def _set_runtime_check_mode_strict(model: torch.nn.Module) -> None:
	"""Make the one-crop smoke use strict checks without changing model weights."""
	for module in model.modules():
		if hasattr(module, 'runtime_checks'):
			module.runtime_checks = RuntimeChecks('strict')


def _legacy_finite_contract(
	config: PerformanceMigrationValidationConfig,
) -> dict[str, object]:
	return {
		'reconstructed_value': config.compatibility['m1_historical_finite_check_mode'],
		'evidence_commit': config.compatibility['historical_finite_check_evidence_commit'],
		'evidence_path': config.compatibility['historical_finite_check_evidence_path'],
		'evidence': HISTORICAL_FINITE_CHECK_EVIDENCE,
		'checkpoint_serialized_field_present': False,
	}


def _legacy_m1_checkpoint_config_override(
	config: PerformanceMigrationValidationConfig,
	serialized: Mapping[str, object],
) -> dict[str, object]:
	"""Apply only the specifically evidenced absent legacy finite-check field."""
	if _mapping(serialized, 'data').get('finite_check_mode') is not None:
		raise ValueError('legacy M1 checkpoint unexpectedly declares finite_check_mode')
	override = deepcopy(dict(serialized))
	data = cast('dict[str, object]', override['data'])
	data['finite_check_mode'] = config.compatibility['m1_historical_finite_check_mode']
	diff = _mapping_diff(serialized, override)
	if diff != {'data.finite_check_mode': {'historical': None, 'current': 'off'}}:
		raise RuntimeError(f'illegal M1 compatibility override diff: {diff!r}')
	return override


def _embedding_completion_payload(  # noqa: PLR0913
	*,
	config: PerformanceMigrationValidationConfig,
	mode: str,
	output_dir: Path,
	config_path: Path,
	resolved_config: Mapping[str, object],
	override: Mapping[str, object],
) -> dict[str, object]:
	files = {
		name: _sha256_path(output_dir / name)
		for name in (_EMBEDDING_FILE_NAME, _VALID_FILE_NAME, _EMBEDDING_METADATA_NAME)
	}
	return {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'stage': f'm1_embedding_{mode}',
		'completion_status': 'COMPLETE',
		'current_git_sha': config.current_git_sha,
		'historical_baseline_sha': config.historical_baseline_sha,
		'mode': mode,
		'config_path': str(config_path),
		'config_sha256': _sha256_path(config_path),
		'resolved_runtime_config': _jsonable(resolved_config),
		'checkpoint_config_override': _jsonable(override),
		'legacy_finite_check_contract': _legacy_finite_contract(config),
		'files': files,
	}


def _embedding_files_complete(root: Path) -> bool:
	return all((root / name).is_file() for name in (_EMBEDDING_FILE_NAME, _VALID_FILE_NAME, _EMBEDDING_METADATA_NAME))


def _embedding_complete_and_valid(
	root: Path,
	*,
	config: PerformanceMigrationValidationConfig,
	mode: str,
) -> bool:
	if not _embedding_files_complete(root):
		return False
	completion_path = root / 'migration_completion.json'
	if not completion_path.is_file():
		return False
	try:
		completion = _load_json(completion_path)
		if completion.get('completion_status') != 'COMPLETE' or completion.get('mode') != mode:
			return False
		if completion.get('current_git_sha') != config.current_git_sha:
			return False
		files = _mapping(completion, 'files')
		return all(
			files.get(name) == _sha256_path(root / name)
			for name in (_EMBEDDING_FILE_NAME, _VALID_FILE_NAME, _EMBEDDING_METADATA_NAME)
		)
	except (OSError, ValueError, TypeError):
		return False


def _prepare_output_for_regeneration(path: Path, *, only_missing: bool) -> None:
	"""Reject normal overwrite; quarantine incomplete/mismatched resume targets."""
	if not path.exists():
		return
	if not only_missing:
		raise FileExistsError(
			f'migration output exists and overwrite is forbidden: {path}; use --only-missing',
		)
	_quarantine_path(path, reason='invalid_or_partial_before_regeneration')


def _hmm_output_hashes(root: Path) -> dict[str, str]:
	paths = {
		'labels': root / 'labels' / 'k6' / 'f3_facies_benchmark.cluster_labels_token.npy',
		'label_metadata': root / 'labels' / 'k6' / 'f3_facies_benchmark.cluster_label_metadata.json',
		'centers': root / 'models' / 'k6' / 'cluster_centers.npy',
		'clustering_metadata': root / 'models' / 'k6' / 'clustering_metadata.json',
		'hmm_model': root / 'models' / 'k6' / 'hmm_model.joblib',
		'preprocessor': root / 'models' / 'k6' / 'preprocessor.joblib',
		'residualizer': root / 'models' / 'residualizer.npz',
	}
	return {name: _sha256_path(path) for name, path in paths.items()}


def _hmm_complete_and_valid(
	root: Path,
	*,
	config: PerformanceMigrationValidationConfig,
) -> bool:
	completion_path = root / 'migration_completion.json'
	if not completion_path.is_file():
		return False
	try:
		completion = _load_json(completion_path)
		if completion.get('completion_status') != 'COMPLETE' or completion.get('current_git_sha') != config.current_git_sha:
			return False
		return _mapping(completion, 'files') == _hmm_output_hashes(root)
	except (OSError, ValueError, TypeError):
		return False


def _pseudo_complete_and_valid(
	root: Path,
	*,
	config: PerformanceMigrationValidationConfig,
) -> bool:
	completion_path = root / 'migration_completion.json'
	prefix = root / 'k6' / 'f3_facies_benchmark'
	paths = {
		'labels': prefix.with_suffix('.hmm_labels_token.npy'),
		'confidence': prefix.with_suffix('.hmm_confidence_token.npy'),
		'valid_tokens': prefix.with_suffix('.valid_tokens.npy'),
		'metadata': prefix.with_suffix('.pseudo_target_metadata.json'),
	}
	if not completion_path.is_file() or any(not path.is_file() for path in paths.values()):
		return False
	try:
		completion = _load_json(completion_path)
		return (
			completion.get('completion_status') == 'COMPLETE'
			and completion.get('current_git_sha') == config.current_git_sha
			and completion.get('boundary_weight_written') is False
			and _mapping(completion, 'files')
			== {name: _sha256_path(path) for name, path in paths.items()}
		)
	except (OSError, ValueError, TypeError):
		return False


def _scientific_residualization(value: Mapping[str, object]) -> dict[str, object]:
	return {
		'enabled': value['enabled'],
		'mode': value['mode'],
		'group_by': value['group_by'],
		'add_global_mean_back': value['add_global_mean_back'],
		'min_group_count': value['min_group_count'],
	}


def _scientific_pca(value: Mapping[str, object]) -> dict[str, object]:
	return {
		'enabled': value['enabled'],
		'n_components': value['n_components'],
		'whiten': value['whiten'],
	}


def _scientific_path_prior(value: Mapping[str, object]) -> dict[str, object]:
	return {
		'enabled': value['enabled'],
		'initial_state': _mapping(value, 'initial_state'),
		'terminal_state': _mapping(value, 'terminal_state'),
		'expected_boundaries': {
			key: item
			for key, item in _mapping(value, 'expected_boundaries').items()
			if key != 'target_resolution'
		},
	}


def _scientific_clustering_view(config: Mapping[str, object]) -> dict[str, object]:
	"""Remove output/cache/timing mechanics before historical replay comparison."""
	clustering = _mapping(config, 'clustering')
	hmm = _mapping(clustering, 'stratigraphic_hmm')
	return {
		'embeddings': {'input_dir': _mapping(config, 'embeddings')['input_dir']},
		'clustering': {
			'embedding_normalization': clustering['embedding_normalization'],
			'residualization': _scientific_residualization(
				_mapping(clustering, 'residualization'),
			),
			'pca': _scientific_pca(_mapping(clustering, 'pca')),
			'sample_tokens': clustering['sample_tokens'],
			'method': clustering['method'],
			'k_values': clustering['k_values'],
			'minibatch_size': clustering['minibatch_size'],
			'prediction_batch_size': clustering['prediction_batch_size'],
			'seed': clustering['seed'],
			'stratigraphic_hmm': {
				'emission_source': hmm['emission_source'],
				'iterations': hmm['iterations'],
				'z_axis': hmm['z_axis'],
				'z_direction': hmm['z_direction'],
				'edge_margin_tokens': hmm.get('edge_margin_tokens', [0, 0, 0]),
				'transition': _mapping(hmm, 'transition'),
				'init': _mapping(hmm, 'init'),
				'update': _mapping(hmm, 'update'),
				'path_prior': _scientific_path_prior(_mapping(hmm, 'path_prior')),
			},
		},
	}


def _mapping_diff(
	left: Mapping[str, object],
	right: Mapping[str, object],
	*,
	prefix: str = '',
) -> dict[str, dict[str, object]]:
	"""Return a stable, leaf-level mapping diff suitable for a contract report."""
	result: dict[str, dict[str, object]] = {}
	for key in sorted(set(left) | set(right)):
		path = f'{prefix}.{key}' if prefix else str(key)
		left_value = left.get(key)
		right_value = right.get(key)
		if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
			result.update(
				_mapping_diff(
					cast('Mapping[str, object]', left_value),
					cast('Mapping[str, object]', right_value),
					prefix=path,
				),
			)
		elif left_value != right_value:
			result[path] = {'historical': _jsonable(left_value), 'current': _jsonable(right_value)}
	return result


def _embedding_scientific_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
	keys = (
		'survey_id',
		'volume_shape_xyz',
		'checkpoint_sha256',
		'model_geometry',
		'patch_size',
		'token_grid_shape',
		'window_size',
		'overlap',
		'output_dtype',
		'min_token_valid_fraction',
		'normalized_clip_abs',
		'amplitude_agc',
		'finite_check_mode',
		'zero_mask',
		'pretraining_objective',
		'stratigraphy_pretext',
	)
	return {key: _jsonable(metadata[key]) for key in keys if key in metadata}


def _embedding_runtime_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
	keys = ('precision', 'preprocessing_cache')
	return {key: _jsonable(metadata[key]) for key in keys if key in metadata}


def _compare_embedding_metadata(
	left: Mapping[str, object],
	right: Mapping[str, object],
) -> dict[str, object]:
	left_science = _embedding_scientific_metadata(left)
	right_science = _embedding_scientific_metadata(right)
	scientific_diff = _mapping_diff(left_science, right_science)
	finite_key = 'finite_check_mode'
	if scientific_diff.get(finite_key) == {'historical': None, 'current': 'off'}:
		scientific_diff.pop(finite_key)
	preprocess_finite = 'preprocessing.finite_check_mode'
	if scientific_diff.get(preprocess_finite) == {'historical': None, 'current': 'off'}:
		scientific_diff.pop(preprocess_finite)
	return {
		'scientific_identity_status': 'EXACT' if not scientific_diff else 'DRIFT',
		'scientific_field_diff': scientific_diff,
		'legacy_finite_check_reconciled': (
			'finite_check_mode' not in left and right.get('finite_check_mode') == 'off'
		),
		'performance_runtime_field_diff': _mapping_diff(
			_embedding_runtime_metadata(left),
			_embedding_runtime_metadata(right),
		),
		'path_only_fields': {
			'checkpoint_path': {'left': left.get('checkpoint_path'), 'right': right.get('checkpoint_path')},
			'source_amplitude_path': {
				'left': left.get('source_amplitude_path'),
				'right': right.get('source_amplitude_path'),
			},
			'normalization_stats_path': {
				'left': left.get('normalization_stats_path'),
				'right': right.get('normalization_stats_path'),
			},
		},
	}


def _distribution_summary(values: np.ndarray) -> dict[str, object]:
	array = np.asarray(values, dtype=np.float64).reshape(-1)
	if array.size == 0:
		return {
			'count': 0,
			'max': 0.0,
			'mean': 0.0,
			'median': 0.0,
			'p95': 0.0,
			'p99': 0.0,
			'p99_9': 0.0,
		}
	return {
		'count': int(array.size),
		'max': float(array.max()),
		'mean': float(array.mean()),
		'median': float(np.median(array)),
		'p95': float(np.percentile(array, 95)),
		'p99': float(np.percentile(array, 99)),
		'p99_9': float(np.percentile(array, 99.9)),
	}


def _per_token_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
	denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
	numerator = np.einsum('nd,nd->n', left, right, optimize=True)
	return np.divide(
		numerator,
		denominator,
		out=np.ones_like(numerator, dtype=np.float32),
		where=denominator > 0.0,
	)


def _embedding_edge_mask(shape: tuple[int, ...]) -> np.ndarray:
	if len(shape) != 3:
		raise ValueError(f'embedding token grid must be 3D; got {shape!r}')
	mask = np.zeros(shape, dtype=bool)
	mask[0] = True
	mask[-1] = True
	mask[:, 0] = True
	mask[:, -1] = True
	mask[:, :, 0] = True
	mask[:, :, -1] = True
	return mask


def _error_summary_for_token_mask(
	left: np.ndarray,
	right: np.ndarray,
	mask: np.ndarray,
) -> dict[str, object]:
	if not bool(mask.any()):
		return {'token_count': 0, 'mean_abs_error': 0.0, 'max_abs_error': 0.0}
	error = np.abs(
		np.asarray(left[mask], dtype=np.float32)
		- np.asarray(right[mask], dtype=np.float32),
	)
	return {
		'token_count': int(np.count_nonzero(mask)),
		'mean_abs_error': float(error.mean()),
		'max_abs_error': float(error.max()),
	}


def _embedding_non_finite_diagnostics(
	left: np.ndarray,
	right: np.ndarray,
	valid: np.ndarray,
) -> dict[str, object]:
	left_valid = np.asarray(left[valid])
	right_valid = np.asarray(right[valid])
	return {
		'left_nan_count_valid': int(np.count_nonzero(np.isnan(left_valid))),
		'right_nan_count_valid': int(np.count_nonzero(np.isnan(right_valid))),
		'left_inf_count_valid': int(np.count_nonzero(np.isinf(left_valid))),
		'right_inf_count_valid': int(np.count_nonzero(np.isinf(right_valid))),
		'invalid_token_non_finite_location_mismatch': False,
	}


def _probe_metrics_exact(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
	for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'mean_iou'):
		if left.get(key) != right.get(key):
			return False
	return True


def _decision_margin(values: np.ndarray) -> float:
	ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
	if ordered.size < 2:
		return 0.0
	return float(ordered[-1] - ordered[-2])


def _row_embedding_error(
	left: Mapping[str, object],
	right: Mapping[str, object],
	coordinate: np.ndarray,
) -> float:
	index = tuple(int(value) for value in coordinate)
	return float(
		np.max(
			np.abs(
				np.asarray(left['embeddings'])[index].astype(np.float32)
				- np.asarray(right['embeddings'])[index].astype(np.float32),
			),
		),
	)


def _hmm_label_mismatch_diagnostics(
	historical: np.ndarray,
	current: np.ndarray,
	mismatch: np.ndarray,
) -> dict[str, object]:
	coords = np.argwhere(mismatch)
	edge = _embedding_edge_mask(mismatch.shape)
	historical_boundary = _z_boundary_mask(historical)
	current_boundary = _z_boundary_mask(current)
	boundary = historical_boundary | current_boundary
	rows = [
		{
			'x': int(coord[0]),
			'y': int(coord[1]),
			'z': int(coord[2]),
			'historical_label': int(historical[tuple(coord)]),
			'current_label': int(current[tuple(coord)]),
			'edge_or_interior': 'edge' if edge[tuple(coord)] else 'interior',
			'boundary_or_interior': 'boundary' if boundary[tuple(coord)] else 'interior',
		}
		for coord in coords
	]
	pairs = np.zeros((6, 6), dtype=np.int64)
	known = mismatch & (historical >= 0) & (current >= 0)
	if known.any():
		np.add.at(pairs, (historical[known], current[known]), 1)
	per_trace = np.count_nonzero(mismatch, axis=2)
	hist_transition = _trace_transition_counts(historical)
	current_transition = _trace_transition_counts(current)
	transition_rows = [
		{
			'metric': key,
			'historical': historical_value,
			'current': current_value,
			'equal': historical_value == current_value,
		}
		for key, historical_value, current_value in (
			('state_occupancy', _occupancy(historical), _occupancy(current)),
			('per_trace_transition_count_total', int(hist_transition.sum()), int(current_transition.sum())),
			('reverse_transition_count', _reverse_count(historical), _reverse_count(current)),
			('ordered_path_violation_count', _ordered_violation_count(historical), _ordered_violation_count(current)),
			('empty_state_count', _empty_state_count(historical), _empty_state_count(current)),
		)
	]
	return {
		'mismatch_rows': rows,
		'transition_rows': transition_rows,
		'state_pair_confusion': pairs.tolist(),
		'x_histogram': np.bincount(coords[:, 0], minlength=mismatch.shape[0]).tolist()
		if coords.size
		else [0] * mismatch.shape[0],
		'y_histogram': np.bincount(coords[:, 1], minlength=mismatch.shape[1]).tolist()
		if coords.size
		else [0] * mismatch.shape[1],
		'z_histogram': np.bincount(coords[:, 2], minlength=mismatch.shape[2]).tolist()
		if coords.size
		else [0] * mismatch.shape[2],
		'edge_mismatch_rate': _masked_rate(mismatch, edge),
		'interior_mismatch_rate': _masked_rate(mismatch, ~edge),
		'boundary_mismatch_rate': _masked_rate(mismatch, boundary),
		'non_boundary_mismatch_rate': _masked_rate(mismatch, ~boundary),
		'per_trace_mismatch_count': per_trace.tolist(),
		'transition_position_difference_count': int(
			np.count_nonzero(hist_transition != current_transition),
		),
		'state_occupancy_difference': (
			np.asarray(_occupancy(current), dtype=np.int64)
			- np.asarray(_occupancy(historical), dtype=np.int64)
		).tolist(),
	}


def _z_boundary_mask(labels: np.ndarray) -> np.ndarray:
	valid = labels >= 0
	mask = np.zeros(labels.shape, dtype=bool)
	if labels.shape[2] < 2:
		return mask
	change = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
	mask[:, :, 1:] |= change
	mask[:, :, :-1] |= change
	return mask


def _trace_transition_counts(labels: np.ndarray) -> np.ndarray:
	valid = labels >= 0
	change = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
	return change.sum(axis=2, dtype=np.int64)


def _occupancy(labels: np.ndarray) -> list[int]:
	return [int(value) for value in np.bincount(labels[labels >= 0], minlength=6)[:6]]


def _reverse_count(labels: np.ndarray) -> int:
	left = labels[:, :, :-1]
	right = labels[:, :, 1:]
	return int(np.count_nonzero((left >= 0) & (right >= 0) & (right < left)))


def _ordered_violation_count(labels: np.ndarray) -> int:
	return _reverse_count(labels)


def _empty_state_count(labels: np.ndarray) -> int:
	return int(sum(count == 0 for count in _occupancy(labels)))


def _masked_rate(values: np.ndarray, mask: np.ndarray) -> float:
	count = int(np.count_nonzero(mask))
	return 0.0 if count == 0 else float(np.count_nonzero(values & mask) / count)


def _numeric_array_comparison(
	left: np.ndarray,
	right: np.ndarray,
	*,
	rtol: float,
	atol: float,
) -> dict[str, object]:
	if left.shape != right.shape:
		return {'shape_equal': False, 'allclose': False}
	left_values = np.asarray(left, dtype=np.float64)
	right_values = np.asarray(right, dtype=np.float64)
	error = np.abs(left_values - right_values)
	denominator = np.maximum(np.maximum(np.abs(left_values), np.abs(right_values)), 1.0e-12)
	return {
		'shape_equal': True,
		'dtype_left': str(left.dtype),
		'dtype_right': str(right.dtype),
		'exact': bool(np.array_equal(left, right)),
		'max_abs': float(error.max()) if error.size else 0.0,
		'mean_abs': float(error.mean()) if error.size else 0.0,
		'max_relative': float((error / denominator).max()) if error.size else 0.0,
		'allclose': bool(np.allclose(left_values, right_values, rtol=rtol, atol=atol)),
	}


def _iteration_summary_comparison(
	historical_model: object,
	current_model: object,
) -> dict[str, object]:
	if not isinstance(historical_model, Mapping) or not isinstance(current_model, Mapping):
		return {'status': 'UNAVAILABLE', 'reason': 'HMM model payload is not a mapping'}
	historical = historical_model.get('iteration_summaries')
	current = current_model.get('iteration_summaries')
	return {
		'historical_recorded': isinstance(historical, list),
		'current_recorded': isinstance(current, list),
		'exact': historical == current,
		'historical': _jsonable(historical),
		'current': _jsonable(current),
		'objective_history': 'NOT_RECORDED_IN_HISTORICAL_ARTIFACT',
	}


def _historical_boundary_weight_present(root: Path) -> bool:
	return bool(list(root.glob('k6/*.hmm_boundary_weight_token.npy')))


def _pseudo_scientific_metadata_comparison(
	historical: Mapping[str, object],
	current: Mapping[str, object],
) -> dict[str, object]:
	keys = ('artifact_type', 'schema_version', 'survey_id', 'k', 'token_grid_shape')
	return {
		'fields': {
			key: {
				'historical': _jsonable(historical.get(key)),
				'current': _jsonable(current.get(key)),
				'equal': historical.get(key) == current.get(key),
			}
			for key in keys
		},
		'boundary_weight_contract': {
			'historical_expected_absent': True,
			'current_expected_absent': True,
		},
	}


def _benchmark_command(
	config: PerformanceMigrationValidationConfig,
	tool: Path,
	output_dir: Path,
) -> list[str]:
	return [
		sys.executable,
		str(tool),
		'--seed',
		str(config.benchmark['seed']),
		'--warm-up',
		str(config.benchmark['warm_up']),
		'--repeat',
		str(config.benchmark['repeat']),
		'--output-json',
		str(output_dir / 'current.json'),
		'--output-markdown',
		str(output_dir / 'current.md'),
	]


def _run_command(
	command: Sequence[str],
	*,
	cwd: Path,
	environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
	completed = subprocess.run(
		list(command),
		cwd=cwd,
		env=None if environment is None else dict(environment),
		check=False,
		capture_output=True,
		text=True,
	)
	return {
		'command': list(command),
		'returncode': completed.returncode,
		'stdout': completed.stdout,
		'stderr': completed.stderr,
	}


def _sha256_path(path: Path) -> str:
	if path.is_dir():
		digest = hashlib.sha256()
		for child in sorted(item for item in path.rglob('*') if item.is_file()):
			digest.update(str(child.relative_to(path)).encode('utf-8'))
			digest.update(b'\0')
			digest.update(_sha256_path(child).encode('ascii'))
			digest.update(b'\n')
		return digest.hexdigest()
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def _sha256_array(array: np.ndarray, *, dtype: np.dtype | None = None) -> str:
	value = np.asarray(array)
	if dtype is not None:
		value = value.astype(dtype, copy=False)
	contiguous = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(contiguous.tobytes(order='C'))
	return digest.hexdigest()


def _path_size(path: Path) -> int:
	if path.is_file():
		return path.stat().st_size
	return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def _write_json_atomic(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.parent / f'.{path.name}.{uuid4().hex}.tmp'
	try:
		temporary.write_text(
			json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + '\n',
			encoding='utf-8',
		)
		with temporary.open('rb') as handle:
			os.fsync(handle.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _write_text_atomic(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.parent / f'.{path.name}.{uuid4().hex}.tmp'
	try:
		temporary.write_text(content.rstrip() + '\n', encoding='utf-8')
		with temporary.open('rb') as handle:
			os.fsync(handle.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _write_csv_atomic(
	path: Path,
	rows: Sequence[Mapping[str, object]],
	*,
	fieldnames: Sequence[str],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.parent / f'.{path.name}.{uuid4().hex}.tmp'
	try:
		with temporary.open('w', encoding='utf-8', newline='') as handle:
			writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
			writer.writeheader()
			writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
		with temporary.open('rb') as handle:
			os.fsync(handle.fileno())
		os.replace(temporary, path)
	finally:
		if temporary.exists():
			temporary.unlink()


def _load_json(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		raise TypeError(f'JSON artifact must contain an object: {path}')
	return cast('dict[str, object]', payload)


def _jsonable(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _jsonable(item) for key, item in value.items()}
	if isinstance(value, tuple | list):
		return [_jsonable(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return _jsonable(value.tolist())
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, torch.Tensor):
		return _jsonable(value.detach().cpu().tolist())
	return value


def _csv_value(value: object) -> object:
	if isinstance(value, Mapping | list | tuple):
		return json.dumps(_jsonable(value), sort_keys=True)
	return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return cast('Mapping[str, object]', value)


def _is_complete_file(path: Path) -> bool:
	try:
		return path.is_file() and bool(_load_json(path))
	except (OSError, ValueError, TypeError):
		return False


def _utc_now() -> str:
	return datetime.now(tz=timezone.utc).isoformat()


def _quarantine_path(path: Path, *, reason: str) -> Path:
	if not path.exists():
		raise FileNotFoundError(path)
	root = path.parent / 'quarantine'
	root.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	destination = root / f'{path.name}.{timestamp}.{reason}.{uuid4().hex[:8]}'
	os.replace(path, destination)
	return destination


def _quarantine_publish_output(
	config: PerformanceMigrationValidationConfig,
	path: Path,
) -> Path:
	"""Preserve an invalid results publish outside the lightweight publish tree."""
	if not path.exists():
		raise FileNotFoundError(path)
	root = config.migration_root / 'quarantine'
	root.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	destination = root / f'{path.name}.{timestamp}.invalid_publish.{uuid4().hex[:8]}'
	os.replace(path, destination)
	return destination


def _quarantine_paths(root: Path) -> list[str]:
	return sorted(str(path) for path in root.rglob('quarantine/*') if path.exists())


def _validate_publish_manifest(root: Path, manifest: Mapping[str, object]) -> bool:
	files = manifest.get('files')
	items = manifest.get('items')
	if not isinstance(files, list) or not isinstance(items, list):
		return False
	if len(files) != len(items):
		return False
	expected: set[str] = {'publish_manifest.json'}
	for file_item, standard_item in zip(files, items, strict=True):
		if not isinstance(file_item, Mapping) or not isinstance(standard_item, Mapping):
			return False
		relative = file_item.get('relative_path')
		if not isinstance(relative, str) or Path(relative).is_absolute():
			return False
		path = root / relative
		if not path.is_file():
			return False
		if path.suffix not in {'.md', '.json', '.csv', '.png'}:
			return False
		if file_item.get('published_sha256') != _sha256_path(path):
			return False
		if file_item.get('byte_size') != path.stat().st_size:
			return False
		source = file_item.get('source_artifact_path')
		if not isinstance(source, str) or not Path(source).is_file():
			return False
		if file_item.get('source_sha256') != _sha256_path(Path(source)):
			return False
		if standard_item.get('source') != source:
			return False
		if standard_item.get('target') != relative:
			return False
		if standard_item.get('size_bytes') != path.stat().st_size:
			return False
		if standard_item.get('sha256') != _sha256_path(path):
			return False
		expected.add(relative)
	actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
	return actual == expected


def _render_inventory_markdown(payload: Mapping[str, object]) -> str:
	inputs = cast('Sequence[Mapping[str, object]]', payload['inputs'])
	lines = [
		'# Performance migration input inventory',
		'',
		f"- Current git SHA: `{payload['current_git_sha']}`",
		f"- Historical baseline SHA: `{payload['historical_baseline_sha']}`",
		f"- Missing inputs: {payload['missing_input_count']}",
		'',
		'| Role | Status | Path | SHA-256 |',
		'| --- | --- | --- | --- |',
	]
	lines.extend(
		f"| {item.get('logical_role')} | {item.get('status')} | `{item.get('path')}` | `{item.get('sha256', '')}` |"
		for item in inputs
	)
	return '\n'.join(lines)


def _render_checkpoint_markdown(payload: Mapping[str, object]) -> str:
	lines = ['# Checkpoint compatibility smoke', '', f"Status: `{payload['status']}`", '']
	if payload['status'] == 'PASS':
		crop = _mapping(payload, 'crop')
		lines.extend(
			[
				f"- Crop: `{crop['start_xyz']}` + `{crop['size_xyz']}`",
				'| Checkpoint | Output shape | Output SHA-256 |',
				'| --- | --- | --- |',
			],
		)
		for role, result in _mapping(payload, 'checkpoints').items():
			output = _mapping(cast('Mapping[str, object]', result), 'encoder_output')
			lines.append(f"| {role} | `{output['shape']}` | `{output['sha256']}` |")
	else:
		lines.extend([f"- Error: `{payload.get('error')}`", '', '```text', str(payload.get('stack_trace', '')), '```'])
	return '\n'.join(lines)


def _render_embedding_markdown(payload: Mapping[str, object]) -> str:
	lines = [
		'# M1 embedding A/B/C parity',
		'',
		f"- Current git SHA: `{payload['current_git_sha']}`",
		f"- Historical baseline SHA: `{payload['historical_baseline_sha']}`",
		'',
		'| Pair | Status | Mask exact | Array exact | Max abs error | Mean cosine |',
		'| --- | --- | --- | --- | --- | --- |',
	]
	for pair, raw in _mapping(payload, 'comparisons').items():
		comparison = cast('Mapping[str, object]', raw)
		if comparison.get('status') == 'BLOCKED_NUMERIC_CONTRACT':
			lines.append(f'| {pair} | BLOCKED_NUMERIC_CONTRACT | false | false | — | — |')
			continue
		absolute = _mapping(comparison, 'absolute_error')
		cosine = _mapping(comparison, 'per_token_cosine_similarity')
		lines.append(
			f"| {pair} | {comparison['status']} | {comparison['valid_token_mask_exact']} | "
			f"{comparison['embedding_array_equal']} | {absolute['max']:.8g} | {cosine['mean']:.10g} |",
		)
	return '\n'.join(lines)


def _render_probe_markdown(payload: Mapping[str, object]) -> str:
	lines = [
		'# Existing linear-probe parity',
		'',
		f"- Validation rows: {payload['validation_row_count']}",
		'',
		'| Pair | Status | Predictions exact | Confusion exact | Primary metrics exact |',
		'| --- | --- | --- | --- | --- |',
	]
	for pair, raw in _mapping(payload, 'parity').items():
		item = cast('Mapping[str, object]', raw)
		lines.append(
			f"| {pair} | {item['status']} | {item['prediction_exact']} | "
			f"{item['confusion_matrix_exact']} | {item['primary_metrics_exact']} |",
		)
	return '\n'.join(lines)


def _render_hmm_markdown(payload: Mapping[str, object]) -> str:
	labels = _mapping(payload, 'labels')
	centers = _mapping(payload, 'centers')
	comparison = _mapping(centers, 'comparison')
	return '\n'.join(
		[
			'# K=6 stratigraphic HMM replay parity',
			'',
			f"- Status: `{payload['status']}`",
			f"- Decoded labels exact: `{labels['decoded_labels_exact']}`",
			f"- Valid-token mask exact: `{labels['valid_token_mask_exact']}`",
			f"- Mismatch tokens: {labels['mismatch_token_count']}",
			f"- Center allclose (rtol/atol 1e-6): `{comparison.get('allclose')}`",
			'- Historical objective history: `NOT_RECORDED_IN_HISTORICAL_ARTIFACT`.',
		],
	)


def _render_pseudo_markdown(payload: Mapping[str, object]) -> str:
	labels = _mapping(payload, 'labels')
	valid = _mapping(payload, 'valid_tokens')
	confidence = _mapping(payload, 'confidence')
	return '\n'.join(
		[
			'# K=6 pseudo-target parity',
			'',
			f"- Status: `{payload['status']}`",
			f"- Labels exact: `{labels['exact']}`",
			f"- Valid tokens exact: `{valid['exact']}`",
			f"- Confidence exact: `{confidence['exact']}`",
			f"- Min-confidence threshold crossings: {confidence['threshold_crossing_count']}",
		],
	)


def _render_summary_markdown(summary: Mapping[str, object]) -> str:
	decision = _mapping(summary, 'migration_decision')
	return '\n'.join(
		[
			'# Performance migration validation summary',
			'',
			f"- Status: `{summary['status']}`",
			f"- Current git SHA: `{summary['current_git_sha']}`",
			f"- Historical baseline SHA: `{summary['historical_baseline_sha']}`",
			f"- Migration decision: `{decision['status']}`",
			f"- Required rerun scope: {summary['required_rerun_scope']}",
			f"- Multi-head policy: {summary['multi_head_baseline_policy']}",
			f"- Atomic path provenance: {summary['atomic_write_provenance']}",
			'',
			'No voxel-decoder or M3-V/M3-V-LB full-job retraining was performed by this migration validation.',
		],
	)


def _render_handoff_markdown(
	summary: Mapping[str, object],
	manifest: Mapping[str, object],
) -> str:
	decision = _mapping(summary, 'migration_decision')
	stage_artifacts = _mapping(manifest, 'stage_artifacts')
	lines = [
		'# Performance migration validation handoff',
		'',
		f"status: {summary['status']}",
		f"current git SHA: {summary['current_git_sha']}",
		f"historical baseline SHA: {summary['historical_baseline_sha']}",
		f"migration decision: {decision['status']}",
		f"required rerun scope: {summary['required_rerun_scope']}",
		f"multi-head baseline policy: {summary['multi_head_baseline_policy']}",
		f"atomic path provenance: {summary['atomic_write_provenance']}",
		'',
		'## Stage artifacts',
		'',
		'| Stage | Exists | SHA-256 |',
		'| --- | --- | --- |',
	]
	for stage, raw in stage_artifacts.items():
		item = cast('Mapping[str, object]', raw)
		lines.append(f"| {stage} | {item['exists']} | `{item['sha256']}` |")
	lines.extend(
		[
			'',
			'## Preserved artifacts',
			'',
			'Historical checkpoints, embeddings, HMM artifacts, pseudo-targets, probes, and M3-V/M3-V-LB outputs were read-only inputs.',
		],
	)
	quarantine_paths = cast('Sequence[str]', manifest.get('quarantine_paths', []))
	lines.extend(['', '## Quarantine', ''])
	if quarantine_paths:
		lines.extend(f'- `{path}`' for path in quarantine_paths)
	else:
		lines.append('None.')
	lines.extend(
		[
			'',
			'## Resume',
			'',
			'Use the documented `validate_performance_migration.py --stage ... --only-missing` commands from the experiment README.',
		],
	)
	return '\n'.join(lines)


__all__ = [
	'build_input_inventory',
	'checkpoint_compatibility_smoke',
	'compare_embedding_artifacts',
	'compare_hmm_replay',
	'compare_probe_predictions',
	'compare_pseudo_targets',
	'export_legacy_m1_pseudo_targets',
	'reconstruct_historical_hmm_config',
	'run_m1_embedding_extraction',
	'run_m1_hmm_replay',
	'run_performance_benchmark',
	'summarize_performance_migration',
]
