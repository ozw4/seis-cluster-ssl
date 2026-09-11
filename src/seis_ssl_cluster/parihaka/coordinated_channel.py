"""Cooperate on exactly the sixty new experiment-40 HMM Channel cells.

Only future CLI processes participate. Operators must first drain the old
noncooperative decoder and GPU1 staging workers, and authorize an arm only after
the main embedding producer exits. A separate singleton scheduler owns GPU1's
three-worker cap. These advisory cell locks neither migrate nor stop jobs.
"""

# ruff: noqa: SLF001 - Compose existing scientific readers and audited POSIX locks.

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import (
	load_config,
	resolve_embedding_extraction_config,
	resolve_strat_hmm_pretext_config,
)
from seis_ssl_cluster.data.schema import read_manifest_json
from seis_ssl_cluster.data.window_preprocessing import resolve_manifest_path
from seis_ssl_cluster.embedding.extractor import (
	UNMASKED_ENCODER_INPUT_MODE,
	build_embedding_metadata,
	extraction_settings_from_config,
)
from seis_ssl_cluster.embedding.sliding_window import token_grid_shape_xyz
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.models.amplitude_encoder_factory import (
	build_model_from_checkpoint_payload,
	checkpoint_config_from_payload,
)
from seis_ssl_cluster.parihaka import channel_decoder as decoder
from seis_ssl_cluster.parihaka import coordinated_hmm as hmm
from seis_ssl_cluster.parihaka.coordinated_channel_state import inspect_channel_state

EXPERIMENT = hmm.EXPERIMENT
LAYOUT_CONFIG = EXPERIMENT.parent / '30_channel_benchmark_v1/02_layouts.yaml'
MODELS = {
	f'{stem}_distill{weight}': f'{source}/distill{weight}'
	for stem, source in (
		('local_barlow_twins_rot90_asym_g060_3ep_hmm_k6', 'local_bt3'),
		('random_init_self_target_hmm_k6', 'random'),
	)
	for weight in ('010', '020')
}
RUNS_STEM = Path('channel_benchmark/pretraining_comparison_completion_v1/runs')
EMBEDDING_STEM = Path(
	'embeddings/parihaka/facies_benchmark_v1/pretraining_comparison_completion_v1'
)
LABEL_STEM = Path('data/parihaka/facies_benchmark_v1')


@dataclass(frozen=True)
class CoordinatedChannelRun:
	"""Immutable scope of one exact canonical Channel cell."""

	config_path: Path
	raw: Mapping[str, object]
	model: str
	layout_id: str
	data_size: str
	layout_config: Path
	output_dir: Path
	lock_path: Path
	receipt_path: Path


def _mapping(value: object) -> Mapping:
	if not isinstance(value, Mapping):
		raise TypeError('coordinated Channel evidence must be a mapping')
	return value


def _safe_path(path: Path, *, regular: bool = False) -> None:
	if not path.is_absolute() or path != path.resolve():
		raise ValueError(f'coordinated Channel refuses path aliases: {path}')
	hmm._no_symlinks(path)
	if regular:
		info = path.stat(follow_symlinks=False)
		if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
			raise ValueError(
				f'coordinated Channel requires single-link regular files: {path}'
			)


def _expected_raw(root: Path, model: str) -> dict[str, object]:
	runs = root / RUNS_STEM
	return {
		'dataset': {'survey_id': 'parihaka'},
		'inputs': {
			'labels_npy': str(root / LABEL_STEM / 'parihaka_labels.npy'),
			'labels_metadata_json': str(
				root / LABEL_STEM / 'parihaka_labels_metadata.json'
			),
			'runs_root': str(runs),
		},
		'embeddings': {
			'models': {
				model: {
					'dir': str(root / EMBEDDING_STEM / model / 'overlap_x64'),
					'checkpoint': str(
						root / hmm.ARTIFACT_STEM / MODELS[model] / 'full_25ep/latest.pt'
					),
				}
			}
		},
		'outputs': {'runs_root': str(runs), 'output_dir': str(runs.parent / 'summary')},
		'decoder': {
			'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
			'embedding_dim': 384,
			'class_count': 2,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {'core_size_tokens': [8, 8, 8], 'context_halo_tokens': [1, 1, 1]},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 0.001,
			'weight_decay': 0.0001,
			'class_weight': 'balanced',
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': False,
			'gradient_clip_norm': 1.0,
		},
	}


def classify_coordinated_channel_run(  # noqa: PLR0913
	raw: Mapping[str, object],
	*,
	config_path: Path,
	model: str,
	layout_id: str,
	data_size: str,
	layout_config: Path,
	max_steps: int | None = None,
	resume: Path | None = None,
	validation_only: bool = False,
	evaluate_completed_validation: Path | None = None,
) -> CoordinatedChannelRun | None:
	"""Classify without writes/GPU calls; reject aliases into the protected scope."""
	definitions = {
		name: EXPERIMENT / f'40_downstream/01_{name}.yaml' for name in MODELS
	}
	definition_model = next(
		(
			name
			for name, path in definitions.items()
			if path.resolve() == config_path.resolve()
		),
		None,
	)
	outputs = _mapping(raw.get('outputs', {}))
	runs = Path(str(outputs.get('runs_root', '')))
	configured = _mapping(_mapping(raw.get('embeddings', {})).get('models', {}))
	protected_output = runs.resolve().parts[
		-len(RUNS_STEM.parts) :
	] == RUNS_STEM.parts and (model in MODELS or bool(set(configured) & set(MODELS)))
	# Generic model names can contain slashes; classify the actual runner target.
	resolved_cell = (
		runs / f'model={model}' / f'layout={layout_id}' / f'size={data_size}'
	).resolve()
	protected_output = protected_output or any(
		resolved_cell.parts[-len(RUNS_STEM.parts) - 3 :]
		== (
			*RUNS_STEM.parts,
			f'model={name}',
			f'layout=layout_{index:03d}',
			f'size={size}',
		)
		for name in MODELS
		for index in range(5)
		for size in ('small', 'medium', 'large')
	)
	if definition_model is None and not protected_output:
		return None
	if definition_model != model or model not in MODELS:
		raise ValueError(
			'canonical Channel output requires its exact model recipe config'
		)
	config_path = config_path.absolute()
	layout_config = layout_config.absolute()
	_safe_path(config_path, regular=True)
	_safe_path(layout_config, regular=True)
	if (
		layout_config != LAYOUT_CONFIG
		or layout_id not in {f'layout_{index:03d}' for index in range(5)}
		or data_size not in {'small', 'medium', 'large'}
	):
		raise ValueError(
			'coordinated Channel layout or size is outside the canonical cells'
		)
	if (
		max_steps is not None
		or validation_only
		or evaluate_completed_validation is not None
	):
		raise ValueError(
			'canonical coordinated Channel cells require full validation/test training'
		)
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	_safe_path(root)
	expected = _expected_raw(root, model)
	if raw != expected or load_config(definitions[model]) != expected:
		raise ValueError(
			'coordinated Channel config differs from the exact scientific recipe'
		)
	_safe_path(runs)
	output = runs / f'model={model}/layout={layout_id}/size={data_size}'
	_safe_path(output)
	if resume is not None:
		_safe_path(resume.absolute())
		if resume.absolute() != output / 'latest.pt':
			raise ValueError(
				'coordinated Channel resume must identify this cell latest.pt'
			)
	lock_root = runs.parent / '.channel_cell_locks'
	_safe_path(lock_root)
	key = f'{model}__{layout_id}__{data_size}'
	return CoordinatedChannelRun(
		config_path,
		dict(raw),
		model,
		layout_id,
		data_size,
		layout_config,
		output,
		lock_root / f'{key}.lock',
		lock_root / f'{key}.inputs.json',
	)


def _hashes(paths: list[Path]) -> dict[str, str]:
	result = {}
	for path in paths:
		_safe_path(path, regular=True)
		before = _stamp(path)
		result[str(path)] = file_sha256(path)
		if _stamp(path) != before:
			raise ValueError(f'Channel input changed while hashing: {path}')
	return result


def _stamp(path: Path) -> tuple[int, ...]:
	info = path.stat(follow_symlinks=False)
	return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _inspect_source(scope: CoordinatedChannelRun) -> tuple[Path, Mapping]:
	recipe = MODELS[scope.model]
	path = EXPERIMENT / f'20_stage2/{recipe}/01_full_25ep.yaml'
	_safe_path(path, regular=True)
	config = resolve_strat_hmm_pretext_config(load_config(path))
	train = _mapping(config['train'])
	if any(
		train[key] != value
		for key, value in {
			'epochs': 25,
			'samples_per_epoch': 10000,
			'batch_size': 16,
			'seed': 42,
			'max_steps': None,
			'amp': False,
			'allow_overwrite_output': False,
		}.items()
	) or (
		_mapping(config['loss'])['distillation_weight']
		!= (0.1 if scope.model.endswith('010') else 0.2)
		or _mapping(config['pseudo_targets'])['k'] != 6
		or _mapping(config['student'])['unfreeze_top_blocks'] != 1
		or _mapping(config['teacher'])['checkpoint']
		!= _mapping(config['student'])['init_checkpoint']
	):
		raise ValueError('Channel HMM source has a different scientific budget')
	root = Path(str(_mapping(config['paths'])['output_root']))
	source = _mapping(_mapping(scope.raw['embeddings'])['models'])[scope.model]
	if root / 'latest.pt' != Path(str(_mapping(source)['checkpoint'])):
		raise ValueError('Channel HMM source output differs from the exact recipe')
	hmm_plan = hmm.CoordinatedHmmRun(
		recipe, path, config, root, root.parent, smoke=False
	)
	if not hmm.inspect_coordinated_hmm_checkpoint(hmm_plan):
		raise ValueError('Channel embeddings require a full 25-epoch HMM source')
	checkpoint = root / 'latest.pt'
	payload = _mapping(
		torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
	)
	for value in _mapping(payload['model_state_dict']).values():
		if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
			raise ValueError('Channel HMM source contains non-finite model tensors')
	return path, payload


def _expected_metadata(
	scope: CoordinatedChannelRun, payload: Mapping, checkpoint_sha: str
) -> tuple[dict[str, object], list[Path]]:
	path = EXPERIMENT / f'30_embeddings/01_extract_{scope.model}.yaml'
	_safe_path(path, regular=True)
	config = resolve_embedding_extraction_config(load_config(path))
	checkpoint_config = checkpoint_config_from_payload(payload)
	settings = extraction_settings_from_config(
		config, checkpoint_config=checkpoint_config
	)
	source = _mapping(
		_mapping(_mapping(scope.raw['embeddings'])['models'])[scope.model]
	)
	if (
		settings.checkpoint_path != Path(str(source['checkpoint']))
		or settings.output_dir != Path(str(source['dir']))
		or settings.window_size_xyz != (128, 128, 128)
		or settings.overlap_xyz != (64, 64, 64)
		or settings.output_dtype != np.dtype('float16')
		or settings.amp is not False
		or settings.min_token_valid_fraction != 0.5
		or settings.preprocessing_cache.mode != 'off'
	):
		raise ValueError('Channel embedding extraction settings differ from the recipe')
	manifest_path = Path(str(_mapping(config['manifests'])['input']))
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	if manifest_path != root / LABEL_STEM / 'parihaka_amplitude_manifest.json':
		raise ValueError(
			'Channel embedding manifest differs from canonical Parihaka input'
		)
	_safe_path(manifest_path, regular=True)
	manifests = read_manifest_json(manifest_path)
	if len(manifests) != 1 or manifests[0].survey_id != 'parihaka':
		raise ValueError('Channel embedding requires exactly one Parihaka manifest')
	manifest = manifests[0]
	amplitude = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats = resolve_manifest_path(manifest, manifest.amplitude.normalization_stats_path)
	with torch.device('cpu'):
		model = build_model_from_checkpoint_payload(payload)
	expected = build_embedding_metadata(
		manifest=manifest,
		amplitude_path=amplitude,
		stats_path=stats,
		settings=settings,
		checkpoint_config=checkpoint_config,
		checkpoint_payload=payload,
		checkpoint_sha256=checkpoint_sha,
		model=model,
		token_grid_shape=token_grid_shape_xyz(
			manifest.amplitude.shape_xyz, model.patch_size_xyz
		),
		device=torch.device('cpu'),
	)
	return expected, [path, manifest_path, stats]


def inspect_coordinated_channel_inputs(  # noqa: C901
	scope: CoordinatedChannelRun,
) -> tuple[decoder.ChannelDecoderPlan, dict[str, str]]:
	"""Read-only source/embedding gate; call only after the producer exits.

	All three published embedding files are hashed twice around exact metadata,
	full-array finite, and fresh plan inspection. Metadata alone is not a commit
	marker: the current producer publishes it before the arrays.
	"""
	config = decoder.channel_decoder_config_from_mapping(scope.raw)
	source = config.models[scope.model]
	paths = output_paths(source.embedding_dir, 'parihaka')
	marker = source.embedding_dir / 'embedding_extraction_execution.json'
	for temporary in (
		paths.embeddings_tmp,
		paths.valid_tokens_tmp,
		paths.sum_tmp,
		paths.count_tmp,
		paths.metadata.with_name(f'.{paths.metadata.name}.tmp'),
		marker.with_name(f'.{marker.name}.tmp'),
	):
		if temporary.exists() or temporary.is_symlink():
			raise ValueError(
				'Channel embedding producer still has unpublished temporary files'
			)
	input_paths = [
		scope.config_path,
		scope.layout_config,
		source.expected_checkpoint,
		paths.embeddings,
		paths.valid_tokens,
		paths.metadata,
		marker,
		config.labels,
		config.labels_metadata,
		EXPERIMENT / f'20_stage2/{MODELS[scope.model]}/01_full_25ep.yaml',
		EXPERIMENT / f'30_embeddings/01_extract_{scope.model}.yaml',
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
		/ LABEL_STEM
		/ 'parihaka_amplitude_manifest.json',
	]
	before = _hashes(input_paths)
	train_path, payload = _inspect_source(scope)
	expected, auxiliary = _expected_metadata(
		scope,
		payload,
		before[str(source.expected_checkpoint)],
	)
	extra = [train_path, *auxiliary]
	for path, digest in _hashes(extra).items():
		if path in before and before[path] != digest:
			raise ValueError(
				'Channel input changed while reconstructing embedding metadata'
			)
		before[path] = digest
	metadata = json.loads(paths.metadata.read_text(encoding='utf-8'))
	if metadata != expected:
		raise ValueError(
			'Channel embedding metadata differs from live source and extraction recipe'
		)
	execution = _mapping(json.loads(marker.read_text(encoding='utf-8')))
	if execution != {
		'artifact_type': 'embedding_extraction_execution',
		'schema_version': 1,
		'encoder_input_mode': UNMASKED_ENCODER_INPUT_MODE,
		'fresh': execution.get('fresh'),
		'reuse': execution.get('reuse'),
		'survey_count': 1,
	} or (
		type(execution.get('fresh')) is not int
		or type(execution.get('reuse')) is not int
		or (execution.get('fresh'), execution.get('reuse')) not in {(1, 0), (0, 1)}
	):
		raise ValueError('Channel embedding execution marker is incomplete')
	array = np.load(paths.embeddings, mmap_mode='r', allow_pickle=False)
	for index in range(array.shape[0]):
		if not np.isfinite(array[index]).all():
			raise ValueError('Channel embedding array contains non-finite values')
	plan = decoder.inspect_channel_decoder_job(
		config,
		model=scope.model,
		layout_id=scope.layout_id,
		data_size=scope.data_size,
		layout_config=scope.layout_config,
	)
	if plan.output_dir != scope.output_dir:
		raise ValueError('fresh Channel plan escaped the locked output directory')
	if _hashes([*input_paths, *extra]) != before:
		raise ValueError('Channel source or embedding inputs changed during inspection')
	return plan, before


def _receipt(scope: CoordinatedChannelRun, hashes: Mapping) -> dict[str, object]:
	return {
		'schema_version': 1,
		'artifact_type': 'coordinated_channel_inputs',
		'output_dir': str(scope.output_dir),
		'input_sha256': {
			path: digest
			for path, digest in hashes.items()
			if Path(path).name != 'embedding_extraction_execution.json'
		},
	}


def _check_receipt(scope: CoordinatedChannelRun, expected: Mapping) -> bool:
	path = scope.receipt_path
	if not path.exists() and not path.is_symlink():
		return False
	_safe_path(path, regular=True)
	if json.loads(path.read_text(encoding='utf-8')) != expected:
		raise ValueError(
			'Channel saved input receipt differs from the live input SHA set'
		)
	return True


def _write_receipt(path: Path, receipt: Mapping) -> None:
	_safe_path(path)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		delete=False,
	) as stream:
		temporary = Path(stream.name)
		try:
			json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
			stream.write('\n')
			stream.flush()
			os.fsync(stream.fileno())
			# Atomic create, never replace even if an uncooperative writer raced us.
			os.link(temporary, path, follow_symlinks=False)
		finally:
			if temporary.exists():
				temporary.unlink()


def _safe_output(scope: CoordinatedChannelRun) -> dict[str, str]:
	_safe_path(scope.output_dir)
	if not scope.output_dir.exists():
		return {}
	return _hashes(list(scope.output_dir.iterdir()))


def run_coordinated_channel_if_scoped(  # noqa: C901, PLR0913
	raw: Mapping[str, object],
	*,
	config_path: Path,
	model: str,
	layout_id: str,
	data_size: str,
	layout_config: Path,
	device: str = 'auto',
	dry_run: bool = False,
	max_steps: int | None = None,
	resume: Path | None = None,
	validation_only: bool = False,
	evaluate_completed_validation: Path | None = None,
) -> Path | None:
	"""Skip a proven cell or finish an owned fresh/partial cell under one lock.

	None means an unrelated invocation, or a scoped read-only dry run whose
	existing CLI inspection should continue. Scope overrides never fall through.
	"""
	arguments = {
		'config_path': config_path,
		'model': model,
		'layout_id': layout_id,
		'data_size': data_size,
		'layout_config': layout_config,
		'max_steps': max_steps,
		'resume': resume,
		'validation_only': validation_only,
		'evaluate_completed_validation': evaluate_completed_validation,
	}
	scope = classify_coordinated_channel_run(raw, **arguments)
	if scope is None or dry_run:
		return None
	with hmm._file_lock(scope.lock_path):
		fresh_raw = load_config(scope.config_path)
		fresh = classify_coordinated_channel_run(fresh_raw, **arguments)
		if fresh != scope:
			raise ValueError(
				'Channel config or scope changed while waiting for the cell lock'
			)
		plan, hashes = inspect_coordinated_channel_inputs(scope)
		identity = decoder._run_identity(plan)
		before = _safe_output(scope)
		state = inspect_channel_state(plan, identity)
		if _safe_output(scope) != before:
			raise ValueError('Channel output changed during its locked state audit')
		expected_receipt = _receipt(scope, hashes)
		sealed = _check_receipt(scope, expected_receipt)
		if _hashes([Path(path) for path in hashes]) != hashes:
			raise ValueError(
				'Channel inputs changed during the locked checkpoint audit'
			)
		print(
			json.dumps(
				{
					'coordinated_channel': state,
					'output_dir': str(scope.output_dir),
					'input_sha256': hashes,
					'historical_embedding_sha_recorded': sealed,
				}
			),
			flush=True,
		)
		if state == 'complete':
			return scope.output_dir / 'metrics.json'
		if state == 'partial' and not sealed:
			raise ValueError(
				'Channel partial has no pre-training input SHA receipt; refusing resume'
			)
		if not sealed:
			_write_receipt(scope.receipt_path, expected_receipt)
		metrics = decoder.run_channel_decoder_job(
			plan,
			device=device,
			resume=scope.output_dir / 'latest.pt' if state == 'partial' else None,
		)
		if metrics != scope.output_dir / 'metrics.json':
			raise RuntimeError(
				'coordinated Channel training did not finish the full budget'
			)
		final_plan, final_hashes = inspect_coordinated_channel_inputs(scope)
		if (
			_receipt(scope, final_hashes) != expected_receipt
			or decoder._run_identity(final_plan) != identity
		):
			raise ValueError('Channel scientific inputs changed during training')
		if not _check_receipt(scope, expected_receipt):
			raise ValueError('Channel input receipt disappeared during training')
		_safe_output(scope)
		if inspect_channel_state(final_plan, identity) != 'complete':
			raise RuntimeError(
				'coordinated Channel result failed strict final completion'
			)
		return metrics
