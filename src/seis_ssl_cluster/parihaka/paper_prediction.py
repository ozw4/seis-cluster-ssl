"""Frozen-decoder full-volume export; no training or metric evaluation."""

from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.models.voxel_decoder import VoxelDecoder3D
from seis_ssl_cluster.parihaka.channel_completion import inspect_completed_channel_job

MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_barlow_twins_rot90_asym_g060_3ep',
	'local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill020',
	'random',
	'random_init_self_target_hmm_k6_distill020',
)
CONDITIONS = ('1', '2b', '3', '4b', '5', '6b')


def read_json(path: Path) -> dict[str, Any]:
	"""Read small provenance files."""
	return json.loads(path.read_text())


def load_config(path: Path) -> dict[str, Any]:
	"""Resolve explicit paths without artifact discovery."""
	for name in ('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', 'SEIS_SSL_CLUSTER_WORKSPACE'):
		if not os.environ.get(name):
			raise ValueError(f'Missing environment variable: {name}')
	config = yaml.safe_load(os.path.expandvars(path.read_text()))
	if tuple(m['model_id'] for m in config['models']) != MODEL_IDS:
		raise ValueError('Unexpected comparison model order')
	if tuple(m['condition_id'] for m in config['models']) != CONDITIONS:
		raise ValueError('Unexpected condition order')
	for model in config['models']:
		for key in ('run_dir', 'embedding_dir'):
			if not Path(model[key]).is_absolute() or '$' in model[key]:
				raise ValueError(f'Unresolved absolute path: {model[key]}')
	expected = (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).resolve()
		/ 'paper_predictions/hmm_v1/parihaka/layout_000/large'
	)
	if Path(config['output_root']).resolve() != expected:
		raise ValueError('Output must use the dedicated paper prediction root')
	return config


def inspect_model(model: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
	"""Verify completed checkpoint and the exact embedding lineage before export."""
	run = Path(model['run_dir'])
	emb = Path(model['embedding_dir'])
	metrics = read_json(run / 'metrics.json')
	completion = inspect_completed_channel_job(run, metrics=metrics)
	identity = metrics['benchmark_identity']
	if (identity['model'], identity['layout_id'], identity['data_size']) != (
		model['model_id'],
		'layout_000',
		'large',
	):
		raise ValueError('Checkpoint model/layout/size mismatch')
	meta_path = emb / 'parihaka.embedding_metadata.json'
	meta = read_json(meta_path)
	if meta['survey_id'] != 'parihaka':
		raise ValueError('Wrong survey')
	for key, value in identity['embedding']['common_metadata'].items():
		if meta.get(key) != value:
			raise ValueError(f'Embedding metadata mismatch: {key}')
	for key in ('checkpoint_path', 'checkpoint_sha256'):
		if meta[key] != identity['embedding'][key]:
			raise ValueError(f'Encoder lineage mismatch: {key}')
	if meta.get('stratigraphy_pretext') != identity['embedding']['model_source'].get(
		'stratigraphy_pretext'
	):
		raise ValueError('HMM provenance mismatch')
	hmm = model['condition_id'].endswith('b')
	if hmm and (
		meta['stratigraphy_pretext']['distillation_weight'] != 0.2
		or meta['stratigraphy_pretext']['head_num_prototypes'] != 6
	):
		raise ValueError('Expected fixed HMM K=6, weight=0.2')
	geometry = identity['geometry']
	valid_path = emb / 'parihaka.valid_tokens.npy'
	if file_sha256(valid_path) != geometry['valid_tokens_sha256']:
		raise ValueError('Valid-token identity mismatch')
	valid = np.load(valid_path, mmap_mode='r', allow_pickle=False)
	embeddings = np.load(
		emb / 'parihaka.embeddings.npy', mmap_mode='r', allow_pickle=False
	)
	if (
		valid.dtype != np.bool_
		or list(valid.shape) != geometry['token_grid_shape_xyz']
		or list(embeddings.shape) != geometry['embedding_shape']
	):
		raise ValueError('Embedding array shape/dtype mismatch')
	if identity['tiles'] != {
		'core_size_tokens': [8, 8, 8],
		'context_halo_tokens': [1, 1, 1],
	} or geometry['patch_size_xyz'] != [8, 8, 8]:
		raise ValueError('Unexpected decoder core/halo/patch geometry')
	return {
		**model,
		'identity': identity,
		'completion': completion,
		'checkpoint_sha256': file_sha256(run / 'best.pt'),
		'prediction_metadata_sources': {
			str(meta_path): file_sha256(meta_path),
			str(run / 'metrics.json'): file_sha256(run / 'metrics.json'),
		},
		'cube_shape': geometry['volume_shape_xyz'],
		'hmm_k': 6 if hmm else None,
		'distillation_weight': 0.2 if hmm else None,
	}


def tile_input(
	embeddings: np.ndarray, valid: np.ndarray, start: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
	"""Match the training dataset's fixed ten-token, zero-padded halo input."""
	source = tuple(
		slice(max(0, s - 1), min(s + 9, n))
		for s, n in zip(start, valid.shape, strict=True)
	)
	dest = tuple(
		slice(max(0, 1 - s), max(0, 1 - s) + a.stop - a.start)
		for s, a in zip(start, source, strict=True)
	)
	values = np.zeros((10, 10, 10, embeddings.shape[-1]), dtype=np.float32)
	mask = np.zeros((10, 10, 10), dtype=np.bool_)
	values[dest] = embeddings[source]
	mask[dest] = valid[source]
	if not np.isfinite(values).all():
		raise ValueError('Nonfinite embedding')
	return np.ascontiguousarray(np.moveaxis(values, -1, 0)), mask


def core_slices(
	start: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
	"""Disjoint global voxel cores and corresponding halo-local crops."""
	global_crop = tuple(
		slice(s * 8, min((s + 8) * 8, n)) for s, n in zip(start, shape, strict=True)
	)
	local_crop = tuple(slice(8, 8 + a.stop - a.start) for a in global_crop)
	return global_crop, local_crop


def validate_prediction(root: Path, shape: tuple[int, ...]) -> dict[str, Any]:
	"""Check full saved volumes slice-wise, including invalid sentinel and class IDs."""
	pred = np.load(
		root / 'parihaka_voxel_predictions.npy', mmap_mode='r', allow_pickle=False
	)
	mask = np.load(
		root / 'parihaka_valid_voxel_mask.npy', mmap_mode='r', allow_pickle=False
	)
	if (
		pred.shape != shape
		or mask.shape != shape
		or pred.dtype != np.int8
		or mask.dtype != np.bool_
	):
		raise ValueError('Prediction shape/dtype mismatch')
	count = 0
	for p, m in zip(pred, mask, strict=True):
		if not np.isin(p[m], [0, 1]).all() or not (p[~m] == -1).all():
			raise ValueError('Invalid class or invalid-voxel sentinel')
		count += int(m.sum())
	return {
		'shape': list(shape),
		'valid_voxels': count,
		'invalid_voxels': int(pred.size) - count,
		'class_ids': [0, 1],
		'invalid_class_id': -1,
	}


def export_model(plan: dict[str, Any], output_root: Path, device: str) -> Path:
	"""Run frozen decoder argmax only and publish complete metadata last."""
	root = output_root / f'model={plan["model_id"]}'
	root.mkdir(parents=True, exist_ok=False)
	identity = plan['identity']
	checkpoint = Path(plan['run_dir']) / 'best.pt'
	if file_sha256(checkpoint) != plan['checkpoint_sha256']:
		raise ValueError('Checkpoint changed since preflight')
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	decoder = VoxelDecoder3D(**identity['decoder'], patch_size_xyz=(8, 8, 8)).to(device)
	decoder.load_state_dict(payload['model_state_dict'], strict=True)
	decoder.eval().requires_grad_(requires_grad=False)
	emb_root = Path(plan['embedding_dir'])
	emb = np.load(
		emb_root / 'parihaka.embeddings.npy', mmap_mode='r', allow_pickle=False
	)
	valid = np.load(
		emb_root / 'parihaka.valid_tokens.npy', mmap_mode='r', allow_pickle=False
	)
	shape = tuple(plan['cube_shape'])
	pred = np.lib.format.open_memmap(
		root / 'parihaka_voxel_predictions.npy', mode='w+', dtype=np.int8, shape=shape
	)
	mask = np.lib.format.open_memmap(
		root / 'parihaka_valid_voxel_mask.npy', mode='w+', dtype=np.bool_, shape=shape
	)
	starts = list(itertools.product(*(range(0, n, 8) for n in valid.shape)))
	with torch.inference_mode():
		for index, start in enumerate(starts):
			values, tokens = tile_input(emb, valid, start)
			global_crop, local_crop = core_slices(start, shape)
			voxel_valid = tokens
			for axis in range(3):
				voxel_valid = voxel_valid.repeat(8, axis=axis)
			core_valid = voxel_valid[local_crop]
			if core_valid.any():
				logits = decoder(
					torch.from_numpy(values[None]).to(device),
					torch.from_numpy(tokens[None]).to(device),
				)
				if not torch.isfinite(logits).all():
					raise ValueError('Nonfinite decoder logits')
				classes = logits.argmax(1)[0].cpu().numpy()[local_crop].astype(np.int8)
				classes[~core_valid] = -1
			else:
				classes = np.full(core_valid.shape, -1, dtype=np.int8)
			pred[global_crop] = classes
			mask[global_crop] = core_valid
			if index % 200 == 0:
				print(f'{plan["model_id"]}: tile {index + 1}/{len(starts)}', flush=True)
	pred.flush()
	mask.flush()
	validation = validate_prediction(root, shape)
	metadata = {key: value for key, value in plan.items() if key != 'identity'}
	metadata.update(
		schema_version=1,
		artifact_type='parihaka_full_volume_raw_voxel_prediction',
		complete=True,
		generated_at_utc=datetime.now(timezone.utc).isoformat(),
		dataset='parihaka',
		layout_id='layout_000',
		data_size='large',
		class_ids=[0, 1],
		class_names=['Non-Channel', 'Channel'],
		training=False,
		metric_evaluation=False,
		smoothing=False,
		ensemble=False,
		postprocessing=False,
		device=device,
		precision='float32',
		core_size_tokens=[8, 8, 8],
		context_halo_tokens=[1, 1, 1],
		validation=validation,
		train_lines=identity['train_lines'],
	)
	metadata['array_sha256'] = {p.name: file_sha256(p) for p in root.glob('*.npy')}
	(root / 'prediction_metadata.json').write_text(
		json.dumps(metadata, indent=2) + '\n'
	)
	return root


def run(config: dict[str, Any], *, dry_run: bool) -> list[dict[str, Any]]:
	"""Preflight all six models before any new output is created."""
	plans = [inspect_model(model) for model in config['models']]
	if any(
		p['identity']['geometry'] != plans[0]['identity']['geometry']
		or p['identity']['train_lines'] != plans[0]['identity']['train_lines']
		for p in plans
	):
		raise ValueError('Models must share geometry, validity and training lines')
	root = Path(config['output_root'])
	for plan in plans:
		if (root / f'model={plan["model_id"]}').exists():
			raise FileExistsError(
				'Prediction output already exists; refusing overwrite'
			)
	if not dry_run:
		torch.set_num_threads(4)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		for plan in plans:
			export_model(plan, root, config['device'])
	return [{k: v for k, v in plan.items() if k != 'identity'} for plan in plans]
