"""Read-only source checks and inference-only exports for Volve paper figures."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from seis_ssl_cluster.f3.lithology.hmm_v1_comparison_visualization import (
	absolute_path,
	read_json,
	sha256,
)
from seis_ssl_cluster.volve.horizon_data import HORIZON_NAMES
from seis_ssl_cluster.volve.horizon_metrics import soft_argmax_global_sample
from seis_ssl_cluster.volve.horizon_model import create_volve_horizon_decoder
from seis_ssl_cluster.volve.horizon_tiles import (
	HorizonTileRecord,
	HorizonTileSettings,
	build_frozen_horizon_tile,
	frozen_survey_output_valid_mask,
)

CONDITIONS = ('1', '2b', '3', '4b', '5', '6b')
MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'rot90_asym_g060_3ep',
	'rot90_asym_g060_3ep_hmm_k6_distill020',
	'random',
	'random_init_hmm_k6_distill020',
)


def load_config(path: str | Path) -> dict[str, Any]:
	"""Resolve explicit environment-owned roots and lock the comparison contract."""
	for key in (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'SEIS_SSL_CLUSTER_VOLVE_ROOT',
	):
		if not os.environ.get(key):
			raise ValueError(f'missing environment variable: {key}')
	resolved = Path(path).resolve()
	config = yaml.safe_load(os.path.expandvars(resolved.read_text()))
	config['config_path'] = str(resolved)
	for value in config['paths'].values():
		absolute_path(value)
	validate_config(config)
	return config


def validate_config(config: dict[str, Any]) -> None:
	"""Reject changes to condition identities or dedicated output boundaries."""
	if tuple(m['condition_id'] for m in config['models']) != CONDITIONS:
		raise ValueError('condition order mismatch')
	if tuple(m['model_id'] for m in config['models']) != MODEL_IDS:
		raise ValueError('model order mismatch')
	if config['comparison'] != {
		'layout_id': 'layout_000',
		'data_size': 'large',
		'fixed_distillation_weight': 0.2,
		'hmm_k': 6,
	}:
		raise ValueError('fixed comparison mismatch')
	if (
		config['inference']['window_start'] != 552
		or config['inference']['window_stop'] != 768
	):
		raise ValueError('only the trained [552, 768) window is supported')
	if config['inference']['precision'] != 'float16':
		raise ValueError('preserve the checkpoint float16 inference contract')
	root = absolute_path(config['paths']['artifact_root'])
	for key, namespace in (
		('predictions', 'paper_predictions'),
		('figures', 'paper_figures'),
	):
		if (
			absolute_path(config['outputs'][key])
			!= root / namespace / 'hmm_v1/volve/layout_000/large'
		):
			raise ValueError(f'unsafe output root: {key}')


def require_hash(path: str | Path, expected: str) -> None:
	"""Verify an existing artifact against its bound identity."""
	if sha256(path) != expected:
		raise ValueError(f'SHA-256 mismatch: {path}')


def inspect_model(config: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
	"""Bind the completed decoder, existing embeddings, and HMM provenance."""
	root = Path(config['paths']['artifact_root'])
	run = (
		root
		/ 'horizon/volve/horizon_benchmark_v1'
		/ model['family']
		/ 'runs'
		/ f'model={model["model_id"]}'
		/ 'layout=layout_000/size=large'
	)
	metrics = read_json(run / 'metrics.json')
	best = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
	latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
	identity = metrics['benchmark_identity']
	validate_checkpoint(run, model['model_id'], metrics, best, latest)
	emb = identity['embedding']
	mask_path = validate_embedding(emb, model)
	decoder = create_volve_horizon_decoder()
	if decoder.architecture != identity['decoder']['architecture']:
		raise ValueError('decoder architecture mismatch')
	decoder.load_state_dict(best['model_state_dict'], strict=True)
	return {
		**model,
		'run_dir': str(run),
		'identity': identity,
		'best_checkpoint_sha256': metrics['best_checkpoint']['sha256'],
		'best_epoch': best['epoch'],
		'metrics_sha256': sha256(run / 'metrics.json'),
		'latest_sha256': sha256(run / 'latest.pt'),
		'history_sha256': sha256(run / 'history.json'),
		'valid_tokens_path': str(mask_path),
		'hmm_k': 6 if model['hmm'] else None,
		'distillation_weight': 0.2 if model['hmm'] else None,
		'provenance_path': emb['metadata_path'],
		'validation': 'passed',
	}


def validate_checkpoint(
	run: Path,
	model_id: str,
	metrics: dict[str, Any],
	best: dict[str, Any],
	latest: dict[str, Any],
) -> None:
	"""Check completion, identity and the previously selected checkpoint."""
	identity = metrics['benchmark_identity']
	if best['run_identity'] != identity or latest['run_identity'] != identity:
		raise ValueError(f'checkpoint/metrics identity mismatch: {run}')
	if (
		latest.get('completed') is not True
		or latest['epoch'] != identity['training']['epochs']
	):
		raise ValueError(f'incomplete decoder: {run}')
	for key, expected in (
		('model', model_id),
		('layout_id', 'layout_000'),
		('data_size', 'large'),
	):
		if identity[key] != expected or metrics[key] != expected:
			raise ValueError(f'identity mismatch: {key}, {run}')
	if best['epoch'] != metrics['best_epoch'] or latest['best_epoch'] != best['epoch']:
		raise ValueError(f'best epoch mismatch: {run}')
	if (
		identity['objective']['checkpoint_selection']
		!= 'strict_lower_validation_macro_mae_v1'
	):
		raise ValueError('checkpoint selection contract mismatch')
	if best['validation'] != metrics['validation']:
		raise ValueError('stored checkpoint validation identity mismatch')
	require_hash(run / 'best.pt', metrics['best_checkpoint']['sha256'])


def validate_embedding(emb: dict[str, Any], model: dict[str, Any]) -> Path:
	"""Verify bound embedding files and HMM weight independently of names."""
	for name in ('embeddings', 'metadata'):
		require_hash(absolute_path(emb[f'{name}_path']), emb[f'{name}_sha256'])
	metadata = read_json(emb['metadata_path'])
	if metadata['checkpoint_sha256'] != emb['checkpoint_sha256']:
		raise ValueError('encoder/embedding identity mismatch')
	hmm = metadata.get('stratigraphy_pretext')
	expected_hmm = model['condition_id'] in ('2b', '4b', '6b')
	if model['hmm'] != expected_hmm:
		raise ValueError('HMM condition mismatch')
	if expected_hmm and (
		not hmm or hmm['distillation_weight'] != 0.2 or hmm['head_num_prototypes'] != 6
	):
		raise ValueError('HMM must have verified weight 0.2 and K=6')
	if not expected_hmm and hmm:
		raise ValueError('unexpected HMM provenance for parent')
	mask_path = Path(emb['embeddings_path']).with_name('volve_st10010.valid_tokens.npy')
	require_hash(mask_path, emb['valid_tokens_sha256'])
	values = np.load(emb['embeddings_path'], mmap_mode='r', allow_pickle=False)
	mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
	if list(values.shape) != emb['embedding_shape'] or values.shape != (
		51,
		90,
		107,
		384,
	):
		raise ValueError('embedding shape mismatch')
	if mask.shape != values.shape[:3] or mask.dtype != np.bool_:
		raise ValueError('token validity schema mismatch')
	return mask_path


def tile_records(settings: HorizonTileSettings) -> list[HorizonTileRecord]:
	"""Enumerate disjoint lateral cores including unlabelled and edge regions."""
	gx, gy, _ = settings.token_grid_shape
	return [
		HorizonTileRecord(i, (x, y), (min(x + 8, gx), min(y + 8, gy)), (0,) * 5)
		for i, (x, y) in enumerate(itertools.product(range(0, gx, 8), range(0, gy, 8)))
	]


def output_slices(
	record: HorizonTileRecord, shape: tuple[int, int]
) -> tuple[slice, slice]:
	"""Map one token core to its clipped canonical lateral footprint."""
	return tuple(
		slice(a * 8, min(b * 8, n))
		for a, b, n in zip(
			record.core_start_token_xy, record.core_stop_token_xy, shape, strict=True
		)
	)


def model_valid_mask(model: dict[str, Any]) -> np.ndarray:
	"""Use the original full-column token validity policy."""
	mask = np.load(model['valid_tokens_path'], mmap_mode='r', allow_pickle=False)
	return frozen_survey_output_valid_mask(
		mask[:, :, 69:96], HorizonTileSettings((401, 720), 1.0)
	)


def validate_prediction(prediction: np.ndarray, valid: np.ndarray) -> None:
	"""Require finite in-window horizon locations and NaN only outside validity."""
	if prediction.shape != (5, *valid.shape) or valid.dtype != np.bool_:
		raise ValueError('horizon prediction shape or mask dtype mismatch')
	for horizon in prediction:
		if not np.isfinite(horizon[valid]).all() or not np.isnan(horizon[~valid]).all():
			raise ValueError('prediction finite/invalid schema mismatch')
		if np.any(horizon[valid] < 552) or np.any(horizon[valid] > 767):
			raise ValueError('prediction outside the trained sample window')


def export_prediction(config: dict[str, Any], source: dict[str, Any]) -> Path:
	"""Run the existing decoder and soft-argmax, never training or scoring."""
	root = Path(config['outputs']['predictions']) / f'model={source["model_id"]}'
	if root.exists():
		raise FileExistsError(root)
	device = torch.device(config['inference']['device'])
	if device.type != 'cuda' or not torch.cuda.is_available():
		raise ValueError('the bound float16 runtime requires an available CUDA device')
	checkpoint = Path(source['run_dir']) / 'best.pt'
	require_hash(checkpoint, source['best_checkpoint_sha256'])
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	model = (
		create_volve_horizon_decoder()
		.to(device)
		.eval()
		.requires_grad_(requires_grad=False)
	)
	model.load_state_dict(payload['model_state_dict'], strict=True)
	del payload
	emb = np.load(
		source['identity']['embedding']['embeddings_path'],
		mmap_mode='r',
		allow_pickle=False,
	)
	mask = np.load(source['valid_tokens_path'], mmap_mode='r', allow_pickle=False)
	settings = HorizonTileSettings((401, 720), 1.0)
	valid = model_valid_mask(source)
	root.mkdir(parents=True)
	path = root / 'horizon_sample_predictions.npy'
	predictions = np.lib.format.open_memmap(
		path, mode='w+', dtype=np.float32, shape=(5, 401, 720)
	)
	predictions[:] = np.nan
	with torch.inference_mode():
		for record in tile_records(settings):
			tile = build_frozen_horizon_tile(
				record=record,
				embeddings=emb[:, :, 69:96],
				valid_tokens=mask[:, :, 69:96],
				settings=settings,
			)
			values = torch.from_numpy(tile.embeddings).unsqueeze(0).to(device)
			valid_tokens = (
				torch.from_numpy(tile.token_valid_mask).unsqueeze(0).to(device)
			)
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				logits = model(values, valid_tokens)
			if not torch.isfinite(logits).all():
				raise ValueError('nonfinite decoder logits')
			predicted = soft_argmax_global_sample(logits).cpu().numpy()[0]
			x, y = output_slices(record, settings.lateral_shape_xy)
			core = predicted[:, : x.stop - x.start, : y.stop - y.start]
			predictions[:, x, y] = np.where(valid[x, y][None], core, np.nan)
	predictions.flush()
	validate_prediction(predictions, valid)
	np.save(root / 'valid_lateral_mask.npy', valid, allow_pickle=False)
	metadata = {
		'schema_version': 1,
		'artifact_type': 'volve_paper_horizon_prediction',
		'complete': True,
		'source': source,
		'horizon_names': list(HORIZON_NAMES),
		'prediction_shape': list(predictions.shape),
		'cube_shape': [401, 720, 850],
		'window_start': 552,
		'window_stop': 768,
		'prediction_unit': 'zero_based_fractional_sample',
		'prediction_method': 'soft_argmax_global_sample',
		'lateral_coverage': 'entire_survey',
		'tile_count': len(tile_records(settings)),
		'inference': config['inference'],
		'training': False,
		'metric_evaluation': False,
		'smoothing': False,
		'ensemble': False,
		'postprocessing': False,
		'horizon_reordering': False,
		'valid_trace_count': int(valid.sum()),
		'array_sha256': {
			p.name: sha256(p) for p in (path, root / 'valid_lateral_mask.npy')
		},
	}
	(root / 'prediction_metadata.json').write_text(
		json.dumps(metadata, indent=2) + '\n'
	)
	return root
