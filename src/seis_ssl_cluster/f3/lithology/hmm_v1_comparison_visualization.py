"""Read-only, fixed-condition F3 HMM_v1 paper section comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.visualization.facies import facies_legend_handles, facies_palette

if TYPE_CHECKING:
	from matplotlib.figure import Figure
	from numpy.typing import NDArray

CONDITION_ORDER = ('1', '2b', '3', '4b', '5', '6b')
MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_bt_nr_rot90_asym_g060_3ep',
	'local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep',
	'random',
	'random_self_target_hmm_k6_distill020',
)
DISPLAY_NAMES = (
	'MAE100 + MAE25',
	'MAE100 + HMM25',
	'asym LocalBT3',
	'asym LocalBT3 + HMM25',
	'Random weight',
	'Random weight + HMM25',
)
DOMAINS = ('inline', 'crossline', 'time')
DATASET = {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v2'}
OUTPUT_SUFFIX = Path('paper_figures/hmm_v1/f3/layout_000/large')


def read_json(path: str | Path) -> dict[str, Any]:
	"""Read a metadata object without loading array data."""
	return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path: str | Path) -> str:
	"""Hash a file with bounded memory."""
	digest = hashlib.sha256()
	with Path(path).open('rb') as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def absolute_path(value: str) -> Path:
	"""Reject unresolved or relative artifact/provenance paths."""
	path = Path(os.path.expandvars(value))
	if '$' in str(path) or not path.is_absolute():
		raise ValueError(f'expected an absolute resolved path: {value}')
	return path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
	"""Resolve the single experiment YAML and enforce the comparison contract."""
	required = (
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		'SEIS_SSL_CLUSTER_WORKSPACE',
		'F3_ROOT',
	)
	missing = [name for name in required if not os.environ.get(name)]
	if missing:
		raise ValueError(
			f'Missing required environment variables: {", ".join(missing)}'
		)
	workspace = absolute_path(os.environ['SEIS_SSL_CLUSTER_WORKSPACE'])
	config_path = Path(path)
	if not config_path.is_absolute():
		config_path = workspace / config_path
	config = yaml.safe_load(os.path.expandvars(config_path.read_text(encoding='utf-8')))
	config['config_path'] = str(config_path.resolve())
	for group in ('paths', 'inputs', 'outputs'):
		config[group] = {
			key: str(absolute_path(value)) for key, value in config[group].items()
		}
	for model in config['models']:
		model['prediction_dir'] = str(absolute_path(model['prediction_dir']))
	validate_config(config)
	return config


def validate_config(config: dict[str, Any]) -> None:
	"""Keep this paper comparison fixed, including panel order and supervision."""
	comparison = config['comparison']
	if config['dataset'] != DATASET or comparison != {
		'hmm_version': 'HMM_v1',
		'layout_id': 'layout_000',
		'data_size': 'large',
		'fixed_distillation_weight': 0.2,
		'condition_order': list(CONDITION_ORDER),
	}:
		raise ValueError(
			'dataset/comparison must use fixed F3 HMM_v1 layout_000 / large conditions'
		)
	validate_condition_mapping(config['models'])
	validate_selection_figure(config)
	root = Path(config['paths']['artifact_root']) / OUTPUT_SUFFIX
	if Path(config['outputs']['root']) != root.resolve():
		raise ValueError(f'output root must be {root}')
	for key, name in [
		('selected_slices_csv', 'selected_slices.csv'),
		('manifest_json', 'manifest.json'),
	]:
		if Path(config['outputs'][key]) != root / name:
			raise ValueError(f'{key} must be inside the dedicated output root')


def validate_condition_mapping(models: list[dict[str, Any]]) -> None:
	"""Reject alternate condition IDs, family weights, and panel names."""
	if len(models) != 6:
		raise ValueError('exactly six models are required')
	for index, model in enumerate(models):
		expected = {
			'condition_id': CONDITION_ORDER[index],
			'model_id': MODEL_IDS[index],
			'display_name': DISPLAY_NAMES[index],
			'hmm': bool(index % 2),
		}
		if any(model.get(key) != value for key, value in expected.items()):
			raise ValueError(f'condition mapping mismatch: {model}')
		if model['hmm'] and (
			model.get('hmm_k') != 6 or model.get('distillation_weight') != 0.2
		):
			raise ValueError(
				'all HMM conditions require K=6 and distillation_weight=0.2'
			)


def validate_selection_figure(config: dict[str, Any]) -> None:
	"""Enforce deterministic section selection and production rendering settings."""
	selection = config['selection']
	if selection != {
		'count_per_domain': 10,
		'quantile_min': 0.05,
		'quantile_max': 0.95,
		'common_valid_thresholds': [0.95, 0.90, 0.80],
		'minimum_ground_truth_classes': 2,
		'exclude_active_training_lines': True,
	}:
		raise ValueError('section selection policy is fixed for this comparison')
	figure = config['figure']
	for key, value in {
		'rows': 2,
		'columns': 4,
		'amplitude_cmap': 'seismic',
		'label_interpolation': 'nearest',
		'invalid_prediction_rgb': [226, 226, 226],
		'output_format': 'png',
		'background': 'white',
	}.items():
		if figure[key] != value:
			raise ValueError(f'figure.{key} must be {value!r}')
	if figure['dpi'] < 300 or figure['amplitude_sample_count'] < 1:
		raise ValueError(
			'production figures require dpi >= 300 and positive sampling count'
		)


def verified_identity(identity: dict[str, Any]) -> Path:
	"""Verify a small provenance file against the prediction's stored identity."""
	path = absolute_path(identity['path'])
	if sha256(path) != identity['sha256']:
		raise ValueError(f'provenance SHA-256 mismatch: {path}')
	return path


def validate_model(
	model: dict[str, Any],
	config: dict[str, Any],
	shape: tuple[int, ...],
	class_ids: list[int],
) -> tuple[Any, dict[str, Any]]:
	"""Use the shared full validator, then check bound F3/layout/HMM provenance."""
	root = Path(model['prediction_dir'])
	artifact = validate_f3_voxel_prediction_artifact(root, mmap_mode='r')
	metadata = artifact.metadata
	if (
		metadata['model_tag'] != model['model_id']
		or tuple(metadata['volume_shape_xyz']) != shape
	):
		raise ValueError(f'model identity or volume shape mismatch: {root}')
	if metadata['class_probability_order'] != class_ids:
		raise ValueError(f'class probability order mismatch: {root}')
	if (
		not metadata['coverage']['exact_once']
		or metadata['prediction_kind'] != 'frozen_embedding_decoder'
	):
		raise ValueError(
			f'expected completed full-volume raw decoder predictions: {root}'
		)
	source = metadata['source_identity']
	decoder_path = verified_identity(source['resolved_decoder_config'])
	embedding_path = verified_identity(
		source['artifact_identities']['embedding_metadata']
	)
	voxel_path = verified_identity(
		source['artifact_identities']['voxel_dataset_metadata']
	)
	decoder, embedding, voxel = map(
		read_json, (decoder_path, embedding_path, voxel_path)
	)
	if decoder['dataset'] != DATASET or voxel['dataset'] != DATASET:
		raise ValueError(f'dataset provenance mismatch: {root}')
	if decoder['model']['tag'] != model['model_id']:
		raise ValueError(f'decoder model mismatch: {decoder_path}')
	section = voxel['section_layout']
	if section['layout_id'] != 'layout_000' or section['data_size'] != 'large':
		raise ValueError(f'layout/data size provenance mismatch: {voxel_path}')
	expected_dataset = Path(config['inputs']['section_layout_metadata']).parent
	if (
		absolute_path(decoder['voxel_dataset']['input_dir']) != expected_dataset
		or voxel_path.parent != expected_dataset
	):
		raise ValueError(f'supervision source mismatch: {root}')
	if absolute_path(metadata['inputs']['embedding_metadata']) != embedding_path:
		raise ValueError(f'embedding source mismatch: {root}')
	if decoder['embeddings']['checkpoint_path'] != embedding['checkpoint_path']:
		raise ValueError(f'encoder checkpoint provenance mismatch: {root}')
	validate_hmm_provenance(model, embedding, embedding_path)
	result = {
		**model,
		'layout_id': section['layout_id'],
		'data_size': section['data_size'],
		'hmm_k': 6 if model['hmm'] else None,
		'distillation_weight': 0.2 if model['hmm'] else None,
		'prediction_metadata_path': str(root / 'prediction_metadata.json'),
		'prediction_metadata_sha256': sha256(root / 'prediction_metadata.json'),
		'prediction_array_path': str(root / 'f3_voxel_predictions.npy'),
		'valid_mask_path': str(root / 'f3_valid_voxel_mask.npy'),
		'resolved_config_path': str(decoder_path),
		'provenance_path': str(embedding_path),
		'provenance_sha256': sha256(embedding_path),
		'voxel_dataset_metadata_path': str(voxel_path),
		'artifact_validation': {
			'status': 'complete',
			'summary': metadata['summary'],
			'coverage': metadata['coverage'],
			'class_probability_order': class_ids,
			'array_validation': (
				'shared validator: full chunked dtype/value/summary validation; '
				'no NPY hashes recomputed'
			),
			'confidence_and_probabilities': (
				'validated by shared schema only; not used for selection or rendering'
			),
		},
	}
	return artifact.arrays, result


def validate_hmm_provenance(
	model: dict[str, Any],
	embedding: dict[str, Any],
	path: Path,
) -> None:
	"""Verify HMM parameters from bound embedding provenance, never from names."""
	if model['hmm']:
		pretext = embedding['stratigraphy_pretext']
		if pretext['distillation_weight'] != 0.2 or pretext['head_num_prototypes'] != 6:
			raise ValueError(f'HMM K/weight mismatch: {path}')


def coordinate_step(values: NDArray) -> float | None:
	"""Check finite strictly monotone coordinates; return step only if uniform."""
	difference = np.diff(values.astype(float))
	if (
		not np.all(np.isfinite(values))
		or not len(difference)
		or not (np.all(difference > 0) or np.all(difference < 0))
	):
		raise ValueError('coordinates must be finite and strictly monotone')
	return (
		float(difference[0])
		if np.allclose(difference, difference[0], rtol=1e-6, atol=1e-9)
		else None
	)


def time_in_seconds(samples: NDArray, interval: int) -> tuple[NDArray, dict[str, Any]]:
	"""Determine sample units by matching differences to the binary interval (µs)."""
	step = coordinate_step(samples)
	differences = np.abs(np.diff(samples))
	units = [
		(unit, factor)
		for unit, factor in [('milliseconds', 0.001), ('seconds', 1.0)]
		if interval > 0
		and np.allclose(differences * factor, interval * 1e-6, rtol=1e-5, atol=1e-9)
	]
	if len(units) != 1:
		raise ValueError(
			f'ambiguous/inconsistent sample units: first={samples[0]}, '
			f'last={samples[-1]}, differences={differences.tolist()}, '
			f'binary sample interval={interval} microseconds'
		)
	unit, factor = units[0]
	seconds = samples.astype(float) * factor
	return seconds, {
		'raw_sample_min': float(min(samples)),
		'raw_sample_max': float(max(samples)),
		'raw_sample_step': step,
		'segy_binary_sample_interval': interval,
		'segy_binary_sample_interval_unit': 'microseconds',
		'detected_sample_unit': unit,
		'conversion_to_seconds': factor,
		'time_conversion_method': (
			'sample differences matched to binary interval in microseconds'
		),
		'time_min_seconds': float(min(seconds)),
		'time_max_seconds': float(max(seconds)),
		'time_step_seconds': coordinate_step(seconds),
	}


def load_coordinates(
	inputs: dict[str, str], shape: tuple[int, ...]
) -> tuple[dict[str, NDArray], dict[str, Any]]:
	"""Read coordinate vectors only and cross-check prepared metadata."""
	import segyio  # noqa: PLC0415

	geometry, volume = (
		read_json(inputs['segy_geometry']),
		read_json(inputs['volume_metadata']),
	)
	for metadata in (geometry, volume):
		axes = metadata['axis_assumption']['cube_to_repo_axes']
		if [axes[f'cube_axis_{i}']['domain_axis'] for i in range(3)] != [
			'inline',
			'crossline',
			'sample/time',
		]:
			raise ValueError('prepared axis order does not match inline/crossline/time')
	for role in ('seismic', 'label'):
		if tuple(volume['volumes'][role]['shape_xyz']) != shape:
			raise ValueError(f'prepared {role} shape mismatch')
	loaded = []
	for role in ('seismic', 'label'):
		with segyio.open(inputs[f'{role}_segy'], 'r') as segy:
			vectors = [
				np.array(segy.ilines),
				np.array(segy.xlines),
				np.array(segy.samples),
			]
			interval = int(segy.bin[segyio.BinField.Interval])
		if tuple(map(len, vectors)) != shape:
			raise ValueError(f'{role} SEG-Y shape mismatch')
		validate_geometry_vectors(vectors, geometry['segy_files'][role])
		loaded.append((vectors, interval))
	if loaded[0][1] != loaded[1][1] or any(
		not np.array_equal(a, b)
		for a, b in zip(loaded[0][0], loaded[1][0], strict=True)
	):
		raise ValueError('seismic and label SEG-Y geometries differ')
	inline, crossline, samples = loaded[0][0]
	seconds, report = time_in_seconds(samples, loaded[0][1])
	coords = dict(zip(DOMAINS, (inline, crossline, seconds), strict=True))
	for name, values in coords.items():
		report.update(
			{
				f'{name}_min': float(min(values)),
				f'{name}_max': float(max(values)),
				f'{name}_count': len(values),
				f'{name}_step': coordinate_step(values),
			}
		)
	return coords, report


def validate_geometry_vectors(vectors: list[NDArray], stored: dict[str, Any]) -> None:
	"""Check each SEG-Y coordinate vector against the inspected geometry."""
	for prefix, vector in zip(('iline', 'xline', 'sample'), vectors, strict=True):
		coordinate_step(vector)
		for key, value in [
			('min', min(vector)),
			('max', max(vector)),
			('count', len(vector)),
		]:
			if stored[f'{prefix}_{key}'] != value:
				raise ValueError(f'SEG-Y {prefix}_{key} mismatch')


def extract_section(volume: NDArray, domain: str, index: int) -> NDArray:
	"""Extract one section with the fixed XYZ orientation, without copying a cube."""
	if domain == 'inline':
		return volume[index, :, :].T
	if domain == 'crossline':
		return volume[:, index, :].T
	if domain == 'time':
		return volume[:, :, index]
	raise ValueError(f'unknown domain: {domain}')


def section_axes(
	coords: dict[str, NDArray], domain: str
) -> tuple[NDArray, NDArray, str, str]:
	"""Resolve actual horizontal/vertical coordinates and publication axis labels."""
	if domain == 'inline':
		return coords['crossline'], coords['time'], 'xline', 'time (s)'
	if domain == 'crossline':
		return coords['inline'], coords['time'], 'inline', 'time (s)'
	if domain == 'time':
		return coords['crossline'], coords['inline'], 'xline', 'inline'
	raise ValueError(f'unknown domain: {domain}')


def load_active_lines(path: str, coords: dict[str, NDArray]) -> dict[str, list[int]]:
	"""Load supervised line numbers, checking layout and coordinate membership."""
	metadata = read_json(path)
	if metadata['layout_id'] != 'layout_000' or metadata['data_size'] != 'large':
		raise ValueError('active training lines must come from layout_000 / large')
	active = metadata['active_lines']
	for domain in ('inline', 'crossline'):
		if not set(active[domain]).issubset(coords[domain].tolist()):
			raise ValueError(f'active {domain} lines are not actual SEG-Y coordinates')
	return active


def slice_eligibility(  # noqa: PLR0913, PLR0917
	seismic: NDArray,
	labels: NDArray,
	masks: list[NDArray],
	class_ids: list[int],
	domain: str,
	coordinates: NDArray,
	active: list[int],
) -> list[dict[str, Any]]:
	"""Scan slices using GT classes, finite amplitude, and common validity masks."""
	rows = []
	low, high = np.quantile(coordinates, [0.05, 0.95])
	for index, coordinate in enumerate(coordinates):
		if not low <= coordinate <= high or coordinate in active:
			continue
		label = extract_section(labels, domain, index)
		gt_valid = np.isin(label, class_ids)
		ids = np.unique(label[gt_valid]).astype(int).tolist()
		base = gt_valid & np.isfinite(extract_section(seismic, domain, index))
		denominator = int(np.count_nonzero(base))
		common = base.copy()
		for mask in masks:
			common &= extract_section(mask, domain, index)
		count = int(np.count_nonzero(common))
		rows.append(
			{
				'domain': domain,
				'array_index': index,
				'actual_coordinate': float(coordinate),
				'coordinate_unit': 'seconds' if domain == 'time' else 'line_number',
				'common_valid_fraction': count / denominator if denominator else 0.0,
				'ground_truth_valid_count': int(np.count_nonzero(gt_valid)),
				'ground_truth_valid_finite_amplitude_count': denominator,
				'common_valid_count': count,
				'ground_truth_class_count': len(ids),
				'ground_truth_class_ids': ids,
				'excluded_training_line_count': len(active),
				'active_training_line': 'not_applicable' if domain == 'time' else False,
			}
		)
	return rows


def select_slices(
	rows: list[dict[str, Any]], coordinates: NDArray
) -> tuple[list[dict[str, Any]], float]:
	"""Choose unused slices nearest fixed coordinate quantiles; tie by index."""
	eligible = []
	for threshold in (0.95, 0.90, 0.80):
		eligible = [
			row
			for row in rows
			if row['ground_truth_class_count'] >= 2
			and row['ground_truth_valid_count'] > 0
			and row['common_valid_fraction'] >= threshold
		]
		if len(eligible) >= 10:
			break
	else:
		fractions = [row['common_valid_fraction'] for row in rows]
		stats = (
			[float(function(fractions)) for function in (np.min, np.median, np.max)]
			if fractions
			else []
		)
		raise ValueError(
			f'fewer than 10 eligible slices at 0.80: candidates={len(eligible)}; '
			f'valid fraction min/median/max={stats}'
		)
	selected = []
	for order, quantile in enumerate(np.linspace(0.05, 0.95, 10), start=1):
		target = float(np.quantile(coordinates, quantile))
		chosen = min(
			eligible,
			key=lambda row: (
				round(abs(row['actual_coordinate'] - target), 10),
				row['array_index'],
			),
		)
		eligible.remove(chosen)
		selected.append(
			{
				**chosen,
				'order': order,
				'selection_quantile': float(quantile),
				'selection_threshold': threshold,
			}
		)
	return selected, threshold


def amplitude_scale(
	seismic: NDArray, stats_path: str, figure: dict[str, Any]
) -> dict[str, Any]:
	"""Use recorded matching percentiles, otherwise bounded deterministic sampling."""
	stats = read_json(stats_path)
	percentiles = figure['amplitude_clip_percentiles']
	if [
		stats.get('clip_low_percentile'),
		stats.get('clip_high_percentile'),
	] == percentiles:
		values = [stats['clip_low'], stats['clip_high']]
		source = stats_path
		count = 0
	else:
		rng = np.random.default_rng(figure['amplitude_sample_seed'])
		indices = rng.integers(0, seismic.size, size=figure['amplitude_sample_count'])
		sample = seismic[np.unravel_index(indices, seismic.shape)]
		sample = sample[np.isfinite(sample)]
		if not sample.size:
			raise ValueError('no finite amplitude samples')
		values = np.percentile(sample, percentiles).tolist()
		source, count = (
			'deterministic uniform flat-index sampling with replacement',
			int(sample.size),
		)
	limit = max(abs(value) for value in values)
	if not np.isfinite(limit) or limit <= 0:
		raise ValueError(f'invalid global amplitude limit: {limit}')
	return {
		'source': source,
		'clip_percentiles': percentiles,
		'lower_percentile_value': values[0],
		'upper_percentile_value': values[1],
		'vmin': -limit,
		'vmax': limit,
		'cmap': 'seismic',
		'sample_count': count,
		'requested_sample_count': figure['amplitude_sample_count'],
		'seed': figure['amplitude_sample_seed'],
	}


def categorical_image(
	labels: NDArray, class_ids: list[int], mask: NDArray | None = None
) -> np.ma.MaskedArray:
	"""Map class IDs to palette positions; mask only schema-invalid predictions."""
	valid = np.ones(labels.shape, dtype=bool) if mask is None else mask
	if not np.all(np.isin(labels[valid], class_ids)):
		raise ValueError('unknown class ID in categorical image')
	if mask is not None and np.any(labels[~valid] != -1):
		raise ValueError('invalid prediction voxels must contain -1')
	indices = np.zeros(labels.shape, dtype=np.int16)
	for position, class_id in enumerate(class_ids):
		indices[labels == class_id] = position
	return np.ma.array(indices, mask=~valid)


def coordinate_edges(values: NDArray) -> NDArray:
	"""Return edges around coordinate centers, including nonuniform grids."""
	midpoints = (values[:-1] + values[1:]) / 2
	return np.concatenate(
		(
			[values[0] - (midpoints[0] - values[0])],
			midpoints,
			[values[-1] + (values[-1] - midpoints[-1])],
		)
	)


def drawing_method(coords: dict[str, NDArray], domain: str) -> str:
	"""Use imshow only when both displayed axes are uniformly spaced."""
	x, y, _, _ = section_axes(coords, domain)
	return (
		'imshow'
		if coordinate_step(x) is not None and coordinate_step(y) is not None
		else 'pcolormesh'
	)


@dataclass
class PreparedComparison:
	"""Validated read-only arrays and serializable publication metadata."""

	config: dict[str, Any]
	seismic: NDArray
	labels: NDArray
	arrays: list[Any]
	classes: list[dict[str, Any]]
	coords: dict[str, NDArray]
	manifest: dict[str, Any]
	rows: list[dict[str, Any]]


def valid_display_bounds(common: NDArray) -> list[int]:
	"""Trim invalid borders; reject internal holes rather than hiding them."""
	y, x = np.nonzero(common)
	if not len(y):
		raise ValueError('no common valid display area')
	top, bottom = int(y.min()), int(y.max()) + 1
	left, right = int(x.min()), int(x.max()) + 1
	while left < right and not common[top:bottom, left].all():
		left += 1
	while right > left and not common[top:bottom, right - 1].all():
		right -= 1
	if right - left < 2 or bottom - top < 2:
		raise ValueError('no rectangular common valid display area')
	if not common[top:bottom, left:right].all():
		raise ValueError('internal invalid prediction hole in display area')
	return [top, bottom, left, right]


def display_crops(prepared: PreparedComparison) -> dict[str, Any]:
	"""Use identical fully valid bounds for all models and sections of each domain."""
	crops = {}
	for domain in DOMAINS:
		common = None
		for row in prepared.rows:
			if row['domain'] != domain:
				continue
			for arrays in prepared.arrays:
				mask = extract_section(arrays.valid_mask, domain, row['array_index'])
				if common is None:
					common = mask.copy()
				else:
					common &= mask
		if common is None:
			raise ValueError(f'no selected sections for {domain}')
		top, bottom, left, right = valid_display_bounds(common)
		x, y, _, _ = section_axes(prepared.coords, domain)
		crops[domain] = {
			'array_bounds_yx_stop_exclusive': [top, bottom, left, right],
			'x_min': float(min(x[left:right])),
			'x_max': float(max(x[left:right])),
			'y_min': float(min(y[top:bottom])),
			'y_max': float(max(y[top:bottom])),
			'invalid_prediction_voxels_displayed': 0,
		}
	return crops


def panel_subtitle(title: str) -> str:
	"""Use concise one-line paper subtitles without changing model provenance."""
	return (
		title.replace('MAE100 + MAE25', 'MAE')
		.replace('asym LocalBT3', 'LocalBT')
		.replace('Random weight', 'Random')
		.replace('MAE100', 'MAE')
		.replace('HMM25', 'HMM')
	)


def render_figure(prepared: PreparedComparison, row: dict[str, Any]) -> Figure:  # noqa: PLR0915
	"""Create the one fixed 2-by-4 figure; caller must close it after saving."""
	import matplotlib.pyplot as plt  # noqa: PLC0415
	from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415
	from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: PLC0415

	settings, domain, index = (
		prepared.config['figure'],
		row['domain'],
		row['array_index'],
	)
	fig, axes = plt.subplots(
		2,
		4,
		figsize=(settings['width_mm'] / 25.4, settings['height_mm'] / 25.4),
		sharex=True,
		sharey=True,
		facecolor='white',
	)
	try:
		x, y, xlabel, ylabel = section_axes(prepared.coords, domain)
		crops = prepared.manifest.get('display_crops_by_domain')
		if crops is None:
			crops = display_crops(prepared)
		top, bottom, left, right = crops[domain]['array_bounds_yx_stop_exclusive']
		x, y = x[left:right], y[top:bottom]
		xedge, yedge = coordinate_edges(x), coordinate_edges(y)
		palette = facies_palette(prepared.classes)
		class_ids = list(palette)
		cmap = ListedColormap(list(palette.values()))
		cmap.set_bad(np.array(settings['invalid_prediction_rgb']) / 255)
		norm = BoundaryNorm(np.arange(len(class_ids) + 1) - 0.5, len(class_ids))
		titles = [
			'Seismic amplitude',
			'Ground truth',
			*[model['display_name'] for model in prepared.config['models']],
		]
		for panel, ax in enumerate(axes.flat):
			if panel == 0:
				data = extract_section(prepared.seismic, domain, index)[
					top:bottom, left:right
				]
				options = {
					'cmap': 'seismic',
					'vmin': prepared.manifest['amplitude']['vmin'],
					'vmax': prepared.manifest['amplitude']['vmax'],
				}
			else:
				volume = (
					prepared.labels
					if panel == 1
					else prepared.arrays[panel - 2].predictions
				)
				mask = (
					None
					if panel == 1
					else extract_section(
						prepared.arrays[panel - 2].valid_mask, domain, index
					)
				)
				data = categorical_image(
					extract_section(volume, domain, index)[top:bottom, left:right],
					class_ids,
					None if mask is None else mask[top:bottom, left:right],
				)
				if np.ma.getmaskarray(data).any():
					raise ValueError('invalid prediction inside display crop')  # noqa: TRY301
				options = {'cmap': cmap, 'norm': norm}
			if drawing_method(prepared.coords, domain) == 'imshow':
				ax.imshow(
					data,
					origin='lower',
					extent=(xedge[0], xedge[-1], yedge[0], yedge[-1]),
					interpolation='nearest',
					aspect='auto',
					**options,
				)
			else:
				ax.pcolormesh(
					xedge, yedge, data, shading='flat', antialiased=False, **options
				)
				ax.set_aspect('auto')
			ax.set_xlim(min(x), max(x))
			ax.set_ylim((min(y), max(y)) if domain == 'time' else (max(y), min(y)))
			ax.set_title(
				panel_subtitle(titles[panel]),
				fontsize=settings['title_size'],
				pad=6,
			)
			ax.annotate(
				f'({chr(97 + panel)})',
				xy=(0, 0.5),
				xycoords=(ax.transAxes, ax.title),
				xytext=(-3, 0),
				textcoords='offset points',
				fontsize=settings['panel_label_size'],
				fontweight='bold',
				va='center',
				ha='right',
			)
			ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
			ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=domain == 'time'))
			ax.yaxis.set_major_formatter(
				FormatStrFormatter('%d' if domain == 'time' else '%.3f')
			)
			ax.tick_params(
				which='both', length=0, labelsize=settings['tick_label_size']
			)
			for spine in ax.spines.values():
				spine.set_linewidth(settings['axis_spine_width'])
			ax.grid(visible=False)
			if panel >= 4:
				ax.set_xlabel(xlabel, fontsize=settings['axis_label_size'])
			if panel % 4 == 0:
				ax.set_ylabel(ylabel, fontsize=settings['axis_label_size'])
		handles = facies_legend_handles(prepared.classes)
		fig.legend(
			handles=handles,
			loc='lower center',
			ncol=3,
			frameon=False,
			fontsize=settings['legend_size'],
		)
		fig.subplots_adjust(
			left=0.11, right=0.985, bottom=0.18, top=0.945, wspace=0.14, hspace=0.25
		)
		apply_font_family(fig, settings['font_family'])
	except BaseException:
		plt.close(fig)
		raise
	return fig


def apply_font_family(fig: Figure, family: str) -> None:
	"""Apply the requested font to all figure text, including generated ticks."""
	from matplotlib.text import Text  # noqa: PLC0415

	for ax in fig.axes:
		ax.get_xticklabels()
		ax.get_yticklabels()
	for text in fig.findobj(match=Text):
		text.set_fontfamily(family)


def resolve_font_files(family: str) -> dict[str, str]:
	"""Resolve actual regular/bold font files without silent font substitution."""
	from matplotlib.font_manager import FontProperties, findfont  # noqa: PLC0415

	return {
		weight: findfont(
			FontProperties(family=family, weight=weight), fallback_to_default=False
		)
		for weight in ('normal', 'bold')
	}


def output_png(row: dict[str, Any]) -> str:
	"""Name a PNG by domain and actual coordinate."""
	domain, coordinate = row['domain'], row['actual_coordinate']
	suffix = (
		f'{coordinate:.3f}s_sample_{row["array_index"]:04d}'.replace('.', 'p')
		if domain == 'time'
		else f'{coordinate:.0f}'
	)
	return f'{domain}/f3_hmm_v1_layout_000_large_{domain}_{suffix}.png'


def validate_input_values(
	seismic: NDArray, labels: NDArray, class_ids: list[int]
) -> None:
	"""Reject NaN and unknown labels slice by slice before any figure is created."""
	if seismic.ndim != 3 or labels.shape != seismic.shape:
		raise ValueError('seismic and label volumes must have identical 3D shapes')
	for index in range(seismic.shape[0]):
		if not np.all(np.isfinite(seismic[index])):
			raise ValueError(f'nonfinite seismic value in inline array index {index}')
		if not np.all(np.isin(labels[index], class_ids)):
			raise ValueError(
				f'unknown/nonfinite ground truth class in inline array index {index}'
			)


def prepare_comparison(config: dict[str, Any]) -> PreparedComparison:
	"""Validate every input and determine all sections without creating output files."""
	inputs = config['inputs']
	seismic, labels = [
		np.load(inputs[key], mmap_mode='r', allow_pickle=False)
		for key in ('seismic_volume', 'label_volume')
	]
	classes = read_json(inputs['class_info'])['classes']
	class_ids = [item['class_id'] for item in classes]
	if len(set(class_ids)) != len(class_ids) or 0 not in class_ids:
		raise ValueError('F3 class IDs must be unique and include valid class 0')
	facies_palette(classes)
	validate_input_values(seismic, labels, class_ids)
	coords, coordinate_report = load_coordinates(inputs, seismic.shape)
	active = load_active_lines(inputs['section_layout_metadata'], coords)
	arrays, models = [], []
	for model in config['models']:
		values, report = validate_model(model, config, seismic.shape, class_ids)
		arrays.append(values)
		models.append(report)
	rows, thresholds = [], {}
	for domain in DOMAINS:
		candidates = slice_eligibility(
			seismic,
			labels,
			[value.valid_mask for value in arrays],
			class_ids,
			domain,
			coords[domain],
			active.get(domain, []),
		)
		selected, thresholds[domain] = select_slices(candidates, coords[domain])
		rows.extend({**row, 'output_png': output_png(row)} for row in selected)
	manifest = build_manifest(
		config,
		models,
		classes,
		seismic.shape,
		coordinate_report,
		active,
		thresholds,
		rows,
	)
	manifest['amplitude'] = amplitude_scale(
		seismic, inputs['normalization_stats'], config['figure']
	)
	manifest['drawing_method_by_domain'] = {
		domain: drawing_method(coords, domain) for domain in DOMAINS
	}
	prepared = PreparedComparison(
		config, seismic, labels, arrays, classes, coords, manifest, rows
	)
	manifest['display_crops_by_domain'] = display_crops(prepared)
	manifest['figure']['font_files'] = resolve_font_files(
		config['figure']['font_family']
	)
	manifest['figure']['invalid_prediction_policy'] = (
		'crop common valid borders across all selected sections per domain'
	)
	return prepared


def build_manifest(  # noqa: PLR0913, PLR0917
	config: dict[str, Any],
	models: list[dict[str, Any]],
	classes: list[dict[str, Any]],
	shape: tuple[int, ...],
	coordinates: dict[str, Any],
	active: dict[str, list[int]],
	thresholds: dict[str, float],
	rows: list[dict[str, Any]],
) -> dict[str, Any]:
	"""Describe the comparison and verified provenance, without report inputs."""
	commit = subprocess.check_output(  # noqa: S603
		['git', '-C', config['paths']['workspace'], 'rev-parse', 'HEAD'],  # noqa: S607
		text=True,
	).strip()
	return {
		'schema_version': 1,
		'artifact_type': 'f3_hmm_v1_paper_sections',
		'generated_at_utc': datetime.now(timezone.utc).isoformat(),
		'dataset_name': DATASET['name'],
		'dataset_version': DATASET['version'],
		'hmm_version': 'HMM_v1',
		'source_freeze_manifest': config['source_freeze_manifest'],
		'source_git_commit': config['source_git_commit'],
		'generation_git_commit': commit,
		'config_path': config['config_path'],
		'config_sha256': sha256(config['config_path']),
		'layout_id': 'layout_000',
		'data_size': 'large',
		'comparison_policy': {
			'purpose': (
				'paired visualization of HMM effect under a fixed distillation weight'
			),
			'fixed_distillation_weight': 0.2,
			'condition_ids': list(CONDITION_ORDER),
			'no_per_family_weight_selection': True,
			'no_model_specific_layout_selection': True,
			'layout_id': 'layout_000',
			'data_size': 'large',
			'ensemble': False,
			'smoothing': False,
			'postprocessing': False,
		},
		'models': models,
		'inputs': {
			**{f'{key}_path': value for key, value in config['inputs'].items()},
			'cube_shape': list(shape),
			'class_ids': [item['class_id'] for item in classes],
			'class_names': [item['class_name'] for item in classes],
			'class_colors': [item['rgb'] for item in classes],
		},
		'coordinates': coordinates,
		'selection': {
			**config['selection'],
			'quantiles': np.linspace(0.05, 0.95, 10).tolist(),
			'active_training_inline_lines': active['inline'],
			'active_training_crossline_lines': active['crossline'],
			'adopted_threshold_by_domain': thresholds,
			**{
				f'selected_{domain}_slices': [
					row for row in rows if row['domain'] == domain
				]
				for domain in DOMAINS
			},
		},
		'figure': {
			**config['figure'],
			'panel_count': 8,
			'aspect_policy': 'auto, shared within figure',
		},
		'outputs': {
			**config['outputs'],
			'expected_inline_pngs': 10,
			'expected_crossline_pngs': 10,
			'expected_time_pngs': 10,
			'expected_total_pngs': 30,
		},
	}


def write_selected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	"""Persist exactly the selected sections and their eligibility evidence."""
	with path.open('w', newline='', encoding='utf-8') as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		for row in rows:
			writer.writerow(
				{
					key: json.dumps(value) if isinstance(value, (list, bool)) else value
					for key, value in row.items()
				}
			)


def verify_outputs(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
	"""Check exact file counts, PNG readability, distinct slices, and compute hashes."""
	from PIL import Image  # noqa: PLC0415

	expected = {row['output_png'] for row in rows}
	actual = {str(path.relative_to(root)) for path in root.rglob('*.png')}
	counts = {domain: len(list((root / domain).glob('*.png'))) for domain in DOMAINS}
	if (
		actual != expected
		or len(actual) != 30
		or any(count != 10 for count in counts.values())
	):
		raise ValueError(
			f'PNG file set mismatch: counts={counts}, unexpected={actual - expected}'
		)
	with (root / 'selected_slices.csv').open(encoding='utf-8') as stream:
		csv_rows = list(csv.DictReader(stream))
	if (
		len(csv_rows) != 30
		or len({(row['domain'], row['array_index']) for row in csv_rows}) != 30
	):
		raise ValueError('selected_slices.csv must contain 30 distinct sections')
	files = []
	for relative in sorted(actual):
		with Image.open(root / relative) as png:
			if min(png.size) <= 0:
				raise ValueError(f'empty PNG: {relative}')
			png.verify()
		files.append({'relative_path': relative, 'sha256': sha256(root / relative)})
	return {
		'actual_counts': {**counts, 'total': len(actual)},
		'pngs': files,
		'selected_slices_csv_sha256': sha256(root / 'selected_slices.csv'),
	}


def check_output_available(config: dict[str, Any]) -> None:
	"""Refuse existing publication output; no overwrite or deletion is implicit."""
	root = Path(config['outputs']['root'])
	if root.exists() and (
		any(root.rglob('*.png'))
		or (root / 'manifest.json').exists()
		or (root / 'selected_slices.csv').exists()
	):
		raise FileExistsError(f'refusing to overwrite existing paper output: {root}')


def write_outputs(
	prepared: PreparedComparison, *, overwrite: bool = False
) -> dict[str, Any]:
	"""Render one figure at a time, then publish CSV and verified manifest."""
	import matplotlib.pyplot as plt  # noqa: PLC0415

	root = Path(prepared.config['outputs']['root'])
	if overwrite:
		expected_root = Path(prepared.config['paths']['artifact_root']) / OUTPUT_SUFFIX
		if root.resolve() != expected_root.resolve():
			raise ValueError('overwrite requires the dedicated paper output root')
		expected = {row['output_png'] for row in prepared.rows}
		if any(
			str(path.relative_to(root)) not in expected for path in root.rglob('*.png')
		):
			raise ValueError('refusing overwrite with unexpected PNGs')
	else:
		check_output_available(prepared.config)
	for domain in DOMAINS:
		(root / domain).mkdir(parents=True, exist_ok=True)
	for row in prepared.rows:
		fig = render_figure(prepared, row)
		try:
			fig.savefig(
				root / row['output_png'],
				dpi=prepared.config['figure']['dpi'],
				facecolor='white',
				bbox_inches=None,
			)
		finally:
			plt.close(fig)
	write_selected_csv(root / 'selected_slices.csv', prepared.rows)
	prepared.manifest['outputs'].update(verify_outputs(root, prepared.rows))
	(root / 'manifest.json').write_text(
		json.dumps(prepared.manifest, indent=2, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	return prepared.manifest


def run(
	config_path: str | Path, *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
	"""Validate and preview or render, without training, inference, or evaluation."""
	config = load_config(config_path)
	if not dry_run and not overwrite:
		check_output_available(config)
	prepared = prepare_comparison(config)
	return (
		prepared.manifest if dry_run else write_outputs(prepared, overwrite=overwrite)
	)
