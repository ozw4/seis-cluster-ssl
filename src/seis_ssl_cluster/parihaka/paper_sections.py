"""Index-coordinate Parihaka comparison using the existing paper renderer."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from seis_ssl_cluster.f3.lithology import hmm_v1_comparison_visualization as paper
from seis_ssl_cluster.parihaka.paper_prediction import (
	CONDITIONS,
	MODEL_IDS,
	validate_prediction,
)

if TYPE_CHECKING:
	from matplotlib.figure import Figure


class BinaryLabels:
	"""Convert only requested sections, never an entire label volume."""

	def __init__(self, labels: np.ndarray) -> None:
		"""Keep the source memory map without copying it."""
		self.labels = labels
		self.shape = labels.shape

	def __getitem__(self, key: tuple[int | slice, ...]) -> np.ndarray:
		"""Map source facies 5 to Channel on the requested section only."""
		section = self.labels[key]
		if not np.isin(section, [1, 2, 3, 4, 5, 6]).all():
			raise ValueError('Unknown Parihaka ground truth class')
		return (section == 5).astype(np.int8)


def load_config(path: Path) -> dict[str, Any]:
	"""Read explicit experiment configuration and enforce the output boundary."""
	for name in ('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', 'SEIS_SSL_CLUSTER_WORKSPACE'):
		if not os.environ.get(name):
			raise ValueError(f'Missing environment variable: {name}')
	config = yaml.safe_load(os.path.expandvars(path.read_text()))
	for key in ('output_root', 'prediction_root', 'data_root'):
		config[key] = str(paper.absolute_path(config[key]))
	expected = (
		Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT']).resolve()
		/ 'paper_figures/hmm_v1/parihaka/layout_000/large'
	)
	if Path(config['output_root']) != expected:
		raise ValueError('Unexpected output root')
	config['config_path'] = str(path.resolve())
	config['models'] = [
		{'display_name': name}
		for name in (
			'MAE',
			'MAE + HMM',
			'LocalBT',
			'LocalBT + HMM',
			'Random',
			'Random + HMM',
		)
	]
	return config


def load_predictions(
	config: dict[str, Any], shape: tuple[int, ...]
) -> tuple[list[Any], list[dict[str, Any]]]:
	"""Require the six completed raw artifacts and verify their stored hashes."""
	arrays, metadata = [], []
	for condition, model in zip(CONDITIONS, MODEL_IDS, strict=True):
		root = Path(config['prediction_root']) / f'model={model}'
		meta = paper.read_json(root / 'prediction_metadata.json')
		if not meta['complete'] or (
			meta['condition_id'],
			meta['model_id'],
			meta['layout_id'],
			meta['data_size'],
			meta['dataset'],
		) != (condition, model, 'layout_000', 'large', 'parihaka'):
			raise ValueError('Prediction identity mismatch')
		if condition.endswith('b') and (meta['hmm_k'], meta['distillation_weight']) != (
			6,
			0.2,
		):
			raise ValueError('HMM K/weight mismatch')
		if any(meta[key] for key in ('smoothing', 'ensemble', 'postprocessing')):
			raise ValueError('Only raw predictions are allowed')
		for name, digest in meta['array_sha256'].items():
			if paper.sha256(root / name) != digest:
				raise ValueError('Prediction hash mismatch')
		validate_prediction(root, shape)
		arrays.append(
			SimpleNamespace(
				predictions=np.load(
					root / 'parihaka_voxel_predictions.npy',
					mmap_mode='r',
					allow_pickle=False,
				),
				valid_mask=np.load(
					root / 'parihaka_valid_voxel_mask.npy',
					mmap_mode='r',
					allow_pickle=False,
				),
			)
		)
		metadata.append(
			{
				**meta,
				'prediction_metadata_path': str(root / 'prediction_metadata.json'),
				'prediction_metadata_sha256': paper.sha256(
					root / 'prediction_metadata.json'
				),
			}
		)
	if any(m['train_lines'] != metadata[0]['train_lines'] for m in metadata):
		raise ValueError('Training layouts differ')
	return arrays, metadata


def prepare(config: dict[str, Any]) -> paper.PreparedComparison:
	"""Select shared sections without consulting prediction classes or metrics."""
	data = Path(config['data_root'])
	amplitude = np.load(
		data / 'parihaka_amplitude.npy', mmap_mode='r', allow_pickle=False
	)
	labels = BinaryLabels(
		np.load(data / 'parihaka_labels.npy', mmap_mode='r', allow_pickle=False)
	)
	if amplitude.shape != labels.shape:
		raise ValueError('Input volume shapes differ')
	arrays, metadata = load_predictions(config, amplitude.shape)
	coords = {
		domain: np.arange(1, n + 1)
		for domain, n in zip(paper.DOMAINS, amplitude.shape, strict=True)
	}
	active = {
		d: [i + 1 for i in metadata[0]['train_lines'][d]]
		for d in ('inline', 'crossline')
	}
	rows, thresholds = [], {}
	for domain in paper.DOMAINS:
		candidates = paper.slice_eligibility(
			amplitude,
			labels,
			[a.valid_mask for a in arrays],
			[0, 1],
			domain,
			coords[domain],
			active.get(domain, []),
		)
		selected, thresholds[domain] = paper.select_slices(candidates, coords[domain])
		for row in selected:
			row['coordinate_unit'] = 'one_based_index'
			axis = {'inline': 'X', 'crossline': 'Y', 'time': 'Z'}[domain]
			row['fixed_axis'] = axis
			row['output_png'] = (
				f'{domain}/parihaka_hmm_v1_layout_000_large_{axis}_{int(row["actual_coordinate"]):04d}.png'
			)
		rows.extend(selected)
	classes = [
		{'class_id': 0, 'class_name': 'Non-Channel', 'rgb': [219, 241, 247]},
		{'class_id': 1, 'class_name': 'Channel', 'rgb': [208, 10, 0]},
	]
	manifest = {
		'schema_version': 1,
		'artifact_type': 'parihaka_paper_sections',
		'generated_at_utc': datetime.now(timezone.utc).isoformat(),
		'coordinate_policy': 'array index + 1; not physical coordinates or seconds',
		'axis_mapping': {
			'inline': ['Y', 'Z'],
			'crossline': ['X', 'Z'],
			'time': ['X', 'Y'],
		},
		'cube_shape': list(amplitude.shape),
		'models': metadata,
		'classes': classes,
		'figure': config['figure'],
		'config_path': config['config_path'],
		'config_sha256': paper.sha256(config['config_path']),
		'active_training_lines_one_based': active,
		'adopted_threshold_by_domain': thresholds,
		'selected_slices': rows,
		'amplitude': paper.amplitude_scale(
			amplitude,
			str(data / 'parihaka_amplitude.normalization_stats.json'),
			config['figure'],
		),
		'font_files': paper.resolve_font_files('Arial'),
	}
	prepared = paper.PreparedComparison(
		config, amplitude, labels, arrays, classes, coords, manifest, rows
	)
	manifest['display_crops_by_domain'] = paper.display_crops(prepared)
	xy_crop = manifest['display_crops_by_domain']['time']
	for suffix in ('min', 'max'):
		xy_crop[f'x_{suffix}'], xy_crop[f'y_{suffix}'] = (
			xy_crop[f'y_{suffix}'],
			xy_crop[f'x_{suffix}'],
		)
	manifest['selection_policy'] = {
		'quantiles': np.linspace(0.05, 0.95, 10).tolist(),
		'common_valid_thresholds': [0.95, 0.90, 0.80],
		'minimum_ground_truth_classes': 2,
		'exclude_training_lines': True,
		'uses_prediction_accuracy': False,
	}
	return prepared


def render(prepared: paper.PreparedComparison, row: dict[str, Any]) -> Figure:
	"""Reuse F3 typography, adapting axis names and ticks to one-based indices."""
	fig = paper.render_figure(prepared, row)
	domain = row['domain']
	xlabel, ylabel = prepared.manifest['axis_mapping'][domain]
	original_x, original_y = fig.axes[0].get_xlim(), fig.axes[0].get_ylim()
	for index, ax in enumerate(fig.axes):
		if domain == 'time':
			artist = ax.images[0]
			x0, x1, y0, y1 = artist.get_extent()
			artist.set_data(artist.get_array().T)
			artist.set_extent((y0, y1, x0, x1))
			ax.set_xlim(original_y)
			ax.set_ylim(original_x)
		ax.set_xlabel(xlabel if index >= 4 else '', fontsize=9)
		ax.set_ylabel(ylabel if index % 4 == 0 else '', fontsize=9)
		for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
			ticks = np.unique(
				np.rint(np.linspace(min(limits), max(limits), 4)).astype(int)
			)
			axis.set_major_locator(FixedLocator(ticks))
			axis.set_major_formatter(FormatStrFormatter('%d'))
	paper.apply_font_family(fig, 'Arial')
	return fig


def run(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
	"""Validate before drawing; refuse existing figure outputs."""
	root = Path(config['output_root'])
	if root.exists() and any(root.iterdir()):
		raise FileExistsError(f'Output already exists: {root}')
	prepared = prepare(config)
	if dry_run:
		return prepared.manifest
	root.mkdir(parents=True, exist_ok=True)
	for domain in paper.DOMAINS:
		(root / domain).mkdir()
	for row in prepared.rows:
		fig = render(prepared, row)
		try:
			fig.savefig(
				root / row['output_png'],
				dpi=config['figure']['dpi'],
				facecolor='white',
				bbox_inches=None,
			)
		finally:
			plt.close(fig)
	paper.write_selected_csv(root / 'selected_slices.csv', prepared.rows)
	prepared.manifest['outputs'] = paper.verify_outputs(root, prepared.rows)
	(root / 'manifest.json').write_text(json.dumps(prepared.manifest, indent=2) + '\n')
	return prepared.manifest
