"""Fixed-layout, real-coordinate Volve horizon comparison figures."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
from PIL import Image

from seis_ssl_cluster.f3.lithology.hmm_v1_comparison_visualization import (
	amplitude_scale,
	apply_font_family,
	coordinate_edges,
	coordinate_step,
	read_json,
	resolve_font_files,
	sha256,
	write_selected_csv,
)
from seis_ssl_cluster.volve.horizon_data import load_volve_horizon_data
from seis_ssl_cluster.volve.paper_prediction import (
	export_prediction,
	inspect_model,
	load_config,
	model_valid_mask,
	require_hash,
	validate_prediction,
)

if TYPE_CHECKING:
	from matplotlib.axes import Axes
	from matplotlib.figure import Figure


def longest_valid_interval(valid: np.ndarray) -> tuple[int, int]:
	"""Return the longest contiguous valid span, ties resolved to lower index."""
	edges = np.diff(np.r_[False, valid, False].astype(np.int8))
	starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
	if not len(starts):
		raise ValueError('no common valid display interval')
	best = int(np.argmax(stops - starts))
	if stops[best] - starts[best] < 2:
		raise ValueError('common valid interval too short')
	return int(starts[best]), int(stops[best])


def horizon_section(volume: np.ndarray, domain: str, index: int) -> np.ndarray:
	"""Extract [horizon, lateral coordinate] without reversing either axis."""
	if domain == 'inline':
		return np.asarray(volume[:, index, :])
	if domain == 'crossline':
		return np.asarray(volume[:, :, index])
	raise ValueError(f'unknown domain: {domain}')


def choose_sections(
	rows: list[dict[str, Any]], coords: np.ndarray
) -> tuple[list[dict[str, Any]], float]:
	"""Select ten nearest unused coordinate quantiles using validity only."""
	for threshold in (0.95, 0.90, 0.80):
		eligible = [
			r
			for r in rows
			if r['common_valid_fraction'] >= threshold
			and r['ground_truth_horizon_count'] >= 2
		]
		if len(eligible) >= 10:
			break
	else:
		fractions = [r['common_valid_fraction'] for r in rows]
		stats = (
			[float(f(fractions)) for f in (np.min, np.median, np.max)]
			if fractions
			else []
		)
		raise ValueError(
			f'fewer than ten candidates: {len(eligible)}; '
			f'valid fraction min/median/max={stats}'
		)
	selected = []
	for order, q in enumerate(np.linspace(0.05, 0.95, 10), 1):
		target = float(np.quantile(coords, q))
		chosen = min(
			eligible,
			key=lambda r: (
				round(abs(r['actual_coordinate'] - target), 10),
				r['array_index'],
			),
		)
		eligible.remove(chosen)
		selected.append(
			{
				**chosen,
				'order': order,
				'selection_quantile': float(q),
				'selection_threshold': threshold,
			}
		)
	return selected, threshold


def select_domain(prepared: dict[str, Any], domain: str) -> list[dict[str, Any]]:
	"""Exclude active training sections and select using GT and shared support."""
	data, common = prepared['data'], prepared['common_valid']
	axis = 0 if domain == 'inline' else 1
	coords = prepared['coords'][domain]
	active = prepared['manifest']['active_training_lines'][domain]
	rows = []
	for i, coordinate in enumerate(coords):
		if not 0.05 <= i / (len(coords) - 1) <= 0.95 or coordinate in active:
			continue
		gt = horizon_section(data.bound_valid_mask, domain, i)
		samples = horizon_section(data.sample_float, domain, i)
		gt = gt & np.isfinite(samples) & (samples >= 552) & (samples < 768)
		volume = prepared['seismic']
		amplitude = volume[i, :, 552:768] if axis == 0 else volume[:, i, 552:768]
		finite = np.isfinite(amplitude).all(axis=-1)
		gt = gt & finite[None]
		valid = common[i, :] if axis == 0 else common[:, i]
		count = int(gt.sum())
		common_count = int((gt & valid[None]).sum())
		rows.append(
			{
				'domain': domain,
				'array_index': i,
				'actual_coordinate': int(coordinate),
				'coordinate_unit': 'line_number',
				'active_training_line': False,
				'excluded_training_line_count': len(active),
				'ground_truth_valid_count': count,
				'common_valid_count': common_count,
				'ground_truth_horizon_count': int(gt.any(axis=1).sum()),
				'common_valid_fraction': common_count / count if count else 0.0,
				'output_png': (
					f'{domain}/volve_hmm_v1_layout_000_large_{domain}_{coordinate}.png'
				),
			}
		)
	selected, threshold = choose_sections(rows, coords)
	prepared['manifest']['adopted_threshold_by_domain'][domain] = threshold
	visible = np.ones(common.shape[1 - axis], dtype=bool)
	for row in selected:
		i = row['array_index']
		visible &= common[i, :] if axis == 0 else common[:, i]
	start, stop = longest_valid_interval(visible)
	prepared['manifest']['display_crops'][domain] = [start, stop]
	return selected


def validate_shared_sources(
	prepared: dict[str, Any], canonical: dict[str, Any]
) -> None:
	"""Match all models to the same canonical inputs, labels, split and time window."""
	sources = prepared['models']
	base = sources[0]['identity']
	for source in sources:
		identity = source['identity']
		for key in ('canonical_scientific_identity', 'horizon_split_plan', 'tiles'):
			if identity[key] != base[key]:
				raise ValueError(f'cross-model mismatch: {key}')
		if (
			identity['horizon_split_plan']['input_identity']
			!= prepared['data'].input_identity()
		):
			raise ValueError('ground truth provenance differs from training')
		if (identity['tiles']['window_start'], identity['tiles']['window_stop']) != (
			552,
			768,
		):
			raise ValueError('time window mismatch')
	identity = base['canonical_scientific_identity']
	for key in ('inline_values_sha256', 'crossline_values_sha256', 'time_axis_sha256'):
		if identity[key] != canonical['scientific_identity'][key]:
			raise ValueError(f'canonical identity mismatch: {key}')
	for name, key in (
		('inline_values.npy', 'inline_values_sha256'),
		('crossline_values.npy', 'crossline_values_sha256'),
		('time_ms.npy', 'time_axis_sha256'),
		('valid_trace_mask.npy', 'valid_trace_mask_sha256'),
	):
		require_hash(canonical['provenance']['public_inputs'][name], identity[key])
	require_hash(
		canonical['provenance']['amplitude']['path'],
		identity['canonical_amplitude_sha256'],
	)
	prepared['manifest']['active_training_lines'] = base['horizon_split_plan'][
		'selected_physical_lines'
	]


def prepare(config: dict[str, Any]) -> dict[str, Any]:
	"""Preflight all scientific inputs and choose sections before inference/writes."""
	root = Path(config['paths']['artifact_root'])
	canonical_path = (
		root / 'data/volve/horizon_benchmark_v1/volve_canonical_input_metadata.json'
	)
	canonical = read_json(canonical_path)
	data = load_volve_horizon_data(config['paths']['volve_root'])
	seismic = np.load(
		canonical['provenance']['amplitude']['path'], mmap_mode='r', allow_pickle=False
	)
	if seismic.shape != (401, 720, 850):
		raise ValueError('canonical volume shape mismatch')
	coords = {
		'inline': data.inline_values,
		'crossline': data.crossline_values,
		'time': data.time_ms.astype(np.float64) / 1000,
	}
	for values in coords.values():
		if not np.all(np.diff(values) > 0) or coordinate_step(values) is None:
			raise ValueError('unexpected nonuniform canonical coordinates')
	models = []
	for model in config['models']:
		print(
			f'Validating {model["condition_id"]}: {model["model_id"]}',
			file=sys.stderr,
			flush=True,
		)
		models.append(inspect_model(config, model))
	manifest = {
		'schema_version': 1,
		'artifact_type': 'volve_hmm_v1_paper_sections',
		'dataset': 'volve_st10010',
		'dataset_version': 'horizon_benchmark_v1',
		'cube_shape': list(seismic.shape),
		'comparison': config['comparison'],
		'config_path': config['config_path'],
		'config_sha256': sha256(config['config_path']),
		'canonical_input_metadata_path': str(canonical_path),
		'canonical_input_metadata_sha256': sha256(canonical_path),
		'canonical_inputs': canonical['provenance'],
		'ground_truth_identity': data.input_identity(),
		'figure': config['figure'],
		'font_files': resolve_font_files(config['figure']['font_family']),
		'coordinate_ranges': {
			k: [float(v[0]), float(v[-1])] for k, v in coords.items()
		},
		'time_unit_source': (
			'validated canonical time_ms.npy; 4 ms sampling; divide by 1000'
		),
		'displayed_time_seconds': [
			float(coords['time'][552]),
			float(coords['time'][767]),
		],
		'horizon_names': list(data.horizon_names),
		'models': models,
		'selection_policy': config['selection'],
		'adopted_threshold_by_domain': {},
		'display_crops': {},
		'display_crop_policy': (
			'longest shared contiguous valid lateral span; tie by lower index'
		),
		'prediction_postprocessing': False,
		'metric_evaluation': False,
		'training': False,
		'amplitude': amplitude_scale(
			seismic,
			str(
				root / 'data/volve/horizon_benchmark_v1/volve.normalization_stats.json'
			),
			config['figure'],
		),
		'output_root': config['outputs']['figures'],
		'expected_png_count': 20,
	}
	prepared = {
		'config': config,
		'data': data,
		'seismic': seismic,
		'coords': coords,
		'models': models,
		'manifest': manifest,
	}
	validate_shared_sources(prepared, canonical)
	common = np.array(data.valid_trace_mask, copy=True)
	for model in models:
		common &= model_valid_mask(model)
	prepared['common_valid'] = common
	rows = [
		row
		for domain in ('inline', 'crossline')
		for row in select_domain(prepared, domain)
	]
	prepared['rows'] = rows
	manifest['selected_slices'] = rows
	return prepared


def load_predictions(prepared: dict[str, Any]) -> list[np.ndarray]:
	"""Reuse only complete exports bound to precisely the inspected sources."""
	root = Path(prepared['config']['outputs']['predictions'])
	arrays = []
	for source in prepared['models']:
		folder = root / f'model={source["model_id"]}'
		meta = read_json(folder / 'prediction_metadata.json')
		if meta['complete'] is not True or meta['source'] != source:
			raise ValueError(f'prediction source mismatch: {folder}')
		for filename, digest in meta['array_sha256'].items():
			require_hash(folder / filename, digest)
		prediction = np.load(
			folder / 'horizon_sample_predictions.npy', mmap_mode='r', allow_pickle=False
		)
		valid = np.load(
			folder / 'valid_lateral_mask.npy', mmap_mode='r', allow_pickle=False
		)
		validate_prediction(prediction, valid)
		if not np.array_equal(valid, model_valid_mask(source)):
			raise ValueError('prediction validity differs from bound token mask')
		arrays.append(prediction)
	prepared['manifest']['prediction_metadata'] = [
		{
			'path': str(root / f'model={s["model_id"]}' / 'prediction_metadata.json'),
			'sha256': sha256(
				root / f'model={s["model_id"]}' / 'prediction_metadata.json'
			),
		}
		for s in prepared['models']
	]
	return arrays


def style_axes(
	ax: Axes, panel: int, title: str, settings: dict[str, Any], xlabel: str
) -> None:
	"""Apply the shared paper typography and tick policy to one panel."""
	ax.set_title(title, fontsize=settings['title_size'], pad=6)
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
	left, right = ax.get_xlim()
	margin = (right - left) * 0.15
	ax.xaxis.set_major_locator(
		FixedLocator(np.round(np.linspace(left + margin, right - margin, 3)))
	)
	ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	ax.tick_params(which='both', length=0, labelsize=settings['tick_label_size'])
	for spine in ax.spines.values():
		spine.set_linewidth(settings['axis_spine_width'])
	ax.grid(visible=False)
	if panel >= 4:
		ax.set_xlabel(xlabel, fontsize=settings['axis_label_size'])
	if panel % 4 == 0:
		ax.set_ylabel('time (s)', fontsize=settings['axis_label_size'])


def render_figure(
	prepared: dict[str, Any], row: dict[str, Any], predictions: list[np.ndarray]
) -> Figure:
	"""Draw amplitude alone, native GT lines, and six unmodified horizon outputs."""
	settings = prepared['config']['figure']
	domain, index = row['domain'], row['array_index']
	left, right = prepared['manifest']['display_crops'][domain]
	lateral = 'crossline' if domain == 'inline' else 'inline'
	x = prepared['coords'][lateral][left:right]
	time = prepared['coords']['time'][552:768]
	fig, axes = plt.subplots(
		2,
		4,
		figsize=(settings['width_mm'] / 25.4, settings['height_mm'] / 25.4),
		sharex=True,
		sharey=True,
		facecolor='white',
	)
	try:
		titles = [
			'Seismic amplitude',
			'Ground truth',
			*[s['display_name'] for s in prepared['models']],
		]
		for panel, ax in enumerate(axes.flat):
			if panel == 0:
				volume = prepared['seismic']
				section = (
					volume[index, left:right, 552:768].T
					if domain == 'inline'
					else volume[left:right, index, 552:768].T
				)
				if not np.isfinite(section).all():
					raise ValueError('nonfinite amplitude inside display crop')  # noqa: TRY301
				xe, te = coordinate_edges(x), coordinate_edges(time)
				scale = prepared['manifest']['amplitude']
				ax.imshow(
					section,
					origin='lower',
					extent=(xe[0], xe[-1], te[0], te[-1]),
					aspect='auto',
					interpolation='nearest',
					cmap='seismic',
					vmin=scale['vmin'],
					vmax=scale['vmax'],
				)
			else:
				volume = (
					prepared['data'].sample_float
					if panel == 1
					else predictions[panel - 2]
				)
				samples = horizon_section(volume, domain, index)[:, left:right]
				if panel == 1:
					valid = horizon_section(
						prepared['data'].bound_valid_mask, domain, index
					)[:, left:right]
					samples = np.where(
						valid & (samples >= 552) & (samples < 768), samples, np.nan
					)
				elif not np.isfinite(samples).all():
					raise ValueError('invalid prediction inside display crop')  # noqa: TRY301
				times = (
					prepared['data'].time_ms[0]
					+ samples * np.diff(prepared['data'].time_ms)[0]
				) / 1000
				for values, color in zip(
					times, settings['horizon_colors'], strict=True
				):
					ax.plot(
						x, values, color=color, linewidth=settings['horizon_line_width']
					)
			ax.set_xlim(x[0], x[-1])
			ax.set_ylim(time[-1], time[0])
			style_axes(
				ax,
				panel,
				titles[panel],
				settings,
				'xline' if domain == 'inline' else 'inline',
			)
		handles = [
			Line2D([], [], color=c, linewidth=settings['horizon_line_width'], label=n)
			for n, c in zip(
				('Ty top', 'Shetland top', 'BCU', 'Hugin top', 'Hugin base'),
				settings['horizon_colors'],
				strict=True,
			)
		]
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


def verify_outputs(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
	"""Check the exact twenty readable PNGs, unique CSV rows, and output hashes."""
	counts = {d: len(list((root / d).glob('*.png'))) for d in ('inline', 'crossline')}
	if (
		counts != {'inline': 10, 'crossline': 10}
		or len(list(root.rglob('*.png'))) != 20
	):
		raise ValueError(f'PNG count mismatch: {counts}')
	with (root / 'selected_slices.csv').open(newline='') as stream:
		stored = list(csv.DictReader(stream))
	if (
		len(stored) != 20
		or len({(r['domain'], r['array_index']) for r in stored}) != 20
	):
		raise ValueError('selected CSV count/uniqueness mismatch')
	outputs = []
	for row in rows:
		path = root / row['output_png']
		with Image.open(path) as image:
			image.load()
			if min(image.size) <= 0:
				raise ValueError('empty PNG')
			outputs.append(
				{
					'path': row['output_png'],
					'sha256': sha256(path),
					'pixel_size': list(image.size),
				}
			)
	return {
		'counts': counts,
		'total_pngs': 20,
		'pngs': outputs,
		'selected_slices_csv_sha256': sha256(root / 'selected_slices.csv'),
	}


def check_output_root(config: dict[str, Any], *, overwrite: bool) -> None:
	"""Refuse existing figures unless explicitly replacing this exact owned file set."""
	root = Path(config['outputs']['figures'])
	if not root.exists() or not any(root.iterdir()):
		return
	if not overwrite:
		raise FileExistsError(root)
	previous = read_json(root / 'manifest.json')
	if previous['artifact_type'] != 'volve_hmm_v1_paper_sections':
		raise ValueError('cannot replace a different artifact type')
	expected = {r['output_png'] for r in previous['selected_slices']}
	actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
	if actual != expected | {'selected_slices.csv', 'manifest.json'}:
		raise ValueError('unexpected files in dedicated figure output root')


def run(
	path: str | Path, *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
	"""Preflight, export missing predictions, then draw twenty PNGs."""
	config = load_config(path)
	root = Path(config['outputs']['figures'])
	check_output_root(config, overwrite=overwrite)
	prepared = prepare(config)
	if overwrite and (root / 'manifest.json').exists():
		old_rows = read_json(root / 'manifest.json')['selected_slices']
		if old_rows != prepared['rows']:
			raise ValueError('overwrite requires the same selected sections')
	if dry_run:
		return prepared['manifest']
	torch.set_num_threads(4)
	for source in prepared['models']:
		folder = Path(config['outputs']['predictions']) / f'model={source["model_id"]}'
		if not folder.exists():
			print(f'Inference only: {source["model_id"]}', file=sys.stderr, flush=True)
			export_prediction(config, source)
	predictions = load_predictions(prepared)
	root.mkdir(parents=True, exist_ok=True)
	for row in prepared['rows']:
		output = root / row['output_png']
		output.parent.mkdir(exist_ok=True)
		fig = render_figure(prepared, row, predictions)
		try:
			fig.savefig(output, dpi=config['figure']['dpi'], facecolor='white')
		finally:
			plt.close(fig)
	write_selected_csv(root / 'selected_slices.csv', prepared['rows'])
	manifest = prepared['manifest']
	manifest['outputs'] = verify_outputs(root, prepared['rows'])
	manifest['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
	manifest['generation_git_commit'] = subprocess.check_output(  # noqa: S603
		[shutil.which('git') or '/usr/bin/git', 'rev-parse', 'HEAD'],
		cwd=config['paths']['workspace'],
		text=True,
	).strip()
	(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
	return manifest
