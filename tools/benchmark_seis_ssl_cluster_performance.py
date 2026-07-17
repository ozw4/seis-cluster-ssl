"""Benchmark deterministic synthetic current and compatible baseline reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.clustering.reconstruct import reconstruct_voxel_labels
from seis_ssl_cluster.clustering.residualization import (
	fit_local_token_position_residualizer,
	token_phase_keys_for_grid,
)
from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	_squared_euclidean_emission_costs_with_center_norms,
)
from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeAgcConfig,
	AmplitudePreprocessSettings,
	CropRequest,
	NpyMemmapVolumeStore,
	SurveyNormalizationStats,
	ZeroMaskConfig,
	read_amplitude_crop,
)
from seis_ssl_cluster.embedding import EmbeddingMerger, SlidingWindow
from seis_ssl_cluster.masking import generate_spatial_block_mask
from seis_ssl_cluster.models.mae import (
	build_3d_sincos_position_embedding,
	select_visible_tokens,
)

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence

TOKEN_GRID_SHAPE = (16, 16, 16)
PATCH_SIZE_XYZ = (8, 8, 8)
VOXEL_SHAPE = (128, 128, 128)
EMBEDDING_DIM = 128


@dataclass(frozen=True)
class BenchmarkCase:
	"""One prepared synthetic benchmark case."""

	name: str
	version: int
	shape: dict[str, object]
	input_fingerprint: str
	run: Callable[[], object]


def build_parser() -> argparse.ArgumentParser:
	"""Build the synthetic benchmark CLI parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--warm-up', '--warmup', dest='warm_up', type=int, default=2)
	parser.add_argument('--repeat', type=int, default=10)
	parser.add_argument('--output-json', type=Path)
	parser.add_argument('--output-markdown', type=Path)
	parser.add_argument('--baseline-json', type=Path)
	parser.add_argument(
		'--smoke',
		action='store_true',
		help='run one measurement per case without warm-up',
	)
	return parser


def run_benchmarks(
	*,
	seed: int,
	warm_up: int,
	repeat: int,
	baseline: dict[str, object] | None = None,
) -> dict[str, object]:
	"""Run every synthetic case and return a JSON-compatible report."""
	_validate_counts(warm_up=warm_up, repeat=repeat)
	with (
		tempfile.TemporaryDirectory(
			prefix='seis_ssl_cluster_benchmark_',
		) as temp_dir,
		ExitStack() as resources,
	):
		cases = _build_cases(seed, Path(temp_dir), resources)
		case_results = [
			_benchmark_case(case, warm_up=warm_up, repeat=repeat) for case in cases
		]
	report: dict[str, object] = {
		'schema_version': 2,
		'seed': seed,
		'warm_up': warm_up,
		'repeat': repeat,
		'environment': _environment(),
		'cases': case_results,
	}
	commit = _git_commit()
	if commit is not None:
		report['git_commit'] = commit
	if baseline is not None:
		report['baseline_comparison'] = compare_reports(report, baseline)
	return report


def main(argv: Sequence[str] | None = None) -> None:
	"""Run the benchmark and write its JSON report."""
	args = build_parser().parse_args(argv)
	warm_up = 0 if args.smoke else args.warm_up
	repeat = 1 if args.smoke else args.repeat
	baseline = _read_json_object(args.baseline_json) if args.baseline_json else None
	report = run_benchmarks(
		seed=args.seed,
		warm_up=warm_up,
		repeat=repeat,
		baseline=baseline,
	)
	if args.output_json is not None:
		_write_text(
			args.output_json,
			json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
		)
		print(f'benchmark JSON: {args.output_json}')
	markdown = render_markdown(report)
	if args.output_markdown is not None:
		_write_text(args.output_markdown, markdown)
		print(f'benchmark Markdown: {args.output_markdown}')
	if args.output_json is None and args.output_markdown is None:
		print(markdown, end='')


def _build_cases(
	seed: int,
	temp_dir: Path,
	resources: ExitStack,
) -> tuple[BenchmarkCase, ...]:
	seed_sequences = np.random.SeedSequence(seed).spawn(7)
	case_seeds = tuple(int(item.generate_state(1)[0]) for item in seed_sequences)
	volume_path = temp_dir / 'synthetic_amplitude.npy'
	volume_rng = np.random.default_rng(case_seeds[0])
	volume = volume_rng.normal(size=(160, 160, 160)).astype(np.float32)
	np.save(volume_path, volume)
	volume_descriptor = _array_descriptor(volume)
	del volume
	return (
		_memmap_case(
			volume_path,
			resources.enter_context(NpyMemmapVolumeStore()),
			volume_descriptor,
		),
		_spatial_mask_case(case_seeds[1]),
		_amplitude_preprocessing_case(
			volume_path,
			resources.enter_context(NpyMemmapVolumeStore()),
			volume_descriptor,
		),
		_position_visible_case(case_seeds[3]),
		_embedding_merge_case(case_seeds[4]),
		_residualization_case(case_seeds[5]),
		_hmm_emission_case(case_seeds[6]),
	)


def _memmap_case(
	volume_path: Path,
	store: NpyMemmapVolumeStore,
	volume_descriptor: dict[str, object],
) -> BenchmarkCase:
	starts = ((0, 0, 0), (8, 8, 8), (16, 16, 16), (24, 24, 24))

	def run() -> float:
		checksum = 0.0
		for start_xyz in starts:
			crop = store.read_crop(volume_path, start_xyz, VOXEL_SHAPE)
			checksum += float(np.asarray(crop).sum(dtype=np.float64))
		return checksum

	return _case(
		name='memmap_repeated_open_crop',
		shape={
			'volume_xyz': [160, 160, 160],
			'crop_xyz': list(VOXEL_SHAPE),
			'open_crop_count': len(starts),
		},
		inputs={
			'volume': volume_descriptor,
			'starts_xyz': starts,
			'crop_size_xyz': VOXEL_SHAPE,
		},
		run=run,
	)


def _spatial_mask_case(seed: int) -> BenchmarkCase:
	mask_ratio = 0.75
	block_size_tokens_xyz = (1, 1, 1)
	generated_mask = generate_spatial_block_mask(
		TOKEN_GRID_SHAPE,
		mask_ratio,
		block_size_tokens_xyz,
		np.random.default_rng(seed),
	)

	def run() -> np.ndarray:
		return generate_spatial_block_mask(
			TOKEN_GRID_SHAPE,
			mask_ratio,
			block_size_tokens_xyz,
			np.random.default_rng(seed),
		)

	return _case(
		name='spatial_mask_16_cubed_m075_block1',
		shape={
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'mask_ratio': mask_ratio,
			'block_size_tokens_xyz': list(block_size_tokens_xyz),
		},
		inputs={
			'seed': seed,
			'token_grid_shape_xyz': TOKEN_GRID_SHAPE,
			'mask_ratio': mask_ratio,
			'block_size_tokens_xyz': block_size_tokens_xyz,
			'generated_mask': _array_descriptor(generated_mask),
		},
		run=run,
	)


def _amplitude_preprocessing_case(
	volume_path: Path,
	store: NpyMemmapVolumeStore,
	volume_descriptor: dict[str, object],
) -> BenchmarkCase:
	settings = AmplitudePreprocessSettings(
		zero_mask=ZeroMaskConfig(enabled=False),
		normalized_clip_abs=8.0,
		amplitude_agc=AmplitudeAgcConfig(
			enabled=True,
			mode='trace_rms_z',
			window_z=65,
			eps=1.0e-6,
			clip_abs=8.0,
		),
		min_token_valid_fraction=1.0,
	)
	stats = SurveyNormalizationStats(
		survey_id='synthetic',
		source_path=volume_path,
		grid_order=GRID_ORDER_XYZ,
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=-3.0,
		clip_high=3.0,
		median=0.0,
		iqr=1.35,
	)
	request = CropRequest(
		survey_id='synthetic',
		start_xyz=(16, 16, 16),
		size_xyz=VOXEL_SHAPE,
	)
	stats_settings = asdict(stats)
	stats_settings.pop('source_path')

	def run() -> object:
		return read_amplitude_crop(
			request=request,
			amplitude_path=volume_path,
			stats=stats,
			store=store,
			patch_size_xyz=PATCH_SIZE_XYZ,
			settings=settings,
		)

	return _case(
		name='amplitude_preprocessing',
		shape={
			'crop_xyz': list(VOXEL_SHAPE),
			'patch_size_xyz': list(PATCH_SIZE_XYZ),
			'agc_window_z': 65,
		},
		inputs={
			'volume': volume_descriptor,
			'request': asdict(request),
			'normalization_stats': stats_settings,
			'preprocess_settings': asdict(settings),
			'patch_size_xyz': PATCH_SIZE_XYZ,
		},
		run=run,
	)


def _position_visible_case(seed: int) -> BenchmarkCase:
	rng = np.random.default_rng(seed)
	tokens = torch.from_numpy(
		rng.normal(size=(1, int(np.prod(TOKEN_GRID_SHAPE)), EMBEDDING_DIM)).astype(
			np.float32,
		),
	)
	visible_mask = torch.from_numpy(
		~generate_spatial_block_mask(
			TOKEN_GRID_SHAPE,
			0.75,
			(1, 1, 1),
			np.random.default_rng(seed),
		),
	).unsqueeze(0)

	def run() -> object:
		positions = build_3d_sincos_position_embedding(
			TOKEN_GRID_SHAPE,
			EMBEDDING_DIM,
		)
		return select_visible_tokens(tokens, positions, visible_mask)

	return _case(
		name='position_embedding_visible_selection',
		shape={
			'batch': 1,
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'embedding_dim': EMBEDDING_DIM,
			'visible_tokens': int(visible_mask.sum().item()),
		},
		inputs={
			'tokens': _array_descriptor(tokens),
			'visible_mask': _array_descriptor(visible_mask),
			'token_grid_shape_xyz': TOKEN_GRID_SHAPE,
			'embedding_dim': EMBEDDING_DIM,
		},
		run=run,
	)


def _embedding_merge_case(seed: int) -> BenchmarkCase:
	rng = np.random.default_rng(seed)
	embeddings = rng.normal(
		size=(*TOKEN_GRID_SHAPE, EMBEDDING_DIM),
	).astype(np.float32)
	valid = np.ones(TOKEN_GRID_SHAPE, dtype=np.bool_)
	labels = np.argmax(embeddings[..., :8], axis=-1).astype(np.int32)
	window = SlidingWindow(start_xyz=(0, 0, 0), size_xyz=VOXEL_SHAPE)

	def run() -> object:
		merger = EmbeddingMerger(
			token_grid_shape_xyz=TOKEN_GRID_SHAPE,
			embedding_dim=EMBEDDING_DIM,
		)
		for _ in range(2):
			merger.add_window(
				window,
				patch_size_xyz=PATCH_SIZE_XYZ,
				token_embeddings=embeddings,
				token_valid_mask=valid,
			)
		merged = merger.finalize(output_dtype=np.float32)
		voxels = reconstruct_voxel_labels(
			labels,
			patch_size_xyz=PATCH_SIZE_XYZ,
			volume_shape_xyz=VOXEL_SHAPE,
		)
		return merged, voxels

	return _case(
		name='embedding_merge_token_to_voxel',
		shape={
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'embedding_dim': EMBEDDING_DIM,
			'merge_window_count': 2,
			'voxel_shape_xyz': list(VOXEL_SHAPE),
		},
		inputs={
			'embeddings': _array_descriptor(embeddings),
			'valid_mask': _array_descriptor(valid),
			'labels': _array_descriptor(labels),
			'window': asdict(window),
			'patch_size_xyz': PATCH_SIZE_XYZ,
			'token_grid_shape_xyz': TOKEN_GRID_SHAPE,
			'embedding_dim': EMBEDDING_DIM,
			'merge_window_count': 2,
			'merge_output_dtype': 'float32',
			'reconstruction_volume_shape_xyz': VOXEL_SHAPE,
		},
		run=run,
	)


def _residualization_case(seed: int) -> BenchmarkCase:
	rng = np.random.default_rng(seed)
	embeddings = rng.normal(
		size=(int(np.prod(TOKEN_GRID_SHAPE)), EMBEDDING_DIM),
	).astype(np.float32)
	group_keys = token_phase_keys_for_grid(
		TOKEN_GRID_SHAPE,
		patch_size_xyz=PATCH_SIZE_XYZ,
		window_size_xyz=VOXEL_SHAPE,
		overlap_xyz=(64, 64, 64),
	)

	def run() -> np.ndarray:
		residualizer = fit_local_token_position_residualizer(
			embeddings,
			group_keys,
			group_by='token_phase',
			add_global_mean_back=True,
			min_group_count=1,
		)
		return residualizer.transform(embeddings, group_keys)

	return _case(
		name='token_phase_residualization',
		shape={
			'tokens': int(embeddings.shape[0]),
			'embedding_dim': EMBEDDING_DIM,
			'token_phase_groups': int(np.unique(group_keys, axis=0).shape[0]),
		},
		inputs={
			'embeddings': _array_descriptor(embeddings),
			'group_keys': _array_descriptor(group_keys),
			'group_by': 'token_phase',
			'add_global_mean_back': True,
			'min_group_count': 1,
		},
		run=run,
	)


def _hmm_emission_case(seed: int) -> BenchmarkCase:
	rng = np.random.default_rng(seed)
	features = rng.normal(size=(4096, EMBEDDING_DIM)).astype(np.float32)
	centers = rng.normal(size=(12, EMBEDDING_DIM)).astype(np.float32)
	center_squared_norms = np.einsum(
		'kd,kd->k',
		centers,
		centers,
		optimize=True,
	)

	def run() -> np.ndarray:
		return _squared_euclidean_emission_costs_with_center_norms(
			features,
			centers,
			center_squared_norms,
		)

	return _case(
		name='hmm_squared_euclidean_emission',
		version=2,
		shape={
			'tokens': int(features.shape[0]),
			'states': int(centers.shape[0]),
			'feature_dim': EMBEDDING_DIM,
			'dtype': str(features.dtype),
		},
		inputs={
			'features': _array_descriptor(features),
			'centers': _array_descriptor(centers),
			'center_squared_norms': _array_descriptor(center_squared_norms),
		},
		run=run,
	)


def _case(
	*,
	name: str,
	shape: dict[str, object],
	inputs: dict[str, object],
	run: Callable[[], object],
	version: int = 1,
) -> BenchmarkCase:
	return BenchmarkCase(
		name=name,
		version=version,
		shape=shape,
		input_fingerprint=_fingerprint({'name': name, 'inputs': inputs}),
		run=run,
	)


def _benchmark_case(
	case: BenchmarkCase,
	*,
	warm_up: int,
	repeat: int,
) -> dict[str, object]:
	for _ in range(warm_up):
		case.run()
	durations = np.empty(repeat, dtype=np.float64)
	for index in range(repeat):
		started_at = time.perf_counter()
		case.run()
		durations[index] = time.perf_counter() - started_at
	return {
		'name': case.name,
		'version': case.version,
		'shape': case.shape,
		'input_fingerprint': case.input_fingerprint,
		'median_seconds': float(np.median(durations)),
		'p25_seconds': float(np.percentile(durations, 25)),
		'p75_seconds': float(np.percentile(durations, 75)),
	}


def compare_reports(
	current: dict[str, object],
	baseline: dict[str, object],
) -> dict[str, object]:
	"""Compare medians only for cases with identical versioned inputs."""
	baseline_cases = _case_mappings(baseline)
	baseline_by_name = {
		str(case.get('name')): case
		for case in baseline_cases
		if isinstance(case.get('name'), str)
	}
	comparisons = []
	for current_case in _case_mappings(current):
		name = str(current_case.get('name'))
		baseline_case = baseline_by_name.get(name)
		reason = _incompatibility_reason(current_case, baseline_case)
		comparison: dict[str, object] = {
			'name': name,
			'comparable': reason is None,
			'current_median_seconds': current_case.get('median_seconds'),
			'baseline_median_seconds': (
				None if baseline_case is None else baseline_case.get('median_seconds')
			),
		}
		if reason is None and baseline_case is not None:
			current_median = _nonnegative_float(
				current_case.get('median_seconds'),
				label=f'{name} current median',
			)
			baseline_median = _nonnegative_float(
				baseline_case.get('median_seconds'),
				label=f'{name} baseline median',
			)
			if current_median == 0.0:
				comparison['note'] = 'current median is zero; multiplier is undefined'
			else:
				comparison['speedup_multiplier'] = baseline_median / current_median
		else:
			comparison['note'] = reason
		comparisons.append(comparison)
	return {
		'baseline_schema_version': baseline.get('schema_version'),
		'baseline_git_commit': baseline.get('git_commit'),
		'cases': comparisons,
	}


def render_markdown(report: dict[str, object]) -> str:
	"""Render benchmark inputs, medians, comparability, and cautions."""
	lines = [
		'# seis-ssl-cluster performance benchmark',
		'',
		'## Input conditions',
		'',
		f"- Schema version: {report.get('schema_version')}",
		f"- Seed: {report.get('seed')}",
		f"- Warm-up iterations: {report.get('warm_up')}",
		f"- Measured iterations: {report.get('repeat')}",
		f"- Environment: `{_compact_json(report.get('environment'))}`",
		'',
		'| Case | Shape and settings |',
		'|---|---|',
		*(
			f"| {case.get('name')} | `{_compact_json(case.get('shape'))}` |"
			for case in _case_mappings(report)
		),
		'',
		'## Results',
		'',
		'| Case | Version | Input fingerprint | Median (s) | P25-P75 (s) '
		'| Comparable | Baseline median (s) | Speedup | Note |',
		'|---|---:|---|---:|---:|---|---:|---:|---|',
	]
	comparisons = _comparison_by_name(report)
	for case in _case_mappings(report):
		name = str(case.get('name'))
		comparison = comparisons.get(name)
		comparable = 'not requested'
		baseline_median = '—'
		speedup = '—'
		note = 'No baseline report supplied.'
		if comparison is not None:
			comparable = 'yes' if comparison.get('comparable') is True else 'no'
			baseline_median = _format_seconds(
				comparison.get('baseline_median_seconds'),
			)
			multiplier = comparison.get('speedup_multiplier')
			if isinstance(multiplier, int | float) and not isinstance(multiplier, bool):
				speedup = f'{float(multiplier):.3f}x'
			note = str(comparison.get('note') or '')
		lines.append(
			f"| {name} | {case.get('version')} | "
			f"`{case.get('input_fingerprint')}` | "
			f"{_format_seconds(case.get('median_seconds'))} | "
			f"{_format_seconds(case.get('p25_seconds'))}-"
			f"{_format_seconds(case.get('p75_seconds'))} | {comparable} | "
			f'{baseline_median} | {speedup} | {note} |',
		)
	lines.extend(
		[
			'',
			'## Cautions',
			'',
			'- Speedup is baseline median divided by current median.',
			'- A multiplier is omitted when the case name, version, or input '
			'fingerprint differs, or when the current median is zero.',
			'- Compare runs from the same machine and software environment; '
			'interquartile overlap can indicate timing noise.',
			'- AMP and lower-precision paths may be numerically close rather than '
			'bitwise identical.',
			'',
		],
	)
	return '\n'.join(lines)


def _incompatibility_reason(
	current: dict[str, object],
	baseline: dict[str, object] | None,
) -> str | None:
	if baseline is None:
		return 'case is absent from baseline report'
	for key in ('name', 'version', 'input_fingerprint'):
		if current.get(key) != baseline.get(key):
			return f'{key} mismatch'
	return None


def _case_mappings(report: dict[str, object]) -> list[dict[str, object]]:
	cases = report.get('cases')
	if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
		raise ValueError('benchmark report cases must be a list of mappings')
	return cases


def _comparison_by_name(
	report: dict[str, object],
) -> dict[str, dict[str, object]]:
	comparison = report.get('baseline_comparison')
	if not isinstance(comparison, dict):
		return {}
	return {
		str(case.get('name')): case
		for case in _case_mappings(comparison)
		if isinstance(case.get('name'), str)
	}


def _fingerprint(payload: object) -> str:
	encoded = json.dumps(
		payload,
		sort_keys=True,
		separators=(',', ':'),
		allow_nan=False,
	).encode()
	return hashlib.sha256(encoded).hexdigest()[:16]


def _array_descriptor(value: np.ndarray | torch.Tensor) -> dict[str, object]:
	"""Describe exact array contents for a stable benchmark input fingerprint."""
	array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
	contiguous = np.ascontiguousarray(array)
	return {
		'shape': list(contiguous.shape),
		'dtype': contiguous.dtype.str,
		'sha256': hashlib.sha256(memoryview(contiguous).cast('B')).hexdigest(),
	}


def _read_json_object(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		raise TypeError(f'benchmark report must be a JSON object: {path}')
	return payload


def _write_text(path: Path, value: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(value, encoding='utf-8')


def _compact_json(value: object) -> str:
	return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _format_seconds(value: object) -> str:
	if isinstance(value, int | float) and not isinstance(value, bool):
		return f'{float(value):.6f}'
	return '—'


def _nonnegative_float(value: object, *, label: str) -> float:
	if (
		not isinstance(value, int | float)
		or isinstance(value, bool)
		or not np.isfinite(value)
		or value < 0
	):
		raise ValueError(f'{label} must be a finite nonnegative number')
	return float(value)


def _environment() -> dict[str, object]:
	return {
		'python_version': platform.python_version(),
		'platform': platform.platform(),
		'machine': platform.machine(),
		'processor': platform.processor(),
		'cpu_count': os.cpu_count(),
		'numpy_version': np.__version__,
		'torch_version': torch.__version__,
		'torch_num_threads': torch.get_num_threads(),
		'device': 'cpu',
	}


def _git_commit() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	result = subprocess.run(  # noqa: S603
		[git, 'rev-parse', 'HEAD'],
		cwd=Path(__file__).resolve().parents[1],
		capture_output=True,
		text=True,
		check=False,
	)
	return result.stdout.strip() if result.returncode == 0 else None


def _validate_counts(*, warm_up: int, repeat: int) -> None:
	if warm_up < 0:
		raise ValueError(f'warm_up must be nonnegative; got {warm_up!r}')
	if repeat <= 0:
		raise ValueError(f'repeat must be positive; got {repeat!r}')


if __name__ == '__main__':
	main()
