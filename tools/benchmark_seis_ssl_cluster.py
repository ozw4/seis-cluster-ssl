"""Run deterministic synthetic performance baselines for core pipeline stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.clustering.reconstruct import reconstruct_voxel_labels
from seis_ssl_cluster.clustering.residualization import (
	fit_local_token_position_residualizer,
	token_phase_keys_for_grid,
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
	shape: dict[str, object]
	run: Callable[[], object]


def build_parser() -> argparse.ArgumentParser:
	"""Build the synthetic benchmark CLI parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--warm-up', '--warmup', dest='warm_up', type=int, default=2)
	parser.add_argument('--repeat', type=int, default=10)
	parser.add_argument('--output-json', type=Path, required=True)
	return parser


def run_benchmarks(*, seed: int, warm_up: int, repeat: int) -> dict[str, object]:
	"""Run every synthetic case and return a JSON-compatible report."""
	_validate_counts(warm_up=warm_up, repeat=repeat)
	with tempfile.TemporaryDirectory(prefix='seis_ssl_cluster_benchmark_') as temp_dir:
		cases = _build_cases(seed, Path(temp_dir))
		case_results = [
			_benchmark_case(case, warm_up=warm_up, repeat=repeat) for case in cases
		]
	report: dict[str, object] = {
		'schema_version': 1,
		'seed': seed,
		'warm_up': warm_up,
		'repeat': repeat,
		'environment': _environment(),
		'cases': case_results,
	}
	commit = _git_commit()
	if commit is not None:
		report['git_commit'] = commit
	return report


def main(argv: Sequence[str] | None = None) -> None:
	"""Run the benchmark and write its JSON report."""
	args = build_parser().parse_args(argv)
	report = run_benchmarks(
		seed=args.seed,
		warm_up=args.warm_up,
		repeat=args.repeat,
	)
	output_path = args.output_json
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	print(f'benchmark JSON: {output_path}')


def _build_cases(seed: int, temp_dir: Path) -> tuple[BenchmarkCase, ...]:
	seed_sequences = np.random.SeedSequence(seed).spawn(6)
	case_seeds = tuple(int(item.generate_state(1)[0]) for item in seed_sequences)
	volume_path = temp_dir / 'synthetic_amplitude.npy'
	volume_rng = np.random.default_rng(case_seeds[0])
	volume = volume_rng.normal(size=(160, 160, 160)).astype(np.float32)
	np.save(volume_path, volume)
	del volume
	return (
		_memmap_case(volume_path),
		_spatial_mask_case(case_seeds[1]),
		_amplitude_preprocessing_case(volume_path),
		_position_visible_case(case_seeds[3]),
		_embedding_merge_case(case_seeds[4]),
		_residualization_case(case_seeds[5]),
	)


def _memmap_case(volume_path: Path) -> BenchmarkCase:
	store = NpyMemmapVolumeStore()
	starts = ((0, 0, 0), (8, 8, 8), (16, 16, 16), (24, 24, 24))

	def run() -> float:
		checksum = 0.0
		for start_xyz in starts:
			crop = store.read_crop(volume_path, start_xyz, VOXEL_SHAPE)
			checksum += float(np.asarray(crop).sum(dtype=np.float64))
		return checksum

	return BenchmarkCase(
		name='memmap_repeated_open_crop',
		shape={
			'volume_xyz': [160, 160, 160],
			'crop_xyz': list(VOXEL_SHAPE),
			'open_crop_count': len(starts),
		},
		run=run,
	)


def _spatial_mask_case(seed: int) -> BenchmarkCase:
	def run() -> np.ndarray:
		return generate_spatial_block_mask(
			TOKEN_GRID_SHAPE,
			0.75,
			(1, 1, 1),
			np.random.default_rng(seed),
		)

	return BenchmarkCase(
		name='spatial_mask_16_cubed_m075_block1',
		shape={
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'mask_ratio': 0.75,
			'block_size_tokens_xyz': [1, 1, 1],
		},
		run=run,
	)


def _amplitude_preprocessing_case(volume_path: Path) -> BenchmarkCase:
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
	store = NpyMemmapVolumeStore()

	def run() -> object:
		return read_amplitude_crop(
			request=request,
			amplitude_path=volume_path,
			stats=stats,
			store=store,
			patch_size_xyz=PATCH_SIZE_XYZ,
			settings=settings,
		)

	return BenchmarkCase(
		name='amplitude_preprocessing',
		shape={
			'crop_xyz': list(VOXEL_SHAPE),
			'patch_size_xyz': list(PATCH_SIZE_XYZ),
			'agc_window_z': 65,
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

	return BenchmarkCase(
		name='position_embedding_visible_selection',
		shape={
			'batch': 1,
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'embedding_dim': EMBEDDING_DIM,
			'visible_tokens': int(visible_mask.sum().item()),
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

	return BenchmarkCase(
		name='embedding_merge_token_to_voxel',
		shape={
			'token_grid_xyz': list(TOKEN_GRID_SHAPE),
			'embedding_dim': EMBEDDING_DIM,
			'merge_window_count': 2,
			'voxel_shape_xyz': list(VOXEL_SHAPE),
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

	return BenchmarkCase(
		name='token_phase_residualization',
		shape={
			'tokens': int(embeddings.shape[0]),
			'embedding_dim': EMBEDDING_DIM,
			'token_phase_groups': int(np.unique(group_keys, axis=0).shape[0]),
		},
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
		'shape': case.shape,
		'median_seconds': float(np.median(durations)),
		'p25_seconds': float(np.percentile(durations, 25)),
		'p75_seconds': float(np.percentile(durations, 75)),
	}


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
