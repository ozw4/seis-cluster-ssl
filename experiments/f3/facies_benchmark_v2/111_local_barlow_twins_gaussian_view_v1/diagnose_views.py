# ruff: noqa: INP001
"""Measure how strongly the F3 Local Barlow Twins views remain correlated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
)
from seis_ssl_cluster.data.amplitude_dataset import AmplitudePretrainDataset
from seis_ssl_cluster.data.barlow_twins_dataset import (
	LocalBarlowTwinsPretrainDataset,
)
from seis_ssl_cluster.data.schema import read_manifest_json
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig

if TYPE_CHECKING:
	from collections.abc import Sequence

	from numpy.typing import NDArray

	from seis_ssl_cluster.data.window_preprocessing import FiniteCheckMode

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
	REPOSITORY_ROOT
	/ 'experiments/f3/facies_benchmark_v1/22_local_barlow_twins_v1/'
	'02_full_100ep.yaml'
)
VIEW_NOISE_STDS = {
	'legacy': 0.0,
	'gaussian_noise_std005': 0.05,
	'gaussian_noise_std010': 0.10,
}


@dataclass
class _MetricAccumulator:
	"""Sufficient statistics for one aligned physical-voxel population."""

	voxel_count: int = 0
	sum_a: float = 0.0
	sum_b: float = 0.0
	sum_base: float = 0.0
	sum_squared_a: float = 0.0
	sum_squared_b: float = 0.0
	sum_squared_base: float = 0.0
	sum_product: float = 0.0
	sum_squared_pair_delta: float = 0.0
	sum_squared_base_delta: float = 0.0

	def add(
		self,
		view_a: NDArray[np.floating],
		view_b: NDArray[np.floating],
		base: NDArray[np.floating],
	) -> None:
		"""Accumulate one crop without retaining its voxel arrays."""
		a = np.asarray(view_a, dtype=np.float64).ravel()
		b = np.asarray(view_b, dtype=np.float64).ravel()
		canonical = np.asarray(base, dtype=np.float64).ravel()
		if a.size == 0 or a.shape != b.shape or a.shape != canonical.shape:
			raise ValueError('metric arrays must have one shared nonempty shape')
		pair_delta = a - b
		delta_a = a - canonical
		delta_b = b - canonical
		self.voxel_count += int(a.size)
		self.sum_a += float(np.sum(a))
		self.sum_b += float(np.sum(b))
		self.sum_base += float(np.sum(canonical))
		self.sum_squared_a += float(np.sum(np.square(a)))
		self.sum_squared_b += float(np.sum(np.square(b)))
		self.sum_squared_base += float(np.sum(np.square(canonical)))
		self.sum_product += float(np.sum(a * b))
		self.sum_squared_pair_delta += float(np.sum(np.square(pair_delta)))
		self.sum_squared_base_delta += float(
			(np.sum(np.square(delta_a)) + np.sum(np.square(delta_b))) / 2.0
		)

	def result(self) -> dict[str, float | int]:
		"""Return correlation and RMS measurements derived from the totals."""
		if self.voxel_count <= 0:
			raise ValueError('cannot summarize an empty metric population')
		count = float(self.voxel_count)
		covariance = self.sum_product - self.sum_a * self.sum_b / count
		variance_a = self.sum_squared_a - self.sum_a**2 / count
		variance_b = self.sum_squared_b - self.sum_b**2 / count
		variance_base = self.sum_squared_base - self.sum_base**2 / count
		denominator = math.sqrt(max(0.0, variance_a * variance_b))
		if denominator == 0.0:
			raise ValueError('paired correlation is undefined for constant views')
		return {
			'voxel_count': self.voxel_count,
			'paired_correlation': covariance / denominator,
			'paired_rms': math.sqrt(self.sum_squared_pair_delta / count),
			'per_view_rms_from_unaugmented': math.sqrt(
				self.sum_squared_base_delta / count
			),
			'unaugmented_amplitude_std': math.sqrt(
				max(0.0, variance_base / count)
			),
		}


def diagnose_views(
	config: Mapping[str, object],
	*,
	config_path: Path,
	epoch: int = 0,
	start_index: int = 0,
	count: int = 16,
) -> dict[str, object]:
	"""Run the CPU-only aligned-view diagnostic on a resolved control config."""
	_validate_sampling(epoch=epoch, start_index=start_index, count=count)
	base_dataset = _build_base_dataset(config)
	if start_index + count > len(base_dataset):
		raise ValueError(
			'selected sample range exceeds train.samples_per_epoch; '
			f'got stop={start_index + count} and length={len(base_dataset)}'
		)
	augmentations = _mapping(config, 'augmentations')
	barlow_twins = _mapping(config, 'barlow_twins')
	if barlow_twins.get('method') != 'local_barlow_twins_3d':
		raise ValueError('control config must use local_barlow_twins_3d')
	if 'policy' in augmentations:
		raise ValueError('control config must use the legacy horizontal-flip views')
	local_pairs_per_crop = _integer(barlow_twins, 'local_pairs_per_crop')
	flip_probability = _floating(
		augmentations,
		'horizontal_flip_probability',
	)
	datasets = {
		name: LocalBarlowTwinsPretrainDataset(
			base_dataset,
			local_pairs_per_crop=local_pairs_per_crop,
			horizontal_flip_probability=flip_probability,
			gaussian_noise_std=noise_std,
		)
		for name, noise_std in VIEW_NOISE_STDS.items()
	}
	for dataset in datasets.values():
		dataset.set_epoch(epoch)

	metrics, integrity = _measure_samples(
		datasets,
		patch_size_xyz=base_dataset.patch_size_xyz,
		token_shape_xyz=base_dataset.token_grid_shape_xyz,
		indices=range(start_index, start_index + count),
	)
	manifest_path = Path(_string(_mapping(config, 'manifests'), 'train'))
	return {
		'schema_version': 1,
		'diagnostic': 'f3_local_barlow_twins_aligned_views',
		'control_config': str(config_path.resolve()),
		'control_config_sha256': _file_sha256(config_path),
		'manifest': str(manifest_path.resolve()),
		'sampling': {
			'seed': _integer(_mapping(config, 'train'), 'seed'),
			'epoch': epoch,
			'indices': list(range(start_index, start_index + count)),
			'local_crop_size_xyz': list(base_dataset.local_crop_size_xyz),
			'patch_size_xyz': list(base_dataset.patch_size_xyz),
			'local_pairs_per_crop': local_pairs_per_crop,
		},
		'views': {
			name: {
				'horizontal_flip_probability': flip_probability,
				'gaussian_noise_std': noise_std,
			}
			for name, noise_std in VIEW_NOISE_STDS.items()
		},
		'metrics': metrics,
		'integrity': integrity,
	}


def _build_base_dataset(
	config: Mapping[str, object],
) -> AmplitudePretrainDataset:
	data = _mapping(config, 'data')
	model = _mapping(config, 'model')
	train = _mapping(config, 'train')
	barlow_twins = _mapping(config, 'barlow_twins')
	zero_mask = _mapping(config, 'zero_mask')
	manifest_path = Path(_string(_mapping(config, 'manifests'), 'train'))
	return AmplitudePretrainDataset(
		read_manifest_json(manifest_path),
		local_crop_size_xyz=_integer_triplet(data, 'local_crop_size'),
		patch_size_xyz=_integer_triplet(model, 'patch_size'),
		emit_spatial_mask=False,
		seed=_integer(train, 'seed'),
		samples_per_epoch=_integer(train, 'samples_per_epoch'),
		zero_mask=ZeroMaskConfig(
			enabled=_boolean(zero_mask, 'enabled'),
			zero_atol=_floating(zero_mask, 'zero_atol'),
			z_sample_influence_radius=_integer(
				zero_mask,
				'z_sample_influence_radius',
			),
			xy_trace_influence_radius=_integer(
				zero_mask,
				'xy_trace_influence_radius',
			),
		),
		min_valid_fraction=_floating(data, 'min_valid_fraction'),
		max_resample_attempts=_integer(data, 'max_resample_attempts'),
		normalized_clip_abs=_optional_float(data, 'normalized_clip_abs'),
		amplitude_agc=cast('Mapping[str, object]', data['amplitude_agc']),
		finite_check_mode=cast('FiniteCheckMode', data['finite_check_mode']),
		min_valid_token_count=_integer(
			barlow_twins,
			'local_pairs_per_crop',
		),
	)


def _measure_samples(
	datasets: Mapping[str, LocalBarlowTwinsPretrainDataset],
	*,
	patch_size_xyz: tuple[int, int, int],
	token_shape_xyz: tuple[int, int, int],
	indices: Sequence[int],
) -> tuple[dict[str, object], dict[str, object]]:
	metric_totals = {
		name: {
			'all_valid_physical_voxels': _MetricAccumulator(),
			'sampled_pair_token_voxels': _MetricAccumulator(),
		}
		for name in VIEW_NOISE_STDS
	}
	integrity_counts = {
		'legacy_aligned_view_mismatched_voxels': 0,
		'aligned_mask_mismatched_views': 0,
		'flip_state_mismatched_views': 0,
		'pair_index_mismatched_views': 0,
		'physical_pair_mismatched_views': 0,
		'coordinate_mismatched_samples': 0,
		'sampled_pair_invalid_voxels': 0,
	}
	invalid_nonzero_values = dict.fromkeys(VIEW_NOISE_STDS, 0)
	invalid_value_mismatches = dict.fromkeys(VIEW_NOISE_STDS, 0)
	valid_counts: list[int] = []
	invalid_voxel_count = 0
	noise_scale_residual = 0.0

	for index in indices:
		samples = {name: dataset[index] for name, dataset in datasets.items()}
		legacy = samples['legacy']
		canonical = _aligned_amplitude(legacy, 'a')[0]
		legacy_b = _aligned_amplitude(legacy, 'b')[0]
		canonical_valid = _aligned_mask(legacy, 'a')
		integrity_counts['legacy_aligned_view_mismatched_voxels'] += int(
			np.count_nonzero(canonical != legacy_b)
		)
		if not np.array_equal(canonical_valid, _aligned_mask(legacy, 'b')):
			integrity_counts['aligned_mask_mismatched_views'] += 1
		valid_counts.append(int(np.count_nonzero(canonical_valid)))
		invalid_voxel_count += int(np.count_nonzero(~canonical_valid))
		legacy_pairs = _canonical_pair_indices(
			legacy,
			'a',
			token_shape_xyz=token_shape_xyz,
		)
		if not np.array_equal(
			legacy_pairs,
			_canonical_pair_indices(
				legacy,
				'b',
				token_shape_xyz=token_shape_xyz,
			),
		):
			integrity_counts['physical_pair_mismatched_views'] += 1
		selected = _selected_voxel_mask(
			legacy_pairs,
			patch_size_xyz=patch_size_xyz,
			token_shape_xyz=token_shape_xyz,
		)
		integrity_counts['sampled_pair_invalid_voxels'] += int(
			np.count_nonzero(selected & ~canonical_valid)
		)
		aligned: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}
		for name, sample in samples.items():
			aligned[name] = (
				_aligned_amplitude(sample, 'a')[0],
				_aligned_amplitude(sample, 'b')[0],
			)
			_integrity_for_sample(
				sample,
				legacy=legacy,
				legacy_pairs=legacy_pairs,
				canonical_valid=canonical_valid,
				token_shape_xyz=token_shape_xyz,
				counts=integrity_counts,
			)
			if sample.get('coords') != legacy.get('coords'):
				integrity_counts['coordinate_mismatched_samples'] += 1
			view_a, view_b = aligned[name]
			invalid_nonzero_values[name] += int(
				np.count_nonzero(view_a[~canonical_valid])
				+ np.count_nonzero(view_b[~canonical_valid])
			)
			invalid_value_mismatches[name] += int(
				np.count_nonzero(
					view_a[~canonical_valid] != canonical[~canonical_valid]
				)
				+ np.count_nonzero(
					view_b[~canonical_valid] != canonical[~canonical_valid]
				)
			)
			metric_totals[name]['all_valid_physical_voxels'].add(
				view_a[canonical_valid],
				view_b[canonical_valid],
				canonical[canonical_valid],
			)
			metric_totals[name]['sampled_pair_token_voxels'].add(
				view_a[selected],
				view_b[selected],
				canonical[selected],
			)
		for suffix_index in (0, 1):
			delta_005 = aligned['gaussian_noise_std005'][suffix_index] - canonical
			delta_010 = aligned['gaussian_noise_std010'][suffix_index] - canonical
			noise_scale_residual = max(
				noise_scale_residual,
				float(
					np.max(
						np.abs(
							delta_010[canonical_valid]
							- 2.0 * delta_005[canonical_valid]
						)
					)
				),
			)

	metrics: dict[str, object] = {
		name: {
			population: accumulator.result()
			for population, accumulator in populations.items()
		}
		for name, populations in metric_totals.items()
	}
	integrity: dict[str, object] = {
		**integrity_counts,
		'invalid_canonical_voxel_count': invalid_voxel_count,
		'valid_voxel_count_min_per_crop': min(valid_counts),
		'valid_voxel_count_max_per_crop': max(valid_counts),
		'invalid_nonzero_values_across_both_views': invalid_nonzero_values,
		'invalid_value_mismatches_vs_legacy': invalid_value_mismatches,
		'max_abs_noise_scaling_residual_std010_minus_2x_std005': (
			noise_scale_residual
		),
	}
	return metrics, integrity


def _integrity_for_sample(  # noqa: PLR0913
	sample: Mapping[str, object],
	*,
	legacy: Mapping[str, object],
	legacy_pairs: NDArray[np.integer],
	canonical_valid: NDArray[np.bool_],
	token_shape_xyz: tuple[int, int, int],
	counts: dict[str, int],
) -> None:
	for suffix in ('a', 'b'):
		if not np.array_equal(
			_array(sample, f'horizontal_flip_state_{suffix}'),
			_array(legacy, f'horizontal_flip_state_{suffix}'),
		):
			counts['flip_state_mismatched_views'] += 1
		if not np.array_equal(
			_array(sample, f'local_pair_indices_{suffix}'),
			_array(legacy, f'local_pair_indices_{suffix}'),
		):
			counts['pair_index_mismatched_views'] += 1
		if not np.array_equal(_aligned_mask(sample, suffix), canonical_valid):
			counts['aligned_mask_mismatched_views'] += 1
		if not np.array_equal(
			_canonical_pair_indices(
				sample,
				suffix,
				token_shape_xyz=token_shape_xyz,
			),
			legacy_pairs,
		):
			counts['physical_pair_mismatched_views'] += 1


def _aligned_amplitude(
	sample: Mapping[str, object],
	suffix: str,
) -> NDArray[np.floating]:
	value = _array(sample, f'view_{suffix}')
	state = _array(sample, f'horizontal_flip_state_{suffix}')
	axes = tuple(
		axis + 1 for axis, flipped in enumerate(state) if bool(flipped)
	)
	return np.flip(value, axis=axes) if axes else value


def _aligned_mask(
	sample: Mapping[str, object],
	suffix: str,
) -> NDArray[np.bool_]:
	value = _array(sample, f'valid_mask_{suffix}')
	state = _array(sample, f'horizontal_flip_state_{suffix}')
	axes = tuple(axis for axis, flipped in enumerate(state) if bool(flipped))
	return cast('NDArray[np.bool_]', np.flip(value, axis=axes) if axes else value)


def _canonical_pair_indices(
	sample: Mapping[str, object],
	suffix: str,
	*,
	token_shape_xyz: tuple[int, int, int],
) -> NDArray[np.integer]:
	indices = _array(sample, f'local_pair_indices_{suffix}')
	coordinates = np.asarray(
		np.unravel_index(indices, token_shape_xyz, order='C'),
		dtype=np.int64,
	)
	state = _array(sample, f'horizontal_flip_state_{suffix}')
	if bool(state[0]):
		coordinates[0] = token_shape_xyz[0] - 1 - coordinates[0]
	if bool(state[1]):
		coordinates[1] = token_shape_xyz[1] - 1 - coordinates[1]
	return np.asarray(
		np.ravel_multi_index(tuple(coordinates), token_shape_xyz, order='C'),
		dtype=np.int64,
	)


def _selected_voxel_mask(
	canonical_pair_indices: NDArray[np.integer],
	*,
	patch_size_xyz: tuple[int, int, int],
	token_shape_xyz: tuple[int, int, int],
) -> NDArray[np.bool_]:
	tokens = np.zeros(token_shape_xyz, dtype=bool)
	tokens.ravel(order='C')[canonical_pair_indices] = True
	selected = tokens
	for axis, patch_size in enumerate(patch_size_xyz):
		selected = np.repeat(selected, patch_size, axis=axis)
	return selected


def _validate_sampling(*, epoch: int, start_index: int, count: int) -> None:
	for name, value in (('epoch', epoch), ('start_index', start_index)):
		if value < 0:
			raise ValueError(f'{name} must be nonnegative; got {value}')
	if count <= 0:
		raise ValueError(f'count must be positive; got {count}')


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description='Diagnose aligned Local Barlow Twins F3 views on CPU.',
	)
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--epoch', type=int, default=0)
	parser.add_argument('--start-index', type=int, default=0)
	parser.add_argument('--count', type=int, default=16)
	parser.add_argument(
		'--output',
		type=Path,
		help='Optional new JSON path; existing files are never overwritten.',
	)
	return parser


def _emit_report(payload: Mapping[str, object], output: Path | None) -> None:
	serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
	if output is not None:
		output.parent.mkdir(parents=True, exist_ok=True)
		with output.open('x', encoding='utf-8') as file_obj:
			file_obj.write(serialized)
	print(serialized, end='')


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(chunk)
	return digest.hexdigest()


def _array(mapping: Mapping[str, object], key: str) -> NDArray[np.generic]:
	value = mapping.get(key)
	if not isinstance(value, np.ndarray):
		raise TypeError(f'{key} must be a NumPy array')
	return value


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = mapping.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return value


def _string(mapping: Mapping[str, object], key: str) -> str:
	value = mapping.get(key)
	if not isinstance(value, str):
		raise TypeError(f'{key} must be a string')
	return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
	value = mapping.get(key)
	if isinstance(value, bool) or not isinstance(value, int):
		raise TypeError(f'{key} must be an integer')
	return value


def _floating(mapping: Mapping[str, object], key: str) -> float:
	value = mapping.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError(f'{key} must be numeric')
	return float(value)


def _optional_float(mapping: Mapping[str, object], key: str) -> float | None:
	value = mapping.get(key)
	return None if value is None else _floating(mapping, key)


def _boolean(mapping: Mapping[str, object], key: str) -> bool:
	value = mapping.get(key)
	if not isinstance(value, bool):
		raise TypeError(f'{key} must be a boolean')
	return value


def _integer_triplet(
	mapping: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	value = mapping.get(key)
	if not isinstance(value, list | tuple) or len(value) != 3:
		raise TypeError(f'{key} must be a length-three sequence')
	if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
		raise TypeError(f'{key} must contain integers')
	return cast('tuple[int, int, int]', tuple(value))


def main() -> None:
	"""Run the diagnostic and emit deterministic JSON."""
	args = _build_parser().parse_args()
	config = resolve_barlow_twins_training_config(load_config(args.config))
	report = diagnose_views(
		config,
		config_path=args.config,
		epoch=args.epoch,
		start_index=args.start_index,
		count=args.count,
	)
	_emit_report(report, args.output)


if __name__ == '__main__':
	main()
