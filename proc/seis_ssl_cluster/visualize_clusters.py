"""Thin entrypoint for amplitude-only cluster visualization."""

from __future__ import annotations

import importlib
import json
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Protocol

import numpy as np

from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	apply_configured_agc,
	load_normalization_stats,
	normalize_amplitude,
)
from seis_ssl_cluster.data.zero_mask import (
	ZeroMaskConfig,
	compute_zero_amplitude_invalid_mask,
)
from seis_ssl_cluster.utils.cli import print_config_summary

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'visualize_clusters.yaml'
)
XYZ = tuple[int, int, int]


@dataclass(frozen=True)
class _SurveyGeometry:
	patch_size_xyz: XYZ
	volume_shape_xyz: XYZ


@dataclass(frozen=True)
class _LabelArtifact:
	k: int
	k_dir: Path
	survey_id: str
	token_path: Path
	metadata_path: Path


@dataclass(frozen=True)
class _SliceBound:
	start: int
	stop: int
	scalar: bool = False


class _SliceableAmplitude(Protocol):
	shape: tuple[int, ...]

	def __getitem__(self, key: object) -> np.ndarray:
		"""Return a requested amplitude subset."""


@dataclass(frozen=True)
class _AmplitudeDisplay:
	array: _SliceableAmplitude
	display_space: str


class _PreprocessedAmplitudeVolume:
	def __init__(
		self,
		source: np.ndarray,
		*,
		stats: SurveyNormalizationStats,
		normalized_clip_abs: float | None,
		agc: AmplitudeAgcConfig,
		zero_mask: ZeroMaskConfig | None,
	) -> None:
		self._source = source
		self._stats = stats
		self._normalized_clip_abs = normalized_clip_abs
		self._agc = agc
		self._zero_mask = zero_mask
		self.shape = tuple(int(axis) for axis in source.shape)

	def __getitem__(self, key: object) -> np.ndarray:
		bounds = _volume_slice_bounds(key, self.shape)
		context = (
			_expand_bound(bounds[0], self.shape[0], self._xy_context_radius),
			_expand_bound(bounds[1], self.shape[1], self._xy_context_radius),
			_expand_bound(bounds[2], self.shape[2], self._z_context_radius),
		)
		raw = np.asarray(
			self._source[
				context[0][0] : context[0][1],
				context[1][0] : context[1][1],
				context[2][0] : context[2][1],
			],
			dtype=np.float32,
		)
		amplitude = _preprocess_amplitude_chunk(
			raw,
			stats=self._stats,
			normalized_clip_abs=self._normalized_clip_abs,
			agc=self._agc,
			zero_mask=self._zero_mask,
		)
		result = amplitude[
			bounds[0].start - context[0][0] : bounds[0].stop - context[0][0],
			bounds[1].start - context[1][0] : bounds[1].stop - context[1][0],
			bounds[2].start - context[2][0] : bounds[2].stop - context[2][0],
		]
		scalar_axes = tuple(
			axis for axis, bound in enumerate(bounds) if bound.scalar
		)
		if scalar_axes:
			result = np.squeeze(result, axis=scalar_axes)
		return result.astype(np.float32, copy=False)

	@property
	def _z_context_radius(self) -> int:
		radius = 0
		if self._agc.enabled:
			radius = max(radius, int(self._agc.window_z) // 2)
		if self._zero_mask is not None and self._zero_mask.enabled:
			radius = max(radius, self._zero_mask.z_sample_influence_radius)
		return radius

	@property
	def _xy_context_radius(self) -> int:
		if self._zero_mask is None or not self._zero_mask.enabled:
			return 0
		return self._zero_mask.xy_trace_influence_radius


class _TokenAmplitudeUnderlay:
	def __init__(
		self,
		amplitude: _SliceableAmplitude,
		*,
		token_shape: XYZ,
		patch: XYZ,
	) -> None:
		self._amplitude = amplitude
		self._patch = patch
		self.shape = token_shape

	def __getitem__(self, key: object) -> np.ndarray:
		bounds = _volume_slice_bounds(key, self.shape)
		if (
			_full_slice(bounds[0], self.shape[0])
			and _full_slice(bounds[1], self.shape[1])
			and bounds[2].scalar
		):
			return self._xy_slice(bounds[2].start)
		if (
			_full_slice(bounds[0], self.shape[0])
			and bounds[1].scalar
			and _full_slice(bounds[2], self.shape[2])
		):
			return self._xz_slice(bounds[1].start)
		msg = f'unsupported token amplitude slice: {key!r}'
		raise IndexError(msg)

	def _xy_slice(self, token_z: int) -> np.ndarray:
		z_start = token_z * self._patch[2]
		z_stop = min(z_start + self._patch[2], self._amplitude.shape[2])
		if z_start >= self._amplitude.shape[2]:
			return np.full(self.shape[:2], np.nan, dtype=np.float32)
		amplitude = self._amplitude[:, :, z_start:z_stop]
		tokens = _downsample_amplitude_to_tokens(
			amplitude,
			(self.shape[0], self.shape[1], 1),
			self._patch,
		)
		return tokens[:, :, 0]

	def _xz_slice(self, token_y: int) -> np.ndarray:
		y_start = token_y * self._patch[1]
		y_stop = min(y_start + self._patch[1], self._amplitude.shape[1])
		if y_start >= self._amplitude.shape[1]:
			return np.full(
				(self.shape[0], self.shape[2]),
				np.nan,
				dtype=np.float32,
			)
		amplitude = self._amplitude[:, y_start:y_stop, :]
		tokens = _downsample_amplitude_to_tokens(
			amplitude,
			(self.shape[0], 1, self.shape[2]),
			self._patch,
		)
		return tokens[:, 0, :]


def main() -> None:
	"""Run amplitude-only cluster visualization or print a dry-run summary."""
	parser = ArgumentParser(description='Visualize amplitude-only clusters.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	args = parser.parse_args()

	config = resolve_cluster_visualization_config(load_config(args.config))
	if args.dry_run:
		print_config_summary(config)
		print('execution: dry-run; visualization skipped')
		return

	result = run_cluster_visualization(config)
	print(
		f'visualization: wrote {result["png_count"]} PNG(s), '
		f'{result["voxel_count"]} voxel label file(s), '
		f'and {result["summary_count"]} summary set(s)',
	)


def run_cluster_visualization(  # noqa: C901, PLR0912, PLR0915
	config: Mapping[str, object],
) -> dict[str, int]:
	"""Run cluster-map reconstruction, summaries, and configured PNG rendering."""
	reconstruct = importlib.import_module('seis_ssl_cluster.clustering.reconstruct')
	summaries = importlib.import_module('seis_ssl_cluster.clustering.summaries')
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	clustering = _required_mapping(config, 'clustering')
	visualization = _required_mapping(config, 'visualization')
	input_dir = _required_path(clustering, 'input_dir', 'clustering')
	output_dir = _required_path(visualization, 'output_dir', 'visualization')
	reconstruct_voxel = _bool(
		visualization.get('reconstruct_voxel', False),
		'visualization.reconstruct_voxel',
	)
	requested_survey_ids = _survey_ids(visualization.get('survey_ids', []))
	allow_all_reconstruction = _bool(
		visualization.get('allow_all_surveys_for_voxel_reconstruction', False),
		'visualization.allow_all_surveys_for_voxel_reconstruction',
	)
	skip_existing_voxel_labels = _bool(
		visualization.get('skip_existing_voxel_labels', True),
		'visualization.skip_existing_voxel_labels',
	)
	max_voxel_output_gib = _nonnegative_float(
		visualization.get('max_voxel_output_gib', 50.0),
		'visualization.max_voxel_output_gib',
	)
	allow_large_voxel_output = _bool(
		visualization.get('allow_large_voxel_output', False),
		'visualization.allow_large_voxel_output',
	)
	modes = _modes(visualization.get('modes', ['token']))
	slice_request = clusters.ClusterSliceRequest(
		xy_slices=_int_tuple(visualization.get('xy_slices', ()), 'xy_slices'),
		xz_slices=_int_tuple(visualization.get('xz_slices', ()), 'xz_slices'),
	)
	_validate_slice_coordinate_space(
		visualization.get('slice_coordinate_space', 'voxel'),
	)
	dpi = _positive_int(visualization.get('dpi', 160), 'visualization.dpi')
	invalid_color = str(visualization.get('invalid_color', 'lightgray'))
	underlay_cfg = _optional_mapping(visualization, 'amplitude_underlay')
	underlay_enabled = _bool(
		underlay_cfg.get('enabled', False),
		'visualization.amplitude_underlay.enabled',
	)
	underlay_alpha = _fraction(
		underlay_cfg.get('alpha', 0.35),
		'visualization.amplitude_underlay.alpha',
	)
	comparison_cfg = _optional_mapping(visualization, 'amplitude_comparison')
	comparison_enabled = _bool(
		comparison_cfg.get('enabled', False),
		'visualization.amplitude_comparison.enabled',
	)
	comparison_alpha = _fraction(
		comparison_cfg.get('alpha', 0.35),
		'visualization.amplitude_comparison.alpha',
	)
	summary_cfg = _optional_mapping(visualization, 'summaries')
	summaries_enabled = _bool(
		summary_cfg.get('enabled', True),
		'visualization.summaries.enabled',
	)
	include_amplitude = _bool(
		summary_cfg.get('include_amplitude_norm', False),
		'visualization.summaries.include_amplitude_norm',
	)
	artifacts = _filter_label_artifacts(
		_discover_label_artifacts(input_dir),
		requested_survey_ids=requested_survey_ids,
	)
	if reconstruct_voxel and not requested_survey_ids and not allow_all_reconstruction:
		msg = (
			'visualization.reconstruct_voxel with an empty survey_ids list would '
			'reconstruct every discovered survey; set visualization.survey_ids '
			'or visualization.allow_all_surveys_for_voxel_reconstruction: true'
		)
		raise ValueError(msg)
	if reconstruct_voxel:
		_validate_voxel_output_estimate(
			artifacts,
			max_gib=max_voxel_output_gib,
			allow_large=allow_large_voxel_output,
		)

	png_count = 0
	voxel_count = 0
	summary_count = 0
	for k, k_artifacts in _artifacts_by_k(artifacts).items():
		summary_inputs = []
		for artifact in k_artifacts:
			survey_id = artifact.survey_id
			token_path = artifact.token_path
			metadata_path = artifact.metadata_path
			metadata = _load_metadata(metadata_path)
			embedding_metadata = _embedding_metadata(metadata)
			embedding_input = metadata.get('embedding_input')
			embeddings_path = None
			if isinstance(embedding_input, Mapping):
				value = embedding_input.get('embeddings_path')
				if isinstance(value, str):
					embeddings_path = Path(value)
			summary_inputs.append(
				summaries.ClusterSummaryInput(
					survey_id=survey_id,
					labels_path=token_path,
					metadata_path=metadata_path,
					embeddings_path=embeddings_path,
				),
			)
			token_labels = np.load(token_path, mmap_mode='r')
			amplitude_display = (
				_open_amplitude_display(embedding_metadata)
				if underlay_enabled or comparison_enabled
				else None
			)
			amplitude = (
				None if amplitude_display is None else amplitude_display.array
			)
			amplitude_display_space = (
				'amplitude'
				if amplitude_display is None
				else amplitude_display.display_space
			)
			needs_geometry = (
				'token' in modes
				or reconstruct_voxel
				or 'voxel' in modes
			)
			geometry = (
				_required_survey_geometry(
					embedding_metadata,
					token_shape_xyz=token_labels.shape,
					survey_id=survey_id,
				)
				if needs_geometry
				else None
			)
			if 'token' in modes:
				if geometry is None:
					msg = 'internal error: token visualization requires geometry'
					raise RuntimeError(msg)
				token_slices = _token_slice_request(
					slice_request,
					token_shape_xyz=token_labels.shape,
					geometry=geometry,
					survey_id=survey_id,
				)
				token_amplitude = _amplitude_underlay_for_labels(
					amplitude,
					token_labels,
					geometry.patch_size_xyz,
				)
				png_count += len(
					clusters.save_cluster_slice_pngs(
						token_labels,
						survey_id=survey_id,
						k=k,
						mode='token',
						output_dir=output_dir / 'token',
						slices=token_slices,
						amplitude=token_amplitude if underlay_enabled else None,
						amplitude_alpha=underlay_alpha,
						invalid_color=invalid_color,
						dpi=dpi,
					),
				)
				if comparison_enabled:
					comparison_amplitude = _require_amplitude_for_comparison(
						token_amplitude,
						survey_id=survey_id,
						mode='token',
					)
					png_count += len(
						clusters.save_cluster_comparison_pngs(
							token_labels,
							survey_id=survey_id,
							k=k,
							mode='token',
							output_dir=output_dir / 'token_comparison',
							slices=token_slices,
							amplitude=comparison_amplitude,
							amplitude_display_space=amplitude_display_space,
							amplitude_alpha=comparison_alpha,
							invalid_color=invalid_color,
							dpi=dpi,
						),
					)
			voxel_path = artifact.k_dir / f'{survey_id}.cluster_labels_voxel.npy'
			if reconstruct_voxel:
				if geometry is None:
					msg = 'internal error: voxel reconstruction requires geometry'
					raise RuntimeError(msg)
				reconstructed = reconstruct.reconstruct_labels_for_survey(
					token_path,
					metadata_path=metadata_path,
					write_voxel_labels=True,
					skip_existing_voxel_labels=skip_existing_voxel_labels,
				)
				if not reconstructed.skipped_existing_voxel_labels:
					voxel_count += 1
			if 'voxel' in modes:
				if geometry is None:
					msg = 'internal error: voxel visualization requires geometry'
					raise RuntimeError(msg)
				if not voxel_path.is_file():
					msg = (
						f'voxel visualization requested for survey {survey_id!r} '
						f'at k={k}, but voxel labels do not exist: {voxel_path}; '
						'set visualization.reconstruct_voxel: true to create them'
					)
					raise FileNotFoundError(msg)
				voxel_slices = _voxel_slice_request(
					slice_request,
					geometry=geometry,
					survey_id=survey_id,
				)
				voxel_labels = np.load(voxel_path, mmap_mode='r')
				png_count += len(
					clusters.save_cluster_slice_pngs(
						voxel_labels,
						survey_id=survey_id,
						k=k,
						mode='voxel',
						output_dir=output_dir / 'voxel',
						slices=voxel_slices,
						amplitude=amplitude if underlay_enabled else None,
						amplitude_alpha=underlay_alpha,
						invalid_color=invalid_color,
						dpi=dpi,
					),
				)
				if comparison_enabled:
					comparison_amplitude = _require_amplitude_for_comparison(
						amplitude,
						survey_id=survey_id,
						mode='voxel',
					)
					png_count += len(
						clusters.save_cluster_comparison_pngs(
							voxel_labels,
							survey_id=survey_id,
							k=k,
							mode='voxel',
							output_dir=output_dir / 'voxel_comparison',
							slices=voxel_slices,
							amplitude=comparison_amplitude,
							amplitude_display_space=amplitude_display_space,
							amplitude_alpha=comparison_alpha,
							invalid_color=invalid_color,
							dpi=dpi,
						),
					)
		if summaries_enabled and summary_inputs:
			summaries.write_cluster_summaries(
				summary_inputs,
				k=k,
				output_dir=output_dir / f'k{k}',
				include_amplitude_norm=include_amplitude,
				selected_survey_ids=(
					requested_survey_ids if requested_survey_ids else None
				),
			)
			summary_count += 1
	return {
		'png_count': png_count,
		'voxel_count': voxel_count,
		'summary_count': summary_count,
	}


def _discover_label_artifacts(input_dir: Path) -> list[_LabelArtifact]:
	artifacts = []
	for k_dir in _label_k_dirs(input_dir):
		k = int(k_dir.name.removeprefix('k'))
		for token_path in sorted(k_dir.glob('*.cluster_labels_token.npy')):
			survey_id = token_path.name.removesuffix('.cluster_labels_token.npy')
			artifacts.append(
				_LabelArtifact(
					k=k,
					k_dir=k_dir,
					survey_id=survey_id,
					token_path=token_path,
					metadata_path=k_dir / f'{survey_id}.cluster_label_metadata.json',
				),
			)
	if not artifacts:
		msg = f'no token label files found under {input_dir / "labels"}'
		raise ValueError(msg)
	return artifacts


def _filter_label_artifacts(
	artifacts: Sequence[_LabelArtifact],
	*,
	requested_survey_ids: tuple[str, ...],
) -> list[_LabelArtifact]:
	if not requested_survey_ids:
		return list(artifacts)
	known = {artifact.survey_id for artifact in artifacts}
	missing = sorted(set(requested_survey_ids) - known)
	if missing:
		examples = ', '.join(sorted(known)[:10])
		msg = (
			'unknown visualization.survey_ids requested: '
			f'{missing!r}; discovered survey IDs include: {examples}'
		)
		raise ValueError(msg)
	requested = set(requested_survey_ids)
	return [artifact for artifact in artifacts if artifact.survey_id in requested]


def _artifacts_by_k(
	artifacts: Sequence[_LabelArtifact],
) -> dict[int, list[_LabelArtifact]]:
	grouped: dict[int, list[_LabelArtifact]] = {}
	for artifact in artifacts:
		grouped.setdefault(artifact.k, []).append(artifact)
	return dict(sorted(grouped.items()))


def _validate_voxel_output_estimate(
	artifacts: Sequence[_LabelArtifact],
	*,
	max_gib: float,
	allow_large: bool,
) -> None:
	file_count = 0
	byte_count = 0
	for artifact in artifacts:
		labels = np.load(artifact.token_path, mmap_mode='r')
		metadata = _embedding_metadata(_load_metadata(artifact.metadata_path))
		geometry = _required_survey_geometry(
			metadata,
			token_shape_xyz=labels.shape,
			survey_id=artifact.survey_id,
		)
		byte_count += (
			int(np.prod(geometry.volume_shape_xyz, dtype=np.int64))
			* np.dtype(np.int32).itemsize
		)
		file_count += 1
	gib = byte_count / (1024.0**3)
	print(
		'voxel reconstruction estimate: '
		f'{file_count} file(s), {byte_count} bytes ({gib:.2f} GiB)',
	)
	if byte_count > max_gib * (1024.0**3) and not allow_large:
		msg = (
			'estimated voxel label output is '
			f'{gib:.2f} GiB, exceeding '
			f'visualization.max_voxel_output_gib={max_gib:g}; set '
			'visualization.allow_large_voxel_output: true to proceed'
		)
		raise ValueError(msg)


def _label_k_dirs(input_dir: Path) -> list[Path]:
	labels_root = input_dir / 'labels'
	if not labels_root.is_dir():
		msg = f'clustering labels directory does not exist: {labels_root}'
		raise FileNotFoundError(msg)
	k_dirs = [
		path
		for path in labels_root.iterdir()
		if path.is_dir()
		and path.name.startswith('k')
		and path.name.removeprefix('k').isdigit()
	]
	if not k_dirs:
		msg = f'no k label directories found under {labels_root}'
		raise ValueError(msg)
	return sorted(k_dirs, key=lambda path: int(path.name.removeprefix('k')))


def _load_metadata(path: Path) -> dict[str, object]:
	if not path.is_file():
		return {}
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		return {}
	return payload


def _embedding_metadata(label_metadata: Mapping[str, object]) -> dict[str, object]:
	embedding_input = label_metadata.get('embedding_input')
	if not isinstance(embedding_input, Mapping):
		return label_metadata.copy()
	metadata_path = embedding_input.get('metadata_path')
	if not isinstance(metadata_path, str) or not Path(metadata_path).is_file():
		return label_metadata.copy()
	payload = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		return label_metadata.copy()
	return {**payload, **label_metadata}


def _open_amplitude_display(metadata: Mapping[str, object]) -> _AmplitudeDisplay | None:
	value = metadata.get('source_amplitude_path')
	if not isinstance(value, str) or not Path(value).is_file():
		return None
	array = np.load(value, mmap_mode='r')
	if array.ndim != 3:
		return None
	preprocessing = metadata.get('preprocessing')
	if not isinstance(preprocessing, Mapping):
		return _AmplitudeDisplay(
			array=array,
			display_space='raw_amplitude',
		)
	agc = AmplitudeAgcConfig.from_mapping(preprocessing.get('amplitude_agc'))
	stats_path = metadata.get('normalization_stats_path')
	if not isinstance(stats_path, str) or not Path(stats_path).is_file():
		return _AmplitudeDisplay(
			array=array,
			display_space='raw_amplitude',
		)
	normalized_clip_abs = _optional_positive_float(
		preprocessing.get('normalized_clip_abs'),
		'preprocessing.normalized_clip_abs',
	)
	return _AmplitudeDisplay(
		array=_PreprocessedAmplitudeVolume(
			array,
			stats=load_normalization_stats(stats_path),
			normalized_clip_abs=normalized_clip_abs,
			agc=agc,
			zero_mask=_zero_mask_config(metadata),
		),
		display_space=_amplitude_display_space(
			normalized_clip_abs=normalized_clip_abs,
			agc=agc,
		),
	)


def _amplitude_valid_mask(
	raw: np.ndarray,
	zero_mask: ZeroMaskConfig | None,
) -> np.ndarray:
	valid_mask = np.ones(raw.shape, dtype=bool)
	if zero_mask is None:
		return valid_mask
	zero_invalid = compute_zero_amplitude_invalid_mask(
		raw,
		valid_mask=valid_mask,
		config=zero_mask,
	)
	return np.logical_and(valid_mask, ~zero_invalid)


def _zero_mask_config(metadata: Mapping[str, object]) -> ZeroMaskConfig | None:
	zero_mask = metadata.get('zero_mask')
	if not isinstance(zero_mask, Mapping):
		return None
	config = ZeroMaskConfig(**dict(zero_mask))
	config.validate()
	return config


def _preprocess_amplitude_chunk(
	raw: np.ndarray,
	*,
	stats: SurveyNormalizationStats,
	normalized_clip_abs: float | None,
	agc: AmplitudeAgcConfig,
	zero_mask: ZeroMaskConfig | None,
) -> np.ndarray:
	local_valid_mask = _amplitude_valid_mask(raw, zero_mask)
	normalized = normalize_amplitude(
		raw,
		stats,
		normalized_clip_abs=normalized_clip_abs,
	)
	amplitude = apply_configured_agc(normalized, local_valid_mask, agc)
	amplitude[~local_valid_mask] = 0.0
	return amplitude.astype(np.float32, copy=False)


def _amplitude_display_space(
	*,
	normalized_clip_abs: float | None,
	agc: AmplitudeAgcConfig,
) -> str:
	if agc.enabled:
		return 'survey_normalized_clipped_agc'
	if normalized_clip_abs is not None:
		return 'survey_normalized_clipped'
	return 'survey_normalized'


def _optional_positive_float(value: object, name: str) -> float | None:
	if value is None:
		return None
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{name} must be a number or null; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not np.isfinite(number) or number <= 0.0:
		msg = f'{name} must be a finite positive number or null; got {value!r}'
		raise ValueError(msg)
	return number


def _volume_slice_bounds(
	key: object,
	shape: tuple[int, ...],
) -> tuple[_SliceBound, _SliceBound, _SliceBound]:
	if len(shape) != 3:
		msg = f'amplitude shape must be 3D; got {shape!r}'
		raise ValueError(msg)
	selectors = key if isinstance(key, tuple) else (key,)
	if len(selectors) > 3:
		msg = f'expected at most 3 slice selectors; got {key!r}'
		raise IndexError(msg)
	padded = (*selectors, *(slice(None) for _ in range(3 - len(selectors))))
	return tuple(  # type: ignore[return-value]
		_slice_bound(selector, size)
		for selector, size in zip(padded, shape, strict=True)
	)


def _slice_bound(selector: object, size: int) -> _SliceBound:
	if isinstance(selector, bool):
		msg = f'boolean amplitude indices are not supported: {selector!r}'
		raise TypeError(msg)
	if isinstance(selector, Integral):
		index = int(selector)
		if index < 0:
			index += size
		if index < 0 or index >= size:
			msg = f'amplitude index out of range: {selector!r}; size={size}'
			raise IndexError(msg)
		return _SliceBound(index, index + 1, scalar=True)
	if isinstance(selector, slice):
		start, stop, step = selector.indices(size)
		if step != 1:
			msg = f'amplitude slices must have step 1; got {selector!r}'
			raise ValueError(msg)
		if stop <= start:
			msg = f'amplitude slice must be non-empty; got {selector!r}'
			raise ValueError(msg)
		return _SliceBound(start, stop)
	msg = f'unsupported amplitude index: {selector!r}'
	raise TypeError(msg)


def _expand_bound(
	bound: _SliceBound,
	size: int,
	radius: int,
) -> tuple[int, int]:
	return (
		max(0, bound.start - radius),
		min(size, bound.stop + radius),
	)


def _full_slice(bound: _SliceBound, size: int) -> bool:
	return not bound.scalar and bound.start == 0 and bound.stop == size


def _amplitude_underlay_for_labels(
	amplitude: _SliceableAmplitude | None,
	labels: np.ndarray,
	patch: XYZ,
) -> _SliceableAmplitude | None:
	if amplitude is None:
		return None
	if amplitude.shape == labels.shape:
		return amplitude
	padded_shape = tuple(
		label_axis * patch_axis
		for label_axis, patch_axis in zip(labels.shape, patch, strict=True)
	)
	if any(
		amplitude_axis > padded_axis
		for amplitude_axis, padded_axis in zip(
			amplitude.shape,
			padded_shape,
			strict=True,
		)
	):
		msg = (
			'amplitude underlay shape is incompatible with token labels; '
			f'got amplitude={amplitude.shape!r}, labels={labels.shape!r}, '
			f'patch_size={patch!r}'
		)
		raise ValueError(msg)
	return _TokenAmplitudeUnderlay(
		amplitude,
		token_shape=tuple(int(axis) for axis in labels.shape),
		patch=patch,
	)


def _require_amplitude_for_comparison(
	amplitude: _SliceableAmplitude | None,
	*,
	survey_id: str,
	mode: str,
) -> _SliceableAmplitude:
	if amplitude is None:
		msg = (
			'amplitude_comparison requested, but no amplitude array could be '
			f'opened for survey {survey_id!r} in {mode} mode; ensure the '
			'cluster label metadata points to a valid source_amplitude_path'
		)
		raise FileNotFoundError(msg)
	return amplitude


def _validate_slice_coordinate_space(value: object) -> None:
	if value != 'voxel':
		msg = (
			'visualization.slice_coordinate_space must be "voxel"; '
			f'got {value!r}'
		)
		raise ValueError(msg)


def _required_survey_geometry(
	metadata: Mapping[str, object],
	*,
	token_shape_xyz: Sequence[int],
	survey_id: str,
) -> _SurveyGeometry:
	if 'patch_size' not in metadata:
		msg = (
			f'cluster visualization for survey {survey_id!r} requires '
			'metadata field patch_size to map voxel-space slices to '
			'token-space slices and reconstruct voxel labels'
		)
		raise ValueError(msg)
	patch = _metadata_xyz(metadata['patch_size'], 'patch_size')
	volume_shape = _required_volume_shape_xyz(metadata, survey_id=survey_id)
	_validate_volume_fits_token_grid(
		volume_shape,
		token_shape_xyz=token_shape_xyz,
		patch_size_xyz=patch,
		survey_id=survey_id,
	)
	return _SurveyGeometry(
		patch_size_xyz=patch,
		volume_shape_xyz=volume_shape,
	)


def _required_volume_shape_xyz(
	metadata: Mapping[str, object],
	*,
	survey_id: str,
) -> XYZ:
	for key in ('volume_shape_xyz', 'volume_shape', 'shape_xyz'):
		if key in metadata:
			return _metadata_xyz(metadata[key], key)
	source_path = metadata.get('source_amplitude_path')
	if isinstance(source_path, str) and source_path:
		path = Path(source_path)
		if path.is_file():
			array = np.load(path, mmap_mode='r')
			if array.ndim == 3:
				return tuple(int(axis) for axis in array.shape)
	msg = (
		f'cluster visualization for survey {survey_id!r} requires '
		'volume_shape_xyz or a valid source_amplitude_path to interpret '
		'configured slices as original voxel coordinates'
	)
	raise ValueError(msg)


def _validate_volume_fits_token_grid(
	volume_shape_xyz: XYZ,
	*,
	token_shape_xyz: Sequence[int],
	patch_size_xyz: XYZ,
	survey_id: str,
) -> None:
	token_shape = _metadata_xyz(token_shape_xyz, 'token_shape_xyz')
	padded_shape = tuple(
		token_axis * patch_axis
		for token_axis, patch_axis in zip(token_shape, patch_size_xyz, strict=True)
	)
	if any(
		volume_axis > padded_axis
		for volume_axis, padded_axis in zip(
			volume_shape_xyz,
			padded_shape,
			strict=True,
		)
	):
		msg = (
			f'cluster visualization metadata for survey {survey_id!r} is '
			'incompatible with token labels; '
			f'volume_shape_xyz={volume_shape_xyz!r}, '
			f'token_shape_xyz={token_shape!r}, '
			f'patch_size={patch_size_xyz!r}'
		)
		raise ValueError(msg)


def _token_slice_request(
	slices: object,
	*,
	token_shape_xyz: Sequence[int],
	geometry: _SurveyGeometry,
	survey_id: str,
) -> object:
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	return clusters.ClusterSliceRequest(
		xy_slices=tuple(
			_token_slice(
				voxel_index,
				view='xy',
				token_shape_xyz=token_shape_xyz,
				geometry=geometry,
				survey_id=survey_id,
			)
			for voxel_index in slices.xy_slices
		),
		xz_slices=tuple(
			_token_slice(
				voxel_index,
				view='xz',
				token_shape_xyz=token_shape_xyz,
				geometry=geometry,
				survey_id=survey_id,
			)
			for voxel_index in slices.xz_slices
		),
	)


def _token_slice(
	voxel_index: int,
	*,
	view: str,
	token_shape_xyz: Sequence[int],
	geometry: _SurveyGeometry,
	survey_id: str,
) -> object:
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	_validate_voxel_slice_index(
		voxel_index,
		view=view,
		volume_shape_xyz=geometry.volume_shape_xyz,
		survey_id=survey_id,
	)
	axis = 2 if view == 'xy' else 1
	token_shape = _metadata_xyz(token_shape_xyz, 'token_shape_xyz')
	token_index = voxel_index // geometry.patch_size_xyz[axis]
	if token_index < 0 or token_index >= token_shape[axis]:
		msg = (
			f'{view} voxel slice {voxel_index} maps to token index '
			f'{token_index}, outside token label shape {token_shape!r} '
			f'for survey {survey_id!r}'
		)
		raise ValueError(msg)
	return clusters.ClusterSlice(
		array_slice_index=token_index,
		voxel_slice_index=voxel_index,
	)


def _voxel_slice_request(
	slices: object,
	*,
	geometry: _SurveyGeometry,
	survey_id: str,
) -> object:
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	return clusters.ClusterSliceRequest(
		xy_slices=tuple(
			_voxel_slice(
				voxel_index,
				view='xy',
				geometry=geometry,
				survey_id=survey_id,
			)
			for voxel_index in slices.xy_slices
		),
		xz_slices=tuple(
			_voxel_slice(
				voxel_index,
				view='xz',
				geometry=geometry,
				survey_id=survey_id,
			)
			for voxel_index in slices.xz_slices
		),
	)


def _voxel_slice(
	voxel_index: int,
	*,
	view: str,
	geometry: _SurveyGeometry,
	survey_id: str,
) -> object:
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	_validate_voxel_slice_index(
		voxel_index,
		view=view,
		volume_shape_xyz=geometry.volume_shape_xyz,
		survey_id=survey_id,
	)
	return clusters.ClusterSlice(
		array_slice_index=voxel_index,
		voxel_slice_index=voxel_index,
	)


def _validate_voxel_slice_index(
	voxel_index: int,
	*,
	view: str,
	volume_shape_xyz: XYZ,
	survey_id: str,
) -> None:
	axis = 2 if view == 'xy' else 1
	if view not in {'xy', 'xz'}:
		msg = f'unknown view: {view!r}'
		raise ValueError(msg)
	if voxel_index < 0 or voxel_index >= volume_shape_xyz[axis]:
		msg = (
			f'{view} voxel slice index out of range for survey {survey_id!r}: '
			f'{voxel_index}; valid=[0, {volume_shape_xyz[axis] - 1}]'
		)
		raise ValueError(msg)


def _downsample_amplitude_to_tokens(
	amplitude: np.ndarray,
	token_shape: tuple[int, int, int],
	patch: tuple[int, int, int],
) -> np.ndarray:
	underlay = np.empty(token_shape, dtype=np.float32)
	for token_x in range(token_shape[0]):
		x_start = token_x * patch[0]
		x_stop = min(x_start + patch[0], amplitude.shape[0])
		for token_y in range(token_shape[1]):
			y_start = token_y * patch[1]
			y_stop = min(y_start + patch[1], amplitude.shape[1])
			for token_z in range(token_shape[2]):
				z_start = token_z * patch[2]
				z_stop = min(z_start + patch[2], amplitude.shape[2])
				values = np.asarray(
					amplitude[x_start:x_stop, y_start:y_stop, z_start:z_stop],
					dtype=np.float32,
				)
				finite = values[np.isfinite(values)]
				underlay[token_x, token_y, token_z] = (
					float(finite.mean()) if finite.size else np.nan
				)
	return underlay


def _metadata_xyz(value: object, name: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or any(
			isinstance(item, bool) or not isinstance(item, Integral)
			for item in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(item) for item in value)
	if any(item <= 0 for item in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key, {})
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_path(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string'
		raise TypeError(msg)
	return Path(value)


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str):
		msg = f'visualization.{name} must be a sequence of integers'
		raise TypeError(msg)
	if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
		msg = f'visualization.{name} must be a sequence of integers'
		raise TypeError(msg)
	return tuple(int(item) for item in value)


def _modes(value: object) -> tuple[str, ...]:
	if isinstance(value, str):
		modes = (value,)
	elif isinstance(value, Sequence):
		modes = tuple(str(item) for item in value)
	else:
		msg = f'visualization.modes must be a string or sequence; got {value!r}'
		raise TypeError(msg)
	unknown = sorted(set(modes) - {'token', 'voxel'})
	if unknown:
		msg = f'unknown visualization modes: {unknown!r}'
		raise ValueError(msg)
	return modes


def _survey_ids(value: object) -> tuple[str, ...]:
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str):
		msg = 'visualization.survey_ids must be a sequence of strings'
		raise TypeError(msg)
	if any(not isinstance(item, str) for item in value):
		msg = 'visualization.survey_ids must be a sequence of strings'
		raise TypeError(msg)
	survey_ids = tuple(value)
	if any(not item for item in survey_ids):
		msg = 'visualization.survey_ids entries must be non-empty strings'
		raise ValueError(msg)
	return survey_ids


def _bool(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _positive_int(value: object, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{name} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return int(value)


def _fraction(value: object, name: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{name} must be a number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if fraction < 0.0 or fraction > 1.0:
		msg = f'{name} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _nonnegative_float(value: object, name: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{name} must be a number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if number < 0.0:
		msg = f'{name} must be non-negative; got {value!r}'
		raise ValueError(msg)
	return number


if __name__ == '__main__':
	main()
