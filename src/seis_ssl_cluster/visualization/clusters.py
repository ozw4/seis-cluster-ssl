"""PNG rendering for seismic cluster label maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.visualization.common import slice_image


@dataclass(frozen=True)
class ClusterSlice:
	"""One rendered slice with array and physical voxel coordinates."""

	array_slice_index: int
	voxel_slice_index: int


@dataclass(frozen=True)
class ClusterSliceRequest:
	"""Configured cluster slices for one rendering mode."""

	xy_slices: tuple[int | ClusterSlice, ...] = ()
	xz_slices: tuple[int | ClusterSlice, ...] = ()


def save_cluster_slice_pngs(  # noqa: PLR0913
	labels: np.ndarray,
	*,
	survey_id: str,
	k: int,
	mode: str,
	output_dir: str | Path,
	slices: ClusterSliceRequest,
	amplitude: np.ndarray | None = None,
	amplitude_alpha: float = 0.35,
	invalid_color: str = 'lightgray',
	dpi: int = 160,
) -> list[Path]:
	"""Render configured XY and XZ cluster-label slices as PNG files."""
	label_array = _validate_labels(labels)
	if amplitude is not None and amplitude.shape != label_array.shape:
		msg = (
			'amplitude underlay shape must match label shape; '
			f'got {amplitude.shape!r} and {label_array.shape!r}'
		)
		raise ValueError(msg)
	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)
	created = [
		_save_one_slice(
			label_array,
			survey_id=survey_id,
			k=k,
			mode=mode,
			view='xy',
			slice_index=index.array_slice_index,
			voxel_slice_index=index.voxel_slice_index,
			output_dir=root,
			amplitude=amplitude,
			amplitude_alpha=amplitude_alpha,
			invalid_color=invalid_color,
			dpi=dpi,
		)
		for index in _normalize_slice_specs(slices.xy_slices)
	]
	created.extend(
		[
			_save_one_slice(
				label_array,
				survey_id=survey_id,
				k=k,
				mode=mode,
				view='xz',
				slice_index=index.array_slice_index,
				voxel_slice_index=index.voxel_slice_index,
				output_dir=root,
				amplitude=amplitude,
				amplitude_alpha=amplitude_alpha,
				invalid_color=invalid_color,
				dpi=dpi,
			)
			for index in _normalize_slice_specs(slices.xz_slices)
		],
	)
	return created


def save_cluster_comparison_pngs(  # noqa: PLR0913
	labels: np.ndarray,
	*,
	survey_id: str,
	k: int,
	mode: str,
	output_dir: str | Path,
	slices: ClusterSliceRequest,
	amplitude: np.ndarray,
	amplitude_alpha: float = 0.35,
	invalid_color: str = 'lightgray',
	dpi: int = 160,
) -> list[Path]:
	"""Render amplitude, cluster, and overlay comparison slices as PNG files."""
	label_array = _validate_labels(labels)
	if amplitude.shape != label_array.shape:
		msg = (
			'amplitude comparison shape must match label shape; '
			f'got {amplitude.shape!r} and {label_array.shape!r}'
		)
		raise ValueError(msg)
	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)
	created = [
		_save_one_comparison_slice(
			label_array,
			survey_id=survey_id,
			k=k,
			mode=mode,
			view='xy',
			slice_index=index.array_slice_index,
			voxel_slice_index=index.voxel_slice_index,
			output_dir=root,
			amplitude=amplitude,
			amplitude_alpha=amplitude_alpha,
			invalid_color=invalid_color,
			dpi=dpi,
		)
		for index in _normalize_slice_specs(slices.xy_slices)
	]
	created.extend(
		[
			_save_one_comparison_slice(
				label_array,
				survey_id=survey_id,
				k=k,
				mode=mode,
				view='xz',
				slice_index=index.array_slice_index,
				voxel_slice_index=index.voxel_slice_index,
				output_dir=root,
				amplitude=amplitude,
				amplitude_alpha=amplitude_alpha,
				invalid_color=invalid_color,
				dpi=dpi,
			)
			for index in _normalize_slice_specs(slices.xz_slices)
		],
	)
	return created


def stable_cluster_colors(k: int, *, invalid_color: str = 'lightgray') -> object:
	"""Return a discrete colormap with stable label-to-color mapping."""
	return _cluster_colors(
		k,
		invalid_color=invalid_color,
		cluster_alpha=1.0,
		name=f'clusters_k{k}',
	)


def _cluster_colors(
	k: int,
	*,
	invalid_color: str,
	cluster_alpha: float,
	name: str,
) -> object:
	if k <= 0:
		msg = f'k must be positive; got {k!r}'
		raise ValueError(msg)
	alpha = _validate_alpha(cluster_alpha)
	color_module = _matplotlib_colors()
	colors = [color_module.to_rgba(invalid_color)]
	base = _plt().get_cmap('tab20', max(k, 1))
	colors.extend((*base(index)[:3], alpha) for index in range(k))
	return color_module.ListedColormap(colors, name=name)


def _save_one_slice(  # noqa: PLR0913
	labels: np.ndarray,
	*,
	survey_id: str,
	k: int,
	mode: str,
	view: str,
	slice_index: int,
	voxel_slice_index: int,
	output_dir: Path,
	amplitude: np.ndarray | None,
	amplitude_alpha: float,
	invalid_color: str,
	dpi: int,
) -> Path:
	_validate_slice_index(labels, view=view, slice_index=slice_index)
	plt = _plt()
	label_slice = slice_image(labels, view=view, slice_index=slice_index)
	display_labels = np.where(label_slice < 0, 0, label_slice + 1)
	fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=dpi)
	underlay_alpha = None
	if amplitude is not None:
		underlay_alpha = _validate_alpha(amplitude_alpha)
		amp_slice = slice_image(amplitude, view=view, slice_index=slice_index)
		vmin, vmax = _robust_limits(amp_slice)
		ax.imshow(
			amp_slice,
			cmap='gray',
			origin='lower',
			interpolation='none',
			vmin=vmin,
			vmax=vmax,
		)
	cmap = (
		stable_cluster_colors(k, invalid_color=invalid_color)
		if underlay_alpha is None
		else _cluster_colors(
			k,
			invalid_color=invalid_color,
			cluster_alpha=1.0 - underlay_alpha,
			name=f'clusters_k{k}_underlay',
		)
	)
	norm = _matplotlib_colors().BoundaryNorm(
		np.arange(-0.5, k + 1.5, 1.0),
		k + 1,
	)
	ax.imshow(
		display_labels,
		cmap=cmap,
		norm=norm,
		origin='lower',
		interpolation='none',
	)
	ax.set_title(
		_slice_title(
			survey_id=survey_id,
			k=k,
			mode=mode,
			view=view,
			array_slice_index=slice_index,
			voxel_slice_index=voxel_slice_index,
		),
	)
	ax.set_xlabel('x')
	ax.set_ylabel('y' if view == 'xy' else 'z')
	ax.tick_params(labelsize=7)
	fig.tight_layout()
	stem = (
		f'{survey_id}_k{k}_xy_z{voxel_slice_index}.png'
		if view == 'xy'
		else f'{survey_id}_k{k}_xz_y{voxel_slice_index}.png'
	)
	out_path = output_dir / stem
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def _save_one_comparison_slice(  # noqa: PLR0913
	labels: np.ndarray,
	*,
	survey_id: str,
	k: int,
	mode: str,
	view: str,
	slice_index: int,
	voxel_slice_index: int,
	output_dir: Path,
	amplitude: np.ndarray,
	amplitude_alpha: float,
	invalid_color: str,
	dpi: int,
) -> Path:
	_validate_slice_index(labels, view=view, slice_index=slice_index)
	plt = _plt()
	label_slice = slice_image(labels, view=view, slice_index=slice_index)
	display_labels = np.where(label_slice < 0, 0, label_slice + 1)
	amp_slice = slice_image(amplitude, view=view, slice_index=slice_index)
	vmin, vmax = _robust_limits(amp_slice)
	fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=dpi)
	amp_ax, cluster_ax, overlay_ax = axes
	for ax in axes:
		ax.set_xlabel('x')
		ax.set_ylabel('y' if view == 'xy' else 'z')
		ax.tick_params(labelsize=7)
	amp_ax.imshow(
		amp_slice,
		cmap='gray',
		origin='lower',
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	amp_ax.set_title('amplitude')
	cluster_cmap = stable_cluster_colors(k, invalid_color=invalid_color)
	norm = _matplotlib_colors().BoundaryNorm(
		np.arange(-0.5, k + 1.5, 1.0),
		k + 1,
	)
	cluster_ax.imshow(
		display_labels,
		cmap=cluster_cmap,
		norm=norm,
		origin='lower',
		interpolation='none',
	)
	cluster_ax.set_title('clusters')
	overlay_ax.imshow(
		amp_slice,
		cmap='gray',
		origin='lower',
		interpolation='none',
		vmin=vmin,
		vmax=vmax,
	)
	overlay_alpha = _validate_alpha(amplitude_alpha)
	overlay_ax.imshow(
		display_labels,
		cmap=_cluster_colors(
			k,
			invalid_color=invalid_color,
			cluster_alpha=1.0 - overlay_alpha,
			name=f'clusters_k{k}_comparison_overlay',
		),
		norm=norm,
		origin='lower',
		interpolation='none',
	)
	overlay_ax.set_title('overlay')
	fig.suptitle(
		_slice_title(
			survey_id=survey_id,
			k=k,
			mode=mode,
			view=view,
			array_slice_index=slice_index,
			voxel_slice_index=voxel_slice_index,
		),
		fontsize=10,
	)
	fig.tight_layout()
	stem = (
		f'{survey_id}_k{k}_xy_z{voxel_slice_index}.png'
		if view == 'xy'
		else f'{survey_id}_k{k}_xz_y{voxel_slice_index}.png'
	)
	out_path = output_dir / stem
	fig.savefig(out_path)
	plt.close(fig)
	return out_path


def _normalize_slice_specs(
	slices: tuple[int | ClusterSlice, ...],
) -> tuple[ClusterSlice, ...]:
	normalized = []
	for item in slices:
		if isinstance(item, ClusterSlice):
			normalized.append(item)
		else:
			normalized.append(
				ClusterSlice(
					array_slice_index=int(item),
					voxel_slice_index=int(item),
				),
			)
	return tuple(normalized)


def _slice_title(  # noqa: PLR0913
	*,
	survey_id: str,
	k: int,
	mode: str,
	view: str,
	array_slice_index: int,
	voxel_slice_index: int,
) -> str:
	axis = 'z' if view == 'xy' else 'y'
	if mode == 'token':
		return (
			f'{survey_id} k={k} token {view.upper()} '
			f'voxel-{axis}={voxel_slice_index} token-{axis}={array_slice_index}'
		)
	return (
		f'{survey_id} k={k} {mode} {view.upper()} '
		f'voxel-{axis}={voxel_slice_index}'
	)


def _validate_labels(labels: np.ndarray) -> np.ndarray:
	array = np.asarray(labels)
	if array.ndim != 3:
		msg = f'labels must be 3D; got shape={array.shape!r}'
		raise ValueError(msg)
	if array.dtype.kind not in {'i', 'u'}:
		msg = f'labels must use an integer dtype; got {array.dtype}'
		raise TypeError(msg)
	return array


def _validate_slice_index(labels: np.ndarray, *, view: str, slice_index: int) -> None:
	axis = 2 if view == 'xy' else 1
	if view not in {'xy', 'xz'}:
		msg = f'unknown view: {view!r}'
		raise ValueError(msg)
	if slice_index < 0 or slice_index >= labels.shape[axis]:
		msg = (
			f'{view} slice index out of range: {slice_index}; '
			f'valid=[0, {labels.shape[axis] - 1}]'
		)
		raise ValueError(msg)


def _robust_limits(image: np.ndarray) -> tuple[float | None, float | None]:
	values = np.asarray(image, dtype=np.float32)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return None, None
	vmin, vmax = np.percentile(values, (1.0, 99.0))
	if np.isclose(vmin, vmax):
		return None, None
	return float(vmin), float(vmax)


def _validate_alpha(value: float) -> float:
	alpha = float(value)
	if not 0.0 <= alpha <= 1.0:
		msg = f'alpha must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return alpha


def _plt() -> object:
	return __import__('matplotlib.pyplot', fromlist=['pyplot'])


def _matplotlib_colors() -> object:
	return __import__('matplotlib.colors', fromlist=['colors'])


__all__ = [
	'ClusterSlice',
	'ClusterSliceRequest',
	'save_cluster_comparison_pngs',
	'save_cluster_slice_pngs',
	'stable_cluster_colors',
]
