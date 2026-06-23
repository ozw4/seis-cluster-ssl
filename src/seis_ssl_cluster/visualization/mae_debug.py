"""Training-time debug renderer for amplitude MAE batches."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DEBUG_VISUALIZATION_COLUMNS,
	MAE_DEBUG_VISUALIZATION_COLUMNS,
)
from seis_ssl_cluster.models.mae.patching import unpatchify_3d
from seis_ssl_cluster.visualization.common import (
	ImagePanel,
	apply_visual_invalid_mask,
	as_numpy,
	display_limits,
	slice_image,
	slice_mask,
	unpatchify_mae_predictions,
	upsample_token_mask_to_voxels,
	validate_positive_int_triple,
)

if TYPE_CHECKING:
	from pathlib import Path


@dataclass(frozen=True)
class MaeDebugVisualizationConfig:
	"""Configuration for amplitude MAE debug PNG rendering."""

	output_dir: Path
	every_steps: int | None = 1000
	every_epochs: int | None = None
	max_samples: int = 1
	xy_slice_index: int | None = None
	xz_slice_y_index: int | None = None
	dpi: int = 160
	clip_percentiles: tuple[float, float] = (1.0, 99.0)
	columns: tuple[str, ...] = DEFAULT_MAE_DEBUG_VISUALIZATION_COLUMNS
	panel_width: float = 2.6
	panel_height: float = 2.4
	invalid_color: str = 'lightgray'


def save_mae_debug_visualization_pngs(  # noqa: PLR0913
	*,
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
	metrics: Mapping[str, float] | None = None,
) -> list[Path]:
	"""Save XY and XZ debug PNGs for selected samples and return paths."""
	_validate_render_inputs(
		patch_size_xyz=patch_size_xyz,
		epoch=epoch,
		global_step=global_step,
		config=config,
	)
	with torch.no_grad():
		target_tensor = _required_tensor(batch, 'target')
		pred_patches = _required_tensor(model_output, 'pred_patches')
		sample_count = min(config.max_samples, int(target_tensor.shape[0]))
		token_grid_shape = _resolve_token_grid_shape(
			model_output.get('token_grid_shape'),
			target_tensor.shape[2:],
			patch_size_xyz,
		)
		created: list[Path] = []
		for sample_index in range(sample_count):
			sample = _sample_arrays(
				batch=batch,
				model_output=model_output,
				pred_patches=pred_patches,
				patch_size_xyz=patch_size_xyz,
				token_grid_shape=token_grid_shape,
				sample_index=sample_index,
			)
			stem = _file_stem(
				epoch=epoch,
				global_step=global_step,
				sample_index=sample_index,
				sample_count=sample_count,
			)
			for view in ('xy', 'xz'):
				out_path = config.output_dir / f'{stem}_{view}.png'
				_save_view_png(
					out_path=out_path,
					view=view,
					sample=sample,
					sample_index=sample_index,
					epoch=epoch,
					global_step=global_step,
					config=config,
					metrics=metrics,
					coords=batch.get('coords'),
				)
				created.append(out_path)
		return created


def _sample_arrays(  # noqa: PLR0913
	*,
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
	pred_patches: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	token_grid_shape: tuple[int, int, int],
	sample_index: int,
) -> dict[str, np.ndarray | None]:
	target = _sample_volume(batch, 'target', sample_index)
	x = _sample_volume(batch, 'x', sample_index)
	if target.shape != x.shape:
		msg = (
			'x and target shapes must match; '
			f'got x={x.shape!r}, target={target.shape!r}'
		)
		raise ValueError(msg)
	if target.ndim != 4 or target.shape[0] != 1:
		msg = (
			'amplitude MAE target must have shape [1, X, Y, Z] per sample; '
			f'got shape={target.shape!r}'
		)
		raise ValueError(msg)
	prediction = as_numpy(
		unpatchify_3d(
			pred_patches[sample_index : sample_index + 1].detach(),
			patch_size_xyz,
			token_grid_shape,
		)[0],
		'prediction',
	)
	if prediction.shape != target.shape:
		msg = (
			'prediction shape must match target shape after unpatchify; '
			f'got prediction={prediction.shape!r}, target={target.shape!r}'
		)
		raise ValueError(msg)

	local_valid_mask = _optional_sample_mask(batch, 'local_valid_mask', sample_index)
	if local_valid_mask is not None:
		_validate_mask_shape(local_valid_mask, target.shape[1:], 'local_valid_mask')

	spatial_mask = _optional_spatial_mask(batch, model_output)
	spatial_mask_voxel = None
	if spatial_mask is not None:
		spatial_mask_voxel = upsample_token_mask_to_voxels(
			spatial_mask[sample_index : sample_index + 1],
			patch_size_xyz=patch_size_xyz,
		)[0]
		_validate_mask_shape(spatial_mask_voxel, target.shape[1:], 'spatial_mask')

	return {
		'x': x[0],
		'target': target[0],
		'prediction': prediction[0],
		'local_valid_mask': local_valid_mask,
		'spatial_mask_voxel': spatial_mask_voxel,
	}


def _save_view_png(  # noqa: PLR0913
	*,
	out_path: Path,
	view: str,
	sample: Mapping[str, np.ndarray | None],
	sample_index: int,
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
	metrics: Mapping[str, float] | None,
	coords: object,
) -> None:
	target = _required_sample_array(sample, 'target')
	slice_index = _resolve_slice_index(view, target.shape, config)
	panels = _build_panels(
		sample=sample,
		view=view,
		slice_index=slice_index,
		columns=config.columns,
	)
	title = _figure_title(
		view=view,
		slice_index=slice_index,
		sample_index=sample_index,
		epoch=epoch,
		global_step=global_step,
		metrics=metrics,
		coords=coords,
	)
	_plot_panels(
		panels,
		title=title,
		out_path=out_path,
		xlabel='x',
		ylabel='y' if view == 'xy' else 'z',
		aspect='equal' if view == 'xy' else 'auto',
		config=config,
	)
	_write_metadata(
		out_path.with_suffix('.json'),
		view=view,
		slice_index=slice_index,
		sample_index=sample_index,
		epoch=epoch,
		global_step=global_step,
		metrics=metrics,
		coords=coords,
		sample=sample,
	)


def _build_panels(
	*,
	sample: Mapping[str, np.ndarray | None],
	view: str,
	slice_index: int,
	columns: Sequence[str],
) -> list[ImagePanel]:
	_validate_columns(columns)
	x = _required_sample_array(sample, 'x')
	target = _required_sample_array(sample, 'target')
	prediction = _required_sample_array(sample, 'prediction')
	local_valid_mask = sample.get('local_valid_mask')
	spatial_mask = sample.get('spatial_mask_voxel')
	valid_slice = slice_mask(local_valid_mask, view=view, slice_index=slice_index)
	spatial_slice = slice_mask(spatial_mask, view=view, slice_index=slice_index)
	panels: list[ImagePanel] = []
	for column in columns:
		if column == 'input':
			panels.append(
				ImagePanel(
					'input',
					slice_image(x, view=view, slice_index=slice_index),
					valid_mask=valid_slice,
				),
			)
		elif column == 'masked_input':
			mask = valid_slice
			if spatial_slice is not None:
				mask = ~spatial_slice if mask is None else mask & ~spatial_slice
			panels.append(
				ImagePanel(
					'masked_input',
					slice_image(x, view=view, slice_index=slice_index),
					valid_mask=mask,
				),
			)
		elif column == 'target':
			panels.append(
				ImagePanel(
					'target',
					slice_image(target, view=view, slice_index=slice_index),
					valid_mask=valid_slice,
				),
			)
		elif column == 'prediction':
			panels.append(
				ImagePanel(
					'prediction',
					slice_image(prediction, view=view, slice_index=slice_index),
					valid_mask=valid_slice,
				),
			)
		elif column == 'abs_error':
			panels.append(
				ImagePanel(
					'abs_error',
					slice_image(
						np.abs(prediction - target),
						view=view,
						slice_index=slice_index,
					),
					valid_mask=valid_slice,
					range_name='error',
				),
			)
		elif column == 'valid_mask':
			if valid_slice is None:
				valid_slice = np.ones_like(
					slice_image(target, view=view, slice_index=slice_index),
					dtype=bool,
				)
			panels.append(
				ImagePanel(
					'valid_mask',
					valid_slice.astype(np.float32),
					range_name='mask',
				),
			)
	return panels


def _plot_panels(  # noqa: PLR0913
	panels: Sequence[ImagePanel],
	*,
	title: str,
	out_path: Path,
	xlabel: str,
	ylabel: str,
	aspect: str,
	config: MaeDebugVisualizationConfig,
) -> None:
	plt = importlib.import_module('matplotlib.pyplot')
	fig = None
	try:
		fig, axes = plt.subplots(
			1,
			len(panels),
			figsize=(config.panel_width * len(panels), config.panel_height),
			squeeze=False,
		)
		amplitude_limits = _shared_panel_limits(
			(panel for panel in panels if panel.range_name == 'amplitude'),
			config.clip_percentiles,
			error=False,
		)
		error_limits = _shared_panel_limits(
			(panel for panel in panels if panel.range_name == 'error'),
			config.clip_percentiles,
			error=True,
		)
		for ax, panel in zip(axes.ravel(), panels, strict=True):
			image = apply_visual_invalid_mask(panel.image, panel.valid_mask)
			if panel.range_name == 'mask':
				vmin, vmax = 0.0, 1.0
			elif panel.range_name == 'error':
				vmin, vmax = error_limits
			else:
				vmin, vmax = amplitude_limits
			im = ax.imshow(
				image,
				origin='upper',
				aspect=aspect,
				cmap=_cmap(panel, config, plt),
				vmin=vmin,
				vmax=vmax,
			)
			ax.set_title(panel.title, fontsize=8)
			ax.set_xlabel(xlabel, fontsize=8)
			ax.set_ylabel(ylabel, fontsize=8)
			ax.tick_params(labelsize=7)
			fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		fig.suptitle(title, fontsize=10)
		fig.tight_layout()
		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=config.dpi, bbox_inches='tight')
	finally:
		if fig is not None:
			plt.close(fig)


def _shared_panel_limits(
	panels: Sequence[ImagePanel] | object,
	clip_percentiles: tuple[float, float],
	*,
	error: bool,
) -> tuple[float | None, float | None]:
	values: list[np.ndarray] = []
	for panel in panels:
		image = apply_visual_invalid_mask(panel.image, panel.valid_mask)
		compressed = np.ma.masked_invalid(image).compressed()
		if compressed.size:
			values.append(compressed)
	if not values:
		return None, None
	return display_limits(
		np.concatenate(values),
		clip_percentiles,
		error=error,
	)


def _cmap(
	panel: ImagePanel,
	config: MaeDebugVisualizationConfig,
	pyplot: object,
) -> object:
	if panel.range_name == 'mask':
		return 'gray'
	name = 'magma' if panel.range_name == 'error' else 'viridis'
	if panel.valid_mask is None or panel.valid_mask.all():
		return name
	cmap = pyplot.get_cmap(name).copy()
	cmap.set_bad(config.invalid_color)
	return cmap


def _write_metadata(  # noqa: PLR0913
	path: Path,
	*,
	view: str,
	slice_index: int,
	sample_index: int,
	epoch: int,
	global_step: int,
	metrics: Mapping[str, float] | None,
	coords: object,
	sample: Mapping[str, np.ndarray | None],
) -> None:
	target = _required_sample_array(sample, 'target')
	prediction = _required_sample_array(sample, 'prediction')
	valid_mask = sample.get('local_valid_mask')
	valid_values = (
		np.ones(target.shape, dtype=bool)
		if valid_mask is None
		else valid_mask.astype(bool, copy=False)
	)
	error = np.abs(prediction - target)
	error_values = error[valid_values & np.isfinite(error)]
	payload = {
		'epoch': int(epoch),
		'global_step': int(global_step),
		'sample_index': int(sample_index),
		'view': view,
		'slice_index': int(slice_index),
		'coords': _json_safe(_sample_coords(coords, sample_index)),
		'survey_id': _json_safe(_coord_value(coords, sample_index, 'survey_id')),
		'local_start_xyz': _json_safe(
			_coord_value(coords, sample_index, 'local_start_xyz'),
		),
		'metrics': _json_safe(dict(metrics or {})),
		'valid_voxels': int(valid_values.sum()),
		'abs_error_mae': (
			None if error_values.size == 0 else float(np.mean(error_values))
		),
	}
	path.write_text(
		f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
		encoding='utf-8',
	)


def _figure_title(  # noqa: PLR0913
	*,
	view: str,
	slice_index: int,
	sample_index: int,
	epoch: int,
	global_step: int,
	metrics: Mapping[str, float] | None,
	coords: object,
) -> str:
	loss_text = ''
	if metrics is not None and 'loss' in metrics:
		loss_text = f' loss={metrics["loss"]:.4g}'
	title = (
		f'Amplitude MAE debug {view.upper()} sample={sample_index} '
		f'epoch={epoch:04d} step={global_step:06d}{loss_text} '
		f'{"z" if view == "xy" else "y"}={slice_index}'
	)
	coord = _sample_coords(coords, sample_index)
	if coord is None:
		return title
	pieces: list[str] = []
	if (survey_id := coord.get('survey_id')) is not None:
		pieces.append(f'survey={survey_id}')
	if (local_start := coord.get('local_start_xyz')) is not None:
		pieces.append(f'local_start_xyz={local_start}')
	if not pieces:
		return title
	return f'{title} | {", ".join(pieces)}'


def _file_stem(
	*,
	epoch: int,
	global_step: int,
	sample_index: int,
	sample_count: int,
) -> str:
	stem = f'epoch_{epoch:04d}_step_{global_step:06d}'
	if sample_count > 1:
		return f'{stem}_sample_{sample_index:02d}'
	return stem


def _resolve_slice_index(
	view: str,
	volume_shape: Sequence[int],
	config: MaeDebugVisualizationConfig,
) -> int:
	_x_size, y_size, z_size = tuple(int(dim) for dim in volume_shape)
	if view == 'xy':
		index = z_size // 2 if config.xy_slice_index is None else config.xy_slice_index
		limit = z_size
		name = 'xy_slice_index'
	elif view == 'xz':
		index = (
			y_size // 2
			if config.xz_slice_y_index is None
			else config.xz_slice_y_index
		)
		limit = y_size
		name = 'xz_slice_y_index'
	else:
		msg = f'unknown view: {view!r}'
		raise ValueError(msg)
	if not 0 <= index < limit:
		msg = f'{name} out of range: {index}, valid=[0, {limit - 1}]'
		raise ValueError(msg)
	return int(index)


def _resolve_token_grid_shape(
	value: object,
	volume_shape: Sequence[int],
	patch_size_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
	px_size, py_size, pz_size = validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	if value is not None:
		if isinstance(value, torch.Tensor):
			value = value.detach().cpu().tolist()
		if isinstance(value, np.ndarray):
			value = value.tolist()
		if not isinstance(value, Sequence):
			msg = f'token_grid_shape must be a sequence; got {type(value).__name__}'
			raise TypeError(msg)
		return validate_positive_int_triple(
			tuple(int(item) for item in value),
			'token_grid_shape',
		)
	x_size, y_size, z_size = tuple(int(dim) for dim in volume_shape)
	if x_size % px_size or y_size % py_size or z_size % pz_size:
		msg = (
			'target volume shape must be divisible by patch_size_xyz when '
			'token_grid_shape is absent; '
			f'got volume_shape={tuple(volume_shape)!r}, '
			f'patch_size_xyz={patch_size_xyz!r}'
		)
		raise ValueError(msg)
	return x_size // px_size, y_size // py_size, z_size // pz_size


def _sample_volume(
	batch: Mapping[str, object],
	key: str,
	sample_index: int,
) -> np.ndarray:
	tensor = _required_tensor(batch, key)
	if tensor.ndim != 5:
		msg = f'{key} must have shape [B, C, X, Y, Z]; got {tuple(tensor.shape)!r}'
		raise ValueError(msg)
	return as_numpy(tensor[sample_index].detach(), key)


def _optional_sample_mask(
	batch: Mapping[str, object],
	key: str,
	sample_index: int,
) -> np.ndarray | None:
	value = batch.get(key)
	if value is None:
		return None
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor; got {type(value).__name__}'
		raise TypeError(msg)
	if value.ndim != 4:
		msg = f'{key} must have shape [B, X, Y, Z]; got {tuple(value.shape)!r}'
		raise ValueError(msg)
	return as_numpy(value[sample_index].detach(), key).astype(bool, copy=False)


def _optional_spatial_mask(
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
) -> torch.Tensor | None:
	value = batch.get('spatial_mask', model_output.get('spatial_mask'))
	if value is None:
		return None
	if not isinstance(value, torch.Tensor):
		msg = f'spatial_mask must be a torch.Tensor; got {type(value).__name__}'
		raise TypeError(msg)
	if value.ndim != 4:
		msg = (
			'spatial_mask must have shape [B, TX, TY, TZ]; '
			f'got {tuple(value.shape)!r}'
		)
		raise ValueError(msg)
	return value.detach()


def _required_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
	value = mapping[key]
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _required_sample_array(
	sample: Mapping[str, np.ndarray | None],
	key: str,
) -> np.ndarray:
	value = sample.get(key)
	if not isinstance(value, np.ndarray):
		msg = f'sample {key} must be present'
		raise TypeError(msg)
	return value


def _validate_render_inputs(
	*,
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
) -> None:
	validate_positive_int_triple(patch_size_xyz, 'patch_size_xyz')
	_validate_nonnegative_int(epoch, 'epoch')
	_validate_nonnegative_int(global_step, 'global_step')
	_validate_positive_int(config.max_samples, 'max_samples')
	_validate_positive_int(config.dpi, 'dpi')
	_validate_positive_float(config.panel_width, 'panel_width')
	_validate_positive_float(config.panel_height, 'panel_height')
	_validate_clip_percentiles(config.clip_percentiles)
	_validate_columns(config.columns)
	_validate_non_empty_string(config.invalid_color, 'invalid_color')


def _validate_mask_shape(
	mask: np.ndarray,
	volume_shape: Sequence[int],
	name: str,
) -> None:
	expected = tuple(int(dim) for dim in volume_shape)
	if tuple(mask.shape) != expected:
		msg = f'{name} shape must be {expected!r}; got {mask.shape!r}'
		raise ValueError(msg)


def _validate_columns(columns: Sequence[str]) -> None:
	if not columns:
		msg = 'visualization columns must not be empty'
		raise ValueError(msg)
	if any(not isinstance(column, str) or not column for column in columns):
		msg = f'visualization columns must be non-empty strings; got {columns!r}'
		raise ValueError(msg)
	if len(set(columns)) != len(columns):
		msg = (
			'visualization columns must not contain duplicates; '
			f'got {list(columns)!r}'
		)
		raise ValueError(msg)
	unknown = sorted(set(columns) - MAE_DEBUG_VISUALIZATION_COLUMNS)
	if unknown:
		msg = f'unknown MAE debug columns: {unknown!r}'
		raise ValueError(msg)


def _validate_clip_percentiles(value: tuple[float, float]) -> None:
	if len(value) != 2:
		msg = f'clip_percentiles must contain two values; got {value!r}'
		raise ValueError(msg)
	low, high = value
	if not 0.0 <= low < high <= 100.0:
		msg = f'clip_percentiles must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_int(value: int, name: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise ValueError(msg)


def _validate_positive_int(value: int, name: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{name} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_positive_float(value: float, name: str) -> None:
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{name} must be a float; got {value!r}'
		raise TypeError(msg)
	if not math.isfinite(float(value)) or float(value) <= 0.0:
		msg = f'{name} must be finite and positive; got {value!r}'
		raise ValueError(msg)


def _validate_non_empty_string(value: str, name: str) -> None:
	if not isinstance(value, str) or not value:
		msg = f'{name} must be a non-empty string; got {value!r}'
		raise TypeError(msg)


def _sample_coords(coords: object, sample_index: int) -> Mapping[str, object] | None:
	if isinstance(coords, Mapping):
		return coords
	if (
		isinstance(coords, Sequence)
		and not isinstance(coords, str)
		and sample_index < len(coords)
		and isinstance(coords[sample_index], Mapping)
	):
		return coords[sample_index]
	return None


def _coord_value(coords: object, sample_index: int, key: str) -> object:
	coord = _sample_coords(coords, sample_index)
	if coord is None:
		return None
	return coord.get(key)


def _json_safe(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _json_safe(child) for key, child in value.items()}
	if isinstance(value, tuple | list):
		return [_json_safe(child) for child in value]
	if isinstance(value, bool | str) or value is None:
		return value
	if isinstance(value, int):
		return int(value)
	if isinstance(value, float):
		return float(value) if math.isfinite(value) else repr(value)
	return repr(value)


__all__ = [
	'MaeDebugVisualizationConfig',
	'apply_visual_invalid_mask',
	'save_mae_debug_visualization_pngs',
	'unpatchify_mae_predictions',
	'upsample_token_mask_to_voxels',
]
