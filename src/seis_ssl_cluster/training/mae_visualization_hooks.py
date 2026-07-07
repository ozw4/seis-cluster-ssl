"""Debug visualization hooks for amplitude MAE training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from seis_ssl_cluster.config.schema import (
	DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS,
	MAE_DEBUG_VISUALIZATION_COLUMNS,
	MAE_DEBUG_VISUALIZATION_KEYS,
)
from seis_ssl_cluster.training.mae_config_completion import (
	_bool_config,
	_float_pair_config,
	_int_config,
	_optional_int_config_with_default,
	_optional_nonnegative_int_config_with_default,
	_positive_float_config,
	_positive_int_value,
	_reject_unknown_runtime_keys,
	_str_config_with_default,
	_string_tuple_config,
	_validate_runtime_output_path_under_root,
)
from seis_ssl_cluster.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
)


def _mae_debug_visualization_config(
	config: Mapping[str, object],
	output_root: Path,
) -> MaeDebugVisualizationConfig | None:
	visualization = config.get('visualization')
	if visualization is None:
		return None
	if not isinstance(visualization, Mapping):
		msg = f'visualization must be a mapping; got {visualization!r}'
		raise TypeError(msg)
	_reject_unknown_runtime_keys(
		visualization,
		frozenset({'mae_debug'}),
		prefix='visualization',
	)
	mae_debug = visualization.get('mae_debug')
	if mae_debug is None:
		return None
	if not isinstance(mae_debug, Mapping):
		msg = f'visualization.mae_debug must be a mapping; got {mae_debug!r}'
		raise TypeError(msg)
	_reject_unknown_runtime_keys(
		mae_debug,
		MAE_DEBUG_VISUALIZATION_KEYS,
		prefix='visualization.mae_debug',
	)
	defaults = DEFAULT_MAE_DEBUG_VISUALIZATION_OPTIONS
	enabled = _bool_config(
		mae_debug,
		'enabled',
		default=bool(defaults['enabled']),
	)

	output_dir_value = mae_debug.get('output_dir')
	if output_dir_value is None:
		output_dir = output_root / 'visualizations' / 'mae_debug'
	elif isinstance(output_dir_value, str) and output_dir_value:
		output_dir = Path(output_dir_value)
		_validate_runtime_output_path_under_root(
			output_dir,
			'visualization.mae_debug.output_dir',
			root=output_root,
			root_label='paths.output_root',
		)
	else:
		msg = (
			'visualization.mae_debug.output_dir must be a non-empty string or null; '
			f'got {output_dir_value!r}'
		)
		raise TypeError(msg)

	resolved = MaeDebugVisualizationConfig(
		output_dir=output_dir,
		every_steps=_optional_int_config_with_default(
			mae_debug,
			'every_steps',
			default=defaults['every_steps'],
		),
		every_epochs=_optional_int_config_with_default(
			mae_debug,
			'every_epochs',
			default=defaults['every_epochs'],
		),
		max_samples=_int_config(mae_debug, 'max_samples', int(defaults['max_samples'])),
		xy_slice_index=_optional_nonnegative_int_config_with_default(
			mae_debug,
			'xy_slice_index',
			default=defaults['xy_slice_index'],
		),
		xz_slice_y_index=_optional_nonnegative_int_config_with_default(
			mae_debug,
			'xz_slice_y_index',
			default=defaults['xz_slice_y_index'],
		),
		dpi=_int_config(mae_debug, 'dpi', int(defaults['dpi'])),
		clip_percentiles=_float_pair_config(
			mae_debug,
			'clip_percentiles',
			default=cast('tuple[float, float]', defaults['clip_percentiles']),
		),
		columns=_string_tuple_config(
			mae_debug,
			'columns',
			default=cast('tuple[str, ...]', defaults['columns']),
		),
		panel_width=_positive_float_config(
			mae_debug,
			'panel_width',
			float(defaults['panel_width']),
		),
		panel_height=_positive_float_config(
			mae_debug,
			'panel_height',
			float(defaults['panel_height']),
		),
		invalid_color=_str_config_with_default(
			mae_debug,
			'invalid_color',
			str(defaults['invalid_color']),
		),
	)
	_validate_mae_debug_columns(resolved.columns)
	if not enabled:
		return None
	if resolved.every_steps is None and resolved.every_epochs is None:
		msg = (
			'visualization.mae_debug requires every_steps or every_epochs '
			'when enabled is true'
		)
		raise ValueError(msg)
	return resolved


def _mae_debug_epoch_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	epoch: int,
	already_triggered: bool,
) -> bool:
	if config.every_epochs is None or already_triggered:
		return False
	interval = _positive_int_value(
		config.every_epochs,
		'visualization.mae_debug.every_epochs',
	)
	return epoch % interval == 0


def _mae_debug_step_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	global_step: int,
) -> bool:
	if config.every_steps is None:
		return False
	interval = _positive_int_value(
		config.every_steps,
		'visualization.mae_debug.every_steps',
	)
	return global_step % interval == 0


def _save_mae_debug_visualization(  # noqa: PLR0913
	*,
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
	metrics: Mapping[str, float],
	target_normalization_config: object = None,
) -> None:
	save_mae_debug_visualization_pngs(
		batch=batch,
		model_output=model_output,
		patch_size_xyz=patch_size_xyz,
		epoch=epoch,
		global_step=global_step,
		config=config,
		metrics=metrics,
		target_normalization_config=target_normalization_config,
	)


def _validate_mae_debug_columns(columns: Sequence[str]) -> None:
	if not columns:
		msg = 'visualization.mae_debug.columns must not be empty'
		raise ValueError(msg)
	if len(set(columns)) != len(columns):
		msg = (
			'visualization.mae_debug.columns must not contain duplicates; '
			f'got {list(columns)!r}'
		)
		raise ValueError(msg)
	unknown = sorted(set(columns) - MAE_DEBUG_VISUALIZATION_COLUMNS)
	if unknown:
		msg = (
			'visualization.mae_debug.columns contains unsupported column(s): '
			f'{unknown!r}'
		)
		raise ValueError(msg)


__all__ = [
	'_mae_debug_epoch_triggered',
	'_mae_debug_step_triggered',
	'_mae_debug_visualization_config',
	'_save_mae_debug_visualization',
]
