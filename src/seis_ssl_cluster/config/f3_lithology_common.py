"""Shared F3 lithology config validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: Path | None = None,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return default
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_path(parent: Mapping[str, object], key: str) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'publish.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_str(
	parent: Mapping[str, object],
	key: str,
	*,
	default: str,
	prefix: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_nullable_str(
	parent: Mapping[str, object],
	key: str,
	*,
	default: str | None,
	prefix: str,
) -> str | None:
	value = parent.get(key, default)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string or null; got {value!r}'
		raise TypeError(msg)
	return value


def _string_item(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} entries must be non-empty strings; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{label} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1); got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction < 1.0:
		msg = f'{label} must be in [0, 1); got {value!r}'
		raise ValueError(msg)
	return fraction


def _optional_positive_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0.0:
		msg = f'{label} must be a positive number; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _optional_nonnegative_float(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool) or value < 0.0:
		msg = f'{label} must be a nonnegative number; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _hidden_dims(value: object) -> tuple[int, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'probe.hidden_dims must be a list of positive integers; got {value!r}'
		raise TypeError(msg)
	dims = tuple(_optional_positive_int(item, 'probe.hidden_dims') for item in value)
	if not dims:
		msg = 'probe.hidden_dims must contain at least one layer width'
		raise ValueError(msg)
	return dims


def _required_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float:
	value = parent.get(key)
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{prefix}.{key} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{prefix}.{key} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _required_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_bool_value(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{label} must be boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a list of integer slice indices; got {value!r}'
		raise TypeError(msg)
	items = tuple(value)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integer slice indices; got {value!r}'
		raise TypeError(msg)
	return items


def _percentiles(value: object) -> tuple[float, float]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
	):
		msg = (
			'visualizations.figure.amplitude_clip_percentiles must contain two '
			f'values; got {value!r}'
		)
		raise TypeError(msg)
	low = float(value[0])
	high = float(value[1])
	return (low, high)


def _publish_optional_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		msg = f'publish.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_non_negative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	default: int,
) -> int:
	value = parent.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		msg = f'publish.{key} must be a non-negative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _max_file_size_bytes(parent: Mapping[str, object]) -> int:
	value = parent.get('max_file_size_mb', 10)
	if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
		msg = f'publish.max_file_size_mb must be positive; got {value!r}'
		raise ValueError(msg)
	return int(value * 1024 * 1024)


def _validate_frozen_encoder(model: Mapping[str, object], *, stage: str) -> None:
	if model.get('freeze_encoder') is not True:
		msg = f'model.freeze_encoder must be true for {stage}'
		raise ValueError(msg)


def _validate_artifact_path_not_f3(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, f3_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _validate_artifact_or_f3_source_path(
	path: Path | None,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if path is None:
		return
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if label == 'labels.source_label_segy':
		if not _is_relative_to(path, f3_root):
			msg = f'{label} must be under paths.f3_root ({f3_root}); got {path}'
			raise ValueError(msg)
		return
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True
