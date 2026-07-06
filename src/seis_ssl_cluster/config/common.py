"""Common primitive validation helpers for config resolvers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral, Real
from pathlib import Path


def _validate_mapping(config: Mapping[str, object]) -> None:
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		labels = [f'{prefix}.{key}' for key in unexpected]
		msg = (
			f'{prefix} key(s) not allowed: {labels!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _validate_required_keys(
	parent: Mapping[str, object],
	keys: frozenset[str],
	*,
	prefix: str,
) -> None:
	for key in sorted(keys):
		_validate_required_key(parent, key, prefix=prefix)


def _validate_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = _validate_path(parent, key, prefix=prefix)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _validate_non_empty_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	return _validate_path(parent, key, prefix=prefix)


def _validate_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _validate_optional_output_path_under_root(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	root: Path,
	root_label: str,
) -> None:
	value = parent.get(key)
	if value is None:
		return
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string or null; got {value!r}'
		raise TypeError(msg)
	_validate_path_under_root(
		Path(value),
		f'{prefix}.{key}',
		root=root,
		root_label=root_label,
	)


def _validate_path_under_root(
	path: Path,
	label: str,
	*,
	root: Path,
	root_label: str,
) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute path; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, root):
		msg = f'{label} must be under {root_label} ({root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _iter_mapping_keys(
	value: object,
	prefix: str = '',
) -> Sequence[tuple[str, str]]:
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		paths: list[tuple[str, str]] = []
		for index, child in enumerate(value):
			path = f'{prefix}[{index}]' if prefix else f'[{index}]'
			paths.extend(_iter_mapping_keys(child, path))
		return paths

	if not isinstance(value, Mapping):
		return ()

	paths: list[tuple[str, str]] = []
	for key, child in value.items():
		if not isinstance(key, str):
			continue
		path = f'{prefix}.{key}' if prefix else key
		paths.append((path, key))
		paths.extend(_iter_mapping_keys(child, path))
	return paths


def _merge_section_defaults(
	config: dict[str, object],
	section: str,
	defaults: Mapping[str, object],
) -> None:
	current = config.get(section)
	if current is None:
		config[section] = deepcopy(dict(defaults))
		return
	if not isinstance(current, dict):
		msg = f'{section} must be a mapping'
		raise TypeError(msg)
	config[section] = {**deepcopy(dict(defaults)), **current}


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_child_mapping(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{prefix}.{key} must be a mapping'
		raise TypeError(msg)
	return value


def _validate_non_empty_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)


def _validate_positive_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three positive integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_nonnegative_int_triplet(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or len(value) != 3
		or not all(_is_int(item) and int(item) >= 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of three nonnegative integers'
		raise ValueError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _validate_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or not value
		or not all(_is_int(item) and int(item) > 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise ValueError(msg)


def _validate_unique_positive_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	_validate_positive_int_list(parent, key, prefix=prefix)
	value = parent.get(key)
	if not isinstance(value, list):
		msg = f'{prefix}.{key} must be a non-empty list of positive integers'
		raise TypeError(msg)
	values = [int(item) for item in value]
	if len(set(values)) != len(values):
		msg = f'{prefix}.{key} must not contain duplicates; got {values!r}'
		raise ValueError(msg)


def _validate_nonnegative_int_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not isinstance(value, list)
		or any(not _is_int(item) or int(item) < 0 for item in value)
	):
		msg = f'{prefix}.{key} must be a list of nonnegative integers'
		raise ValueError(msg)


def _validate_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) <= 0:
		msg = f'{prefix}.{key} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_required_key(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent:
		msg = f'{prefix}.{key} is required'
		raise ValueError(msg)


def _validate_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_int(value) or int(value) < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)


def _validate_optional_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_positive_int(parent, key, prefix=prefix)


def _validate_optional_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key not in parent or parent.get(key) is None:
		return
	_validate_nonnegative_int(parent, key, prefix=prefix)


def _validate_positive_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) <= 0.0:
		msg = f'{prefix}.{key} must be positive; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not _is_number(value) or float(value) < 0.0:
		msg = f'{prefix}.{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)


def _validate_positive_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) <= 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a finite positive number; got {value!r}'
		raise ValueError(msg)


def _validate_nonnegative_finite_number(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be a nonnegative finite number; got {value!r}'
		raise ValueError(msg)


def _validate_optional_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	if key in parent:
		_validate_fraction(parent, key, prefix=prefix)


def _validate_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if (
		not _is_number(value)
		or float(value) < 0.0
		or float(value) > 1.0
		or not math.isfinite(float(value))
	):
		msg = f'{prefix}.{key} must be between 0 and 1; got {value!r}'
		raise ValueError(msg)


def _validate_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> None:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a boolean; got {value!r}'
		raise TypeError(msg)


def _is_int(value: object) -> bool:
	return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
	return isinstance(value, Real) and not isinstance(value, bool)
