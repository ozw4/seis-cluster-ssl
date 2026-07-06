from __future__ import annotations

from pathlib import Path

import pytest

from seis_ssl_cluster.config.common import (
	_required_child_mapping,
	_required_mapping,
	_validate_absolute_path,
	_validate_allowed_keys,
	_validate_fraction,
	_validate_non_empty_str,
	_validate_path,
	_validate_positive_finite_number,
	_validate_positive_int,
)


def test_positive_int_validation_accepts_positive_integer() -> None:
	_validate_positive_int({'count': 3}, 'count', prefix='section')


@pytest.mark.parametrize('value', [0, -1, 1.2, '1', None])
def test_positive_int_validation_rejects_invalid_values(value: object) -> None:
	with pytest.raises(
		ValueError,
		match=r'section\.count must be a positive integer',
	):
		_validate_positive_int({'count': value}, 'count', prefix='section')


@pytest.mark.parametrize(
	'validator',
	[_validate_positive_int, _validate_positive_finite_number],
)
def test_numeric_validators_reject_bool_values(validator) -> None:
	with pytest.raises(ValueError, match=r'section\.value'):
		validator({'value': True}, 'value', prefix='section')


def test_positive_finite_number_validation_accepts_float() -> None:
	_validate_positive_finite_number({'value': 1.25}, 'value', prefix='section')


@pytest.mark.parametrize('value', [0.0, -0.1, float('inf'), float('nan'), '1.0'])
def test_positive_finite_number_validation_rejects_invalid_values(
	value: object,
) -> None:
	with pytest.raises(
		ValueError,
		match=r'section\.value must be a finite positive number',
	):
		_validate_positive_finite_number({'value': value}, 'value', prefix='section')


@pytest.mark.parametrize('value', [0.0, 0.5, 1.0])
def test_fraction_validation_accepts_closed_unit_interval(value: float) -> None:
	_validate_fraction({'ratio': value}, 'ratio', prefix='section')


@pytest.mark.parametrize('value', [-0.1, 1.1, float('inf'), True])
def test_fraction_validation_rejects_invalid_values(value: object) -> None:
	with pytest.raises(
		ValueError,
		match=r'section\.ratio must be between 0 and 1',
	):
		_validate_fraction({'ratio': value}, 'ratio', prefix='section')


def test_unknown_key_rejection_reports_dotted_key() -> None:
	with pytest.raises(ValueError, match=r'root\.section\.extra'):
		_validate_allowed_keys(
			{'known': 1, 'extra': 2},
			frozenset({'known'}),
			prefix='root.section',
		)


def test_required_mapping_validation_returns_child_mapping() -> None:
	child = {'value': 1}

	assert _required_mapping({'child': child}, 'child') is child


def test_required_mapping_validation_rejects_missing_or_scalar_child() -> None:
	with pytest.raises(TypeError, match='child must be a mapping'):
		_required_mapping({}, 'child')
	with pytest.raises(TypeError, match='child must be a mapping'):
		_required_mapping({'child': 'value'}, 'child')


def test_required_child_mapping_validation_reports_dotted_key() -> None:
	with pytest.raises(TypeError, match=r'parent\.child must be a mapping'):
		_required_child_mapping({'child': None}, 'child', prefix='parent')


def test_string_validation_accepts_non_empty_string() -> None:
	_validate_non_empty_str({'name': 'example'}, 'name', prefix='section')


def test_string_validation_rejects_empty_string_with_dotted_key() -> None:
	with pytest.raises(
		TypeError,
		match=r'section\.name must be a non-empty string',
	):
		_validate_non_empty_str({'name': ''}, 'name', prefix='section')


def test_path_validation_returns_path() -> None:
	assert (
		_validate_path({'path': '/workspace/example'}, 'path', prefix='section')
		== Path('/workspace/example')
	)


def test_path_validation_rejects_empty_string_with_dotted_key() -> None:
	with pytest.raises(
		TypeError,
		match=r'section\.path must be a non-empty string',
	):
		_validate_path({'path': ''}, 'path', prefix='section')


def test_absolute_path_validation_rejects_relative_path_with_dotted_key() -> None:
	with pytest.raises(
		ValueError,
		match=r'section\.path must be an absolute path',
	):
		_validate_absolute_path({'path': 'relative'}, 'path', prefix='section')
