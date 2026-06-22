from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.data import (
	load_npy_path_list,
	make_survey_id_from_path,
	resolve_npy_path_list,
)

if TYPE_CHECKING:
	from pathlib import Path


def _write_volume(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
	return path


def test_path_list_ignores_empty_and_comment_lines(tmp_path: Path) -> None:
	path_list = tmp_path / 'paths.txt'
	path_list.write_text(
		'\n# comment\nsurvey_a/base.npy\n\nsurvey_b/base.npy\n',
		encoding='utf-8',
	)

	assert load_npy_path_list(path_list) == [
		'survey_a/base.npy',
		'survey_b/base.npy',
	]


def test_resolve_npy_path_list_preserves_order_and_resolves_relative_paths(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	first = _write_volume(root / 'survey_b' / 'base.npy')
	second = _write_volume(root / 'survey_a' / 'base.npy')
	path_list = tmp_path / 'paths.txt'
	path_list.write_text(
		f'{first}\nsurvey_a/base.npy\n',
		encoding='utf-8',
	)

	assert resolve_npy_path_list(path_list, root) == [first, second]


def test_resolve_npy_path_list_rejects_duplicate_resolved_paths(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	volume = _write_volume(root / 'survey_a' / 'base.npy')
	path_list = tmp_path / 'paths.txt'
	path_list.write_text(f'{volume}\nsurvey_a/base.npy\n', encoding='utf-8')

	with pytest.raises(ValueError, match='duplicate path-list entry'):
		resolve_npy_path_list(path_list, root)


def test_resolve_npy_path_list_rejects_non_npy_and_missing_paths(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	text_path = root / 'survey_a' / 'base.txt'
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text('not npy', encoding='utf-8')
	path_list = tmp_path / 'paths.txt'
	path_list.write_text(str(text_path), encoding='utf-8')

	with pytest.raises(ValueError, match=r'\.npy'):
		resolve_npy_path_list(path_list, root)

	path_list.write_text('missing.npy', encoding='utf-8')
	with pytest.raises(FileNotFoundError, match='does not exist'):
		resolve_npy_path_list(path_list, root)


def test_make_survey_id_from_path_is_deterministic_and_collision_safe(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	first = root / 'survey a' / 'base.npy'
	second = root / 'survey_a' / 'base.npy'

	first_id = make_survey_id_from_path(first, root)
	second_id = make_survey_id_from_path(second, root)

	assert first_id == make_survey_id_from_path(first, root)
	assert first_id != second_id
	assert ' ' not in first_id
