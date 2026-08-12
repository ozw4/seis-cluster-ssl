# ruff: noqa: CPY001, PT018, TC003

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import seis_ssl_cluster.parihaka.channel_data as data
from proc.seis_ssl_cluster import prepare_parihaka_channel_labels as prepare_cli


@pytest.fixture
def small_geometry(monkeypatch: pytest.MonkeyPatch) -> tuple[int, int, int]:
	"""Use a bounded synthetic source while retaining the fixed contract logic."""
	shape = (3, 4, 5)
	monkeypatch.setattr(data, 'SOURCE_SHAPE_ZXY', shape)
	monkeypatch.setattr(data, 'PREPARED_SHAPE_XYZ', (4, 5, 3))
	return shape


def _source(shape: tuple[int, int, int]) -> np.ndarray:
	values = np.resize(np.asarray(data.CLASS_IDS, dtype=np.int8), np.prod(shape))
	return values.reshape(shape)


def _label_config(tmp_path: Path) -> data.ChannelLabelConfig:
	return data.ChannelLabelConfig(
		source_npz=tmp_path / 'labels.npz',
		output_labels=tmp_path / 'out' / 'parihaka_labels.npy',
		output_metadata=tmp_path / 'out' / 'parihaka_labels_metadata.json',
		chunk_size_z=2,
	)


def test_label_transpose_and_coordinate_mapping(
	tmp_path: Path, small_geometry: tuple[int, int, int]
) -> None:
	config = _label_config(tmp_path)
	source = _source(small_geometry)
	np.savez(config.source_npz, labels=source)
	data.prepare_channel_labels(config)
	prepared = np.load(config.output_labels, allow_pickle=False)
	assert prepared.flags.c_contiguous
	assert prepared.dtype == np.int8
	assert np.array_equal(prepared, source.transpose(1, 2, 0))
	for z, x, y in ((0, 0, 0), (2, 3, 4), (1, 2, 3)):
		assert prepared[x, y, z] == source[z, x, y]


@pytest.mark.parametrize('failure', ['key', 'shape', 'dtype', 'classes'])
def test_invalid_source_contract_is_rejected(
	tmp_path: Path,
	small_geometry: tuple[int, int, int],
	failure: str,
) -> None:
	config = _label_config(tmp_path)
	array = _source(small_geometry)
	key = 'labels'
	if failure == 'key':
		key = 'wrong'
	elif failure == 'shape':
		array = array[:-1]
	elif failure == 'dtype':
		array = array.astype(np.int16)
	elif failure == 'classes':
		array[array == 6] = 5
	np.savez(config.source_npz, **{key: array})
	with pytest.raises((ValueError, TypeError)):
		data.inspect_source_labels(config)


def test_size_prefixes_are_nested(tmp_path: Path) -> None:
	path = _write_layout(tmp_path)
	layouts = data.load_channel_layouts(path, (30, 30, 8))
	small = data.selected_training_lines(layouts, 'layout_000', 'small')
	medium = data.selected_training_lines(layouts, 'layout_000', 'medium')
	large = data.selected_training_lines(layouts, 'layout_000', 'large')
	assert small.inline == large.inline[:1]
	assert medium.inline == large.inline[:2]
	assert small.crossline == large.crossline[:1]
	assert medium.crossline == large.crossline[:2]
	assert set(small.inline) < set(medium.inline) < set(large.inline)


def test_inline_crossline_intersection_is_counted_once() -> None:
	mask = data.split_mask_for_crop(
		shape=(3, 4, 2),
		start_xyz=(0, 0, 0),
		train=data.SectionLines((1,), (2,)),
		validation=data.SectionLines((8,), (8,)),
		test=data.SectionLines((9,), (9,)),
		split='train',
	)
	assert np.count_nonzero(mask) == 4 * 2 + 3 * 2 - 2


def test_split_priority_is_test_then_validation_then_train() -> None:
	lines = data.SectionLines((1,), (1,))
	validation = data.SectionLines((2,), (2,))
	test = data.SectionLines((3,), (3,))
	kwargs = {
		'shape': (5, 5, 1),
		'start_xyz': (0, 0, 0),
		'train': lines,
		'validation': validation,
		'test': test,
	}
	train = data.split_mask_for_crop(**kwargs, split='train')
	valid = data.split_mask_for_crop(**kwargs, split='validation')
	test_mask = data.split_mask_for_crop(**kwargs, split='test')
	assert test_mask[3, 2, 0]
	assert test_mask[1, 3, 0] and not train[1, 3, 0]
	assert valid[2, 1, 0] and not train[2, 1, 0]
	assert not np.any(train & valid)
	assert not np.any(train & test_mask)
	assert not np.any(valid & test_mask)


def test_training_and_held_out_same_orientation_overlap_is_rejected(
	tmp_path: Path,
) -> None:
	path = _write_layout(tmp_path)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	payload['layouts']['layout_003']['inline'][2] = payload['validation']['inline'][0]
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='training and held-out inline'):
		data.load_channel_layouts(path, (30, 30, 8))


@pytest.mark.parametrize(
	('data_size', 'inline', 'crossline'),
	[
		('small', [1, 10, 15, 20], [1, 10, 15, 20]),
		('medium', [6, 1, 15, 20], [6, 1, 15, 20]),
		('large', [16, 11, 6, 1], [16, 11, 6, 1]),
	],
)
def test_training_section_sets_must_be_unique_for_every_size(
	tmp_path: Path,
	data_size: str,
	inline: list[int],
	crossline: list[int],
) -> None:
	path = _write_layout(tmp_path)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	payload['layouts']['layout_004'] = {
		'inline': inline,
		'crossline': crossline,
	}
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises(ValueError, match=rf'{data_size} training section sets'):
		data.load_channel_layouts(path, (30, 30, 8))


def test_prepare_cli_dry_run_writes_nothing(
	tmp_path: Path,
	small_geometry: tuple[int, int, int],
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	config = _label_config(tmp_path)
	np.savez(config.source_npz, labels=_source(small_geometry))
	config_path = tmp_path / 'prepare.yaml'
	config_path.write_text(
		yaml.safe_dump(
			{
				'inputs': {'labels_npz': str(config.source_npz), 'array_key': 'labels'},
				'outputs': {
					'labels_npy': str(config.output_labels),
					'metadata_json': str(config.output_metadata),
				},
				'conversion': {'chunk_size_z': 2},
			}
		),
		encoding='utf-8',
	)
	monkeypatch.setattr(
		sys, 'argv', ['prepare', '--config', str(config_path), '--dry-run']
	)
	prepare_cli.main()
	assert not config.output_labels.exists()
	assert not config.output_metadata.exists()


def _write_layout(tmp_path: Path) -> Path:
	payload = {
		'axis_mapping': {'inline': 'x', 'crossline': 'y'},
		'validation': {'inline': [28], 'crossline': [28]},
		'test': {'inline': [29], 'crossline': [29]},
		'layouts': {
			layout_id: {
				'inline': [1 + index, 6 + index, 11 + index, 16 + index],
				'crossline': [1 + index, 6 + index, 11 + index, 16 + index],
			}
			for index, layout_id in enumerate(data.LAYOUT_IDS)
		},
	}
	path = tmp_path / 'layouts.yaml'
	path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
	return path
