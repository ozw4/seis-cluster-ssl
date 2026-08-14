# ruff: noqa: PT018, TC003

from __future__ import annotations

import json
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
	metadata = json.loads(config.output_metadata.read_text(encoding='utf-8'))
	identity = metadata['prepared_label_identity']
	assert identity == data.inspect_prepared_label_identity(
		config.output_labels, config.output_metadata
	)
	assert identity['source_npz_path'] == str(config.source_npz)
	assert identity['source_key'] == 'labels'
	assert identity['shape'] == list(prepared.shape)
	assert identity['dtype'] == 'int8'
	assert identity['class_definition'] == {
		'positive_class_id': 5,
		'negative_class_ids': [1, 2, 3, 4, 6],
	}


def test_prepared_label_identity_rejects_replaced_labels_npy(
	tmp_path: Path, small_geometry: tuple[int, int, int]
) -> None:
	config = _label_config(tmp_path)
	source = _source(small_geometry)
	np.savez(config.source_npz, labels=source)
	data.prepare_channel_labels(config)
	np.save(config.output_labels, source.transpose(1, 2, 0)[::-1])
	with pytest.raises(ValueError, match='SHA-256 does not match'):
		data.inspect_prepared_label_identity(
			config.output_labels, config.output_metadata
		)


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


def test_axis_mapping_is_fixed_and_not_a_layout_setting(tmp_path: Path) -> None:
	path = _write_layout(tmp_path)
	data.load_channel_layouts(path, (30, 30, 8))
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	payload['axis_mapping'] = {'inline': 'x', 'crossline': 'y'}
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises(
		ValueError, match='inline is fixed to X and crossline is fixed to Y'
	):
		data.load_channel_layouts(path, (30, 30, 8))


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		(lambda value: value.pop('training_selection'), 'must contain exactly'),
		(
			lambda value: value['training_selection'].__setitem__('unknown', 1),
			'training_selection must contain exactly',
		),
		(
			lambda value: value['training_selection'][
				'target_train_voxel_counts'
			].__setitem__('small', 0),
			'positive integer',
		),
		(
			lambda value: value['training_selection'].__setitem__(
				'allowed_relative_error', 0.2
			),
			'finite and in',
		),
		(
			lambda value: value['training_selection'][
				'target_train_voxel_counts'
			].update({'small': 300, 'medium': 200}),
			'small < medium < large',
		),
	],
)
def test_layout_training_selection_schema_is_strict(
	tmp_path: Path, mutation: object, message: str
) -> None:
	path = _write_layout(tmp_path)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	assert callable(mutation)
	mutation(payload)
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises((TypeError, ValueError), match=message):
		data.load_channel_layouts(path, (30, 30, 8))


def test_training_selection_is_deterministic_nested_and_binary() -> None:
	shape = (30, 30, 2)
	layout_lines = {
		layout_id: data.SectionLines(
			tuple(1 + index + 5 * layout_index for index in range(4)),
			tuple(1 + index + 5 * layout_index for index in range(4)),
		)
		for layout_index, layout_id in enumerate(data.LAYOUT_IDS)
	}
	layouts = data.ChannelLayouts(
		training_selection=data.ChannelTrainingSelection(
			semantics=data.CHANNEL_SELECTION_SEMANTICS,
			allowed_relative_error=0.05,
			target_train_voxel_counts={'small': 20, 'medium': 40, 'large': 80},
		),
		validation=data.SectionLines((28,), (28,)),
		layouts=layout_lines,
	)
	labels = np.ones(shape, dtype=np.int8)
	labels[:, :, ::2] = data.CHANNEL_CLASS_ID
	valid = np.ones(shape, dtype=np.bool_)
	first = data.select_channel_training(
		layouts, 'layout_000', valid, labels, (1, 1, 1)
	)
	np.random.seed(123)  # noqa: NPY002
	_ = np.random.random(100)  # noqa: NPY002
	second = data.select_channel_training(
		layouts, 'layout_000', valid, labels, (1, 1, 1)
	)
	assert first == second
	selected = [set(item.selected_token_xyz) for item in first]
	assert selected[0] < selected[1] < selected[2]
	assert [item.actual_train_voxel_count for item in first] == [20, 40, 80]
	assert all(
		sum(item.per_line_contributions.values())
		== item.actual_train_voxel_count
		for item in first
	)
	assert all(all(value > 0 for value in item.class_counts) for item in first)
	assert all(
		item.selected_token_xyz_sha256
		== data.selected_token_xyz_sha256(item.selected_token_xyz)
		for item in first
	)


def test_last_token_is_added_only_when_it_is_closer_to_target() -> None:
	shape = (30, 30, 2)
	labels = np.ones(shape, dtype=np.int8)
	labels[:, :, ::2] = data.CHANNEL_CLASS_ID
	valid = np.ones((15, 15, 2), dtype=np.bool_)
	line = data.SectionLines((1, 2, 3, 4), (1, 2, 3, 4))

	def result(target: int) -> data.ChannelSelectionResult:
		layouts = data.ChannelLayouts(
			training_selection=data.ChannelTrainingSelection(
				data.CHANNEL_SELECTION_SEMANTICS,
				0.1,
				{'small': target, 'medium': 80, 'large': 160},
			),
			validation=data.SectionLines((28,), (28,)),
			layouts=dict.fromkeys(data.LAYOUT_IDS, line),
		)
		return data.select_channel_training(
			layouts, 'layout_000', valid, labels, (2, 2, 1)
		)[0]

	assert result(37).actual_train_voxel_count == 36
	assert result(38).actual_train_voxel_count == 39


def test_inline_crossline_intersection_is_counted_once() -> None:
	mask = data.split_mask_for_crop(
		shape=(3, 4, 2),
		start_xyz=(0, 0, 0),
		train=data.SectionLines((1,), (2,)),
		validation=data.SectionLines((8,), (8,)),
		reserved_training=data.SectionLines((1,), (2,)),
		split='train',
	)
	assert np.count_nonzero(mask) == 4 * 2 + 3 * 2 - 2


def test_split_masks_use_common_voxel_complement_and_validation_priority() -> None:
	validation = data.SectionLines((2,), (2,))
	reserved = data.SectionLines((1, 3), (1, 3))
	kwargs = {
		'shape': (5, 5, 1),
		'start_xyz': (0, 0, 0),
		'train': data.SectionLines((1, 2), (1, 2)),
		'validation': validation,
		'reserved_training': reserved,
	}
	train = data.split_mask_for_crop(**kwargs, split='train')
	valid = data.split_mask_for_crop(**kwargs, split='validation')
	test_mask = data.split_mask_for_crop(**kwargs, split='test')
	assert not test_mask[3, 4, 0]
	assert not test_mask[4, 3, 0]
	assert not test_mask[2, 4, 0]
	assert test_mask[4, 4, 0]
	assert valid[2, 1, 0] and not train[2, 1, 0]
	assert valid[1, 2, 0] and not train[1, 2, 0]
	assert not np.any(train & valid)
	assert not np.any(train & test_mask)
	assert not np.any(valid & test_mask)


def test_common_reserved_lines_are_sorted_union_and_test_is_job_invariant(
	tmp_path: Path,
) -> None:
	layouts = data.load_channel_layouts(_write_layout(tmp_path), (30, 30, 8))
	reserved = data.common_reserved_training_lines(layouts)
	assert reserved.inline == tuple(range(1, 21))
	assert reserved.crossline == tuple(range(1, 21))
	masks = [
		data.split_mask_for_crop(
			shape=(30, 30, 1),
			start_xyz=(0, 0, 0),
			train=data.selected_training_lines(layouts, layout_id, data_size),
			validation=layouts.validation,
			reserved_training=reserved,
			split='test',
		)
		for layout_id in data.LAYOUT_IDS
		for data_size in data.DATA_SIZE_PREFIX
	]
	assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
	assert not masks[0][4, 29, 0]
	assert not masks[0][29, 4, 0]
	assert not masks[0][28, 29, 0]
	assert masks[0][29, 29, 0]


def test_training_and_held_out_same_orientation_overlap_is_rejected(
	tmp_path: Path,
) -> None:
	path = _write_layout(tmp_path)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	payload['layouts']['layout_003']['inline'][2] = payload['validation']['inline'][0]
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='training and validation inline'):
		data.load_channel_layouts(path, (30, 30, 8))


def test_explicit_test_lines_schema_is_rejected(tmp_path: Path) -> None:
	path = _write_layout(tmp_path)
	payload = yaml.safe_load(path.read_text(encoding='utf-8'))
	payload['test'] = {'inline': [29], 'crossline': [29]}
	path.write_text(yaml.safe_dump(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='must contain exactly'):
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
		'training_selection': {
			'semantics': data.CHANNEL_SELECTION_SEMANTICS,
			'allowed_relative_error': 0.05,
			'target_train_voxel_counts': {
				'small': 100,
				'medium': 200,
				'large': 400,
			},
		},
		'validation': {'inline': [28], 'crossline': [28]},
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


def test_inspection_config_reuses_preparation_outputs(tmp_path: Path) -> None:
	labels = tmp_path / 'parihaka_labels.npy'
	section_counts = tmp_path / 'parihaka_channel_section_counts.csv'

	config = data.channel_inspection_config_from_mapping(
		{
			'inputs': {
				'labels_npz': str(tmp_path / 'parihaka_labels_train.npz'),
				'array_key': 'labels',
			},
			'outputs': {
				'labels_npy': str(labels),
				'metadata_json': str(tmp_path / 'parihaka_labels_metadata.json'),
				'section_counts_csv': str(section_counts),
			},
			'conversion': {'chunk_size_z': 8},
		}
	)

	assert config.labels == labels
	assert config.output_csv == section_counts
