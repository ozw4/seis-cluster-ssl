from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.io.labels import F3ClassInfo
from seis_ssl_cluster.f3.lithology import common_test as subject
from seis_ssl_cluster.f3.lithology.metrics import (
	lithology_metrics_from_confusion_matrix,
)


def test_common_mask_reserves_all_training_candidates_and_validation():
	canonical = np.array([0, 1, 2, 0], dtype=np.uint8)
	valid = np.array([True, True, True, False])
	np.testing.assert_array_equal(
		subject.common_test_mask(valid, canonical), [True, False, False, False]
	)
	with pytest.raises(ValueError, match='split code'):
		subject.common_test_mask(valid, np.array([0, 1, 2, 3]))
	with pytest.raises(ValueError, match='bool'):
		subject.common_test_mask(valid.astype(int), canonical)


def test_confusion_matches_direct_counts_and_chunking():
	labels = np.arange(180, dtype=np.int16).reshape(5, 6, 6) % 6
	pred = (labels + 1) % 6
	mask = np.indices(labels.shape).sum(axis=0) % 3 != 0
	valid = np.ones_like(mask)
	whole = subject.masked_confusion(labels, pred, mask, valid)
	chunked = sum(
		(
			subject.masked_confusion(
				labels[i : i + 1], pred[i : i + 1], mask[i : i + 1], valid[i : i + 1]
			)
			for i in range(5)
		),
		np.zeros((6, 6), dtype=np.int64),
	)
	np.testing.assert_array_equal(whole, chunked)
	assert whole.sum() == mask.sum()
	for true in range(6):
		assert whole[true, (true + 1) % 6] == np.count_nonzero(mask & (labels == true))


@pytest.mark.parametrize('bad', [-1, 6, 20])
def test_unknown_prediction_is_not_silently_removed(bad):
	with pytest.raises(ValueError, match='unknown'):
		subject.masked_confusion(
			np.array([0]), np.array([bad]), np.array([True]), np.array([True])
		)


def test_unknown_truth_and_missing_coverage_fail():
	with pytest.raises(ValueError, match='unknown'):
		subject.masked_confusion(
			np.array([9]), np.array([0]), np.array([True]), np.array([True])
		)
	with pytest.raises(ValueError, match='coverage'):
		subject.masked_confusion(
			np.array([0]), np.array([-1]), np.array([True]), np.array([False])
		)


def _fixture(tmp_path, *, chunk_size=3):
	shape = (8, 8, 8)
	labels = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) % 6
	canonical = np.zeros(shape, dtype=np.uint8)
	canonical[0, 0, 0] = 1
	canonical[0, 0, 1] = 2
	paths = {}
	for key, array in [
		('labels', labels),
		('canonical_grid', canonical),
		('reference_valid_tokens', np.ones((1, 1, 1), dtype=bool)),
	]:
		path = tmp_path / (key + '.npy')
		np.save(path, array)
		paths[key] = {'path': str(path), 'sha256': file_sha256(path)}
	condition = {
		'audit_id': 'none__mae__no_hmm',
		'version': 'none',
		'source': 'mae',
		'head_ks': '',
		'distillation_weight': '',
		'consistency_weight': '',
	}
	cells = []
	for layout in subject.LAYOUTS:
		for size in subject.SIZES:
			root = tmp_path / 'old' / layout / size
			root.mkdir(parents=True)
			pred, valid, cp = root / 'pred.npy', root / 'valid.npy', root / 'best.pt'
			np.save(pred, labels)
			np.save(valid, np.ones(shape, dtype=bool))
			cp.write_bytes(b'frozen-checkpoint-not-loaded-by-test-evaluator')
			evidence = {
				name: {'path': str(path), 'sha256': file_sha256(path)}
				for name, path in [
					('predictions', pred),
					('valid_mask', valid),
					('decoder_checkpoint', cp),
				]
			}
			cells.append(
				subject.TestCell(condition, layout, size, root, pred, valid, evidence)
			)
	config = {
		'output_root': str(tmp_path / 'new'),
		'inputs': paths,
		'volume_shape_xyz': list(shape),
		'expected_test_voxels': 510,
		'chunk_size_x': chunk_size,
		'mask_policy': 'reference_valid_and_canonical_split_zero_v1',
	}
	classes = tuple(F3ClassInfo(i, str(i), (i, i, i)) for i in range(6))
	return subject.TestPlan(
		config,
		classes,
		tuple(cells),
		(tmp_path / 'canonical_grid.npy',),
		{'test_fixture': True},
		'frozen-identity',
	)


def test_end_to_end_publication_resume_and_old_inputs_unchanged(tmp_path):
	plan = _fixture(tmp_path)
	original = {p: file_sha256(p) for p in (tmp_path / 'old').rglob('*') if p.is_file()}
	summary = subject.run_test(plan)
	assert {p.name for p in summary.iterdir()} == subject.SUMMARY_FILES
	assert json.loads((summary / 'summary.json').read_text())['cell_count'] == 15
	with (summary / 'per_layout.csv').open() as stream:
		rows = list(csv.DictReader(stream))
	assert len(rows) == 15
	assert all(float(row['mean_iou']) == 1 for row in rows)
	with (summary / 'comparison.csv').open() as stream:
		rows = list(csv.DictReader(stream))
	assert len(rows) == 3
	assert all(float(row['mean_iou_sample_sd']) == 0 for row in rows)
	for p in (tmp_path / 'new/cells').glob('*/*/*'):
		assert {q.name for q in p.iterdir()} == subject.CELL_FILES
	assert all(file_sha256(p) == sha for p, sha in original.items())
	with pytest.raises(FileExistsError, match='exists'):
		subject.run_test(plan)
	before = {p: file_sha256(p) for p in summary.iterdir()}
	subject.run_test(plan, resume=True)
	assert all(file_sha256(p) == sha for p, sha in before.items())
	with pytest.raises(ValueError, match='identity drift'):
		subject.run_test(replace(plan, identity='changed'), resume=True)


def test_mask_rejects_training_overlap(tmp_path):
	plan = _fixture(tmp_path)
	other = np.zeros((8, 8, 8), dtype=np.uint8)
	other[7, 7, 7] = 1
	path = tmp_path / 'overlap.npy'
	np.save(path, other)
	plan = replace(plan, supervision_grids=(path,))
	with pytest.raises(ValueError, match='overlaps'):
		subject.run_test(plan)
	assert not (tmp_path / 'new/inputs').exists()


def test_prediction_hash_drift_is_rejected_without_publication(tmp_path):
	plan = _fixture(tmp_path)
	root = tmp_path / 'new'
	root.mkdir()
	mask = subject._build_test_inputs(plan, root)  # noqa: SLF001
	cell = plan.cells[0]
	pred = np.load(cell.predictions)
	pred[2, 2, 2] = (pred[2, 2, 2] + 1) % 6
	np.save(cell.predictions, pred)
	output = root / 'bad'
	with pytest.raises(ValueError, match='hash drift'):
		subject.evaluate_cell(plan, cell, mask, output)
	assert not output.exists()


def test_changed_best_checkpoint_is_rejected(tmp_path):
	plan = _fixture(tmp_path)
	root = tmp_path / 'new'
	root.mkdir()
	mask = subject._build_test_inputs(plan, root)  # noqa: SLF001
	cell = plan.cells[0]
	(cell.root / 'best.pt').write_bytes(b'different-best-epoch')
	with pytest.raises(ValueError, match='hash drift'):
		subject.evaluate_cell(plan, cell, mask, root / 'bad')


def test_summary_uses_sample_sd_and_rejects_duplicate_or_missing_layout():
	rows = []
	for index, layout in enumerate(subject.LAYOUTS):
		rows.append(
			dict(
				condition_id='test',
				version='v1',
				source='mae',
				head_ks='6',
				distillation_weight='0.2',
				consistency_weight='0',
				data_size='small',
				layout_id=layout,
				**dict.fromkeys(
					['mean_iou', 'macro_f1', 'accuracy', 'balanced_accuracy'],
					float(index),
				),
			)
		)
	result = subject.aggregate_rows(rows)[0]
	assert result['mean_iou_mean'] == 2
	assert result['mean_iou_sample_sd'] == pytest.approx(np.std(range(5), ddof=1))
	for bad in [rows[:-1], [*rows[:-1], rows[0]]]:
		with pytest.raises(ValueError, match='five distinct'):
			subject.aggregate_rows(bad)


def test_absent_classes_retain_fixed_six_class_policy():
	classes = tuple(F3ClassInfo(i, str(i), (0, 0, 0)) for i in range(6))
	matrix = np.zeros((6, 6), dtype=np.int64)
	matrix[0, 0] = 10
	metrics = lithology_metrics_from_confusion_matrix(matrix, classes)
	assert metrics['mean_iou'] == pytest.approx(1 / 6)
	assert metrics['macro_f1'] == pytest.approx(1 / 6)


def test_inventory_scope_is_exactly_23_existing_conditions():
	expected = subject._expected_conditions()  # noqa: SLF001
	assert len(expected) == 23
	assert ('v2', 'mae', '4|6', '0.2', '0') in expected
	assert ('v2', 'local_bt', '4|6', '0.2', '0') not in expected


def test_v3_scope_retains_all_and_only_completed_conditions():
	expected = subject._expected_conditions(subject.V3_NAMESPACE)  # noqa: SLF001
	assert len(expected) == 9
	assert ('v3', 'mae', '6|8|10', '0.2', '0.1', '1e-4', '0.5', '0.5') in expected
	assert (
		'v3',
		'local_bt',
		'6|8|10',
		'0.2',
		'0.1',
		'1e-4',
		'0.5',
		'0.5',
	) not in expected
	config = {
		'experiment_id': subject.V3_NAMESPACE,
		'expected_conditions': 9,
		'expected_cells': 135,
		'inputs': {'reference_test_mask': {}},
	}
	keys, actual = subject._scope(config)  # noqa: SLF001
	assert actual == expected
	assert keys == subject.CONDITION_KEYS + subject.V3_KEYS
	with pytest.raises(ValueError, match='reference test mask'):
		subject._scope(dict(config, inputs={}))  # noqa: SLF001
	with pytest.raises(ValueError, match='scope/count'):
		subject._scope(dict(config, expected_cells=345))  # noqa: SLF001
	with pytest.raises(ValueError, match='unsupported'):
		subject._scope(dict(config, experiment_id='other'))  # noqa: SLF001


def test_v3_publication_keeps_lr_and_branch_weights(tmp_path):
	plan = _fixture(tmp_path)
	condition = dict(
		plan.cells[0].condition,
		audit_id='v3__mae__masklr3e-5__mw0.25',
		version='v3',
		head_ks='6|8|10',
		distillation_weight='0.2',
		consistency_weight='0.1',
		mask_token_lr='3e-5',  # noqa: S106 - optimizer learning rate, not a password
		masked_prototype_weight='0.25',
		visible_prototype_weight='0.75',
	)
	cells = tuple(replace(c, condition=condition) for c in plan.cells)
	summary = subject.run_test(replace(plan, cells=cells))
	assert {p.name for p in summary.iterdir()} == subject.SUMMARY_FILES
	for filename, count in [('comparison.csv', 3), ('per_layout.csv', 15)]:
		with (summary / filename).open() as stream:
			rows = list(csv.DictReader(stream))
		assert len(rows) == count
		assert all(all(r[k] == condition[k] for k in subject.V3_KEYS) for r in rows)


def test_v3_rejects_a_different_reference_test_mask(tmp_path):
	plan = _fixture(tmp_path)
	config = dict(plan.config, inputs=dict(plan.config['inputs']))
	config['inputs']['reference_test_mask'] = {'sha256': 'different-existing-mask'}
	with pytest.raises(ValueError, match='previously agreed reference'):
		subject.run_test(replace(plan, config=config))
	assert not (tmp_path / 'new/inputs').exists()


def test_v3_recipe_must_match_declared_mask_parameters(tmp_path):
	config = {
		'head': {'ks': [6, 8, 10]},
		'loss': {'distillation_weight': 0.2, 'consistency_weight': 0.1},
		'spatial_context': {
			'column_fraction': 0.1,
			'masked_prototype_weight': 0.5,
			'visible_prototype_weight': 0.5,
		},
		'train': {'epochs': 25, 'seed': 42, 'lr': 1e-5, 'encoder_lr': 1e-5},
	}
	path = tmp_path / 'training.yaml'
	path.write_text(json.dumps(config))
	condition = {
		'pretraining_config_workspace_relative': path.name,
		'pretraining_config_sha256': file_sha256(path),
		'mask_token_lr': '1e-5',
		'masked_prototype_weight': '0.5',
		'visible_prototype_weight': '0.5',
	}
	subject._inspect_v3_parameters(condition, tmp_path)  # noqa: SLF001
	with pytest.raises(ValueError, match='pinned training recipe'):
		subject._inspect_v3_parameters(  # noqa: SLF001
			dict(condition, masked_prototype_weight='0.25'), tmp_path
		)
	path.write_text('{}')
	with pytest.raises(ValueError, match='hash drift'):
		subject._inspect_v3_parameters(condition, tmp_path)  # noqa: SLF001


@pytest.mark.parametrize(
	('name', 'condition_count', 'cell_count'),
	[('hmm_f3_common_test_v1', 23, 345), ('hmm_f3_common_test_v3_v1', 9, 135)],
)
def test_curated_common_test_reports_have_exact_outputs_and_paired_statistics(
	name, condition_count, cell_count
):
	report = Path(__file__).resolve().parents[2] / 'reports' / name
	assert {path.name for path in report.iterdir()} == {
		'README.md',
		'comparison.csv',
		'per_layout.csv',
	}
	with (report / 'comparison.csv').open(newline='') as stream:
		groups = list(csv.DictReader(stream))
	with (report / 'per_layout.csv').open(newline='') as stream:
		cells = list(csv.DictReader(stream))
	assert len(groups) == condition_count * 3
	assert len(cells) == cell_count
	assert len({row['condition_id'] for row in cells}) == condition_count
	assert (
		len({(r['condition_id'], r['layout_id'], r['data_size']) for r in cells})
		== cell_count
	)
	for group in groups:
		selected = [
			row
			for row in cells
			if (row['condition_id'], row['data_size'])
			== (group['condition_id'], group['data_size'])
		]
		assert {row['layout_id'] for row in selected} == {
			f'layout_{i:03d}' for i in range(5)
		}
		assert all(int(row['evaluation_voxel_count']) == 113979624 for row in selected)
		for metric in ('mean_iou', 'macro_f1', 'accuracy', 'balanced_accuracy'):
			values = [float(row[metric]) for row in selected]
			assert float(group[metric + '_mean']) == pytest.approx(mean(values))
			assert float(group[metric + '_sample_sd']) == pytest.approx(stdev(values))
