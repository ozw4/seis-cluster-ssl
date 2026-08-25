from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
	from types import ModuleType

SCRIPT = Path(
	'experiments/parihaka/facies_benchmark_v1/'
	'36_channel_hmm_transition_balance_v1/scripts/summarize_final_test.py'
)
MODEL = 'mae_hmm_k6_persist003'
VALUES = (0.31, 0.45, 0.37, 0.52, 0.40)


def _load_summary_module() -> ModuleType:
	spec = importlib.util.spec_from_file_location(
		'parihaka_transition_balance_final_summary',
		SCRIPT,
	)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


SUMMARY = _load_summary_module()


def _write_metrics(
	runs_root: Path,
	*,
	layouts: tuple[str, ...] = SUMMARY.LAYOUTS,
	overrides: dict[str, Any] | None = None,
) -> None:
	for index, layout in enumerate(layouts):
		payload = {
			'model': MODEL,
			'layout_id': layout,
			'data_size': 'medium',
			'evaluation_mode': 'validation_and_test',
			'test': {'channel_iou': VALUES[index]},
		}
		if overrides:
			payload.update(overrides)
		path = (
			runs_root
			/ f'model={MODEL}'
			/ f'layout={layout}'
			/ 'size=medium/metrics.json'
		)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(payload) + '\n', encoding='utf-8')


def test_known_layout_values_write_expected_summary_files(tmp_path: Path) -> None:
	runs_root = tmp_path / 'final_runs'
	report_root = tmp_path / 'report'
	_write_metrics(runs_root)

	result = SUMMARY.summarize_final_test(runs_root, MODEL, report_root)

	assert result == {
		'model': MODEL,
		'metric': 'test.channel_iou',
		'data_size': 'medium',
		'layout_count': 5,
		'layouts': dict(zip(SUMMARY.LAYOUTS, VALUES, strict=True)),
		'mean': pytest.approx(statistics.mean(VALUES)),
		'median': pytest.approx(statistics.median(VALUES)),
		'sample_standard_deviation': pytest.approx(statistics.stdev(VALUES)),
	}
	with (report_root / 'final_test_layouts.csv').open(
		encoding='utf-8',
		newline='',
	) as stream:
		rows = list(csv.DictReader(stream))
	assert [row['layout_id'] for row in rows] == list(SUMMARY.LAYOUTS)
	assert [float(row['test_channel_iou']) for row in rows] == list(VALUES)
	assert all(row['model'] == MODEL for row in rows)
	assert all(row['data_size'] == 'medium' for row in rows)
	assert json.loads(
		(report_root / 'final_test_summary.json').read_text(encoding='utf-8')
	) == result
	markdown = (report_root / 'final_test_summary.md').read_text(encoding='utf-8')
	assert 'All five layouts are reported as repeated evaluations.' in markdown
	assert 'They are not ranked.' in markdown


def test_missing_predefined_layout_is_rejected(tmp_path: Path) -> None:
	runs_root = tmp_path / 'final_runs'
	_write_metrics(runs_root, layouts=SUMMARY.LAYOUTS[:-1])

	with pytest.raises(ValueError, match=r'missing=.*layout_004'):
		SUMMARY.summarize_final_test(runs_root, MODEL, tmp_path / 'report')


def test_unexpected_layout_does_not_complete_the_five_job_set(
	tmp_path: Path,
) -> None:
	runs_root = tmp_path / 'final_runs'
	layouts = (*SUMMARY.LAYOUTS[:-1], 'layout_999')
	_write_metrics(runs_root, layouts=layouts)

	with pytest.raises(
		ValueError,
		match=r'missing=.*layout_004.*unexpected=.*layout_999',
	):
		SUMMARY.summarize_final_test(runs_root, MODEL, tmp_path / 'report')


@pytest.mark.parametrize(
	('override', 'message'),
	[
		({'model': 'local_barlow_twins_hmm_k6_persist003'}, 'model identity'),
		({'data_size': 'small'}, 'data_size must be medium'),
		({'evaluation_mode': 'validation_only'}, 'evaluation_mode must be'),
	],
)
def test_job_identity_mismatch_is_rejected(
	tmp_path: Path,
	override: dict[str, Any],
	message: str,
) -> None:
	runs_root = tmp_path / 'final_runs'
	_write_metrics(runs_root, overrides=override)

	with pytest.raises(ValueError, match=message):
		SUMMARY.summarize_final_test(runs_root, MODEL, tmp_path / 'report')


@pytest.mark.parametrize(
	'test_metrics',
	[None, {}, {'other_metric': 0.4}],
)
def test_missing_test_channel_iou_is_rejected(
	tmp_path: Path,
	test_metrics: dict[str, float] | None,
) -> None:
	runs_root = tmp_path / 'final_runs'
	_write_metrics(runs_root, overrides={'test': test_metrics})

	with pytest.raises((TypeError, ValueError), match=r'test metrics|test Channel IoU'):
		SUMMARY.summarize_final_test(runs_root, MODEL, tmp_path / 'report')
