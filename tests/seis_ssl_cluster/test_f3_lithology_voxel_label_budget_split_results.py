from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from seis_ssl_cluster.f3.lithology.voxel_label_budget_split_results import (
	OUTPUT_NAMES,
	REQUIRED_METRICS,
	_validate_published_tree,
	aggregate_low_label_split_results,
	publish_low_label_split_summary,
	write_low_label_split_summary,
)

if TYPE_CHECKING:
	from pathlib import Path


def _rows() -> list[dict[str, object]]:
	rows = []
	for split_index in range(6):
		for budget_id in ('cap25', 'cap50'):
			for role, value in (
				('mae', 0.4),
				('m1_current_k6', 0.5),
				('mh_nocons', 0.6),
			):
				row: dict[str, object] = {
					'split_id': f'split_{split_index:03d}',
					'budget_id': budget_id,
					'model_role': role,
				}
				row.update(dict.fromkeys(REQUIRED_METRICS, value))
				rows.append(row)
	return rows


def _hold_rows() -> list[dict[str, object]]:
	"""Create a complete matrix whose cap50 Mean IoU misses only the win gate."""
	rows = _rows()
	for row in rows:
		role = str(row['model_role'])
		budget = str(row['budget_id'])
		split_index = int(str(row['split_id']).removeprefix('split_'))
		if role == 'mae':
			row.update(dict.fromkeys(REQUIRED_METRICS, 0.4))
		elif role == 'm1_current_k6':
			row.update(dict.fromkeys(REQUIRED_METRICS, 0.5))
		else:
			row.update(dict.fromkeys(REQUIRED_METRICS, 0.6))
			if budget == 'cap50':
				row['macro_f1'] = 0.56 if split_index == 0 else 0.51
				row['mean_iou'] = (0.52, 0.53, 0.54, 0.49, 0.48, 0.47)[
					split_index
				]
			if budget == 'cap25':
				row['vertical_boundary_position_mae'] = 0.6
			else:
				row['vertical_boundary_position_mae'] = 0.4
	return rows


def _aggregate_row(
	aggregates: list[dict[str, object]], *, budget: str, comparison: str,
	metric: str,
) -> dict[str, object]:
	return next(
		row
		for row in aggregates
		if row['budget_id'] == budget
		and row['comparison'] == comparison
		and row['metric'] == metric
	)


def test_aggregate_requires_complete_unique_metric_matrix() -> None:
	deltas, aggregates, decision = aggregate_low_label_split_results(_rows()[:-1])
	assert deltas == []
	assert aggregates == []
	assert decision['status'] == 'M4_MH_SPLIT_BLOCKED'
	assert 'coverage failure' in str(decision['blocked_reason'])


def test_aggregate_reports_fixed_confirmatory_comparisons() -> None:
	deltas, aggregates, decision = aggregate_low_label_split_results(_rows())
	assert len(deltas) == 6 * 2 * 3 * len(REQUIRED_METRICS)
	assert len(aggregates) == 2 * 3 * len(REQUIRED_METRICS)
	assert decision['status'] == 'M4_MH_SPLIT_CONFIRMED'


def test_lower_is_better_metric_reverses_the_win_direction() -> None:
	rows = _rows()
	for row in rows:
		row['vertical_boundary_position_mae'] = {
			'mae': 0.6, 'm1_current_k6': 0.5, 'mh_nocons': 0.4,
		}[str(row['model_role'])]
	deltas, _, _ = aggregate_low_label_split_results(rows)
	value = next(
		item['delta'] for item in deltas
		if item['comparison'] == 'mh_nocons_minus_m1_current_k6'
		and item['metric'] == 'vertical_boundary_position_mae'
	)
	assert value == pytest.approx(0.1)


def test_summary_publish_writes_and_validates_exact_lightweight_tree(
	tmp_path: Path,
) -> None:
	paths = write_low_label_split_summary(_rows(), tmp_path / 'artifacts')
	config = SimpleNamespace(results_root=tmp_path / 'reports')
	published_files = publish_low_label_split_summary(config, paths)
	publish_dir = (
		config.results_root
		/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'
	)
	assert {path.name for path in publish_dir.iterdir()} == {
		*OUTPUT_NAMES,
	}
	assert {path.name for path in published_files} == set(OUTPUT_NAMES)
	_validate_published_tree(publish_dir, published_files)


def test_summary_separates_formal_hold_from_project_adoption_and_uses_aggregates(
	tmp_path: Path,
) -> None:
	rows = _hold_rows()
	_, aggregates, decision = aggregate_low_label_split_results(rows)
	assert decision == {
		'status': 'M4_MH_SPLIT_HOLD',
		'systematic_major_degradation': False,
	}
	paths = write_low_label_split_summary(rows, tmp_path / 'artifacts')
	decisions = json.loads(paths['decisions'].read_text(encoding='utf-8'))
	summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
	markdown = paths['markdown'].read_text(encoding='utf-8')

	assert decisions == {
		'status': 'M4_MH_SPLIT_HOLD',
		'decision': {
			'status': 'M4_MH_SPLIT_HOLD',
			'systematic_major_degradation': False,
		},
		'job_count': 36,
		'aggregate_count': 120,
	}
	assert summary['formal_decision'] == decisions['decision']
	assert summary['project_decision']['status'] == 'ADOPT_MH_NOCONS_FOR_M5'
	assert summary['project_decision']['selected_model_role'] == 'mh_nocons'
	assert summary['project_decision']['selected_model_tag'] == (
		'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
	)
	assert summary['project_decision']['additional_decoder_seed_gate_required'] is False
	assert summary['formal_decision']['systematic_major_degradation'] is False
	assert summary['next_stage']['milestone'] == 'M5_U_SOFT_POSTERIOR'
	assert summary['next_stage']['status'] == 'PLANNED_UNVALIDATED'

	primary = {
		(row['comparison'], row['budget_id']): row
		for row in summary['primary_evidence']['rows']
	}
	for budget in ('cap25', 'cap50'):
		for metric in ('macro_f1', 'mean_iou'):
			expected = _aggregate_row(
				aggregates,
				budget=budget,
				comparison='mh_nocons_minus_m1_current_k6',
				metric=metric,
			)
			actual = primary[('mh_nocons_minus_m1_current_k6', budget)][metric]
			assert actual['mean'] == pytest.approx(float(expected['mean']))
			assert actual['median'] == pytest.approx(float(expected['median']))
			assert actual['wins'] == int(expected['wins'])

	hold_reason = summary['hold_reason']
	assert hold_reason['budget_id'] == 'cap50'
	assert hold_reason['metric'] == 'mean_iou'
	assert hold_reason['observed']['wins'] == 3
	assert hold_reason['requirement']['minimum_wins'] == 4
	assert hold_reason['mean_positive'] is True
	assert hold_reason['median_positive'] is True
	assert hold_reason['wins_requirement_met'] is False
	assert '## Why the formal result is HOLD' in markdown
	assert 'wins `3/6`' in markdown
	assert 'wins ≥ `4/6`' in markdown
	assert 'M4_MH_SPLIT_CONFIRMED' not in markdown
	assert 'robust superiority across all splits' not in markdown
	assert 'cap25 is robust' in markdown
	assert 'cap50 is split-dependent' in markdown


def test_handoff_carries_mh_nocons_without_requiring_extra_decoder_seeds(
	tmp_path: Path,
) -> None:
	paths = write_low_label_split_summary(_hold_rows(), tmp_path / 'artifacts')
	handoff = paths['handoff'].read_text(encoding='utf-8')
	carry_forward = handoff.split('## Do not carry forward as primary candidate')[0]

	assert 'Formal result: `M4_MH_SPLIT_HOLD`' in handoff
	assert 'Project decision: `ADOPT_MH_NOCONS_FOR_M5`' in handoff
	assert 'HOLD is preserved' in handoff
	assert '- mh_nocons' in carry_forward
	assert 'mh_cons010' not in carry_forward
	assert '## Do not carry forward as primary candidate\n\n- mh_cons010' in handoff
	assert 'M5-U soft posterior' in handoff
	assert 'effectiveness is unverified' in handoff
	assert 'Decoder seeds `42001/42002`' in handoff
	assert '## No longer required as a gate' in handoff


def test_summary_publish_rejects_noncanonical_existing_tree(tmp_path: Path) -> None:
	paths = write_low_label_split_summary(_rows(), tmp_path / 'artifacts')
	config = SimpleNamespace(results_root=tmp_path / 'reports')
	publish_dir = (
		config.results_root
		/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'
	)
	publish_dir.mkdir(parents=True)
	(publish_dir / 'raw_predictions.npy').write_bytes(b'raw')
	with pytest.raises(FileExistsError, match='unexpected file set'):
		publish_low_label_split_summary(config, paths)


def test_summary_publish_rejects_missing_required_source(
	tmp_path: Path,
) -> None:
	paths = write_low_label_split_summary(_rows(), tmp_path / 'artifacts')
	config = SimpleNamespace(results_root=tmp_path / 'reports')
	paths['paired_metrics'].unlink()
	with pytest.raises(FileNotFoundError, match='required six-split publish source'):
		publish_low_label_split_summary(config, paths)
