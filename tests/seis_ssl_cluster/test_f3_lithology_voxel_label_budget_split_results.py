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
	config = SimpleNamespace(results_root=tmp_path / 'results')
	manifest = publish_low_label_split_summary(config, paths)
	publish_dir = (
		config.results_root
		/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'
	)
	assert {path.name for path in publish_dir.iterdir()} == {
		*OUTPUT_NAMES,
		'publish_manifest.json',
	}
	payload = json.loads(manifest.manifest_path.read_text(encoding='utf-8'))
	assert {item['target'] for item in payload['items']} == set(OUTPUT_NAMES)
	_validate_published_tree(publish_dir, manifest)


def test_summary_publish_rejects_noncanonical_existing_tree(tmp_path: Path) -> None:
	paths = write_low_label_split_summary(_rows(), tmp_path / 'artifacts')
	config = SimpleNamespace(results_root=tmp_path / 'results')
	publish_dir = (
		config.results_root
		/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'
	)
	publish_dir.mkdir(parents=True)
	(publish_dir / 'raw_predictions.npy').write_bytes(b'raw')
	with pytest.raises(FileExistsError, match='unexpected file set'):
		publish_low_label_split_summary(config, paths)


def test_summary_publish_detects_source_or_target_hash_tampering(
	tmp_path: Path,
) -> None:
	paths = write_low_label_split_summary(_rows(), tmp_path / 'artifacts')
	config = SimpleNamespace(results_root=tmp_path / 'results')
	manifest = publish_low_label_split_summary(config, paths)
	publish_dir = (
		config.results_root
		/ 'f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_six_split_v1'
	)
	paths['paired_metrics'].write_text('tampered\n', encoding='utf-8')
	with pytest.raises(ValueError, match='SHA-256'):
		_validate_published_tree(publish_dir, manifest)
