'''Contract tests for the Volve horizon recipe-arm paired summary.'''

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, cast

import pytest

from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.volve.horizon_recipe_arm import RECIPE_ARM_BASELINE_MODEL_ID
from seis_ssl_cluster.volve.horizon_recipe_arm_results import (
	PER_CELL_CSV_NAME,
	SUMMARY_JSON_NAME,
	SUMMARY_MD_NAME,
	inspect_volve_horizon_recipe_arm_results,
	summarize_volve_horizon_recipe_arm,
)
from tests.seis_ssl_cluster.helpers_volve_recipe_arm import (
	ARM_ID,
	write_recipe_arm_runs,
	write_recipe_arm_universe,
)

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.volve.horizon_recipe_arm import (
		VolveHorizonRecipeArmConfig,
	)

CELLS = [(layout, size) for layout in LAYOUT_IDS for size in DATA_SIZE_PREFIX]


def _config(tmp_path: Path) -> VolveHorizonRecipeArmConfig:
	universe = write_recipe_arm_universe(tmp_path, embeddings=False)
	return cast('VolveHorizonRecipeArmConfig', universe['config'])


def _flat(value: float) -> dict[tuple[str, str], float]:
	return dict.fromkeys(CELLS, value)


def test_summary_reports_a_positive_delta_when_the_arm_has_lower_error(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(
		config,
		arm_mae=_flat(10.0),
		baseline_mae=_flat(15.0),
	)
	result = summarize_volve_horizon_recipe_arm(config)
	summary = cast('dict[str, object]', result['summary'])
	assert result['complete_cells'] == 30
	assert summary['mean'] == pytest.approx(5.0)
	assert summary['arm_better_count'] == 15
	assert summary['arm_worse_count'] == 0
	assert summary['n'] == 15


def test_summary_reports_a_negative_delta_when_the_arm_is_worse(
	tmp_path: Path,
) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(
		config,
		arm_mae=_flat(24.0),
		baseline_mae=_flat(15.0),
	)
	summary = cast(
		'dict[str, object]',
		summarize_volve_horizon_recipe_arm(config)['summary'],
	)
	assert summary['mean'] == pytest.approx(-9.0)
	assert summary['arm_better_count'] == 0
	assert summary['arm_worse_count'] == 15


def test_summary_writes_json_markdown_and_per_cell_csv(tmp_path: Path) -> None:
	config = _config(tmp_path)
	arm = {cell: 10.0 + index for index, cell in enumerate(CELLS)}
	write_recipe_arm_runs(config, arm_mae=arm, baseline_mae=_flat(20.0))
	summarize_volve_horizon_recipe_arm(config)
	payload = json.loads(
		(config.summary_root / SUMMARY_JSON_NAME).read_text(encoding='utf-8')
	)
	assert payload['arm_id'] == ARM_ID
	assert payload['baseline_id'] == RECIPE_ARM_BASELINE_MODEL_ID
	assert payload['primary_metric'] == 'macro_mae_samples'
	assert set(payload['by_size']) == set(DATA_SIZE_PREFIX)
	assert payload['model_means'][RECIPE_ARM_BASELINE_MODEL_ID] == pytest.approx(20.0)
	markdown = (config.summary_root / SUMMARY_MD_NAME).read_text(encoding='utf-8')
	assert ARM_ID in markdown
	assert 'random - arm' in markdown
	rows = list(
		csv.DictReader(
			(config.summary_root / PER_CELL_CSV_NAME)
			.read_text(encoding='utf-8')
			.splitlines()
		)
	)
	assert len(rows) == 15
	assert {row['data_size'] for row in rows} == set(DATA_SIZE_PREFIX)
	for row in rows:
		assert float(row['random_minus_arm']) == pytest.approx(
			float(row['random_macro_mae_samples'])
			- float(row['arm_macro_mae_samples'])
		)


def test_summary_rejects_a_missing_cell(tmp_path: Path) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(
		config,
		arm_mae=_flat(10.0),
		baseline_mae=_flat(15.0),
		omit=((ARM_ID, 'layout_003', 'medium'),),
	)
	with pytest.raises(FileNotFoundError, match='missing recipe-arm cell metrics'):
		inspect_volve_horizon_recipe_arm_results(config)


def test_summary_rejects_a_cell_from_another_benchmark(tmp_path: Path) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(config, arm_mae=_flat(10.0), baseline_mae=_flat(15.0))
	path = (
		config.runs_root
		/ f'model={ARM_ID}'
		/ 'layout=layout_001'
		/ 'size=small'
		/ 'metrics.json'
	)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['benchmark'] = 'mae_local_bt_hmm_five_way_v1'
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='produced by another benchmark'):
		inspect_volve_horizon_recipe_arm_results(config)


def test_summary_rejects_a_changed_decoder_contract(tmp_path: Path) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(config, arm_mae=_flat(10.0), baseline_mae=_flat(15.0))
	path = (
		config.runs_root
		/ f'model={ARM_ID}'
		/ 'layout=layout_002'
		/ 'size=large'
		/ 'metrics.json'
	)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['decoder']['initialization_seed'] = 7
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='changed the shared decoder contract'):
		inspect_volve_horizon_recipe_arm_results(config)


def test_summary_rejects_an_unpaired_evaluation_support(tmp_path: Path) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(config, arm_mae=_flat(10.0), baseline_mae=_flat(15.0))
	path = (
		config.runs_root
		/ f'model={ARM_ID}'
		/ 'layout=layout_004'
		/ 'size=small'
		/ 'metrics.json'
	)
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['benchmark_identity']['effective_model_valid_observation_counts'] = {
		'bcu': 1
	}
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='does not share the evaluation support'):
		inspect_volve_horizon_recipe_arm_results(config)


def test_summary_accepts_cell_specific_tile_counts(tmp_path: Path) -> None:
	config = _config(tmp_path)
	write_recipe_arm_runs(config, arm_mae=_flat(10.0), baseline_mae=_flat(15.0))
	for model_id in config.model_ids:
		path = (
			config.runs_root
			/ f'model={model_id}'
			/ 'layout=layout_000'
			/ 'size=large'
			/ 'metrics.json'
		)
		payload = json.loads(path.read_text(encoding='utf-8'))
		payload['benchmark_identity']['tiles']['counts']['train'] = 60
		path.write_text(json.dumps(payload), encoding='utf-8')
	report = inspect_volve_horizon_recipe_arm_results(config)
	assert report['complete_cells'] == 30
