"""Tests for the paired Parihaka Channel recipe-transfer summary."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import pytest

from proc.seis_ssl_cluster import (
	summarize_parihaka_channel_recipe_transfer as summary_cli,
)
from seis_ssl_cluster.parihaka import (
	channel_recipe_transfer_results as recipe_results,
)
from seis_ssl_cluster.parihaka.channel_data import DATA_SIZE_PREFIX, LAYOUT_IDS
from seis_ssl_cluster.parihaka.channel_recipe_transfer_results import (
	OUTPUT_NAMES,
	ChannelRecipeTransferSummaryConfig,
	channel_recipe_transfer_summary_config_from_mapping,
	inspect_channel_recipe_transfer_results,
	summarize_channel_recipe_transfer,
)

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from _pytest.monkeypatch import MonkeyPatch

CANDIDATE = 'local_barlow_twins_rot90_asym_g060_3ep'
REFERENCE = 'local_barlow_twins'


def _raw_config(tmp_path: Path) -> dict[str, object]:
	return {
		'inputs': {'runs_root': str(tmp_path / 'learned-runs')},
		'comparison': {
			'candidate_model_id': CANDIDATE,
			'reference_model_id': REFERENCE,
			'legacy_random_runs_root': str(tmp_path / 'legacy-runs'),
			'legacy_random_model_id': 'random',
		},
		'outputs': {'output_dir': str(tmp_path / 'summary')},
	}


def _config(tmp_path: Path) -> ChannelRecipeTransferSummaryConfig:
	return channel_recipe_transfer_summary_config_from_mapping(_raw_config(tmp_path))


def _identity(
	model_id: str,
	layout_id: str,
	data_size: str,
	*,
	legacy: bool,
) -> dict[str, object]:
	common_metadata: dict[str, object] = {
		'survey_id': 'parihaka',
		'volume_shape_xyz': [782, 590, 1006],
		'patch_size': [8, 8, 8],
		'preprocessing': {'amplitude_agc': 'trace_rms_z'},
	}
	if legacy:
		common_metadata['pretraining_objective'] = {'name': 'mae'}
	return {
		'model': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'embedding': {
			'checkpoint_path': f'/checkpoints/{model_id}.pt',
			'checkpoint_sha256': model_id,
			'model_source': {'model_id': model_id},
			'common_metadata': common_metadata,
		},
		'decoder': {'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1'},
		'training': {'epochs': 50, 'seed': 42000},
		'selection': {'layout_id': layout_id, 'data_size': data_size},
	}


def _payload(
	model_id: str,
	layout_id: str,
	data_size: str,
	channel_iou: float,
	*,
	legacy: bool,
) -> dict[str, object]:
	return {
		'test': {'channel_iou': channel_iou},
		'benchmark_identity': _identity(model_id, layout_id, data_size, legacy=legacy),
	}


def _jobs() -> tuple[
	dict[tuple[str, str, str], Mapping[str, object]],
	dict[tuple[str, str, str], Mapping[str, object]],
]:
	learned: dict[tuple[str, str, str], Mapping[str, object]] = {}
	legacy: dict[tuple[str, str, str], Mapping[str, object]] = {}
	for size_index, data_size in enumerate(DATA_SIZE_PREFIX):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			index = size_index * len(LAYOUT_IDS) + layout_index
			candidate = 0.3 + 0.01 * index
			candidate_minus_reference = 0.01 * (layout_index - 2)
			candidate_minus_random = 0.02 + 0.01 * layout_index
			learned[(CANDIDATE, layout_id, data_size)] = _payload(
				CANDIDATE,
				layout_id,
				data_size,
				candidate,
				legacy=False,
			)
			learned[(REFERENCE, layout_id, data_size)] = _payload(
				REFERENCE,
				layout_id,
				data_size,
				candidate - candidate_minus_reference,
				legacy=False,
			)
			legacy[('random', layout_id, data_size)] = _payload(
				'random',
				layout_id,
				data_size,
				candidate - candidate_minus_random,
				legacy=True,
			)
			legacy[('pretrained', layout_id, data_size)] = _payload(
				'pretrained',
				layout_id,
				data_size,
				candidate - 0.01,
				legacy=True,
			)
	return learned, legacy


def _patch_inspectors(
	monkeypatch: MonkeyPatch,
	learned: dict[tuple[str, str, str], Mapping[str, object]],
	legacy: dict[tuple[str, str, str], Mapping[str, object]],
	*,
	calls: list[tuple[str, object]] | None = None,
) -> None:
	def inspect_learned(
		config: object, *, model_ids: tuple[str, ...]
	) -> dict[tuple[str, str, str], Mapping[str, object]]:
		if calls is not None:
			calls.extend((('learned_config', config), ('model_ids', model_ids)))
		return learned

	def inspect_legacy(
		config: object,
	) -> dict[tuple[str, str, str], Mapping[str, object]]:
		if calls is not None:
			calls.append(('legacy_config', config))
		return legacy

	monkeypatch.setattr(
		recipe_results, 'inspect_channel_model_results', inspect_learned
	)
	monkeypatch.setattr(
		recipe_results, 'inspect_channel_benchmark_results', inspect_legacy
	)


def test_config_resolves_both_run_roots_and_model_ids(tmp_path: Path) -> None:
	config = _config(tmp_path)
	assert config.learned_runs_root == tmp_path / 'learned-runs'
	assert config.legacy_random_runs_root == tmp_path / 'legacy-runs'
	assert config.output_dir == tmp_path / 'summary'
	assert config.candidate_model_id == CANDIDATE
	assert config.reference_model_id == REFERENCE
	assert config.legacy_random_model_id == 'random'


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('reference_model_id', CANDIDATE, 'must differ'),
		('legacy_random_model_id', 'random_seed_7', 'must be random'),
		('candidate_model_id', 'pretrained', 'legacy benchmark model IDs'),
	],
)
def test_config_rejects_ambiguous_model_ids(
	tmp_path: Path, field: str, value: str, message: str
) -> None:
	raw = _raw_config(tmp_path)
	comparison = cast('dict[str, object]', raw['comparison'])
	comparison[field] = value
	with pytest.raises(ValueError, match=message):
		channel_recipe_transfer_summary_config_from_mapping(raw)


def test_inspection_uses_both_canonical_auditors(
	tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
	config = _config(tmp_path)
	learned, legacy = _jobs()
	calls: list[tuple[str, object]] = []
	_patch_inspectors(monkeypatch, learned, legacy, calls=calls)
	report = inspect_channel_recipe_transfer_results(config)
	assert report['complete_learned_jobs'] == 30
	assert report['complete_legacy_jobs'] == 30
	assert report['compared_cells'] == 15
	assert report['downstream_identity_parity'] is True
	assert ('model_ids', (CANDIDATE, REFERENCE)) in calls
	learned_config = next(value for name, value in calls if name == 'learned_config')
	legacy_config = next(value for name, value in calls if name == 'legacy_config')
	assert learned_config.runs_root == config.learned_runs_root
	assert legacy_config.runs_root == config.legacy_random_runs_root


def test_inspection_rejects_cross_root_downstream_drift(
	tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
	config = _config(tmp_path)
	learned, legacy = _jobs()
	legacy = deepcopy(legacy)
	drifted = cast('dict[str, object]', legacy[('random', 'layout_003', 'medium')])
	identity = cast('dict[str, object]', drifted['benchmark_identity'])
	identity['decoder'] = {'spec': 'different_decoder'}
	_patch_inspectors(monkeypatch, learned, legacy)
	with pytest.raises(ValueError, match='learned/legacy-random downstream'):
		inspect_channel_recipe_transfer_results(config)


def test_summary_writes_exact_file_set_and_paired_statistics(
	tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
	config = _config(tmp_path)
	learned, legacy = _jobs()
	_patch_inspectors(monkeypatch, learned, legacy)
	paths = summarize_channel_recipe_transfer(config)
	assert tuple(path.name for path in paths) == OUTPUT_NAMES
	assert {path.name for path in config.output_dir.iterdir()} == set(OUTPUT_NAMES)
	payload = json.loads((config.output_dir / 'summary.json').read_text())
	assert payload['job_counts'] == {
		'learned': 30,
		'legacy_pretrained_random': 30,
		'compared_cells': 15,
	}
	assert payload['downstream_identity_parity'] is True
	reference_stats = payload['aggregates']['candidate_minus_reference']
	assert reference_stats['small']['wins'] == 2
	assert reference_stats['small']['ties'] == 1
	assert reference_stats['small']['losses'] == 2
	assert reference_stats['all_15']['n'] == 15
	assert reference_stats['all_15']['mean'] == pytest.approx(0.0, abs=1e-12)
	random_stats = payload['aggregates']['candidate_minus_random']['all_15']
	assert random_stats['mean'] == pytest.approx(0.04)
	assert random_stats['wins'] == 15
	assert random_stats['paired_t_statistic'] is not None
	rows = list(
		csv.DictReader(
			(config.output_dir / 'comparison.csv')
			.read_text(encoding='utf-8')
			.splitlines()
		)
	)
	assert len(rows) == 15
	assert float(rows[0]['candidate_minus_random']) == pytest.approx(0.02)
	markdown = (config.output_dir / 'summary.md').read_text(encoding='utf-8')
	assert CANDIDATE in markdown
	assert 'all_15' in markdown


@pytest.mark.parametrize('flag', ['--check-only', '--dry-run'])
def test_read_only_cli_aliases_never_write(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	flag: str,
) -> None:
	raw = _raw_config(tmp_path)

	def load_raw(_path: Path, *, loader: object) -> Mapping[str, object]:
		del loader
		return raw

	def inspect(
		config: ChannelRecipeTransferSummaryConfig,
	) -> dict[str, object]:
		return {
			'candidate_model_id': config.candidate_model_id,
			'complete_learned_jobs': 30,
			'complete_legacy_jobs': 30,
			'compared_cells': 15,
		}

	def reject_write(_config: ChannelRecipeTransferSummaryConfig) -> None:
		pytest.fail('read-only CLI attempted to write a summary')

	monkeypatch.setattr(summary_cli, 'load_config_for_cli', load_raw)
	monkeypatch.setattr(summary_cli, 'inspect_channel_recipe_transfer_results', inspect)
	monkeypatch.setattr(summary_cli, 'summarize_channel_recipe_transfer', reject_write)
	monkeypatch.setattr(
		sys, 'argv', ['summary', '--config', str(tmp_path / 'config.yaml'), flag]
	)
	summary_cli.main()
	assert 'execution: check-only; summary files skipped' in capsys.readouterr().out
	assert not (tmp_path / 'summary').exists()
