"""Portable screening config and all-or-nothing summary contracts."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from proc.seis_ssl_cluster import summarize_f3_multi_head_screening as cli
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import multi_head_screening_results as screening
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)
from seis_ssl_cluster.f3.lithology.hmm_v1_freeze_receipt import (
	canonical_control_cells_sha256,
)
from seis_ssl_cluster.f3.lithology.paired_candidate_results import _audit_job_rows
from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	CANDIDATES,
	EXPERIMENT,
)
from tests.seis_ssl_cluster.test_f3_lithology_paired_candidate_results import (
	CANDIDATE,
	CONTROL,
	_canonical,
	_config,
	_patch_auditors,
	_source_provenance,
	_write_jobs,
)

CONFIG = EXPERIMENT / '60_summary/01_paired_screening.yaml'


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	return screening.screening_config_from_mapping(load_config(CONFIG))


def test_config_binds_four_recipes_and_historical_control(
	config: dict[str, object],
) -> None:
	assert [entry['head_ks'] for entry in config['candidates']] == list(
		CANDIDATES.values()
	)
	assert config['control_id'] == 'mae_hmm_k6'
	assert config['control_freeze_receipt'].is_file()
	assert config['control_training_config'].is_file()
	baseline = load_config(config['control_training_config'])
	for entry in config['candidates']:
		candidate = f3_lithology_candidate_config_from_mapping(
			load_config(entry['downstream_config'])
		)
		assert entry['candidate_id'] == candidate.candidate_id
		canonical = load_f3_lithology_candidate_canonical_config(candidate)
		assert (
			canonical.model_by_id('mae_hmm_k6').checkpoint
			== Path(baseline['paths']['output_root']) / 'latest.pt'
		)
		assert candidate.runs_root != canonical.runs_root
		assert (
			entry['training_config']
			== EXPERIMENT.resolve()
			/ '30_pretraining'
			/ candidate.candidate_id
			/ '02_full_25ep.yaml'
		)
	assert not config['summary_root'].exists()


@pytest.fixture
def evidence(
	config: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
	pair = _config(tmp_path / 'synthetic')
	_write_jobs(pair)
	_patch_auditors(monkeypatch, pair)
	rows = _audit_job_rows(
		pair,
		_canonical(pair.runs_root.parent),
		{
			'candidate': _source_provenance(pair, CANDIDATE),
			'control': _source_provenance(pair, CONTROL),
		},
	)
	state = {
		'rows': rows,
		'head_drift': False,
		'checkpoint_drift': False,
		'source_drift': False,
		'control_checkpoint_sha256': 'e' * 64,
	}
	config['control_freeze_receipt'] = tmp_path / 'control_receipt.json'
	config['control_freeze_receipt'].write_text(
		json.dumps(
			{
				'control_id': 'mae_hmm_k6',
				'condition_id': '2b',
				'checkpoint_sha256': state['control_checkpoint_sha256'],
				'f3_cell_count': 15,
				'canonical_control_cells_sha256': canonical_control_cells_sha256(
					[
						{
							'layout_id': row['layout_id'],
							'data_size': row['data_size'],
							'mean_iou': json.loads(
								Path(row['control_metrics_path']).read_text()
							)['mean_iou'],
							'macro_f1': row['control_macro_f1'],
						}
						for row in rows
					]
				),
			}
		)
	)
	state['config_path'] = tmp_path / 'screening.json'
	state['config_path'].write_text(json.dumps(config, default=str))
	candidates = {
		entry['training_config']: (
			entry,
			f3_lithology_candidate_config_from_mapping(
				load_config(entry['downstream_config'])
			),
		)
		for entry in config['candidates']
	}
	canonical = load_f3_lithology_candidate_canonical_config(
		next(iter(candidates.values()))[1]
	)
	control = canonical.model_by_id(config['control_id'])

	def audit_source(candidate: object, _canonical_config: object) -> dict[str, object]:
		role = CONTROL if candidate.candidate_id == config['control_id'] else CANDIDATE
		result = _source_provenance(pair, role)
		result['checkpoint_path'] = str(candidate.checkpoint)
		result['candidate_id'] = candidate.candidate_id
		if role == CONTROL:
			result['checkpoint_sha256'] = state['control_checkpoint_sha256']
		return result

	def audit_training(path: Path) -> dict[str, object]:
		entry, candidate = candidates[path]
		return {
			'checkpoint': {
				'path': str(candidate.checkpoint),
				'sha256': 'z' * 64 if state['checkpoint_drift'] else 'a' * 64,
			},
			'model_tag': candidate.candidate_id,
			'head_ks': [4, 8] if state['head_drift'] else entry['head_ks'],
			'source_embedding': str(path)
			if state['source_drift']
			else {'identity': 'shared'},
		}

	def audit_rows(
		pair_config: object,
		canonical_config: object,
		_sources: object,
		*,
		control_runs_root: Path,
	) -> list[dict[str, object]]:
		assert control_runs_root == canonical_config.runs_root
		assert control_runs_root != pair_config.runs_root
		result = deepcopy(state['rows'])
		for row in result:
			row['candidate_id'] = pair_config.candidate.model_id
			row['control_id'] = config['control_id']
		return result

	monkeypatch.setattr(
		screening,
		'audit_f3_hmm_final_source',
		lambda _path: {
			'checkpoint': {
				'path': str(control.checkpoint),
				'sha256': state['control_checkpoint_sha256'],
			}
		},
	)
	monkeypatch.setattr(screening, 'audit_multi_head_source', audit_training)
	monkeypatch.setattr(screening, 'audit_f3_lithology_candidate_source', audit_source)
	monkeypatch.setattr(screening, '_audit_job_rows', audit_rows)
	return state


def test_summary_has_exact_file_set_sixty_pairs_and_layout_statistics(
	config: dict[str, object], evidence: dict[str, object]
) -> None:
	del evidence
	assert screening.inspect_multi_head_screening(config)['candidate_cells'] == 60
	assert not config['summary_root'].exists()
	paths = screening.summarize_multi_head_screening(config)
	assert {p.name for p in paths} == {'comparison.csv', 'summary.json', 'summary.md'}
	assert {p.name for p in config['summary_root'].iterdir()} == {p.name for p in paths}
	with paths[0].open() as handle:
		reader = csv.DictReader(handle)
		assert 'candidate_minus_control' not in reader.fieldnames
		assert reader.fieldnames[-6:] == [
			'candidate_mean_iou',
			'control_mean_iou',
			'mean_iou_candidate_minus_control',
			'candidate_macro_f1',
			'control_macro_f1',
			'macro_f1_candidate_minus_control',
		]
		csv_rows = list(reader)
		assert len(csv_rows) == 60
	payload = json.loads(paths[1].read_text())
	assert payload['control_cells'] == 15
	assert payload['primary_metric'] == 'mean_iou'
	assert payload['secondary_metric'] == 'macro_f1'
	assert payload['control_freeze_receipt']['sha256'] == file_sha256(
		config['control_freeze_receipt']
	)
	for row in payload['comparison']:
		assert 'candidate_minus_control' not in row
		for metric in ('mean_iou', 'macro_f1'):
			assert row[f'{metric}_candidate_minus_control'] == pytest.approx(
				row[f'candidate_{metric}'] - row[f'control_{metric}']
			)
	for candidate in payload['candidates']:
		assert candidate['overall_layout_clustered']['n'] == 5
		assert candidate['overall_layout_clustered'][
			'mean_iou_candidate_minus_control'
		]['mean'] == pytest.approx(0.005)
		assert candidate['secondary_macro_f1_by_size']['medium'][
			'macro_f1_candidate_minus_control'
		]['mean'] == pytest.approx(0.01)
		assert candidate['all_15_descriptive']['paired_t_statistical_unit'] is None
		assert candidate['by_size']['medium']['mean_iou_candidate_minus_control'][
			'mean'
		] == pytest.approx(0.005)
	before = {path: path.read_bytes() for path in paths}
	with pytest.raises(FileExistsError, match='overwrite'):
		screening.summarize_multi_head_screening(config)
	assert before == {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize(
	'drift',
	[
		'head_drift',
		'checkpoint_drift',
		'source_drift',
		'missing',
		'duplicate',
		'validation',
	],
)
def test_drift_rejects_without_output(
	config: dict[str, object], evidence: dict[str, object], drift: str
) -> None:
	if drift == 'missing':
		evidence['rows'].pop()
	elif drift == 'duplicate':
		evidence['rows'][-1] = deepcopy(evidence['rows'][0])
	elif drift == 'validation':
		evidence['rows'][0]['validation_mask_sha256'] = 'f' * 64
	else:
		evidence[drift] = True
	with pytest.raises(
		ValueError,
		match=r'head set|checkpoint|source identity|cells|duplicate|not shared',
	):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()


@pytest.mark.parametrize('flag', ['--check-only', '--dry-run'])
def test_cli_check_only_writes_nothing(
	config: dict[str, object],
	evidence: dict[str, object],
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	flag: str,
) -> None:
	monkeypatch.setattr(
		sys, 'argv', ['summary', '--config', str(evidence['config_path']), flag]
	)
	cli.main()
	assert json.loads(capsys.readouterr().out)['candidate_cells'] == 60
	assert not config['summary_root'].exists()


def test_missing_live_artifacts_fail_closed(config: dict[str, object]) -> None:
	with pytest.raises((FileNotFoundError, ValueError)):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()


@pytest.mark.parametrize('missing_control', [False, True])
def test_pair_reader_uses_historical_root_directly(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, missing_control: bool
) -> None:
	pair = _config(tmp_path)
	_write_jobs(pair)
	control_root = tmp_path / 'historical-runs'
	control_root.mkdir()
	(pair.runs_root / f'model={CONTROL}').rename(control_root / f'model={CONTROL}')
	_patch_auditors(monkeypatch, pair, control_runs_root=control_root)
	if missing_control:
		(
			control_root
			/ f'model={CONTROL}/layout=layout_000/size=small/evaluation/metrics.json'
		).unlink()
	before = {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	}
	sources = {
		'candidate': _source_provenance(pair, CANDIDATE),
		'control': _source_provenance(pair, CONTROL),
	}
	if missing_control:
		with pytest.raises(FileNotFoundError):
			_audit_job_rows(
				pair, _canonical(tmp_path), sources, control_runs_root=control_root
			)
	else:
		rows = _audit_job_rows(
			pair, _canonical(tmp_path), sources, control_runs_root=control_root
		)
		assert len(rows) == 15
		assert all(str(control_root) in row['control_metrics_path'] for row in rows)
	assert before == {
		path: (path.read_bytes(), path.stat().st_mtime_ns)
		for path in tmp_path.rglob('*')
		if path.is_file()
	}


@pytest.mark.parametrize('ks', [[6, 4], [4, 4], [6], [True, 6]])
def test_invalid_head_sets_rejected(config: dict[str, object], ks: list[int]) -> None:
	del config
	raw = load_config(CONFIG)
	raw['candidates'][0]['head_ks'] = ks
	with pytest.raises(ValueError, match='head_ks'):
		screening.screening_config_from_mapping(raw)


def test_duplicate_recipes_rejected(config: dict[str, object]) -> None:
	del config
	raw = load_config(CONFIG)
	raw['candidates'][1] = deepcopy(raw['candidates'][0])
	with pytest.raises(ValueError, match='duplicate candidate'):
		screening.screening_config_from_mapping(raw)


@pytest.mark.parametrize('drift', ['missing', 'extra', 'id', 'head_ks', 'control'])
def test_screening_config_requires_exact_candidate_mapping(
	config: dict[str, object], drift: str
) -> None:
	del config
	raw = load_config(CONFIG)
	if drift == 'missing':
		raw['candidates'].pop()
	elif drift == 'extra':
		raw['candidates'].append(deepcopy(raw['candidates'][0]))
	elif drift == 'id':
		raw['candidates'][0]['candidate_id'] = 'unplanned_candidate'
	elif drift == 'head_ks':
		raw['candidates'][0]['head_ks'] = [4, 8]
	else:
		raw['control_id'] = 'random_init'
	with pytest.raises(ValueError, match=r'four|control_id'):
		screening.screening_config_from_mapping(raw)


def test_candidate_order_is_not_part_of_scientific_contract(
	config: dict[str, object],
) -> None:
	raw = load_config(CONFIG)
	raw['candidates'].reverse()
	resolved = screening.screening_config_from_mapping(raw)
	assert resolved['candidates'] == list(reversed(config['candidates']))


def test_downstream_candidate_id_must_match_declared_id(
	config: dict[str, object], evidence: dict[str, object]
) -> None:
	del evidence
	config['candidates'][0]['downstream_config'] = config['candidates'][1][
		'downstream_config'
	]
	with pytest.raises(ValueError, match='candidate ID differs from downstream'):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()


@pytest.mark.parametrize('drift', ['mean_iou', 'macro_f1', 'checkpoint', 'missing'])
def test_valid_live_control_must_match_frozen_receipt(
	config: dict[str, object], evidence: dict[str, object], drift: str
) -> None:
	if drift == 'checkpoint':
		evidence['control_checkpoint_sha256'] = 'f' * 64
	elif drift == 'missing':
		evidence['rows'].pop()
	else:
		row = evidence['rows'][0]
		path = Path(row['control_metrics_path'])
		metrics = json.loads(path.read_text())
		metrics[drift] += 0.01
		path.write_text(json.dumps(metrics))
		row['control_metrics_sha256'] = file_sha256(path)
		if drift == 'macro_f1':
			row['control_macro_f1'] = metrics[drift]
	with pytest.raises(ValueError, match=r'freeze|frozen|15.*cells'):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()


def test_missing_freeze_receipt_rejected_without_output(
	config: dict[str, object], evidence: dict[str, object]
) -> None:
	del evidence
	config['control_freeze_receipt'].unlink()
	with pytest.raises(FileNotFoundError):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()


def test_summary_cannot_overlap_historical_runs(
	config: dict[str, object], evidence: dict[str, object]
) -> None:
	del evidence
	candidate = f3_lithology_candidate_config_from_mapping(
		load_config(config['candidates'][0]['downstream_config'])
	)
	canonical = load_f3_lithology_candidate_canonical_config(candidate)
	config['summary_root'] = canonical.runs_root
	with pytest.raises(ValueError, match='overlaps'):
		screening.inspect_multi_head_screening(config)
	assert not config['summary_root'].exists()


@pytest.mark.parametrize(
	'value', [None, True, float('nan'), float('inf'), -0.1, 1.1, 0.9]
)
def test_invalid_or_changed_primary_metric_rejected(
	config: dict[str, object], evidence: dict[str, object], value: object
) -> None:
	path = Path(evidence['rows'][0]['candidate_metrics_path'])
	payload = json.loads(path.read_text())
	payload['mean_iou'] = value
	path.write_text(json.dumps(payload))
	with pytest.raises(ValueError, match=r'mean_iou|metrics changed'):
		screening.summarize_multi_head_screening(config)
	assert not config['summary_root'].exists()
