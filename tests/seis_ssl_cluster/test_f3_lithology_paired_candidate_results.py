"""Tests for exact F3 lithology paired-candidate summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from proc.seis_ssl_cluster import (
	summarize_f3_lithology_paired_candidates as summary_cli,
)
from seis_ssl_cluster.config.f3_lithology_five_way import F3FiveWayConfig
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.config.io import load_config
from seis_ssl_cluster.f3.lithology import paired_candidate_results as paired_results
from seis_ssl_cluster.f3.lithology.paired_candidate_results import (
	OUTPUT_NAMES,
	F3PairedCandidateSummaryConfig,
	f3_paired_candidate_summary_config_from_mapping,
	inspect_f3_paired_candidate_results,
	summarize_f3_paired_candidate_results,
)

if TYPE_CHECKING:
	from collections.abc import Mapping

	from _pytest.monkeypatch import MonkeyPatch

CANDIDATE = 'local_bt_nr_rot90_asym_g060_3ep_hmm_k6_25ep'
CONTROL = 'local_bt_nr_rot90_asym_g060_3ep'
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = (
	REPO_ROOT
	/ 'experiments/f3/facies_benchmark_v2'
	/ '123_local_bt_noise_rotation_search_v1/80_hmm_summary'
	/ '01_hmm_k6_25ep_vs_rot90_asym_g060_3ep.yaml'
)
SOURCE_SHAS = {
	CANDIDATE: {
		'checkpoint_sha256': 'a' * 64,
		'embeddings_sha256': 'b' * 64,
		'embedding_metadata_sha256': 'c' * 64,
		'valid_tokens_sha256': 'd' * 64,
		'decoder_checkpoint_sha256': '2' * 64,
	},
	CONTROL: {
		'checkpoint_sha256': 'e' * 64,
		'embeddings_sha256': 'f' * 64,
		'embedding_metadata_sha256': '1' * 64,
		'valid_tokens_sha256': 'd' * 64,
		'decoder_checkpoint_sha256': '3' * 64,
	},
}


def _raw_config(tmp_path: Path) -> dict[str, object]:
	return {
		'benchmark': {'canonical_config': str(tmp_path / 'canonical.yaml')},
		'inputs': {'runs_root': str(tmp_path / 'runs')},
		'models': {
			'candidate': {
				'id': CANDIDATE,
				'checkpoint': str(tmp_path / 'candidate.pt'),
				'embeddings_dir': str(tmp_path / 'candidate-embeddings'),
				'expected': {
					'base_training_config': str(tmp_path / 'base.yaml'),
					'stratigraphy_training_config': str(tmp_path / 'hmm.yaml'),
					'epoch': 25,
					'global_step': 15_625,
				},
			},
			'control': {
				'id': CONTROL,
				'checkpoint': str(tmp_path / 'control.pt'),
				'embeddings_dir': str(tmp_path / 'control-embeddings'),
				'expected': {
					'base_training_config': str(tmp_path / 'base.yaml'),
					'stratigraphy_training_config': None,
					'epoch': 3,
					'global_step': 1_875,
				},
			},
		},
		'outputs': {'summary_root': str(tmp_path / 'paired-summary')},
	}


def _config(tmp_path: Path) -> F3PairedCandidateSummaryConfig:
	return f3_paired_candidate_summary_config_from_mapping(_raw_config(tmp_path))


def _canonical(tmp_path: Path) -> F3FiveWayConfig:
	return F3FiveWayConfig(
		artifact_root=tmp_path,
		f3_root=tmp_path,
		dataset={'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v2'},
		labels=_canonical_labels(tmp_path),
		section_layout_dataset_root=tmp_path / 'conditions',
		models=(),
		runs_root=tmp_path / 'canonical-runs',
		summary_root=tmp_path / 'canonical-summary',
	)


def _canonical_labels(tmp_path: Path) -> dict[str, Path]:
	return {
		'source_label_volume': tmp_path / 'ground-truth/f3_facies_labels.npy',
		'source_label_segy': tmp_path / 'ground-truth/f3_labels.sgy',
		'png_label_inventory': tmp_path / 'ground-truth/section_inventory_v2.csv',
		'segy_geometry_json': tmp_path / 'ground-truth/segy_geometry.json',
		'class_info': tmp_path / 'ground-truth/class_info.json',
	}


def _cell_values(size_index: int, layout_index: int) -> tuple[float, float]:
	candidate = 0.45 + 0.02 * size_index + 0.005 * layout_index
	delta = 0.01 * (layout_index - 2 + size_index)
	control = candidate if delta == 0.0 else candidate - delta
	return candidate, control


def _write_jobs(config: F3PairedCandidateSummaryConfig) -> None:
	for size_index, data_size in enumerate(DATA_SIZES):
		for layout_index, layout_id in enumerate(LAYOUT_IDS):
			candidate, control = _cell_values(size_index, layout_index)
			for model, value in (
				(config.candidate, candidate),
				(config.control, control),
			):
				path = (
					config.runs_root
					/ f'model={model.model_id}'
					/ f'layout={layout_id}'
					/ f'size={data_size}'
					/ 'evaluation/metrics.json'
				)
				path.parent.mkdir(parents=True)
				path.write_text(
					json.dumps(
						{
							'aggregation_unit': 'unique_validation_voxel',
							'evaluation_voxel_count': 1000,
							'macro_f1': value,
							'mean_iou': value * 0.5,
						}
					),
					encoding='utf-8',
				)


def _source_provenance(
	config: F3PairedCandidateSummaryConfig,
	model_id: str,
) -> dict[str, object]:
	model = config.candidate if model_id == CANDIDATE else config.control
	shas = SOURCE_SHAS[model_id]
	return {
		'candidate_id': model_id,
		'checkpoint_path': str(model.checkpoint),
		'checkpoint_sha256': shas['checkpoint_sha256'],
		'embeddings_path': str(model.embeddings_dir / 'embeddings.npy'),
		'embeddings_sha256': shas['embeddings_sha256'],
		'embedding_metadata_path': str(model.embeddings_dir / 'metadata.json'),
		'embedding_metadata_sha256': shas['embedding_metadata_sha256'],
		'valid_tokens_path': str(model.embeddings_dir / 'valid.npy'),
		'valid_tokens_sha256': shas['valid_tokens_sha256'],
		'canonical_random': {},
	}


def _patch_auditors(  # noqa: C901, PLR0913
	monkeypatch: MonkeyPatch,
	config: F3PairedCandidateSummaryConfig,
	*,
	evidence_drift: tuple[str, str, str, str, object] | None = None,
	decoder_drift: tuple[str, str, str, str, object] | None = None,
	evaluation_drift: tuple[str, str, str, str, object] | None = None,
	control_runs_root: Path | None = None,
) -> list[str]:
	config.canonical_config.parent.mkdir(parents=True, exist_ok=True)
	config.canonical_config.write_text('canonical fixture\n', encoding='utf-8')
	audited_sources: list[str] = []
	monkeypatch.setattr(
		paired_results,
		'load_f3_lithology_candidate_canonical_config',
		lambda _config: _canonical(config.runs_root.parent),
	)

	def audit_source(candidate_config: object, _canonical_config: object) -> object:
		model_id = candidate_config.candidate_id
		audited_sources.append(model_id)
		return _source_provenance(config, model_id)

	def validate_source(  # noqa: PLR0913
		checkpoint: Path,
		embedding_metadata: Path,
		*,
		role: str,
		base_training_config: Path,
		stratigraphy_training_config: Path | None,
		epoch: int,
		global_step: int,
	) -> dict[str, object]:
		model = config.candidate if role == 'candidate' else config.control
		assert checkpoint == model.checkpoint
		assert embedding_metadata == model.embeddings_dir / 'metadata.json'
		assert base_training_config == model.expected.base_training_config
		assert (
			stratigraphy_training_config == model.expected.stratigraphy_training_config
		)
		assert epoch == model.expected.epoch
		assert global_step == model.expected.global_step
		return {'schema_version': 1, 'role': role}

	def read_evidence(
		_canonical_config: object,
		*,
		model: object,
		layout_id: str,
		data_size: str,
		job_dir: Path,
	) -> dict[str, object]:
		model_id = model.model_id
		assert job_dir == (
			(
				control_runs_root
				if model_id == CONTROL and control_runs_root
				else config.runs_root
			)
			/ f'model={model_id}'
			/ f'layout={layout_id}'
			/ f'size={data_size}'
		)
		shas = SOURCE_SHAS[model_id]
		result: dict[str, object] = {
			'encoder_checkpoint_sha256': shas['checkpoint_sha256'],
			'embeddings_sha256': shas['embeddings_sha256'],
			'embedding_metadata_sha256': shas['embedding_metadata_sha256'],
			'valid_tokens_sha256': shas['valid_tokens_sha256'],
			'decoder_checkpoint_sha256': shas['decoder_checkpoint_sha256'],
			'supervision_identity': hashlib.sha256(
				f'supervision/{layout_id}/{data_size}'.encode()
			).hexdigest(),
			'validation_identity': 'a' * 64,
			'_validation_tile_manifest_sha256': 'b' * 64,
			'validation_voxel_count': 1000,
		}
		if evidence_drift is not None:
			drift_model, drift_layout, drift_size, key, value = evidence_drift
			if (
				drift_model in {model_id, 'both'}
				and layout_id == drift_layout
				and data_size == drift_size
			):
				result[key] = value
		return result

	def read_decoder(
		job_dir: Path,
		*,
		model_id: str,
		layout_id: str,
		data_size: str,
		freeze_encoder: bool,
	) -> dict[str, object]:
		assert freeze_encoder is True
		assert job_dir == (
			(
				control_runs_root
				if model_id == CONTROL and control_runs_root
				else config.runs_root
			)
			/ f'model={model_id}'
			/ f'layout={layout_id}'
			/ f'size={data_size}'
		)
		shas = SOURCE_SHAS[model_id]
		result: dict[str, object] = {
			'decoder_initial_state_sha256': '4' * 64,
			'decoder_contract_identity_sha256': '5' * 64,
			'decoder_resolved_config_sha256': (
				'6' * 64 if model_id == CANDIDATE else '7' * 64
			),
			'decoder_resolved_config_path': str(
				job_dir / 'decoder/resolved_config.json'
			),
			'decoder_run_metadata_sha256': (
				'8' * 64 if model_id == CANDIDATE else '9' * 64
			),
			'decoder_latest_checkpoint_path': str(job_dir / 'decoder/latest.pt'),
			'decoder_latest_checkpoint_sha256': 'a' * 64,
			'decoder_best_checkpoint_path': str(job_dir / 'decoder/best.pt'),
			'decoder_best_checkpoint_sha256': shas['decoder_checkpoint_sha256'],
			'decoder_completed_epoch': 49,
			'decoder_completed_global_step': 22_000,
			'decoder_best_epoch': 22,
			'decoder_best_global_step': 10_120,
			'decoder_train_tile_manifest_path': str(
				job_dir / 'decoder/train_tile_manifest.json'
			),
			'decoder_train_tile_manifest_sha256': 'b' * 64,
			'decoder_train_tile_manifest_identity_sha256': 'c' * 64,
			'decoder_validation_tile_manifest_path': str(
				job_dir / 'decoder/validation_tile_manifest.json'
			),
			'decoder_validation_tile_manifest_sha256': 'd' * 64,
			'decoder_validation_tile_manifest_identity_sha256': 'b' * 64,
			'prediction_metadata_path': str(
				job_dir / 'prediction/prediction_metadata.json'
			),
			'prediction_metadata_sha256': 'b' * 64,
		}
		if decoder_drift is not None:
			drift_model, drift_layout, drift_size, key, value = decoder_drift
			if (model_id, layout_id, data_size) == (
				drift_model,
				drift_layout,
				drift_size,
			):
				result[key] = value
		return result

	def read_evaluation(
		job_dir: Path,
		*,
		model_id: str,
		layout_id: str,
		data_size: str,
	) -> dict[str, object]:
		assert job_dir == (
			(
				control_runs_root
				if model_id == CONTROL and control_runs_root
				else config.runs_root
			)
			/ f'model={model_id}'
			/ f'layout={layout_id}'
			/ f'size={data_size}'
		)
		metrics_path = job_dir / 'evaluation/metrics.json'
		value = paired_results._read_macro_f1(  # noqa: SLF001
			metrics_path,
			evidence={'validation_voxel_count': 1000},
			label=f'{model_id}/{layout_id}/{data_size}',
		)
		metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
		shas = SOURCE_SHAS[model_id]
		labels = _canonical_labels(config.runs_root.parent)
		result: dict[str, object] = {
			'model_id': model_id,
			'layout_id': layout_id,
			'data_size': data_size,
			'aggregation_unit': 'unique_validation_voxel',
			'macro_f1': value,
			'evaluation_voxel_count': 1000,
			'metrics_path': str(metrics_path),
			'metrics_sha256': metrics_sha256,
			'evaluation_metadata_path': str(
				job_dir / 'evaluation/evaluation_metadata.json'
			),
			'evaluation_metadata_sha256': 'a' * 64,
			'prediction_metadata_path': str(
				job_dir / 'prediction/prediction_metadata.json'
			),
			'prediction_metadata_sha256': 'b' * 64,
			'voxel_predictions_path': str(
				job_dir / 'prediction/f3_voxel_predictions.npy'
			),
			'voxel_predictions_sha256': 'c' * 64,
			'voxel_confidence_path': str(
				job_dir / 'prediction/f3_voxel_confidence.npy'
			),
			'voxel_confidence_sha256': 'd' * 64,
			'voxel_valid_mask_path': str(
				job_dir / 'prediction/f3_valid_voxel_mask.npy'
			),
			'voxel_valid_mask_sha256': 'e' * 64,
			'decoder_checkpoint_path': str(job_dir / 'decoder/best.pt'),
			'decoder_checkpoint_sha256': shas['decoder_checkpoint_sha256'],
			'decoder_resolved_config_path': str(
				job_dir / 'decoder/resolved_config.json'
			),
			'decoder_resolved_config_sha256': (
				'6' * 64 if model_id == CANDIDATE else '7' * 64
			),
			'voxel_dataset_metadata_path': str(
				config.runs_root.parent
				/ 'conditions/datasets'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
				/ 'voxel_supervision/voxel_dataset_metadata.json'
			),
			'voxel_dataset_metadata_sha256': '0' * 64,
			'label_volume_path': str(labels['source_label_volume']),
			'label_volume_sha256': '1' * 64,
			'class_info_path': str(labels['class_info']),
			'class_info_sha256': '2' * 64,
			'source_label_segy_path': str(labels['source_label_segy']),
			'source_label_segy_sha256': '3' * 64,
			'png_label_inventory_path': str(labels['png_label_inventory']),
			'png_label_inventory_sha256': '4' * 64,
			'segy_geometry_json_path': str(labels['segy_geometry_json']),
			'segy_geometry_json_sha256': '5' * 64,
		}
		if evaluation_drift is not None:
			drift_model, drift_layout, drift_size, key, drift_value = evaluation_drift
			if (model_id, layout_id, data_size) == (
				drift_model,
				drift_layout,
				drift_size,
			):
				result[key] = drift_value
		return result

	monkeypatch.setattr(
		paired_results,
		'audit_f3_lithology_candidate_source',
		audit_source,
	)
	monkeypatch.setattr(
		paired_results,
		'validate_f3_local_bt_candidate_source',
		validate_source,
	)
	monkeypatch.setattr(
		paired_results,
		'read_f3_lithology_job_evidence',
		read_evidence,
	)
	monkeypatch.setattr(
		paired_results,
		'read_f3_completed_decoder_contract',
		read_decoder,
	)
	monkeypatch.setattr(
		paired_results,
		'read_f3_completed_evaluation_provenance',
		read_evaluation,
	)
	return audited_sources


def _assert_statistical_aggregates(payload: dict[str, object]) -> None:
	aggregates = payload['aggregates']
	assert isinstance(aggregates, dict)
	by_size_aggregates = aggregates['by_size']
	assert isinstance(by_size_aggregates, dict)
	assert by_size_aggregates['small']['candidate_minus_control']['wins'] == 2
	assert by_size_aggregates['small']['candidate_minus_control']['ties'] == 1
	assert by_size_aggregates['small']['candidate_minus_control']['losses'] == 2
	for data_size in DATA_SIZES:
		by_size = by_size_aggregates[data_size]
		assert by_size['n'] == 5
		assert by_size['descriptive_unit'] == 'layout_id'
		assert by_size['paired_t_statistical_unit'] == 'layout_id'
		assert by_size['candidate_minus_control']['n'] == 5
		assert 'paired_t_statistic' in by_size['candidate_minus_control']
	all_15 = aggregates['all_15']
	assert all_15['n'] == 15
	assert all_15['descriptive_unit'] == 'layout_id_x_data_size_cell'
	assert all_15['paired_t_statistical_unit'] is None
	assert all_15['candidate_minus_control']['mean'] == pytest.approx(0.01)
	assert 'paired_t_statistic' not in all_15['candidate_minus_control']
	overall = aggregates['overall_layout_clustered']
	assert overall['n'] == 5
	assert overall['statistical_unit'] == 'layout_id'
	assert overall['within_layout_reduction'] == 'mean_across_data_sizes'
	assert overall['data_sizes_per_layout'] == 3
	assert overall['layout_mean_deltas'] == pytest.approx(
		{
			'layout_000': -0.01,
			'layout_001': 0.0,
			'layout_002': 0.01,
			'layout_003': 0.02,
			'layout_004': 0.03,
		}
	)
	assert overall['candidate_minus_control']['n'] == 5
	assert overall['candidate_minus_control']['mean'] == pytest.approx(0.01)
	assert overall['candidate_minus_control']['paired_t_statistic'] == pytest.approx(
		math.sqrt(2.0)
	)
	assert payload['statistical_units'] == {
		'by_size_paired_t': {'unit': 'layout_id', 'n': 5},
		'all_15_descriptive': {
			'unit': 'layout_id_x_data_size_cell',
			'n': 15,
			'paired_t': None,
		},
		'overall_layout_clustered_paired_t': {
			'unit': 'layout_id',
			'n': 5,
			'within_layout_reduction': 'mean_across_data_sizes',
		},
	}


def test_config_resolves_exact_pair_and_paths(tmp_path: Path) -> None:
	config = _config(tmp_path)
	assert config.canonical_config == tmp_path / 'canonical.yaml'
	assert config.runs_root == tmp_path / 'runs'
	assert config.summary_root == tmp_path / 'paired-summary'
	assert config.candidate.model_id == CANDIDATE
	assert config.control.model_id == CONTROL
	assert config.candidate.expected.epoch == 25
	assert config.candidate.expected.global_step == 15_625
	assert config.candidate.expected.stratigraphy_training_config == (
		tmp_path / 'hmm.yaml'
	)
	assert config.control.expected.epoch == 3
	assert config.control.expected.global_step == 1_875
	assert config.control.expected.stratigraphy_training_config is None


def test_experiment_config_pins_requested_pair_and_dedicated_output(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(REPO_ROOT))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path))
	config = f3_paired_candidate_summary_config_from_mapping(
		load_config(EXPERIMENT_CONFIG)
	)
	assert EXPERIMENT_CONFIG == summary_cli.DEFAULT_CONFIG
	assert config.candidate.model_id == CANDIDATE
	assert config.control.model_id == CONTROL
	assert config.candidate.expected.base_training_config == (
		REPO_ROOT
		/ 'experiments/f3/facies_benchmark_v2'
		/ '123_local_bt_noise_rotation_search_v1/10_pretraining/'
		'rot90_asym_g060_3ep.yaml'
	)
	assert config.candidate.expected.stratigraphy_training_config == (
		REPO_ROOT
		/ 'experiments/f3/facies_benchmark_v2'
		/ '123_local_bt_noise_rotation_search_v1/50_hmm_pretraining/'
		'rot90_asym_g060_3ep/hmm/k6/02_full_25ep.yaml'
	)
	assert config.candidate.expected.epoch == 25
	assert config.candidate.expected.global_step == 15_625
	assert config.control.expected.base_training_config == (
		config.candidate.expected.base_training_config
	)
	assert config.control.expected.stratigraphy_training_config is None
	assert config.control.expected.epoch == 3
	assert config.control.expected.global_step == 1_875
	assert config.runs_root == (
		tmp_path / 'f3_lithology_benchmark/local_bt_noise_rotation_search_v1/runs'
	)
	assert config.summary_root == (
		tmp_path
		/ 'f3_lithology_benchmark/local_bt_noise_rotation_search_v1/paired_summary'
		/ f'{CANDIDATE}_vs_{CONTROL}'
	)


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		('same_id', 'model IDs must differ'),
		('same_checkpoint', 'checkpoints must differ'),
		('nested_summary', 'must not overlap'),
		('invalid_id', 'one path segment'),
	],
)
def test_config_rejects_ambiguous_pair(
	tmp_path: Path,
	mutation: str,
	message: str,
) -> None:
	raw = _raw_config(tmp_path)
	models = raw['models']
	assert isinstance(models, dict)
	candidate = models['candidate']
	control = models['control']
	assert isinstance(candidate, dict)
	assert isinstance(control, dict)
	if mutation == 'same_id':
		control['id'] = CANDIDATE
	elif mutation == 'same_checkpoint':
		control['checkpoint'] = candidate['checkpoint']
	elif mutation == 'nested_summary':
		raw['outputs'] = {'summary_root': str(tmp_path / 'runs/summary')}
	else:
		candidate['id'] = 'nested/model'
	with pytest.raises(ValueError, match=message):
		f3_paired_candidate_summary_config_from_mapping(raw)


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		('candidate_missing_hmm', 'candidate stratigraphy training config'),
		('control_has_hmm', 'control stratigraphy training config'),
		('base_mismatch', 'base training configs must match'),
		('zero_epoch', 'must be a positive integer'),
		('unexpected_expected_key', 'keys must be exactly'),
	],
)
def test_config_rejects_invalid_source_expectations(
	tmp_path: Path,
	mutation: str,
	message: str,
) -> None:
	raw = _raw_config(tmp_path)
	models = raw['models']
	assert isinstance(models, dict)
	candidate = models['candidate']
	control = models['control']
	assert isinstance(candidate, dict)
	assert isinstance(control, dict)
	candidate_expected = candidate['expected']
	control_expected = control['expected']
	assert isinstance(candidate_expected, dict)
	assert isinstance(control_expected, dict)
	if mutation == 'candidate_missing_hmm':
		candidate_expected['stratigraphy_training_config'] = None
	elif mutation == 'control_has_hmm':
		control_expected['stratigraphy_training_config'] = str(tmp_path / 'hmm.yaml')
	elif mutation == 'base_mismatch':
		control_expected['base_training_config'] = str(tmp_path / 'other.yaml')
	elif mutation == 'zero_epoch':
		candidate_expected['epoch'] = 0
	else:
		candidate_expected['unregistered'] = True
	with pytest.raises(ValueError, match=message):
		f3_paired_candidate_summary_config_from_mapping(raw)


def test_inspect_and_summary_audit_exact_30_jobs_and_write_three_files(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	audited_sources = _patch_auditors(monkeypatch, config)
	report = inspect_f3_paired_candidate_results(config)
	assert report['complete_candidate_jobs'] == 15
	assert report['complete_control_jobs'] == 15
	assert report['compared_cells'] == 15
	assert report['source_lineages_audited'] is True
	assert report['downstream_identity_parity'] is True
	assert audited_sources == [CANDIDATE, CONTROL]

	paths = summarize_f3_paired_candidate_results(config)
	assert tuple(path.name for path in paths) == OUTPUT_NAMES
	assert {path.name for path in config.summary_root.iterdir()} == set(OUTPUT_NAMES)
	assert audited_sources == [CANDIDATE, CONTROL, CANDIDATE, CONTROL]
	rows = list(
		csv.DictReader(
			(config.summary_root / 'comparison.csv')
			.read_text(encoding='utf-8')
			.splitlines()
		)
	)
	assert len(rows) == 15
	assert [(row['data_size'], row['layout_id']) for row in rows] == [
		(data_size, layout_id) for data_size in DATA_SIZES for layout_id in LAYOUT_IDS
	]
	payload = json.loads((config.summary_root / 'summary.json').read_text())
	assert payload['job_counts'] == {
		'candidate': 15,
		'control': 15,
		'compared_cells': 15,
	}
	assert set(payload['sources']) == {'candidate', 'control'}
	assert payload['sources']['candidate']['checkpoint_sha256'] == 'a' * 64
	assert payload['sources']['control']['checkpoint_sha256'] == 'e' * 64
	assert payload['sources']['candidate']['model_id'] == CANDIDATE
	assert payload['sources']['control']['model_id'] == CONTROL
	assert payload['sources']['candidate']['semantic_lineage']['role'] == 'candidate'
	assert payload['sources']['control']['semantic_lineage']['role'] == 'control'
	assert payload['benchmark'] == {
		'canonical_config_path': str(config.canonical_config),
		'canonical_config_sha256': hashlib.sha256(b'canonical fixture\n').hexdigest(),
	}
	assert payload['shared_validation_identity'] == {
		'validation_mask_sha256': 'a' * 64,
		'validation_tile_manifest_sha256': 'b' * 64,
		'voxel_valid_mask_sha256': 'e' * 64,
		'evaluation_voxel_count': 1000,
	}
	assert payload['shared_evaluation_inputs'] == {
		'label_volume': {
			'path': str(
				_canonical_labels(config.runs_root.parent)['source_label_volume']
			),
			'sha256': '1' * 64,
		},
		'class_info': {
			'path': str(_canonical_labels(config.runs_root.parent)['class_info']),
			'sha256': '2' * 64,
		},
		'source_label_segy': {
			'path': str(
				_canonical_labels(config.runs_root.parent)['source_label_segy']
			),
			'sha256': '3' * 64,
		},
		'png_label_inventory': {
			'path': str(
				_canonical_labels(config.runs_root.parent)['png_label_inventory']
			),
			'sha256': '4' * 64,
		},
		'segy_geometry_json': {
			'path': str(
				_canonical_labels(config.runs_root.parent)['segy_geometry_json']
			),
			'sha256': '5' * 64,
		},
	}
	assert all(len(row['supervision_identity_sha256']) == 64 for row in rows)
	assert {row['candidate_decoder_initial_state_sha256'] for row in rows} == {'4' * 64}
	assert {row['control_decoder_initial_state_sha256'] for row in rows} == {'4' * 64}
	_assert_statistical_aggregates(payload)
	all_15 = payload['aggregates']['all_15']
	assert payload['model_means']['all_15'][CANDIDATE] == pytest.approx(
		all_15['candidate_mean']
	)
	markdown = (config.summary_root / 'summary.md').read_text(encoding='utf-8')
	assert CANDIDATE in markdown
	assert CONTROL in markdown
	assert 'all_15' in markdown
	assert 'all_15` describes the 15 layout-by-size cells' in markdown
	assert 'Overall layout-clustered inference' in markdown
	assert 'five `layout_id` means' in markdown


@pytest.mark.parametrize(
	('drift', 'message'),
	[
		(
			(CANDIDATE, 'embeddings_sha256', '9' * 64),
			'does not match current source',
		),
		(
			(CONTROL, 'supervision_identity', 'c' * 64),
			'supervision_identity differ',
		),
		(
			(CONTROL, '_validation_tile_manifest_sha256', 'c' * 64),
			'decoder validation tile identity does not match job evidence',
		),
		(
			('both', 'validation_identity', 'c' * 64),
			'validation_mask_sha256 values are not shared',
		),
	],
)
def test_inspection_rejects_stale_source_or_paired_condition_drift(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	drift: tuple[str, str, str],
	message: str,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	model_id, key, value = drift
	_patch_auditors(
		monkeypatch,
		config,
		evidence_drift=(model_id, 'layout_003', 'medium', key, value),
	)
	with pytest.raises(ValueError, match=message):
		inspect_f3_paired_candidate_results(config)
	assert not config.summary_root.exists()


@pytest.mark.parametrize(
	'key',
	['decoder_contract_identity_sha256', 'decoder_initial_state_sha256'],
)
def test_inspection_rejects_decoder_contract_or_initialization_drift(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	key: str,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	_patch_auditors(
		monkeypatch,
		config,
		decoder_drift=(CONTROL, 'layout_002', 'large', key, 'f' * 64),
	)
	with pytest.raises(ValueError, match=key):
		inspect_f3_paired_candidate_results(config)


@pytest.mark.parametrize(
	('key', 'message'),
	[
		('decoder_checkpoint_sha256', 'does not match job evidence'),
		('decoder_resolved_config_sha256', 'does not match decoder completion'),
		('prediction_metadata_sha256', 'does not match decoder completion'),
		(
			'voxel_valid_mask_sha256',
			'candidate/control evaluation voxel_valid_mask_sha256 differ',
		),
		('label_volume_path', 'does not match the canonical benchmark config'),
	],
)
def test_inspection_rejects_evaluation_cross_binding_drift(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	key: str,
	message: str,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	_patch_auditors(
		monkeypatch,
		config,
		evaluation_drift=(
			CANDIDATE,
			'layout_001',
			'medium',
			key,
			'f' * 64,
		),
	)
	with pytest.raises(ValueError, match=message):
		inspect_f3_paired_candidate_results(config)


def test_inspection_rejects_missing_or_unexpected_cell(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	_patch_auditors(monkeypatch, config)
	missing = (
		config.runs_root
		/ f'model={CANDIDATE}/layout=layout_004/size=large/evaluation/metrics.json'
	)
	missing.unlink()
	with pytest.raises(FileNotFoundError, match='missing completed metrics'):
		inspect_f3_paired_candidate_results(config)
	missing.parent.mkdir(parents=True, exist_ok=True)
	missing.write_text(
		json.dumps(
			{
				'aggregation_unit': 'unique_validation_voxel',
				'evaluation_voxel_count': 1000,
				'macro_f1': 0.5,
			}
		),
		encoding='utf-8',
	)
	(config.runs_root / f'model={CONTROL}/layout=layout_999').mkdir()
	with pytest.raises(ValueError, match='unexpected paired run condition'):
		inspect_f3_paired_candidate_results(config)


@pytest.mark.parametrize(
	('field', 'value', 'message'),
	[
		('macro_f1', True, 'must be numeric'),
		('macro_f1', float('nan'), 'must be finite'),
		('macro_f1', -0.01, r'must be within \[0, 1\]'),
		('macro_f1', 1.01, r'must be within \[0, 1\]'),
		('aggregation_unit', 'trace', 'aggregation_unit'),
		('evaluation_voxel_count', 999, 'voxel counts differ'),
	],
)
def test_inspection_rejects_malformed_primary_metric(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	field: str,
	value: object,
	message: str,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	_patch_auditors(monkeypatch, config)
	path = (
		config.runs_root
		/ f'model={CANDIDATE}/layout=layout_000/size=small/evaluation/metrics.json'
	)
	payload = json.loads(path.read_text())
	payload[field] = value
	path.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises((TypeError, ValueError), match=message):
		inspect_f3_paired_candidate_results(config)


def test_summary_refuses_overwrite_and_preserves_existing_files(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
) -> None:
	config = _config(tmp_path)
	_write_jobs(config)
	_patch_auditors(monkeypatch, config)
	config.summary_root.mkdir()
	sentinel = config.summary_root / 'sentinel.txt'
	sentinel.write_text('keep\n', encoding='utf-8')
	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_f3_paired_candidate_results(config)
	assert sentinel.read_text(encoding='utf-8') == 'keep\n'
	assert {path.name for path in config.summary_root.iterdir()} == {'sentinel.txt'}
	assert not list(config.summary_root.parent.glob('.*.staging-*'))


@pytest.mark.parametrize('flag', ['--check-only', '--dry-run'])
def test_read_only_cli_aliases_audit_without_writing(
	tmp_path: Path,
	monkeypatch: MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	flag: str,
) -> None:
	raw = _raw_config(tmp_path)
	config = _config(tmp_path)
	calls: list[str] = []

	def load_raw(_path: Path, *, loader: object) -> Mapping[str, object]:
		del loader
		return deepcopy(raw)

	def inspect(_config: F3PairedCandidateSummaryConfig) -> dict[str, object]:
		calls.append('inspect')
		return {
			'candidate_id': CANDIDATE,
			'control_id': CONTROL,
			'complete_candidate_jobs': 15,
			'complete_control_jobs': 15,
			'compared_cells': 15,
		}

	def reject_write(_config: F3PairedCandidateSummaryConfig) -> object:
		pytest.fail('read-only CLI attempted to write a summary')

	monkeypatch.setattr(summary_cli, 'load_config_for_cli', load_raw)
	monkeypatch.setattr(summary_cli, 'inspect_f3_paired_candidate_results', inspect)
	monkeypatch.setattr(
		summary_cli,
		'summarize_f3_paired_candidate_results',
		reject_write,
	)
	monkeypatch.setattr(
		sys,
		'argv',
		['summary', '--config', str(tmp_path / 'config.yaml'), flag],
	)
	summary_cli.main()
	assert calls == ['inspect']
	assert 'execution: check-only; summary files skipped' in capsys.readouterr().out
	assert not config.summary_root.exists()
