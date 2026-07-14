from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
)
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	BASELINE_MODEL_TAG,
	CANDIDATE_MODEL_TAG,
	F3VoxelDecoderSplitSuiteConfig,
	F3VoxelSplitDatasetSuiteConfig,
	F3VoxelSplitRobustnessPublishConfig,
	F3VoxelSplitRobustnessSummaryConfig,
	F3VoxelV0SplitSuiteConfig,
	VoxelRobustnessModel,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import voxel_robustness
from seis_ssl_cluster.f3.lithology.voxel_evaluation import EVALUATION_OUTPUT_FILES
from seis_ssl_cluster.f3.lithology.voxel_results import (
	EXPECTED_MODEL_TAGS,
	F3LithologyVoxelResultsConfig,
	F3LithologyVoxelResultsRun,
	summarize_f3_lithology_voxel_results,
)
from seis_ssl_cluster.f3.lithology.voxel_results import (
	SUMMARY_JSON as ORIGINAL_SUMMARY_JSON,
)
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	V0_RUN_MANIFEST_TYPE,
	V1_RUN_MANIFEST_TYPE,
	_read_manifest_rows,
	build_f3_lithology_voxel_split_datasets,
	run_f3_lithology_voxel_decoder_split_suite,
	run_f3_lithology_voxel_v0_split_suite,
	summarize_f3_lithology_voxel_split_robustness,
)
from seis_ssl_cluster.models.voxel_decoder import (
	voxel_decoder_architecture_mapping,
)


class _SyntheticProbe:
	classes_ = np.asarray([3, 5], dtype=np.int64)

	def predict(self, features: np.ndarray) -> np.ndarray:
		return np.where(features[:, 0] < 0, 3, 5)

	def predict_proba(self, features: np.ndarray) -> np.ndarray:
		positive = (features[:, 0] >= 0).astype(np.float64)
		return np.stack((1.0 - positive, positive), axis=1)


class _IdentityScaler:
	def transform(self, features: np.ndarray) -> np.ndarray:
		return features


def test_two_split_voxel_workflow_runs_end_to_end(tmp_path: Path) -> None:
	build_config, v0_config, v1_config = _synthetic_workflow_configs(tmp_path)

	datasets = build_f3_lithology_voxel_split_datasets(build_config)
	v0 = run_f3_lithology_voxel_v0_split_suite(v0_config)
	resumed_v0 = run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	probe_path = (
		v0_config.artifact_root / 'probes' / 'split_000' / 'baseline' / 'probe.joblib'
	)
	_probe_bytes = probe_path.read_bytes()
	probe_path.write_bytes(_probe_bytes + b'tampered')
	with pytest.raises(ValueError, match='probe_joblib hash identity mismatch'):
		run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	probe_path.write_bytes(_probe_bytes)
	validation_tokens = (
		v0_config.artifact_root
		/ 'tokens'
		/ 'split_000'
		/ 'baseline'
		/ 'validation_tokens.npz'
	)
	_token_bytes = validation_tokens.read_bytes()
	validation_tokens.write_bytes(_token_bytes + b'tampered')
	with pytest.raises(ValueError, match='lithology identity does not match config'):
		run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	validation_tokens.write_bytes(_token_bytes)
	baseline_v0 = next(
		row
		for row in v0.rows
		if row['split_id'] == 'split_000' and row['model_role'] == 'baseline'
	)
	candidate_v0 = next(
		row
		for row in v0.rows
		if row['split_id'] == 'split_000' and row['model_role'] == 'candidate'
	)
	projection_metadata = (
		Path(str(baseline_v0['prediction_dir'])) / 'prediction_metadata.json'
	)
	projection_bytes = projection_metadata.read_bytes()
	projection_payload = json.loads(projection_bytes)
	candidate_token_dir = Path(str(candidate_v0['token_prediction_dir']))
	projection_payload['inputs']['token_prediction_dir'] = str(candidate_token_dir)
	projection_payload['source_identity']['token_artifact_files'] = {
		key: _file_identity(candidate_token_dir / name)
		for key, name in (
			('token_predictions', 'f3_token_predictions.npy'),
			('token_probabilities', 'f3_token_probabilities.npy'),
			('valid_token_grid', 'f3_valid_token_grid.npy'),
			('prediction_metadata', 'prediction_metadata.json'),
		)
	}
	projection_metadata.write_text(json.dumps(projection_payload), encoding='utf-8')
	with pytest.raises(ValueError, match='V0 prediction source token_predictions path'):
		run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	projection_metadata.write_bytes(projection_bytes)
	v1 = run_f3_lithology_voxel_decoder_split_suite(v1_config, device='cpu')
	baseline_v1 = next(
		row
		for row in v1.rows
		if row['split_id'] == 'split_000' and row['model_role'] == 'baseline'
	)
	candidate_v1 = next(
		row
		for row in v1.rows
		if row['split_id'] == 'split_000' and row['model_role'] == 'candidate'
	)
	decoder_prediction_metadata = (
		Path(str(baseline_v1['prediction_dir'])) / 'prediction_metadata.json'
	)
	decoder_prediction_bytes = decoder_prediction_metadata.read_bytes()
	decoder_prediction_payload = json.loads(decoder_prediction_bytes)
	candidate_best = Path(str(candidate_v1['decoder_dir'])) / 'best.pt'
	decoder_prediction_payload['source_identity']['decoder_checkpoint'] = (
		_file_identity(candidate_best)
	)
	decoder_prediction_metadata.write_text(
		json.dumps(decoder_prediction_payload), encoding='utf-8'
	)
	with pytest.raises(ValueError, match='V1 prediction decoder checkpoint path'):
		run_f3_lithology_voxel_decoder_split_suite(
			v1_config, only_missing=True, device='cpu'
		)
	decoder_prediction_metadata.write_bytes(decoder_prediction_bytes)

	assert [row['split_id'] for row in datasets.rows] == [
		'split_000',
		'split_001',
	]
	assert len(v0.rows) == len(resumed_v0.rows) == len(v1.rows) == 4
	assert all(row['status'] == 'complete' for row in (*v0.rows, *v1.rows))
	assert all(
		Path(str(row['evaluation_dir']), 'evaluation_metadata.json').is_file()
		for row in (*v0.rows, *v1.rows)
	)


def test_v0_rejects_cross_model_paired_identity_mismatch(tmp_path: Path) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	for path in (v0_config.split_dataset_manifest, v0_config.probe_run_manifest):
		payload = json.loads(path.read_text(encoding='utf-8'))
		candidate = next(
			row
			for row in payload['rows']
			if row['split_id'] == 'split_000' and row['model_role'] == 'candidate'
		)
		candidate['paired_identity_hash'] = 'mismatched-cross-model-pair'
		path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='paired identity hash mismatch for split_000'):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_v0_rejects_noncanonical_probe_spec(tmp_path: Path) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	payload = json.loads(v0_config.probe_run_manifest.read_text(encoding='utf-8'))
	payload['probe']['spec'] = 'replacement_probe'
	v0_config.probe_run_manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='must use probe spec'):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_v0_rejects_probe_settings_different_from_manifest(tmp_path: Path) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	payload = json.loads(v0_config.probe_run_manifest.read_text(encoding='utf-8'))
	row = payload['rows'][0]
	resolved_path = Path(row['probe_output_dir']) / 'probe_config_resolved.json'
	resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
	resolved['probe']['random_state'] = 7
	resolved_path.write_text(json.dumps(resolved), encoding='utf-8')

	with pytest.raises(
		ValueError, match='settings do not match the prior run manifest'
	):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_v0_rejects_model_valid_mask_different_from_canonical(
	tmp_path: Path,
) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	candidate = next(model for model in v0_config.models if model.role == 'candidate')
	valid_tokens = (
		candidate.embeddings_dir / f'{v0_config.dataset["name"]}.valid_tokens.npy'
	)
	np.save(valid_tokens, np.zeros((2, 2, 2), dtype=np.bool_), allow_pickle=False)

	with pytest.raises(
		ValueError,
		match='does not match the voxel dataset canonical valid-token identity',
	):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_only_missing_rejects_stale_split_dataset_inventory(tmp_path: Path) -> None:
	build_config, _, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	manifest = json.loads(
		build_config.split_inventory_manifest.read_text(encoding='utf-8')
	)
	stale_source = Path(manifest['rows'][0]['png_label_inventory'])
	replacement = stale_source.with_name('replacement_inventory.csv')
	replacement.write_bytes(stale_source.read_bytes())
	manifest['rows'][0]['png_label_inventory'] = str(replacement)
	build_config.split_inventory_manifest.write_text(
		json.dumps(manifest), encoding='utf-8'
	)

	with pytest.raises(ValueError, match='voxel inventory path identity mismatch'):
		build_f3_lithology_voxel_split_datasets(build_config, only_missing=True)


def test_only_missing_rejects_altered_split_grid(tmp_path: Path) -> None:
	build_config, _, _ = _synthetic_workflow_configs(tmp_path)
	result = build_f3_lithology_voxel_split_datasets(build_config)
	grid_path = Path(str(result.rows[0]['split_grid']['path']))
	grid = np.load(grid_path)
	grid.flat[0] = (int(grid.flat[0]) + 1) % 3
	np.save(grid_path, grid, allow_pickle=False)

	with pytest.raises(
		ValueError, match='split grid does not match the source inventory'
	):
		build_f3_lithology_voxel_split_datasets(build_config, only_missing=True)


def test_v0_rejects_token_suite_from_different_split_inventory(
	tmp_path: Path,
) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	payload = json.loads(v0_config.split_dataset_manifest.read_text(encoding='utf-8'))
	payload['suite']['split_inventory_manifest'] = str(tmp_path / 'other.json')
	v0_config.split_dataset_manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='different split inventories'):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_synthetic_split_summary_uses_paired_split_deltas(  # noqa: PLR0915
	tmp_path: Path,
) -> None:
	v0_rows = []
	v1_rows = []
	for split_index in range(6):
		split_id = f'split_{split_index:03d}'
		grid = tmp_path / 'grids' / f'{split_id}.npy'
		grid.parent.mkdir(parents=True, exist_ok=True)
		np.save(grid, np.full((1,), split_index, dtype=np.uint8))
		voxel_identity = file_sha256(grid)
		for role, tag, offset in (
			('baseline', BASELINE_MODEL_TAG, 0.0),
			('candidate', CANDIDATE_MODEL_TAG, 0.04),
		):
			v0_dir = tmp_path / 'v0' / split_id / role
			v1_dir = tmp_path / 'v1' / split_id / role
			v0_identity = _write_evaluation(
				v0_dir,
				score=0.50 + offset,
				boundary_mae=2.0 - offset,
				model_tag=tag,
				prediction_kind='token_projection_nearest',
				split_grid=grid,
			)
			v1_identity = _write_evaluation(
				v1_dir,
				score=0.60 + offset,
				boundary_mae=1.5 - offset,
				model_tag=tag,
				prediction_kind='frozen_embedding_decoder',
				split_grid=grid,
			)
			base = {
				'split_id': split_id,
				'model_role': role,
				'model_tag': tag,
				'voxel_dataset_identity': voxel_identity,
				'status': 'complete',
			}
			v0_rows.append(
				{**base, **v0_identity, 'evaluation_dir': str(v0_dir)}
			)
			v1_rows.append(
				{**base, **v1_identity, 'evaluation_dir': str(v1_dir)}
			)
	v0_manifest = tmp_path / 'v0.json'
	v1_manifest = tmp_path / 'v1.json'
	_write_manifest(v0_manifest, V0_RUN_MANIFEST_TYPE, v0_rows)
	_write_manifest(v1_manifest, V1_RUN_MANIFEST_TYPE, v1_rows)

	result = summarize_f3_lithology_voxel_split_robustness(
		F3VoxelSplitRobustnessSummaryConfig(
			suite_root=tmp_path / 'suite',
			v0_run_manifest=v0_manifest,
			v1_run_manifest=v1_manifest,
			baseline_model_tag=BASELINE_MODEL_TAG,
			candidate_model_tag=CANDIDATE_MODEL_TAG,
		)
	)

	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert result.status == 'positive'
	assert payload['m2a_vs_m1_voxel_robustness'] == 'positive'
	assert payload['statistical_unit'] == 'split'
	assert payload['voxel_level_significance_computed'] is False
	assert payload['p_values_computed'] is False
	assert payload['confidence_intervals_computed'] is False
	assert payload['decoder_architecture']['upsample_mode'] == 'nearest'
	assert result.summary_markdown is not None
	assert 'voxelwise_layer_norm' in result.summary_markdown.read_text(
		encoding='utf-8'
	)
	primary = next(
		row
		for row in payload['aggregates']
		if row['comparison'] == 'v1_candidate_minus_baseline'
		and row['metric'] == 'primary_metrics_simultaneous'
	)
	assert primary['win_count'] == 6
	assert primary['win_rate'] == 1.0
	assert any(
		row['comparison'] == 'candidate_v1_minus_v0' for row in payload['raw_rows']
	)
	first_metrics = Path(v1_rows[0]['evaluation_dir']) / 'metrics.json'
	first_metadata = Path(v1_rows[0]['evaluation_dir']) / 'evaluation_metadata.json'
	metrics_bytes = first_metrics.read_bytes()
	metadata_bytes = first_metadata.read_bytes()
	partial_metrics = json.loads(metrics_bytes)
	partial_metrics['macro_f1'] = None
	first_metrics.write_text(json.dumps(partial_metrics), encoding='utf-8')
	partial_metadata = json.loads(metadata_bytes)
	partial_metadata['outputs']['metrics.json'] = _file_identity(first_metrics)
	first_metadata.write_text(json.dumps(partial_metadata), encoding='utf-8')
	with pytest.raises(ValueError, match='incomplete run row'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'null-primary-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)
	first_metrics.write_bytes(metrics_bytes)
	first_metadata.write_bytes(metadata_bytes)
	tampered_metrics = json.loads(metrics_bytes)
	tampered_metrics['macro_f1'] += 0.01
	first_metrics.write_text(json.dumps(tampered_metrics), encoding='utf-8')
	with pytest.raises(ValueError, match='incomplete run row'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'tampered-metrics-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)
	first_metrics.write_bytes(metrics_bytes)
	original = tmp_path / 'original'
	_write_original_summary_bundle(tmp_path / 'original-runs', original)
	publish_dir = tmp_path / 'results' / 'voxel'
	published = summarize_f3_lithology_voxel_split_robustness(
		F3VoxelSplitRobustnessSummaryConfig(
			suite_root=tmp_path / 'published-suite',
			v0_run_manifest=v0_manifest,
			v1_run_manifest=v1_manifest,
			baseline_model_tag=BASELINE_MODEL_TAG,
			candidate_model_tag=CANDIDATE_MODEL_TAG,
			original_summary_dir=original,
			publish=F3VoxelSplitRobustnessPublishConfig(
				enabled=True,
				results_root=tmp_path / 'results',
				output_dir=publish_dir,
			),
		)
	)
	assert published.publish_manifest is not None
	targets = {
		item.target.relative_to(publish_dir).as_posix()
		for item in published.publish_manifest.items
	}
	assert f'robustness/{published.summary_json.name}' in targets
	assert ORIGINAL_SUMMARY_JSON in targets
	assert not any(target.endswith(('.npy', '.pt', '.joblib')) for target in targets)

	first_evaluation = Path(v1_rows[0]['evaluation_dir']) / 'evaluation_metadata.json'
	evaluation_payload = json.loads(first_evaluation.read_text(encoding='utf-8'))
	evaluation_payload['prediction_kind'] = 'token_projection_nearest'
	first_evaluation.write_text(json.dumps(evaluation_payload), encoding='utf-8')
	with pytest.raises(ValueError, match='prediction kind'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'wrong-kind-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)
	evaluation_payload['prediction_kind'] = 'frozen_embedding_decoder'
	first_evaluation.write_text(json.dumps(evaluation_payload), encoding='utf-8')

	v1_payload = json.loads(v1_manifest.read_text(encoding='utf-8'))
	v1_payload['rows'][0]['decoder_dir'] = str(tmp_path / 'other-decoder')
	v1_manifest.write_text(json.dumps(v1_payload), encoding='utf-8')
	with pytest.raises(ValueError, match='V1 prediction decoder checkpoint path'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'wrong-decoder-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)
	v1_payload['rows'][0]['decoder_dir'] = v1_rows[0]['decoder_dir']
	v1_manifest.write_text(json.dumps(v1_payload), encoding='utf-8')
	v1_payload['rows'][0]['voxel_dataset_identity'] = 'different-voxel-dataset'
	v1_manifest.write_text(json.dumps(v1_payload), encoding='utf-8')
	with pytest.raises(ValueError, match='split-grid identity'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'rejected-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)


def test_publish_rejects_corrupt_original_summary_bundle(tmp_path: Path) -> None:
	original = tmp_path / 'original'
	original.mkdir()
	(original / ORIGINAL_SUMMARY_JSON).write_text('not JSON', encoding='utf-8')
	result = voxel_robustness.VoxelSplitRobustnessSummaryResult(
		output_dir=tmp_path / 'suite',
		summary_json=tmp_path / 'suite' / 'summary.json',
		paired_rows_csv=tmp_path / 'suite' / 'paired.csv',
		aggregates_csv=tmp_path / 'suite' / 'aggregates.csv',
		status='hold',
	)
	config = F3VoxelSplitRobustnessSummaryConfig(
		suite_root=tmp_path / 'suite',
		v0_run_manifest=tmp_path / 'v0.json',
		v1_run_manifest=tmp_path / 'v1.json',
		baseline_model_tag=BASELINE_MODEL_TAG,
		candidate_model_tag=CANDIDATE_MODEL_TAG,
		original_summary_dir=original,
		publish=F3VoxelSplitRobustnessPublishConfig(
			enabled=True,
			results_root=tmp_path / 'results',
			output_dir=tmp_path / 'results' / 'voxel',
		),
	)

	with pytest.raises(json.JSONDecodeError):
		voxel_robustness._publish_robustness_summary(result, config)  # noqa: SLF001


def test_partial_split_summary_is_rejected(tmp_path: Path) -> None:
	v0_rows = []
	v1_rows = []
	grid = tmp_path / 'grid.npy'
	np.save(grid, np.zeros((1,), dtype=np.uint8))
	for role, tag in (
		('baseline', BASELINE_MODEL_TAG),
		('candidate', CANDIDATE_MODEL_TAG),
	):
		for prediction_kind, rows, stage in (
			('token_projection_nearest', v0_rows, 'v0'),
			('frozen_embedding_decoder', v1_rows, 'v1'),
		):
			evaluation = tmp_path / stage / role
			identity = _write_evaluation(
				evaluation,
				score=0.5,
				boundary_mae=2.0,
				model_tag=tag,
				prediction_kind=prediction_kind,
				split_grid=grid,
			)
			rows.append(
			{
				'split_id': 'split_000',
				'model_role': role,
				'model_tag': tag,
				'voxel_dataset_identity': file_sha256(grid),
				'evaluation_dir': str(evaluation),
				'status': 'complete',
				**identity,
			}
			)
	v0_manifest = tmp_path / 'v0.json'
	v1_manifest = tmp_path / 'v1.json'
	_write_manifest(v0_manifest, V0_RUN_MANIFEST_TYPE, v0_rows)
	_write_manifest(v1_manifest, V1_RUN_MANIFEST_TYPE, v1_rows)

	with pytest.raises(ValueError, match='all six'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)


def test_v1_baseline_candidate_decoder_architecture_mismatch_is_rejected() -> None:
	architecture = voxel_decoder_architecture_mapping(
		embedding_dim=2,
		class_count=2,
		hidden_channels=(2,),
		upsample_factors=((2, 2, 2),),
	)
	rows = (
		{
			'split_id': 'split_000',
			'model_role': 'baseline',
			'status': 'complete',
			'decoder_architecture': architecture,
		},
		{
			'split_id': 'split_000',
			'model_role': 'candidate',
			'status': 'complete',
			'decoder_architecture': {
				**architecture,
				'hidden_channels': [4],
			},
		},
	)
	with pytest.raises(ValueError, match='paired decoder decoder_architecture'):
		voxel_robustness._validate_paired_decoder_identity(  # noqa: SLF001
			rows, split_id='split_000'
		)


def test_null_deltas_remain_in_split_denominators() -> None:
	rows = []
	for split_index in range(6):
		for metric in voxel_robustness.SUMMARY_METRICS:
			delta = (
				(-0.1 if metric == 'boundary_position_mae' else 0.1)
				if split_index == 0
				else None
			)
			rows.append(
				{
					'split_id': f'split_{split_index:03d}',
					'comparison': 'v1_candidate_minus_baseline',
					'metric': metric,
					'delta': delta,
				}
			)

	aggregates = voxel_robustness._aggregate_metric_rows(rows)  # noqa: SLF001
	macro = next(row for row in aggregates if row['metric'] == 'macro_f1')
	primary = next(
		row for row in aggregates if row['metric'] == 'primary_metrics_simultaneous'
	)
	assert macro['split_count'] == 6
	assert macro['win_count'] == 1
	assert macro['win_rate'] == pytest.approx(1 / 6)
	assert primary['split_count'] == 6
	assert primary['win_count'] == 1
	assert primary['win_rate'] == pytest.approx(1 / 6)
	status, _ = voxel_robustness._provisional_status(  # noqa: SLF001
		rows, aggregates
	)
	assert status == 'hold'


def test_all_tied_primary_result_remains_hold() -> None:
	rows = [
		{
			'split_id': f'split_{split_index:03d}',
			'comparison': 'v1_candidate_minus_baseline',
			'metric': metric,
			'delta': 0.0,
		}
		for split_index in range(6)
		for metric in voxel_robustness.SUMMARY_METRICS
	]

	aggregates = voxel_robustness._aggregate_metric_rows(rows)  # noqa: SLF001
	status, _ = voxel_robustness._provisional_status(  # noqa: SLF001
		rows, aggregates
	)

	assert status == 'hold'


def test_one_loss_with_all_other_metrics_tied_remains_hold() -> None:
	rows = [
		{
			'split_id': f'split_{split_index:03d}',
			'comparison': 'v1_candidate_minus_baseline',
			'metric': metric,
			'delta': -0.1 if split_index == 0 and metric == 'macro_f1' else 0.0,
		}
		for split_index in range(6)
		for metric in voxel_robustness.SUMMARY_METRICS
	]

	aggregates = voxel_robustness._aggregate_metric_rows(rows)  # noqa: SLF001
	status, _ = voxel_robustness._provisional_status(  # noqa: SLF001
		rows, aggregates
	)

	assert status == 'hold'


def test_duplicate_split_run_is_rejected(tmp_path: Path) -> None:
	path = tmp_path / 'manifest.json'
	row = {
		'split_id': 'split_000',
		'model_role': 'baseline',
		'model_tag': BASELINE_MODEL_TAG,
	}
	_write_manifest(path, V0_RUN_MANIFEST_TYPE, [row, row])
	with pytest.raises(ValueError, match='duplicate split/run'):
		_read_manifest_rows(
			path,
			artifact_type=V0_RUN_MANIFEST_TYPE,
			required=('split_id', 'model_role', 'model_tag'),
		)


def test_v1_paired_identity_failure_is_recorded_for_resume(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	build_config, _, config = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	config = replace(config, train=replace(config.train, epochs=2))
	real_runner = voxel_robustness.run_f3_lithology_voxel_decoder

	def interrupt_first_job(config, *, device, resume=None):
		return real_runner(config, device=device, resume=resume, max_steps=1)

	monkeypatch.setattr(
		voxel_robustness,
		'run_f3_lithology_voxel_decoder',
		interrupt_first_job,
	)

	with pytest.raises(RuntimeError, match='decoder job did not complete'):
		run_f3_lithology_voxel_decoder_split_suite(config, device='cpu')

	payload = json.loads(
		(config.output_root / 'v1_split_run_manifest.json').read_text(encoding='utf-8')
	)
	assert [row['status'] for row in payload['rows']] == ['failed']
	failed = payload['rows'][0]
	resume_path = Path(failed['resume_path'])
	assert resume_path.name == 'latest.pt'
	assert torch.load(resume_path, weights_only=False)['checkpoint_kind'] != 'completed'

	monkeypatch.setattr(
		voxel_robustness,
		'run_f3_lithology_voxel_decoder',
		real_runner,
	)
	resumed = run_f3_lithology_voxel_decoder_split_suite(
		config, only_missing=True, device='cpu'
	)
	assert all(row['status'] == 'complete' for row in resumed.rows)
	assert torch.load(resume_path, weights_only=False)['checkpoint_kind'] == 'completed'


def _write_evaluation(  # noqa: PLR0913
	root: Path,
	*,
	score: float,
	boundary_mae: float,
	model_tag: str,
	prediction_kind: str,
	split_grid: Path,
) -> dict[str, object]:
	root.mkdir(parents=True)
	metrics = {
		'macro_f1': score,
		'mean_iou': score,
		'balanced_accuracy': score,
		'per_class_f1': {'3': score, '5': score},
		'per_class_iou': {'3': score, '5': score},
		'evaluation_voxel_count': 10,
		'aggregation_unit': 'unique_validation_voxel',
	}
	boundary = {
		'vertical_boundary_f1_at_2': score,
		'vertical_boundary_f1_at_4': score,
		'vertical_boundary_position_mae_at_4': boundary_mae,
		'vertical_boundary_class_3_recall_at_2': score,
		'vertical_boundary_class_3_recall_at_4': score,
		'vertical_boundary_class_5_recall_at_2': score,
		'vertical_boundary_class_5_recall_at_4': score,
	}
	(root / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
	(root / 'boundary_metrics.json').write_text(json.dumps(boundary), encoding='utf-8')
	prediction_metadata = root / 'prediction_metadata.json'
	prediction_payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_voxel_prediction',
		'model_tag': model_tag,
		'prediction_kind': prediction_kind,
	}
	row_identity: dict[str, object] = {'prediction_dir': str(root)}
	if prediction_kind == 'frozen_embedding_decoder':
		architecture = voxel_decoder_architecture_mapping(
			embedding_dim=2,
			class_count=2,
			hidden_channels=(2,),
			upsample_factors=((2, 2, 2),),
		)
		prediction_payload['decoder_architecture'] = architecture
		sources = root / 'decoder_sources'
		sources.mkdir()
		valid_tokens = sources / 'valid_tokens.npy'
		np.save(valid_tokens, np.ones((1,), dtype=np.bool_))
		train_tiles = sources / 'train_tile_manifest.json'
		validation_tiles = sources / 'validation_tile_manifest.json'
		train_tiles.write_text('{"tiles":[0]}\n', encoding='utf-8')
		validation_tiles.write_text('{"tiles":[1]}\n', encoding='utf-8')
		checkpoint = sources / 'best.pt'
		class_weights = [1.0, 2.0]
		torch.save({'class_weights': torch.tensor(class_weights)}, checkpoint)
		prediction_payload['source_identity'] = {
			'decoder_checkpoint': _file_identity(checkpoint),
			'artifact_identities': {
				'valid_tokens': _file_identity(valid_tokens),
			},
			'tile_manifests': {
				'train': _file_identity(train_tiles),
				'validation': _file_identity(validation_tiles),
			},
		}
		row_identity = {
			**row_identity,
			'decoder_dir': str(sources),
			'source_valid_tokens': _file_identity(valid_tokens),
			'train_tile_manifest': _file_identity(train_tiles),
			'validation_tile_manifest': _file_identity(validation_tiles),
			'class_weights': class_weights,
			'decoder_architecture': architecture,
		}
	else:
		sources = root / 'token_sources'
		sources.mkdir()
		source_files = {}
		for key, name in (
			('token_predictions', 'f3_token_predictions.npy'),
			('token_probabilities', 'f3_token_probabilities.npy'),
			('valid_token_grid', 'f3_valid_token_grid.npy'),
			('prediction_metadata', 'prediction_metadata.json'),
		):
			path = sources / name
			path.write_bytes(f'{model_tag}:{key}'.encode())
			source_files[key] = _file_identity(path)
		prediction_payload['inputs'] = {'token_prediction_dir': str(sources)}
		prediction_payload['source_identity'] = {
			'token_artifact_files': source_files
		}
		row_identity['token_prediction_dir'] = str(sources)
	prediction_metadata.write_text(json.dumps(prediction_payload), encoding='utf-8')
	with (root / 'boundary_region_metrics.csv').open(
		'w', encoding='utf-8', newline=''
	) as file_obj:
		writer = csv.DictWriter(
			file_obj, fieldnames=('region', 'radius', 'macro_f1', 'mean_iou')
		)
		writer.writeheader()
		for radius in (2, 4):
			writer.writerow(
				{
					'region': 'boundary',
					'radius': radius,
					'macro_f1': score,
					'mean_iou': score,
				}
			)
	for name in EVALUATION_OUTPUT_FILES:
		path = root / name
		if not path.exists():
			path.write_text(
				'{}\n' if path.suffix == '.json' else '\n', encoding='utf-8'
			)
	evaluation_payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_voxel_evaluation',
		'schema_version': 2,
		'model_tag': model_tag,
		'prediction_kind': prediction_kind,
		'aggregation': {
			'primary_unit': 'unique_validation_voxel',
			'split_code': 2,
			'intersection_voxels_counted_once': True,
			'per_slice_planes_evaluated_independently': True,
			'voxel_independence_p_values_computed': False,
		},
		'inputs': {
			'prediction_metadata': _file_identity(prediction_metadata),
			'voxel_split_grid': _file_identity(split_grid),
		},
		'summary': {'unique_validation_voxel_count': 10},
		'outputs': {
			name: _file_identity(root / name) for name in EVALUATION_OUTPUT_FILES
		},
	}
	if prediction_kind == 'frozen_embedding_decoder':
		evaluation_payload['decoder_architecture'] = architecture
	(root / 'evaluation_metadata.json').write_text(
		json.dumps(
			evaluation_payload
		),
		encoding='utf-8',
	)
	return row_identity


def _file_identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_manifest(
	path: Path,
	artifact_type: str,
	rows: object,
	**extra: object,
) -> None:
	path.write_text(
		json.dumps({'artifact_type': artifact_type, **extra, 'rows': rows}),
		encoding='utf-8',
	)


def _write_original_summary_bundle(run_root: Path, output_dir: Path) -> None:
	grid = run_root / 'supervision_split_grid.npy'
	grid.parent.mkdir(parents=True)
	np.save(grid, np.ones((1,), dtype=np.uint8))
	runs = []
	for model_index, model in enumerate(('MAE', 'M1', 'M2-A')):
		for version in ('V0', 'V1'):
			input_dir = run_root / model.replace('-', '') / version
			score = 0.5 + model_index * 0.05 + (0.02 if version == 'V1' else 0.0)
			_write_evaluation(
				input_dir,
				score=score,
				boundary_mae=1.0 - score,
				model_tag=EXPECTED_MODEL_TAGS[model],
				prediction_kind=(
					'token_projection_nearest'
					if version == 'V0'
					else 'frozen_embedding_decoder'
				),
				split_grid=grid,
			)
			metrics_path = input_dir / 'metrics.json'
			metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
			metrics['class_ids'] = [3, 5]
			metrics['evaluation_voxel_count'] = 10
			metrics_path.write_text(json.dumps(metrics), encoding='utf-8')
			metadata_path = input_dir / 'evaluation_metadata.json'
			metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
			metadata['summary'] = {'unique_validation_voxel_count': 10}
			metadata['outputs']['metrics.json'] = _file_identity(metrics_path)
			metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
			runs.append(F3LithologyVoxelResultsRun(model, version, input_dir))
	summarize_f3_lithology_voxel_results(
		F3LithologyVoxelResultsConfig(runs=tuple(runs), output_dir=output_dir)
	)


def _synthetic_workflow_configs(  # noqa: PLR0915
	tmp_path: Path,
) -> tuple[
	F3VoxelSplitDatasetSuiteConfig,
	F3VoxelV0SplitSuiteConfig,
	F3VoxelDecoderSplitSuiteConfig,
]:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	suite_root = artifact_root / 'voxel_suite'
	artifact_root.mkdir()
	f3_root.mkdir()
	dataset = {'name': 'synthetic_f3', 'version': 'voxel_robustness_test'}
	labels = np.empty((4, 4, 4), dtype=np.int16)
	for x, y, z in np.ndindex(labels.shape):
		labels[x, y, z] = 3 if (x // 2 + y // 2 + z // 2) % 2 == 0 else 5
	label_volume = artifact_root / 'labels.npy'
	np.save(label_volume, labels, allow_pickle=False)
	source_label_segy = f3_root / 'labels.sgy'
	source_label_segy.write_bytes(b'synthetic label segy')
	class_info = artifact_root / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				'3': {'name': 'class three', 'color': [3, 3, 3]},
				'5': {'name': 'class five', 'color': [5, 5, 5]},
			}
		),
		encoding='utf-8',
	)
	geometry = artifact_root / 'geometry.json'
	geometry.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [4, 4, 4],
						'iline_min': 100,
						'iline_max': 103,
						'xline_min': 200,
						'xline_max': 203,
					}
				}
			}
		),
		encoding='utf-8',
	)
	inventory_rows = []
	for split_index in range(2):
		split_id = f'split_{split_index:03d}'
		inventory = artifact_root / f'{split_id}_inventory.csv'
		with inventory.open('w', encoding='utf-8', newline='') as file_obj:
			writer = csv.DictWriter(
				file_obj,
				fieldnames=('relative_path', 'split', 'slice_type', 'slice_index'),
			)
			writer.writeheader()
			writer.writerows(
				(
					{
						'relative_path': f'{split_id}_train.png',
						'split': 'train',
						'slice_type': 'inline',
						'slice_index': 100 + split_index,
					},
					{
						'relative_path': f'{split_id}_validation.png',
						'split': 'validation',
						'slice_type': 'crossline',
						'slice_index': 203 - split_index,
					},
				)
			)
		inventory_rows.append(
			{'split_id': split_id, 'png_label_inventory': str(inventory)}
		)
	inventory_manifest = artifact_root / 'split_inventory_manifest.json'
	_write_manifest(
		inventory_manifest,
		voxel_robustness.INVENTORY_ARTIFACT_TYPE,
		inventory_rows,
	)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	model_specs = (
		('baseline', BASELINE_MODEL_TAG),
		('candidate', CANDIDATE_MODEL_TAG),
	)
	models = []
	for role, model_tag in model_specs:
		checkpoint = artifact_root / 'pretraining' / model_tag / 'best.pt'
		checkpoint.parent.mkdir(parents=True)
		checkpoint.write_bytes(f'{model_tag} checkpoint'.encode())
		embeddings_dir = (
			artifact_root
			/ 'embeddings'
			/ 'f3'
			/ dataset['version']
			/ model_tag
			/ 'synthetic'
		)
		embeddings_dir.mkdir(parents=True)
		token_classes = np.asarray([[[3, 5], [5, 3]], [[5, 3], [3, 5]]], dtype=np.int16)
		embeddings = np.stack(
			(
				np.where(token_classes == 3, -1.0, 1.0),
				np.ones(token_classes.shape),
			),
			axis=-1,
		).astype(np.float32)
		np.save(
			embeddings_dir / f'{dataset["name"]}.embeddings.npy',
			embeddings,
			allow_pickle=False,
		)
		np.save(
			embeddings_dir / f'{dataset["name"]}.valid_tokens.npy',
			valid_tokens,
			allow_pickle=False,
		)
		metadata = {
			'volume_shape_xyz': [4, 4, 4],
			'patch_size': [2, 2, 2],
			'token_grid_shape': [2, 2, 2],
			'embedding_dim': 2,
			'checkpoint_path': str(checkpoint),
			'checkpoint_sha256': file_sha256(checkpoint),
			'preprocessing': {'kind': 'synthetic'},
			'zero_mask': {'enabled': True},
		}
		(embeddings_dir / f'{dataset["name"]}.embedding_metadata.json').write_text(
			json.dumps(metadata), encoding='utf-8'
		)
		models.append(VoxelRobustnessModel(role, model_tag, embeddings_dir, checkpoint))
	reference_metadata = (
		models[0].embeddings_dir / f'{dataset["name"]}.embedding_metadata.json'
	)
	reference_valid_tokens = (
		models[0].embeddings_dir / f'{dataset["name"]}.valid_tokens.npy'
	)
	build_config = F3VoxelSplitDatasetSuiteConfig(
		split_inventory_manifest=inventory_manifest,
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		reference_metadata_json=reference_metadata,
		reference_valid_tokens=reference_valid_tokens,
		ignore_z_border_samples=0,
		overwrite=False,
	)
	dataset_rows = []
	probe_rows = []
	probe_settings = {
		'spec': 'linear_balanced_v1',
		'type': 'logistic_regression',
		'feature_scaling': 'standard',
		'class_weight': 'balanced',
		'max_iter': 2000,
		'hidden_dims': [256, 128],
		'dropout': 0.2,
		'max_epochs': 200,
		'early_stopping_patience': 20,
		'batch_size': 1024,
		'learning_rate': 1.0e-3,
		'weight_decay': 0.0,
		'random_state': 42,
	}
	for split_index in range(2):
		split_id = f'split_{split_index:03d}'
		for role, model_tag in model_specs:
			token_root = artifact_root / 'tokens' / split_id / role
			token_root.mkdir(parents=True)
			train_tokens_path = token_root / 'train_tokens.npz'
			validation_tokens_path = token_root / 'validation_tokens.npz'
			_write_empty_validation_tokens(train_tokens_path, dataset['name'])
			_write_empty_validation_tokens(validation_tokens_path, dataset['name'])
			metadata_path = token_root / 'token_dataset_metadata.json'
			metadata_path.write_text('{}\n', encoding='utf-8')
			paired_hash = voxel_robustness.paired_token_identity_hash(
				voxel_robustness.load_token_dataset_npz(train_tokens_path),
				voxel_robustness.load_token_dataset_npz(validation_tokens_path),
			)
			probe_dir = artifact_root / 'probes' / split_id / role
			probe_dir.mkdir(parents=True)
			joblib.dump(_SyntheticProbe(), probe_dir / 'probe.joblib')
			joblib.dump(_IdentityScaler(), probe_dir / 'scaler.joblib')
			(probe_dir / 'probe_config_resolved.json').write_text(
				json.dumps(
					{
						'artifact_type': 'f3_lithology_probe',
						'model': {'tag': model_tag, 'role': role},
						'probe': probe_settings,
						'token_dataset': {
							'input_dir': str(token_root),
							'split_id': split_id,
							'paired_identity_hash': paired_hash,
						},
						'inputs': {
							'train_tokens': str(train_tokens_path),
							'validation_tokens': str(validation_tokens_path),
						},
						'outputs': {
							'probe_joblib': str(probe_dir / 'probe.joblib'),
							'scaler_joblib': str(probe_dir / 'scaler.joblib'),
						},
					}
				),
				encoding='utf-8',
			)
			dataset_rows.append(
				{
					'split_id': split_id,
					'model_role': role,
					'model_tag': model_tag,
					'token_dataset_root': str(token_root),
					'train_tokens': str(train_tokens_path),
					'validation_tokens': str(validation_tokens_path),
					'metadata_json': str(metadata_path),
					'paired_identity_hash': paired_hash,
				}
			)
			probe_rows.append(
				{
					'split_id': split_id,
					'model_role': role,
					'model_tag': model_tag,
					'token_dataset_root': str(token_root),
					'probe_output_dir': str(probe_dir),
					'probe_spec': 'linear_balanced_v1',
					'probe_joblib': _file_identity(probe_dir / 'probe.joblib'),
					'scaler_joblib': _file_identity(probe_dir / 'scaler.joblib'),
					'paired_identity_hash': paired_hash,
				}
			)
	dataset_manifest = artifact_root / 'split_dataset_manifest.json'
	probe_manifest = artifact_root / 'split_probe_run_manifest.json'
	_write_manifest(
		dataset_manifest,
		'f3_lithology_split_sweep_token_dataset_manifest',
		dataset_rows,
		suite={'split_inventory_manifest': str(inventory_manifest)},
	)
	_write_manifest(
		probe_manifest,
		'f3_lithology_split_probe_run_manifest',
		probe_rows,
		probe=probe_settings,
	)
	evaluation = {
		'monitored_class_ids': [3, 5],
		'boundary_tolerances': [2, 4],
		'boundary_region_radii': [2, 4],
		'chunk_size_x': 2,
	}
	v0_config = F3VoxelV0SplitSuiteConfig(
		voxel_dataset_manifest=suite_root / 'voxel_split_dataset_manifest.json',
		split_dataset_manifest=dataset_manifest,
		probe_run_manifest=probe_manifest,
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		models=tuple(models),
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		batch_size=8,
		tokenization={
			'min_labeled_fraction': 1.0,
			'min_majority_fraction': 1.0,
			'ignore_z_border_samples': 0,
		},
		evaluation=evaluation,
		overwrite=False,
	)
	v1_models = tuple(
		VoxelRobustnessModel(model.role, model.model_tag, model.embeddings_dir)
		for model in models
	)
	v1_config = F3VoxelDecoderSplitSuiteConfig(
		voxel_dataset_manifest=suite_root / 'voxel_split_dataset_manifest.json',
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		models=v1_models,
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		decoder=VoxelDecoderSpec(
			spec='frozen_embedding_decoder_nearest_voxel_ln_v1',
			embedding_dim=2,
			class_count=2,
			hidden_channels=(2,),
			upsample_factors=((2, 2, 2),),
			upsample_mode='nearest',
			normalization='voxelwise_layer_norm',
		),
		tiles=VoxelDecoderTileSettings(
			core_size_tokens=(2, 2, 2), context_halo_tokens=(0, 0, 0)
		),
		train=VoxelDecoderTrainSettings(
			epochs=1,
			batch_size=1,
			learning_rate=0.01,
			weight_decay=0.0,
			class_weight='balanced',
			seed=7,
			num_workers=0,
			amp=False,
			gradient_clip_norm=1.0,
		),
		evaluation=evaluation,
		write_probabilities=False,
		overwrite=False,
	)
	return build_config, v0_config, v1_config


def _write_empty_validation_tokens(path: Path, survey_id: str) -> None:
	np.savez_compressed(
		path,
		features=np.empty((0, 2), dtype=np.float32),
		labels=np.empty((0,), dtype=np.int64),
		survey_id=np.empty((0,), dtype=f'<U{len(survey_id)}'),
		split=np.empty((0,), dtype='<U10'),
		slice_type=np.empty((0,), dtype='<U9'),
		slice_index=np.empty((0,), dtype=np.int64),
		token_xyz=np.empty((0, 3), dtype=np.int64),
		voxel_center_xyz=np.empty((0, 3), dtype=np.float32),
		majority_fraction=np.empty((0,), dtype=np.float32),
		labeled_fraction=np.empty((0,), dtype=np.float32),
	)
