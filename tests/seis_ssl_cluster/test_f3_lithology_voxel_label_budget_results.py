from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

import seis_ssl_cluster.f3.lithology.voxel_label_budget_results as results_module
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_results import (
	F3VoxelLabelBudgetDecisionThresholds,
	F3VoxelLabelBudgetResultsConfig,
	F3VoxelLabelBudgetResultsPublishConfig,
	f3_lithology_voxel_label_budget_results_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	EXPECTED_MODEL_TAGS,
	FIGURE_NAMES,
	MODEL_ROLES,
	REQUIRED_BUDGETS,
	REQUIRED_SEEDS,
	RUN_MANIFEST_ARTIFACT_TYPE,
	SUMMARY_JSON,
	SUMMARY_MARKDOWN,
	TABLE_NAMES,
	inspect_f3_lithology_voxel_label_budget_results,
	summarize_f3_lithology_voxel_label_budget_results,
)
from seis_ssl_cluster.models.voxel_decoder import (
	voxel_decoder_architecture_mapping,
)


def test_complete_summary_pairing_decisions_and_lightweight_publish(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path, publish=True)
	_patch_synthetic_completion_validators(monkeypatch)
	result = summarize_f3_lithology_voxel_label_budget_results(config)

	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert payload['status'] == 'COMPLETE'
	assert payload['completion'] == {
		'dataset_count': 15,
		'job_count': 45,
		'best_checkpoint_inference_count': 45,
		'complete_evaluation_count': 45,
		'paired_identity_mismatch_count': 0,
	}
	assert payload['scientific_decisions']['structured_pretext_vs_mae'][
		'm1_vs_mae'
	]['label'] == 'POSITIVE'
	assert payload['scientific_decisions']['structured_pretext_vs_mae'][
		'm2a_vs_mae'
	]['label'] == 'POSITIVE'
	assert payload['scientific_decisions']['boundary_aware_increment'][
		'm2a_vs_m1'
	]['label'] == 'POSITIVE'
	assert len(payload['paired_deltas']) == 45
	assert all(row['budget_id'] != 'full' for row in payload['paired_deltas'])
	position = next(
		row
		for row in payload['summary_by_budget']
		if row['budget_id'] == 'cap25'
		and row['comparison_id'] == 'm1_vs_mae'
		and row['metric'] == 'vertical_boundary_position_mae'
	)
	assert position['mean_delta'] < 0
	assert position['positive_win_count'] == 5
	assert {path.name for path in result.table_paths} == set(TABLE_NAMES)
	assert {path.name for path in result.figure_paths} == set(FIGURE_NAMES)
	assert result.summary_markdown.name == SUMMARY_MARKDOWN
	assert result.summary_json.name == SUMMARY_JSON
	assert result.published_files
	publish_root = config.publish.output_dir
	assert publish_root is not None
	published = {
		path.relative_to(publish_root).as_posix()
		for path in publish_root.rglob('*')
		if path.is_file()
	}
	assert published == {
		SUMMARY_JSON,
		SUMMARY_MARKDOWN,
		'README.md',
		*(f'tables/{name}' for name in TABLE_NAMES),
		*(f'figures/{name}' for name in FIGURE_NAMES),
	}
	assert {
		path.relative_to(publish_root).as_posix()
		for path in result.published_files
	} == published
	assert not (publish_root / 'publish_manifest.json').exists()
	assert not any(
		Path(name).suffix in {'.pt', '.npy', '.npz', '.joblib'} for name in published
	)


def test_rejects_incomplete_45_job_manifest(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	_patch_synthetic_completion_validators(monkeypatch)
	payload = json.loads(config.run_manifest.read_text(encoding='utf-8'))
	payload['rows'].pop()
	payload['row_count'] = 44
	payload['complete_count'] = 44
	_write_json(config.run_manifest, payload)

	with pytest.raises(ValueError, match='exactly 45'):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_rejects_triplet_initial_state_identity_mismatch(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	_patch_synthetic_completion_validators(monkeypatch)
	payload = json.loads(config.run_manifest.read_text(encoding='utf-8'))
	target = next(
		row
		for row in payload['rows']
		if row['budget_id'] == 'cap25'
		and row['subsample_seed'] == 0
		and row['model_role'] == 'm2a'
	)
	target['initial_model_state_sha256'] = 'f' * 64
	_write_json(config.run_manifest, payload)

	with pytest.raises(ValueError, match='initial_model_state_sha256'):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_rejects_prediction_training_sampling_mismatch(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	_patch_synthetic_dataset_validator(monkeypatch)
	monkeypatch.setattr(
		results_module, '_validate_completed_decoder_artifact', lambda **_: None
	)
	monkeypatch.setattr(
		results_module, '_validate_full_label_anchor_decoder', lambda **_: None
	)
	monkeypatch.setattr(
		results_module,
		'validate_f3_voxel_prediction_artifact',
		_fake_prediction_artifact,
	)
	payload = json.loads(config.run_manifest.read_text(encoding='utf-8'))
	target = next(
		row
		for row in payload['rows']
		if row['budget_id'] == 'cap25'
		and row['subsample_seed'] == 0
		and row['model_role'] == 'mae'
	)
	prediction_path = Path(target['prediction_metadata']['path'])
	prediction = json.loads(prediction_path.read_text(encoding='utf-8'))
	prediction['training_sampling']['steps_per_epoch'] = 5
	_write_json(prediction_path, prediction)
	target['prediction_metadata'] = _identity(prediction_path)
	_write_json(config.run_manifest, payload)

	with pytest.raises(ValueError, match='training-sampling contract mismatch'):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_rejects_missing_six_file_dataset_artifact(tmp_path: Path) -> None:
	config = _fixture(tmp_path)

	with pytest.raises(FileNotFoundError, match='condition is missing required files'):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_rejects_missing_prediction_arrays(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	_patch_synthetic_dataset_validator(monkeypatch)
	monkeypatch.setattr(
		results_module, '_validate_completed_decoder_artifact', lambda **_: None
	)
	monkeypatch.setattr(
		results_module, '_validate_full_label_anchor_decoder', lambda **_: None
	)

	with pytest.raises(
		FileNotFoundError, match='incomplete voxel prediction artifact'
	):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_rejects_fake_completed_checkpoint_without_history(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	config = _fixture(tmp_path)
	_patch_synthetic_dataset_validator(monkeypatch)
	monkeypatch.setattr(
		results_module, '_validate_prediction_metadata', lambda *_, **__: None
	)
	monkeypatch.setattr(
		results_module, '_validate_full_label_anchor_decoder', lambda **_: None
	)

	with pytest.raises(FileNotFoundError, match='decoder artifact is incomplete'):
		inspect_f3_lithology_voxel_label_budget_results(config)


def test_best_checkpoint_kind_is_completed_when_final_epoch_wins() -> None:
	assert results_module._expected_best_checkpoint_kind(48, epochs=50) == 'epoch'  # noqa: SLF001
	final_kind = results_module._expected_best_checkpoint_kind(49, epochs=50)  # noqa: SLF001
	assert final_kind == 'completed'
	with pytest.raises(ValueError, match='outside the training range'):
		results_module._expected_best_checkpoint_kind(50, epochs=50)  # noqa: SLF001


@pytest.mark.parametrize(
	'case', ['cross_budget', 'mean_median', 'mixed_metric', 'weak_wins']
)
def test_decision_holds_for_direction_ambiguity(case: str) -> None:
	thresholds = F3VoxelLabelBudgetDecisionThresholds(
		minimum_positive_budgets=2,
		minimum_primary_wins=4,
		negative_budget_count=2,
		monitored_class_ids=(3, 5),
		major_degradation_delta=-0.05,
		systematic_degradation_budget_count=2,
	)
	rows = []
	for budget_index, budget in enumerate(REQUIRED_BUDGETS):
		if case == 'cross_budget':
			means = (-0.02, -0.02) if budget_index == 2 else (0.02, 0.02)
			medians = means
		elif case == 'mean_median':
			means = (-0.02, -0.02) if budget_index < 2 else (0.0, 0.0)
			medians = (0.01, 0.01) if budget_index < 2 else (0.0, 0.0)
		elif case == 'mixed_metric':
			means = (0.02, -0.02) if budget_index == 2 else (0.02, 0.02)
			medians = means
		else:
			means = (0.02, 0.02)
			medians = means
		rows.extend(
			[
				{
					'budget_id': budget,
					'comparison_id': 'm1_vs_mae',
					'metric': metric,
					'mean_delta': mean,
					'median_delta': median,
					'positive_win_count': (
						3
						if case == 'weak_wins' and budget_index == 2
						else 4 if mean > 0 else 1
					),
				}
				for metric, mean, median in zip(
					('macro_f1', 'mean_iou'), means, medians, strict=True
				)
			]
		)
		rows.extend(
			[
					{
						'budget_id': budget,
						'comparison_id': 'm1_vs_mae',
						'metric': f'class_{class_id}_{metric}',
						'mean_delta': 0.01,
					}
					for class_id in (3, 5)
					for metric in (
						'f1',
						'iou',
						'boundary_recall_t2',
						'boundary_recall_t4',
					)
			]
		)

	decision = results_module._decision_for_comparison(  # noqa: SLF001
		'm1_vs_mae', rows, thresholds
	)

	assert decision['label'] == 'HOLD'
	assert decision['hold_for_direction_ambiguity'] is True


def test_decision_blocks_systematic_boundary_recall_degradation() -> None:
	thresholds = F3VoxelLabelBudgetDecisionThresholds(
		minimum_positive_budgets=2,
		minimum_primary_wins=4,
		negative_budget_count=2,
		monitored_class_ids=(3, 5),
		major_degradation_delta=-0.05,
		systematic_degradation_budget_count=2,
	)
	rows = []
	for budget_index, budget in enumerate(REQUIRED_BUDGETS):
		rows.extend(
			{
				'budget_id': budget,
				'comparison_id': 'm1_vs_mae',
				'metric': metric,
				'mean_delta': 0.02,
				'median_delta': 0.02,
				'positive_win_count': 4,
			}
			for metric in ('macro_f1', 'mean_iou')
		)
		for class_id in (3, 5):
			for metric in (
				'f1',
				'iou',
				'boundary_recall_t2',
				'boundary_recall_t4',
			):
				delta = 0.01
				if (
					class_id == 3
					and metric == 'boundary_recall_t4'
					and budget_index < 2
				):
					delta = -0.06
				rows.append(
					{
						'budget_id': budget,
						'comparison_id': 'm1_vs_mae',
						'metric': f'class_{class_id}_{metric}',
						'mean_delta': delta,
					}
				)

	decision = results_module._decision_for_comparison(  # noqa: SLF001
		'm1_vs_mae', rows, thresholds
	)

	assert decision['label'] == 'HOLD'
	assert decision['major_monitored_class_degradation_budgets']['3'] == [
		'cap25',
		'cap50',
	]
	assert decision['major_monitored_class_degradation_metrics']['3'] == {
		'cap25': ['boundary_recall_t4'],
		'cap50': ['boundary_recall_t4'],
	}


def test_strict_config_rejects_unknown_key(tmp_path: Path) -> None:
	raw = _raw_config(tmp_path)
	raw['decision']['after_results_override'] = True

	with pytest.raises(ValueError, match=r'key\(s\) not allowed'):
		f3_lithology_voxel_label_budget_results_config_from_mapping(raw)


def _patch_synthetic_dataset_validator(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def synthetic_validator(root: str | Path) -> Mapping[str, object]:
		return json.loads(
			(Path(root) / 'voxel_label_budget_metadata.json').read_text(
				encoding='utf-8'
			)
		)

	monkeypatch.setattr(
		results_module,
		'validate_voxel_label_budget_condition_artifact',
		synthetic_validator,
	)


def _patch_synthetic_completion_validators(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	_patch_synthetic_dataset_validator(monkeypatch)
	monkeypatch.setattr(
		results_module, '_validate_prediction_metadata', lambda *_, **__: None
	)
	monkeypatch.setattr(
		results_module, '_validate_completed_decoder_artifact', lambda **_: None
	)
	monkeypatch.setattr(
		results_module, '_validate_full_label_anchor_decoder', lambda **_: None
	)


def _fake_prediction_artifact(
	paths_or_dir: str | Path, *, mmap_mode: str | None = 'r'
) -> SimpleNamespace:
	del mmap_mode
	root = Path(paths_or_dir)
	metadata_path = root / 'prediction_metadata.json'
	return SimpleNamespace(
		output_dir=root,
		metadata=json.loads(metadata_path.read_text(encoding='utf-8')),
		arrays=SimpleNamespace(probabilities=None, valid_mask=None),
	)


def _fixture(
	tmp_path: Path, *, publish: bool = False
) -> F3VoxelLabelBudgetResultsConfig:
	artifact_root = tmp_path / 'artifacts'
	suite_root = artifact_root / 'voxel_label_budget'
	anchor_grid = suite_root / 'full_anchor_grid.npy'
	anchor_grid.parent.mkdir(parents=True, exist_ok=True)
	anchor_grid.write_bytes(b'full-grid')
	dataset_manifest, dataset_rows = _dataset_manifest(
		suite_root, canonical_full_grid=anchor_grid
	)
	run_manifest = _run_manifest(suite_root, dataset_rows)
	anchors = {}
	for role in MODEL_ROLES:
		root = artifact_root / 'full' / role / 'evaluation'
		_write_evaluation(
			root,
			role=role,
			score=0.80 + 0.02 * MODEL_ROLES.index(role),
			position_mae=0.8 - 0.05 * MODEL_ROLES.index(role),
			grid_identity=_identity(anchor_grid),
		)
		anchors[role] = root
	results_root = tmp_path / 'results'
	return F3VoxelLabelBudgetResultsConfig(
		artifact_root=artifact_root,
		suite_root=suite_root,
		dataset_manifest=dataset_manifest,
		run_manifest=run_manifest,
		full_label_evaluations=anchors,
		decision=F3VoxelLabelBudgetDecisionThresholds(
			minimum_positive_budgets=2,
			minimum_primary_wins=4,
			negative_budget_count=2,
			monitored_class_ids=(3, 5),
			major_degradation_delta=-0.05,
			systematic_degradation_budget_count=2,
		),
		publish=F3VoxelLabelBudgetResultsPublishConfig(
			enabled=publish,
			results_root=results_root,
			output_dir=(results_root / 'f3' / 'low_label' if publish else None),
		),
	)


def _dataset_manifest(
	suite_root: Path, *, canonical_full_grid: Path
) -> tuple[Path, dict[tuple[str, int], Mapping[str, object]]]:
	rows = []
	by_key = {}
	validation_hash = _digest('shared-validation-mask')
	for budget in REQUIRED_BUDGETS:
		for seed in REQUIRED_SEEDS:
			root = (
				suite_root
				/ 'datasets'
				/ f'budget={budget}'
				/ f'subsample_seed={seed}'
				/ 'voxel_supervision'
			)
			root.mkdir(parents=True)
			grid = root / 'supervision_split_grid.npy'
			grid.write_bytes(f'{budget}-{seed}-grid'.encode())
			selected = _digest(f'{budget}-{seed}-selected')
			unique = _digest(f'{budget}-{seed}-unique')
			identity = {
				'budget_id': budget,
				'per_class_cap': int(budget.removeprefix('cap')),
				'subsample_seed': seed,
				'class_order': list(range(6)),
				'selected_token_identity_sha256': selected,
				'unique_token_xyz_sha256': unique,
				'actual_train_voxel_count': 120 + seed,
				'validation_voxel_count': 100,
				'validation_mask_sha256': validation_hash,
			}
			metadata = root / 'voxel_label_budget_metadata.json'
			_write_json(
				metadata,
				{
					'artifact_type': 'f3_lithology_voxel_label_budget_dataset',
					'schema_version': 1,
					'identity': identity,
					'sources': {'common_grid': _identity(canonical_full_grid)},
				},
			)
			row = {
				'budget_id': budget,
				'per_class_cap': int(budget.removeprefix('cap')),
				'subsample_seed': seed,
				'voxel_dataset_root': str(root),
				'train_voxel_count': 120 + seed,
				'validation_voxel_count': 100,
				'class_order': list(range(6)),
				'selected_token_identity_sha256': selected,
				'unique_token_xyz_sha256': unique,
				'validation_mask_sha256': validation_hash,
				'supervision_split_grid': _identity(grid),
				'voxel_label_budget_metadata': _identity(metadata),
			}
			rows.append(row)
			by_key[(budget, seed)] = row
	manifest = suite_root / 'voxel_label_budget_dataset_manifest.json'
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_voxel_label_budget_dataset_manifest',
			'schema_version': 1,
			'contract': {
				'budgets': list(REQUIRED_BUDGETS),
				'subsample_seeds': list(REQUIRED_SEEDS),
			},
			'models': EXPECTED_MODEL_TAGS,
			'common_validation_mask_sha256': validation_hash,
			'condition_count': 15,
			'rows': rows,
		},
	)
	return manifest, by_key


def _run_manifest(
	suite_root: Path,
	datasets: Mapping[tuple[str, int], Mapping[str, object]],
) -> Path:
	rows = []
	architecture = voxel_decoder_architecture_mapping(
		embedding_dim=384,
		class_count=6,
		hidden_channels=(128, 64, 32),
		upsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
	)
	for budget_index, budget in enumerate(REQUIRED_BUDGETS):
		for seed in REQUIRED_SEEDS:
			dataset = datasets[(budget, seed)]
			grid = dataset['supervision_split_grid']
			assert isinstance(grid, Mapping)
			condition = f'{budget}-{seed}'
			train_tile_identity = _digest(
				f'{condition}-train-tile-identity'
			)
			validation_tile_identity = _digest(
				f'{condition}-validation-tile-identity'
			)
			training_sampling = {
				'sampling_mode': 'uniform_tiles_with_replacement',
				'steps_per_epoch': 4,
				'train_seed': 42000 + seed,
				'train_tile_manifest_sha256': train_tile_identity,
				'validation_tile_manifest_sha256': validation_tile_identity,
			}
			for role_index, role in enumerate(MODEL_ROLES):
				job_root = (
					suite_root
					/ 'jobs'
					/ f'budget={budget}'
					/ f'subsample_seed={seed}'
					/ f'model={EXPECTED_MODEL_TAGS[role]}'
				)
				evaluation_root = job_root / 'evaluation'
				base = 0.52 + 0.025 * budget_index + 0.001 * seed
				score = base + 0.02 * role_index
				artifacts = _write_evaluation(
					evaluation_root,
					role=role,
					score=score,
					position_mae=1.2 - 0.1 * role_index + 0.01 * seed,
					grid_identity=grid,
					training_sampling=training_sampling,
				)
				rows.append(
					{
						'budget_id': budget,
						'per_class_cap': dataset['per_class_cap'],
						'subsample_seed': seed,
						'decoder_seed': 42000 + seed,
						'model_role': role,
						'model_tag': EXPECTED_MODEL_TAGS[role],
						'status': 'completed',
						'action': 'NEW',
						'voxel_dataset': grid,
						'voxel_supervision_grid_sha256': grid['sha256'],
						'selected_token_identity_sha256': dataset[
							'selected_token_identity_sha256'
						],
						'unique_token_xyz_sha256': dataset[
							'unique_token_xyz_sha256'
						],
						'train_voxel_count': dataset['train_voxel_count'],
						'validation_voxel_count': dataset[
							'validation_voxel_count'
						],
						'class_order': dataset['class_order'],
						'validation_mask_sha256': dataset[
							'validation_mask_sha256'
						],
						'class_weights': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
						'canonical_valid_token_sha256': _digest(
							'canonical-valid-tokens'
						),
						'initial_model_state_sha256': _digest(
							f'{condition}-initial-state'
						),
						'sampling_mode': 'uniform_tiles_with_replacement',
						'steps_per_epoch': 4,
						'sampling_sequence_sha256': _digest(
							f'{condition}-sampling-sequence'
						),
						'train_tile_manifest_sha256': _digest(
							f'{condition}-train-tiles'
						),
						'validation_tile_manifest_sha256': _digest(
							f'{condition}-validation-tiles'
						),
						'train_tile_identity_sha256': train_tile_identity,
						'validation_tile_identity_sha256': (
							validation_tile_identity
						),
						'global_step': 200,
						'latest_checkpoint': artifacts['latest'],
						'best_checkpoint': artifacts['best'],
						'best_selection_epoch': 37,
						'best_selection_metrics': {
							'macro_f1': score,
							'mean_iou': score - 0.08,
						},
						'prediction_metadata': artifacts['prediction_metadata'],
						'prediction_checkpoint_kind': 'best',
						'evaluation_metadata': artifacts['evaluation_metadata'],
						'evaluation_metrics': artifacts['metrics'],
						'evaluation_boundary_metrics': artifacts['boundary'],
						'evaluation_boundary_region_metrics': artifacts['regions'],
						'uncovered_validation_voxel_count': 0,
						'report': artifacts['report'],
						'metric_schema_sha256': artifacts['metric_schema_sha256'],
						'decoder_architecture': architecture,
					}
				)
	manifest = suite_root / 'voxel_label_budget_run_manifest.json'
	_write_json(
		manifest,
		{
			'artifact_type': RUN_MANIFEST_ARTIFACT_TYPE,
			'schema_version': 1,
			'preregistered_contract': {
				'budgets': list(REQUIRED_BUDGETS),
				'subsample_seeds': list(REQUIRED_SEEDS),
				'model_order': list(MODEL_ROLES),
				'epochs': 50,
				'sampling_mode': 'uniform_tiles_with_replacement',
				'steps_per_epoch': 4,
			},
			'row_count': 45,
			'complete_count': 45,
			'rows': rows,
		},
	)
	return manifest


def _write_evaluation(  # noqa: PLR0913
	root: Path,
	*,
	role: str,
	score: float,
	position_mae: float,
	grid_identity: Mapping[str, object],
	training_sampling: Mapping[str, object] | None = None,
) -> dict[str, object]:
	root.mkdir(parents=True, exist_ok=True)
	architecture = voxel_decoder_architecture_mapping(
		embedding_dim=384,
		class_count=6,
		hidden_channels=(128, 64, 32),
		upsample_factors=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
	)
	latest = root.parent / 'decoder' / 'latest.pt'
	best = root.parent / 'decoder' / 'best.pt'
	latest.parent.mkdir(parents=True, exist_ok=True)
	latest.write_bytes(f'{root}-latest'.encode())
	best.write_bytes(f'{root}-best'.encode())
	metrics_path = root / 'metrics.json'
	boundary_path = root / 'boundary_metrics.json'
	regions_path = root / 'boundary_region_metrics.csv'
	prediction_metadata_path = root.parent / 'prediction' / 'prediction_metadata.json'
	prediction_metadata_path.parent.mkdir(parents=True, exist_ok=True)
	report_path = root.parent / 'report' / 'voxel_report.md'
	report_path.parent.mkdir(parents=True, exist_ok=True)
	per_class_f1 = {
		str(class_id): score - 0.01 + class_id * 0.001 for class_id in range(6)
	}
	per_class_iou = {
		str(class_id): score - 0.09 + class_id * 0.001 for class_id in range(6)
	}
	metrics = {
		'aggregation_unit': 'unique_validation_voxel',
		'evaluation_voxel_count': 100,
		'class_ids': list(range(6)),
		'macro_f1': score,
		'mean_iou': score - 0.08,
		'balanced_accuracy': score - 0.02,
		'accuracy': score + 0.03,
		'weighted_f1': score + 0.01,
		'per_class_f1': per_class_f1,
		'per_class_iou': per_class_iou,
	}
	boundary = {
		'vertical_boundary_f1_at_2': score - 0.04,
		'vertical_boundary_f1_at_4': score - 0.02,
		'vertical_boundary_position_mae_at_4': position_mae,
		'vertical_boundary_class_3_recall_at_2': score - 0.06,
		'vertical_boundary_class_3_recall_at_4': score - 0.03,
		'vertical_boundary_class_5_recall_at_2': score - 0.07,
		'vertical_boundary_class_5_recall_at_4': score - 0.04,
	}
	_write_json(metrics_path, metrics)
	_write_json(boundary_path, boundary)
	with regions_path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle, fieldnames=['region', 'radius', 'macro_f1', 'mean_iou']
		)
		writer.writeheader()
		writer.writerows(
			[
				{
					'region': 'boundary',
					'radius': radius,
					'macro_f1': score - 0.03,
					'mean_iou': score - 0.11,
				}
				for radius in (2, 4)
			]
		)
	best_identity = _identity(best)
	prediction_metadata = {
		'artifact_type': 'f3_lithology_voxel_predictions',
		'schema_version': 1,
		'model_tag': EXPECTED_MODEL_TAGS[role],
		'prediction_kind': 'frozen_embedding_decoder',
		'write_probabilities': False,
		'inputs': {'decoder_checkpoint': str(best)},
		'source_identity': {'decoder_checkpoint': best_identity},
	}
	if training_sampling is not None:
		prediction_metadata['training_sampling'] = dict(training_sampling)
	_write_json(prediction_metadata_path, prediction_metadata)
	report_path.write_text('# synthetic voxel report\n', encoding='utf-8')
	metrics_identity = _identity(metrics_path)
	boundary_identity = _identity(boundary_path)
	regions_identity = _identity(regions_path)
	prediction_identity = _identity(prediction_metadata_path)
	evaluation_metadata_path = root / 'evaluation_metadata.json'
	_write_json(
		evaluation_metadata_path,
		{
			'artifact_type': 'f3_lithology_voxel_evaluation',
			'schema_version': 2,
			'model_tag': EXPECTED_MODEL_TAGS[role],
			'prediction_kind': 'frozen_embedding_decoder',
			'decoder_architecture': architecture,
			'aggregation': {
				'primary_unit': 'unique_validation_voxel',
				'split_code': 2,
				'intersection_voxels_counted_once': True,
				'per_slice_planes_evaluated_independently': True,
				'voxel_independence_p_values_computed': False,
			},
			'policy': {
				'monitored_class_ids': [3, 5],
				'boundary_tolerances': [2, 4],
				'boundary_region_radii': [2, 4],
				'primary_trace_boundary_tolerance': 4,
				'chunk_size_x': 8,
			},
			'inputs': {
				'voxel_split_grid': dict(grid_identity),
				'prediction_metadata': prediction_identity,
			},
			'outputs': {
				'metrics.json': metrics_identity,
				'boundary_metrics.json': boundary_identity,
				'boundary_region_metrics.csv': regions_identity,
			},
			'summary': {'unique_validation_voxel_count': 100},
		},
	)
	schema = {
		'metrics': sorted(metrics),
		'boundary': sorted(boundary),
		'boundary_region_columns': [
			'region',
			'radius',
			'macro_f1',
			'mean_iou',
		],
	}
	return {
		'latest': _identity(latest),
		'best': best_identity,
		'prediction_metadata': prediction_identity,
		'evaluation_metadata': _identity(evaluation_metadata_path),
		'metrics': metrics_identity,
		'boundary': boundary_identity,
		'regions': regions_identity,
		'report': _identity(report_path),
		'metric_schema_sha256': hashlib.sha256(
			json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()
		).hexdigest(),
	}


def _raw_config(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	suite_root = artifact_root / 'voxel_label_budget'
	results_root = tmp_path / 'results'
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'results_root': str(results_root),
		},
		'suite': {
			'root': str(suite_root),
			'dataset_manifest': str(
				suite_root / 'voxel_label_budget_dataset_manifest.json'
			),
			'run_manifest': str(
				suite_root / 'voxel_label_budget_run_manifest.json'
			),
		},
		'full_label_reference': {
			role: str(artifact_root / 'full' / role / 'evaluation')
			for role in MODEL_ROLES
		},
		'decision': {
			'minimum_positive_budgets': 2,
			'minimum_primary_wins': 4,
			'negative_budget_count': 2,
			'monitored_class_ids': [3, 5],
			'major_degradation_delta': -0.05,
			'systematic_degradation_budget_count': 2,
		},
		'outputs': {'overwrite': False},
		'publish': {
			'enabled': False,
			'output_dir': str(results_root / 'f3' / 'low_label'),
			'max_file_size_mb': 10,
			'overwrite': True,
		},
	}


def _identity(path: Path) -> dict[str, object]:
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _digest(value: str) -> str:
	return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)
