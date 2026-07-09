from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from proc.seis_ssl_cluster.run_f3_lithology_split_sweep_probes import (
	f3_lithology_split_sweep_probe_config_from_mapping,
)
from seis_ssl_cluster.config import (
	load_config,
	resolve_cluster_visualization_config,
	resolve_clustering_config,
	resolve_embedding_extraction_config,
	resolve_f3_facies_inspection_config,
	resolve_mae_training_config,
	resolve_strat_hmm_pretext_config,
	resolve_strat_hmm_pseudo_target_config,
)
from seis_ssl_cluster.config.f3_baselines import (
	f3_lithology_baseline_token_dataset_config_from_mapping,
	f3_lithology_comparison_publish_config_from_mapping,
	f3_lithology_comparison_report_config_from_mapping,
	random_mae_checkpoint_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology import (
	f3_lithology_prediction_config_from_mapping,
	f3_lithology_probe_config_from_mapping,
	f3_lithology_publish_config_from_mapping,
	f3_lithology_report_config_from_mapping,
	f3_lithology_token_dataset_config_from_mapping,
	f3_lithology_visualization_config_from_mapping,
	f3_prepare_volume_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
	f3_lithology_label_budget_probe_config_from_mapping,
	f3_lithology_split_inventory_config_from_mapping,
	f3_lithology_split_sweep_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.schema import (
	STAGE_F3_INSPECT_FILES,
	STAGE_F3_INSPECTION_REPORT,
	STAGE_F3_LABEL_CONSISTENCY,
	STAGE_F3_PNG_LABELS,
	STAGE_F3_QUICKLOOK,
	STAGE_F3_SEGY_GEOMETRY,
	STAGE_F3_TOKENIZATION_PREVIEW,
)
from seis_ssl_cluster.paths import DEFAULT_ARTIFACT_ROOT, ArtifactPaths, ExperimentKey

NOPIMS_ROOT = Path('experiments/nopims/pretrain_v1')
NOPIMS_PRETRAINING_CONFIGS = sorted((NOPIMS_ROOT / '10_pretrain').rglob('*.yaml'))
NOPIMS_EMBEDDING_CONFIGS = sorted((NOPIMS_ROOT / '20_embedding').rglob('*.yaml'))
NOPIMS_CLUSTERING_CONFIGS = sorted((NOPIMS_ROOT / '30_clustering').rglob('*.yaml'))
NOPIMS_VISUALIZATION_CONFIGS = sorted(
	(NOPIMS_ROOT / '40_visualization').rglob('*.yaml'),
)
F3_ROOT = Path('experiments/f3/facies_benchmark_v1')
F3_INSPECTION_STAGES = {
	'01_inspect_files.yaml': STAGE_F3_INSPECT_FILES,
	'02_inspect_segy_geometry.yaml': STAGE_F3_SEGY_GEOMETRY,
	'03_inspect_png_labels.yaml': STAGE_F3_PNG_LABELS,
	'04_make_quicklook_figures.yaml': STAGE_F3_QUICKLOOK,
	'05_check_label_consistency.yaml': STAGE_F3_LABEL_CONSISTENCY,
	'06_make_tokenization_preview.yaml': STAGE_F3_TOKENIZATION_PREVIEW,
	'07_build_inspection_report.yaml': STAGE_F3_INSPECTION_REPORT,
}
F3_INSPECTION_CONFIGS = [
	(path, F3_INSPECTION_STAGES[path.name])
	for path in sorted((F3_ROOT / '00_inspection').rglob('*.yaml'))
]
F3_PREPARE_CONFIGS = sorted((F3_ROOT / '10_prepare').rglob('*.yaml'))
F3_EMBEDDING_CONFIGS = sorted((F3_ROOT / '20_embedding').rglob('*.yaml'))
F3_STRATIGRAPHIC_CLUSTERING_CONFIGS = sorted(
	(F3_ROOT / '60_stratigraphic_clustering').rglob('*.yaml'),
)
F3_STRAT_HMM_PRETRAINING_M1_ROOT = F3_ROOT / '80_strat_hmm_pretraining_m1'
F3_STRAT_HMM_M1_GUARDRAIL_ROOT = F3_ROOT / '83_strat_hmm_m1_guardrails'
F3_STRAT_HMM_PRETEXT_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '02_train_single_head_topblock_distill_smoke.yaml',
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '03_train_single_head_topblock_distill_full.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '01_train_distillation_only_smoke.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '02_train_distillation_only_full.yaml',
	],
)
F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '04_extract_student_embeddings.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '03_extract_distillation_only_embeddings.yaml',
	],
)
F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '05_build_lithology_token_dataset.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '04_build_distillation_only_token_dataset.yaml',
	],
)
F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '06_train_lithology_probe.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '05_train_distillation_only_probe.yaml',
	],
)
F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT / '07_build_lithology_report.yaml',
		F3_STRAT_HMM_M1_GUARDRAIL_ROOT
		/ '06_build_distillation_only_report.yaml',
	],
)
F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS = sorted(
	[
		F3_STRAT_HMM_PRETRAINING_M1_ROOT
		/ '08_refresh_pseudo_targets_from_logits_smoke.yaml',
	],
)
F3_STRAT_HMM_M1_ROBUSTNESS_ROOT = F3_ROOT / '81_strat_hmm_m1_robustness'
F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS = sorted(
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT.rglob('*.yaml'),
)
F3_STRAT_HMM_M1_LABEL_BUDGET_BUILD_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '01_build_label_budget_datasets.yaml',
]
F3_STRAT_HMM_M1_LABEL_BUDGET_PROBE_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '02_run_label_budget_probes.yaml',
]
F3_STRAT_HMM_M1_SPLIT_INVENTORY_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '04_generate_split_inventories.yaml',
]
F3_STRAT_HMM_M1_SPLIT_DATASET_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '05_build_split_sweep_datasets.yaml',
]
F3_STRAT_HMM_M1_SPLIT_PROBE_CONFIGS = [
	F3_STRAT_HMM_M1_ROBUSTNESS_ROOT / '06_run_split_sweep_probes.yaml',
]
F3_LITHOLOGY_ROOT = F3_ROOT / '50_lithology'
F3_LITHOLOGY_TOKEN_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('01_build_token_dataset.yaml'),
)
F3_LITHOLOGY_PROBE_CONFIGS = sorted(
	[
		*F3_LITHOLOGY_ROOT.rglob('02_train_linear_probe.yaml'),
		*F3_LITHOLOGY_ROOT.rglob('03_train_mlp_probe.yaml'),
	],
)
F3_LITHOLOGY_PREDICTION_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('04_predict_volume.yaml'),
)
F3_LITHOLOGY_VISUALIZATION_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('05_visualize_predictions.yaml'),
)
F3_LITHOLOGY_REPORT_CONFIGS = sorted(
	F3_LITHOLOGY_ROOT.rglob('06_build_lithology_report.yaml'),
)
F3_BASELINE_ROOT = F3_ROOT / '50_lithology_baselines'
F3_BASELINE_TOKEN_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('01_build_baseline_token_dataset.yaml'),
)
F3_RANDOM_ENCODER_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('01_create_random_checkpoint.yaml'),
)
F3_RANDOM_ENCODER_EMBEDDING_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('02_extract_embeddings.yaml'),
)
F3_RANDOM_ENCODER_TOKEN_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('03_build_token_dataset.yaml'),
)
F3_BASELINE_PROBE_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('02_train_linear_probe.yaml'),
)
F3_RANDOM_ENCODER_PROBE_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('04_train_linear_probe.yaml'),
)
F3_BASELINE_REPORT_CONFIGS = sorted(
	[
		*F3_BASELINE_ROOT.rglob('03_build_report.yaml'),
		*F3_BASELINE_ROOT.rglob('05_build_report.yaml'),
	],
)
F3_BASELINE_COMPARISON_CONFIGS = sorted(
	F3_BASELINE_ROOT.rglob('05_build_baseline_comparison_report.yaml'),
)
REQUIRED_ACTIVE_CONFIG_GROUPS = (
	('nopims pretraining', NOPIMS_PRETRAINING_CONFIGS),
	('nopims embedding', NOPIMS_EMBEDDING_CONFIGS),
	('nopims clustering', NOPIMS_CLUSTERING_CONFIGS),
	('nopims visualization', NOPIMS_VISUALIZATION_CONFIGS),
	('f3 inspection', F3_INSPECTION_CONFIGS),
	('f3 prepare', F3_PREPARE_CONFIGS),
	('f3 embedding', F3_EMBEDDING_CONFIGS),
	('f3 stratigraphic clustering', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS),
	('f3 strat hmm pretext', F3_STRAT_HMM_PRETEXT_CONFIGS),
	(
		'f3 strat hmm student embedding',
		F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
	),
	(
		'f3 strat hmm student lithology token dataset',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS,
	),
	(
		'f3 strat hmm student lithology probe',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS,
	),
	(
		'f3 strat hmm student lithology report',
		F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS,
	),
	(
		'f3 strat hmm pseudo-target refresh',
		F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS,
	),
	('f3 strat hmm m1 robustness', F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS),
	('f3 lithology token dataset', F3_LITHOLOGY_TOKEN_CONFIGS),
	('f3 lithology probe', F3_LITHOLOGY_PROBE_CONFIGS),
	('f3 lithology prediction', F3_LITHOLOGY_PREDICTION_CONFIGS),
	('f3 lithology visualization', F3_LITHOLOGY_VISUALIZATION_CONFIGS),
	('f3 lithology report', F3_LITHOLOGY_REPORT_CONFIGS),
	('f3 baseline token dataset', F3_BASELINE_TOKEN_CONFIGS),
	('f3 random encoder', F3_RANDOM_ENCODER_CONFIGS),
	('f3 random encoder embedding', F3_RANDOM_ENCODER_EMBEDDING_CONFIGS),
	('f3 random encoder token dataset', F3_RANDOM_ENCODER_TOKEN_CONFIGS),
	('f3 baseline probe', F3_BASELINE_PROBE_CONFIGS),
	('f3 random encoder probe', F3_RANDOM_ENCODER_PROBE_CONFIGS),
	('f3 baseline report', F3_BASELINE_REPORT_CONFIGS),
	('f3 baseline comparison', F3_BASELINE_COMPARISON_CONFIGS),
)


@pytest.mark.parametrize(('group_name', 'configs'), REQUIRED_ACTIVE_CONFIG_GROUPS)
def test_active_config_groups_are_not_empty(
	group_name: str,
	configs: list[Path] | list[tuple[Path, str]],
) -> None:
	assert configs, f'{group_name} active config list must not be empty'


@pytest.mark.parametrize('config_path', NOPIMS_PRETRAINING_CONFIGS)
def test_active_nopims_pretraining_configs_resolve(config_path: Path) -> None:
	resolve_mae_training_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_EMBEDDING_CONFIGS)
def test_active_nopims_embedding_configs_resolve(config_path: Path) -> None:
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_CLUSTERING_CONFIGS)
def test_active_nopims_clustering_configs_resolve(config_path: Path) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', NOPIMS_VISUALIZATION_CONFIGS)
def test_active_nopims_cluster_visualization_configs_resolve(
	config_path: Path,
) -> None:
	resolve_cluster_visualization_config(load_config(config_path))


@pytest.mark.parametrize(('config_path', 'stage'), F3_INSPECTION_CONFIGS)
def test_active_f3_inspection_configs_resolve(
	config_path: Path,
	stage: str,
) -> None:
	resolve_f3_facies_inspection_config(load_config(config_path), stage=stage)


@pytest.mark.parametrize('config_path', F3_PREPARE_CONFIGS)
def test_active_f3_prepare_configs_resolve(config_path: Path) -> None:
	f3_prepare_volume_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_EMBEDDING_CONFIGS,
		*F3_RANDOM_ENCODER_EMBEDDING_CONFIGS,
		*F3_STRAT_HMM_STUDENT_EMBEDDING_CONFIGS,
	],
)
def test_active_f3_embedding_configs_resolve(config_path: Path) -> None:
	resolve_embedding_extraction_config(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRATIGRAPHIC_CLUSTERING_CONFIGS)
def test_active_f3_stratigraphic_clustering_configs_resolve(
	config_path: Path,
) -> None:
	resolve_clustering_config(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PRETEXT_CONFIGS)
def test_active_f3_strat_hmm_pretext_configs_resolve(
	config_path: Path,
	tmp_path: Path,
) -> None:
	resolve_strat_hmm_pretext_config(
		_config_with_existing_strat_hmm_pretext_inputs(config_path, tmp_path),
	)


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_PSEUDO_TARGET_REFRESH_CONFIGS)
def test_active_f3_strat_hmm_pseudo_target_refresh_configs_resolve(
	config_path: Path,
	tmp_path: Path,
) -> None:
	resolve_strat_hmm_pseudo_target_config(
		_config_with_existing_strat_hmm_refresh_inputs(config_path, tmp_path),
	)


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_ROBUSTNESS_CONFIGS)
def test_active_f3_strat_hmm_m1_robustness_configs_parse(
	config_path: Path,
) -> None:
	assert load_config(config_path)


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_LABEL_BUDGET_BUILD_CONFIGS)
def test_active_f3_strat_hmm_m1_label_budget_build_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_label_budget_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_LABEL_BUDGET_PROBE_CONFIGS)
def test_active_f3_strat_hmm_m1_label_budget_probe_configs_resolve_schema(
	config_path: Path,
	tmp_path: Path,
) -> None:
	raw = load_config(config_path)
	raw['suite']['manifest'] = str(_write_label_budget_manifest(tmp_path))
	raw['labels']['class_info'] = str(_write_class_info(tmp_path))

	config = f3_lithology_label_budget_probe_config_from_mapping(raw)

	assert config.probe.random_state == 42
	assert len(config.probe_configs) == 2


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_INVENTORY_CONFIGS)
def test_active_f3_strat_hmm_m1_split_inventory_configs_resolve(
	config_path: Path,
) -> None:
	config = f3_lithology_split_inventory_config_from_mapping(load_config(config_path))

	assert config.include_base_split_as_split_000 is True
	assert config.random_seeds == (0, 1, 2, 3, 4)
	assert config.min_validation_tokens_per_class == {'default': 1, '3': 100, '5': 20}


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_DATASET_CONFIGS)
def test_active_f3_strat_hmm_m1_split_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_split_sweep_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_STRAT_HMM_M1_SPLIT_PROBE_CONFIGS)
def test_active_f3_strat_hmm_m1_split_probe_configs_resolve_schema(
	config_path: Path,
	tmp_path: Path,
) -> None:
	raw = load_config(config_path)
	raw['suite']['dataset_manifest'] = str(_write_split_dataset_manifest(tmp_path))
	raw['labels']['class_info'] = str(_write_class_info(tmp_path))

	config = f3_lithology_split_sweep_probe_config_from_mapping(raw)

	assert config.probe.random_state == 42
	assert len(config.probe_configs) == 2


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_TOKEN_CONFIGS,
		*F3_RANDOM_ENCODER_TOKEN_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_TOKEN_CONFIGS,
	],
)
def test_active_f3_lithology_token_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_token_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_BASELINE_TOKEN_CONFIGS)
def test_active_f3_baseline_token_dataset_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_baseline_token_dataset_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize('config_path', F3_RANDOM_ENCODER_CONFIGS)
def test_active_f3_random_encoder_configs_resolve(config_path: Path) -> None:
	random_mae_checkpoint_config_from_mapping(load_config(config_path))


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_PROBE_CONFIGS,
		*F3_BASELINE_PROBE_CONFIGS,
		*F3_RANDOM_ENCODER_PROBE_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_PROBE_CONFIGS,
	],
)
def test_active_f3_lithology_probe_configs_resolve(config_path: Path) -> None:
	f3_lithology_probe_config_from_mapping(load_config(config_path), load_classes=False)


@pytest.mark.parametrize('config_path', F3_LITHOLOGY_PREDICTION_CONFIGS)
def test_active_f3_lithology_prediction_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_prediction_config_from_mapping(
		load_config(config_path),
		load_classes=False,
	)


@pytest.mark.parametrize('config_path', F3_LITHOLOGY_VISUALIZATION_CONFIGS)
def test_active_f3_lithology_visualization_configs_resolve(
	config_path: Path,
) -> None:
	f3_lithology_visualization_config_from_mapping(
		load_config(config_path),
		load_classes=False,
	)


@pytest.mark.parametrize(
	'config_path',
	[
		*F3_LITHOLOGY_REPORT_CONFIGS,
		*F3_BASELINE_REPORT_CONFIGS,
		*F3_STRAT_HMM_STUDENT_LITHOLOGY_REPORT_CONFIGS,
	],
)
def test_active_f3_lithology_report_configs_resolve(config_path: Path) -> None:
	raw = load_config(config_path)

	f3_lithology_report_config_from_mapping(raw)
	f3_lithology_publish_config_from_mapping(raw.get('publish'))


@pytest.mark.parametrize('config_path', F3_BASELINE_COMPARISON_CONFIGS)
def test_active_f3_baseline_comparison_configs_resolve(
	config_path: Path,
) -> None:
	raw = load_config(config_path)

	f3_lithology_comparison_report_config_from_mapping(raw)
	f3_lithology_comparison_publish_config_from_mapping(raw.get('publish'))


def test_active_nopims_overlap_x16_paths_match_artifact_paths_contract() -> None:
	model_tag = 'amp_mae_m075_mse_g0_patchnorm_clip8_vis01_v1'
	paths = ArtifactPaths(DEFAULT_ARTIFACT_ROOT)
	key = ExperimentKey(
		dataset='nopims',
		version='pretrain_v1',
		model_tag=model_tag,
		subset='ten_surveys',
		embed_spec='overlap_x16',
		cluster_spec='k6_8_whiten',
		viz_spec='voxel_cmp_xy750_xz150',
	)

	embedding = load_config(
		NOPIMS_ROOT
		/ '20_embedding'
		/ model_tag
		/ '01_ten_surveys_overlap_x16.yaml',
	)
	clustering = load_config(
		NOPIMS_ROOT
		/ '30_clustering'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_k6_8_whiten.yaml',
	)
	visualization = load_config(
		NOPIMS_ROOT
		/ '40_visualization'
		/ model_tag
		/ '01_ten_surveys_overlap_x16_whiten.yaml',
	)

	assert Path(embedding['embeddings']['output_dir']) == paths.embeddings(key)
	assert Path(clustering['embeddings']['input_dir']) == paths.embeddings(key)
	assert Path(clustering['clustering']['output_dir']) == paths.clustering(key)
	assert Path(visualization['clustering']['input_dir']) == paths.clustering(key)
	assert (
		Path(visualization['visualization']['output_dir'])
		== paths.cluster_visualization(key)
	)


def _config_with_existing_strat_hmm_pretext_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	pseudo_target_dir = tmp_path / 'pseudo_targets'
	pseudo_target_dir.mkdir()
	checkpoint = tmp_path / 'mae_best.pt'
	checkpoint.touch()

	config['paths']['artifact_root'] = str(artifact_root)
	config['paths']['output_root'] = str(
		artifact_root / 'pretraining' / 'f3' / config_path.stem,
	)
	config['pseudo_targets']['input_dir'] = str(pseudo_target_dir)
	config['teacher']['checkpoint'] = str(checkpoint)
	config['student']['init_checkpoint'] = str(checkpoint)
	return config


def _config_with_existing_strat_hmm_refresh_inputs(
	config_path: Path,
	tmp_path: Path,
) -> dict[str, object]:
	config = load_config(config_path)
	artifact_root = tmp_path / 'artifacts'
	checkpoint = tmp_path / 'latest.pt'
	torch.save(
		{
			'stratigraphy_config': {
				'head': {'num_prototypes': config['hmm']['k']},
			},
		},
		checkpoint,
	)

	config['paths']['artifact_root'] = str(artifact_root)
	config['checkpoint']['path'] = str(checkpoint)
	config['outputs']['pseudo_target_root'] = str(
		artifact_root / 'pseudo_targets' / 'f3' / config_path.stem,
	)
	return config


def _write_class_info(tmp_path: Path) -> Path:
	path = tmp_path / 'class_info.json'
	path.write_text(
		json.dumps(
			{
				'classes': [
					{'class_id': 0, 'class_name': 'class 0', 'rgb': [230, 159, 0]},
					{'class_id': 1, 'class_name': 'class 1', 'rgb': [86, 180, 233]},
				],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	return path


def _write_label_budget_manifest(tmp_path: Path) -> Path:
	manifest = tmp_path / 'label_budget_suite_manifest.json'
	token_dataset_root = tmp_path / 'tokens'
	rows = [
		_label_budget_manifest_row(
			role='baseline',
			model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			token_dataset_root=token_dataset_root / 'baseline',
		),
		_label_budget_manifest_row(
			role='candidate',
			model_tag='strat_hmm_pretext_m1_k6_topblock1_distill',
			token_dataset_root=token_dataset_root / 'candidate',
		),
	]
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_label_budget_suite_manifest',
			'rows': rows,
		},
	)
	return manifest


def _label_budget_manifest_row(
	*,
	role: str,
	model_tag: str,
	token_dataset_root: Path,
) -> dict[str, object]:
	return {
		'model_role': role,
		'model_tag': model_tag,
		'budget_id': 'cap_25',
		'per_class_cap': 25,
		'subsample_seed': 0,
		'token_dataset_root': str(token_dataset_root),
		'train_tokens': str(token_dataset_root / 'train_tokens.npz'),
		'validation_tokens': str(token_dataset_root / 'validation_tokens.npz'),
		'metadata_json': str(token_dataset_root / 'token_dataset_metadata.json'),
		'selected_train_token_count': 50,
		'validation_token_count': 20,
		'paired_identity_hash': 'paired-hash',
	}


def _write_split_dataset_manifest(tmp_path: Path) -> Path:
	manifest = tmp_path / 'split_dataset_manifest.json'
	token_dataset_root = tmp_path / 'split_tokens'
	rows = [
		_split_dataset_manifest_row(
			role='baseline',
			model_tag='amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
			token_dataset_root=token_dataset_root / 'baseline',
		),
		_split_dataset_manifest_row(
			role='candidate',
			model_tag='strat_hmm_pretext_m1_k6_topblock1_distill',
			token_dataset_root=token_dataset_root / 'candidate',
		),
	]
	_write_json(
		manifest,
		{
			'artifact_type': 'f3_lithology_split_sweep_token_dataset_manifest',
			'rows': rows,
		},
	)
	return manifest


def _split_dataset_manifest_row(
	*,
	role: str,
	model_tag: str,
	token_dataset_root: Path,
) -> dict[str, object]:
	return {
		'split_id': 'split_000',
		'model_role': role,
		'model_tag': model_tag,
		'token_dataset_root': str(token_dataset_root),
		'train_tokens': str(token_dataset_root / 'train_tokens.npz'),
		'validation_tokens': str(token_dataset_root / 'validation_tokens.npz'),
		'metadata_json': str(token_dataset_root / 'token_dataset_metadata.json'),
		'train_token_count': 50,
		'validation_token_count': 20,
		'paired_identity_hash': 'paired-hash',
	}


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
