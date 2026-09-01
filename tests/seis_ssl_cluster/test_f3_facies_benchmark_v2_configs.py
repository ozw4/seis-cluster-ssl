"""Contract tests for the F3 facies_benchmark_v2 five-way preparation."""

from __future__ import annotations

import re
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config import (
	load_config,
	resolve_barlow_twins_training_config,
	resolve_embedding_extraction_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.f3_lithology_five_way import (
	f3_lithology_five_way_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_dataset import (
	f3_lithology_voxel_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout_dataset import (
	f3_lithology_voxel_section_layout_dataset_config_from_mapping,
)
from seis_ssl_cluster.config.schema import (
	STAGE_F3_INSPECT_FILES,
	STAGE_F3_LABEL_CONSISTENCY,
	STAGE_F3_PNG_LABELS,
	STAGE_F3_QUICKLOOK,
	STAGE_F3_SEGY_GEOMETRY,
	STAGE_F3_TOKENIZATION_PREVIEW,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	plan_f3_lithology_five_way_jobs,
)
from seis_ssl_cluster.f3.lithology.voxel_section_layout_calibration import (
	f3_section_layout_calibration_config_from_mapping,
)
from seis_ssl_cluster.f3.prepare_volume import f3_prepare_volume_config_from_mapping
from seis_ssl_cluster.f3.splits import load_f3_slice_split_records

V2_ROOT = Path('experiments/f3/facies_benchmark_v2')
INSPECTION_ROOT = V2_ROOT / '00_inspection'
PREPARE_ROOT = V2_ROOT / '10_prepare'
LAYOUT_ROOT = V2_ROOT / '109_f3_voxel_section_layout_v2'
FIVE_WAY_ROOT = V2_ROOT / '110_lithology_mae_local_bt_five_way_v2'
LAYOUT_V3_ROOT = V2_ROOT / '109_f3_voxel_section_layout_v3'
FIVE_WAY_V3_ROOT = V2_ROOT / '110_lithology_mae_local_bt_five_way_v3'
GAUSSIAN_VIEW_ROOT = V2_ROOT / '111_local_barlow_twins_gaussian_view_v1'
TRACE_DROP_VIEW_ROOT = V2_ROOT / '112_local_barlow_twins_trace_drop_view_v1'
TRACE_DROP_P002_VIEW_ROOT = V2_ROOT / '113_local_barlow_twins_trace_drop_p002_view_v1'
ZERO_PHASE_Z_FILTER_VIEW_ROOT = (
	V2_ROOT / '114_local_barlow_twins_zero_phase_z_filter_view_v1'
)
README = FIVE_WAY_ROOT / 'README.md'
ARTIFACT_ROOT = '/test/artifacts/seis_ssl_cluster'
RAW_F3_ROOT = '/test/f3'

# The v2 experiment contract is restated here on purpose instead of being
# imported from the production constants it is meant to check.
V2_VERSION = 'facies_benchmark_v2'
V2_DATASET = {'name': 'f3_facies_benchmark', 'version': V2_VERSION}
MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'random',
)
LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
SIZES = ('small', 'medium', 'large')
PREFIX_COUNTS = {'small': 1, 'medium': 2, 'large': 4}
JOB_COUNT = 75
RANDOM_SEED = 42
TARGET_RULE = 'fixed_train_voxel_counts_v1'
TARGET_TRAIN_VOXEL_COUNTS = {
	'small': 10_152,
	'medium': 20_184,
	'large': 40_520,
}
SELECTION_SEMANTICS = 'stable_hash_partial_section_token_footprints_v1'
ALLOWED_RELATIVE_ERROR = 0.05
V3_SELECTION_SEMANTICS = 'seeded_nested_class_balanced_section_token_rows_v1'
V3_TARGET_RULE = 'fixed_per_class_token_row_caps_v1'
V3_PER_CLASS_TOKEN_ROW_CAPS = {'small': 25, 'medium': 50, 'large': 100}
V3_NOMINAL_TRAIN_VOXEL_COUNTS = {
	'small': 9_600,
	'medium': 19_200,
	'large': 38_400,
}
V3_LAYOUT_SUBSAMPLE_SEEDS = {
	'layout_000': 0,
	'layout_001': 1,
	'layout_002': 2,
	'layout_003': 3,
	'layout_004': 4,
}
V3_TOKENIZATION_POLICY = {
	'min_labeled_fraction': 0.5,
	'min_majority_fraction': 0.7,
	'ignore_z_border_samples': 1,
}
V3_LAYOUT_002_INLINE_ORDER = [290, 590, 390, 490]
V3_SUMMARY_NAME = 'f3_lithology_mae_local_bt_five_way_v3'
EXTRACTION_CONTRACT = {
	'window_size': [128, 128, 128],
	'overlap': [64, 64, 64],
	'output_dtype': 'float16',
	'amp': False,
	'min_token_valid_fraction': 0.5,
}
PREPROCESSING_CONTRACT = {
	'clipping_percentiles': [0.5, 99.5],
	'epsilon': 1.0e-6,
	'max_samples': 1000000,
	'seed': 42,
}
TRAIN_INLINES = tuple(250 + 20 * index for index in range(20))
TRAIN_CROSSLINES = (
	*(450 + 25 * index for index in range(8)),
	*(850 + 25 * index for index in range(12)),
)
VALIDATION_LINES = {'inline': (150,), 'crossline': (350, 750)}
VALIDATION_GUARD = 100
INLINE_BOUNDS = (100, 700)
CROSSLINE_BOUNDS = (300, 1200)
UPSTREAM_CHECKPOINTS = {
	'mae': (
		'pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/'
		'mae100/mae_continue/full_25ep/latest.pt'
	),
	'mae_hmm_k6': (
		'pretraining/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/stage2/'
		'mae100/hmm/k6/full_25ep/latest.pt'
	),
	'local_barlow_twins': (
		'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/'
		'local_bt100/local_bt_continue/full_25ep/latest.pt'
	),
	'local_barlow_twins_hmm_k6': (
		'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/stage2/'
		'local_bt100/hmm/k6/full_25ep/latest.pt'
	),
	'random': (
		'pretraining/f3/facies_benchmark_v1/mae_local_bt_five_way_v1/random/'
		'random_init.pt'
	),
}
REFERENCE_GEOMETRY_CHECKPOINT = (
	'pretraining/f3/facies_benchmark_v1/'
	'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/'
	'random_init/mae_random_seed42.pt'
)
RAW_QC_REPORT = 'inspection/f3/facies_benchmark_v1/report.json'
GAUSSIAN_VIEW_VARIANTS = (
	'gaussian_noise_std005',
	'gaussian_noise_std010',
	'identity_gaussian_noise_std010',
	'legacy_flip_25ep',
)
GAUSSIAN_VIEW_PRETRAINING_ROOT = (
	'pretraining/f3/facies_benchmark_v1/local_barlow_twins_gaussian_view_v1'
)
TRACE_DROP_VIEW_PRETRAINING_ROOT = (
	'pretraining/f3/facies_benchmark_v1/local_barlow_twins_trace_drop_view_v1/base1ep'
)
TRACE_DROP_P002_VIEW_PRETRAINING_ROOT = (
	'pretraining/f3/facies_benchmark_v1/'
	'local_barlow_twins_trace_drop_p002_view_v1/base1ep'
)
ZERO_PHASE_Z_FILTER_VIEW_PRETRAINING_ROOT = (
	'pretraining/f3/facies_benchmark_v1/'
	'local_barlow_twins_zero_phase_z_filter_view_v1/base1ep'
)
GAUSSIAN_VIEW_BASE5_VARIANTS = (
	'gaussian_noise_std010_base5ep',
	'legacy_flip_base5ep',
)
GAUSSIAN_VIEW_BASE1_VARIANTS = (
	'gaussian_noise_std010_base1ep',
	'legacy_flip_base1ep',
)
# The search changes the existing v1 pretraining lineage, then evaluates its
# frozen continuation outputs with the v2 downstream contract.
GAUSSIAN_VIEW_V1_REFERENCES = (
	'registry/manifests/f3/facies_benchmark_v1/f3_amplitude_manifest.json',
	'registry/splits/f3/facies_benchmark_v1/f3_npy_paths.txt',
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/stage1/{variant}/full_25ep'
		for variant in GAUSSIAN_VIEW_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/stage1/{variant}/full_25ep/latest.pt'
		for variant in GAUSSIAN_VIEW_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/stage2/{variant}/local_bt_continue/full_25ep'
		for variant in GAUSSIAN_VIEW_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/stage2/{variant}/'
		'local_bt_continue/full_25ep/latest.pt'
		for variant in GAUSSIAN_VIEW_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base5ep/stage1/{variant}/full_5ep'
		for variant in GAUSSIAN_VIEW_BASE5_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base5ep/stage1/{variant}/full_5ep/latest.pt'
		for variant in GAUSSIAN_VIEW_BASE5_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base5ep/stage2/{variant}/'
		'local_bt_continue/full_25ep'
		for variant in GAUSSIAN_VIEW_BASE5_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base5ep/stage2/{variant}/'
		'local_bt_continue/full_25ep/latest.pt'
		for variant in GAUSSIAN_VIEW_BASE5_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base1ep/stage1/{variant}/full_1ep'
		for variant in GAUSSIAN_VIEW_BASE1_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base1ep/stage1/{variant}/full_1ep/latest.pt'
		for variant in GAUSSIAN_VIEW_BASE1_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base1ep/stage2/{variant}/'
		'local_bt_continue/full_25ep'
		for variant in GAUSSIAN_VIEW_BASE1_VARIANTS
	),
	*(
		f'{GAUSSIAN_VIEW_PRETRAINING_ROOT}/base1ep/stage2/{variant}/'
		'local_bt_continue/full_25ep/latest.pt'
		for variant in GAUSSIAN_VIEW_BASE1_VARIANTS
	),
)
TRACE_DROP_VIEW_V1_REFERENCES = (
	(
		f'{TRACE_DROP_VIEW_PRETRAINING_ROOT}/stage1/'
		'horizontal_trace_drop_p001_base1ep/full_1ep'
	),
	(
		f'{TRACE_DROP_VIEW_PRETRAINING_ROOT}/stage1/'
		'horizontal_trace_drop_p001_base1ep/full_1ep/latest.pt'
	),
	(
		f'{TRACE_DROP_VIEW_PRETRAINING_ROOT}/stage2/'
		'horizontal_trace_drop_p001_base1ep/local_bt_continue/full_25ep'
	),
	(
		f'{TRACE_DROP_VIEW_PRETRAINING_ROOT}/stage2/'
		'horizontal_trace_drop_p001_base1ep/local_bt_continue/'
		'full_25ep/latest.pt'
	),
)
TRACE_DROP_P002_VIEW_V1_REFERENCES = (
	(
		f'{TRACE_DROP_P002_VIEW_PRETRAINING_ROOT}/stage1/'
		'horizontal_trace_drop_p002_base1ep/full_1ep'
	),
	(
		f'{TRACE_DROP_P002_VIEW_PRETRAINING_ROOT}/stage1/'
		'horizontal_trace_drop_p002_base1ep/full_1ep/latest.pt'
	),
	(
		f'{TRACE_DROP_P002_VIEW_PRETRAINING_ROOT}/stage2/'
		'horizontal_trace_drop_p002_base1ep/local_bt_continue/full_25ep'
	),
	(
		f'{TRACE_DROP_P002_VIEW_PRETRAINING_ROOT}/stage2/'
		'horizontal_trace_drop_p002_base1ep/local_bt_continue/'
		'full_25ep/latest.pt'
	),
)
ZERO_PHASE_Z_FILTER_VIEW_V1_REFERENCES = (
	(
		f'{ZERO_PHASE_Z_FILTER_VIEW_PRETRAINING_ROOT}/stage1/'
		'zero_phase_z_filter_w025_base1ep/full_1ep'
	),
	(
		f'{ZERO_PHASE_Z_FILTER_VIEW_PRETRAINING_ROOT}/stage1/'
		'zero_phase_z_filter_w025_base1ep/full_1ep/latest.pt'
	),
	(
		f'{ZERO_PHASE_Z_FILTER_VIEW_PRETRAINING_ROOT}/stage2/'
		'zero_phase_z_filter_w025_base1ep/local_bt_continue/full_25ep'
	),
	(
		f'{ZERO_PHASE_Z_FILTER_VIEW_PRETRAINING_ROOT}/stage2/'
		'zero_phase_z_filter_w025_base1ep/local_bt_continue/full_25ep/latest.pt'
	),
)
ALLOWED_V1_REFERENCES = (
	*UPSTREAM_CHECKPOINTS.values(),
	REFERENCE_GEOMETRY_CHECKPOINT,
	RAW_QC_REPORT,
	*GAUSSIAN_VIEW_V1_REFERENCES,
	*TRACE_DROP_VIEW_V1_REFERENCES,
	*TRACE_DROP_P002_VIEW_V1_REFERENCES,
	*ZERO_PHASE_Z_FILTER_VIEW_V1_REFERENCES,
)
REFERENCE_GEOMETRY_DIR = (
	'embeddings/f3/facies_benchmark_v2/reference_token_geometry/'
	'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/'
	'overlap_x64'
)
INSPECTION_STAGES = {
	'01_inspect_files.yaml': STAGE_F3_INSPECT_FILES,
	'02_inspect_segy_geometry.yaml': STAGE_F3_SEGY_GEOMETRY,
	'03_inspect_png_labels.yaml': STAGE_F3_PNG_LABELS,
	'04_make_quicklook_figures.yaml': STAGE_F3_QUICKLOOK,
	'05_check_label_consistency.yaml': STAGE_F3_LABEL_CONSISTENCY,
	'06_make_tokenization_preview.yaml': STAGE_F3_TOKENIZATION_PREVIEW,
}
EXTRACTION_CONFIGS = {
	'mae': '01_extract_mae.yaml',
	'mae_hmm_k6': '02_extract_mae_hmm_k6.yaml',
	'local_barlow_twins': '03_extract_local_barlow_twins.yaml',
	'local_barlow_twins_hmm_k6': '04_extract_local_barlow_twins_hmm_k6.yaml',
	'random': '05_extract_random.yaml',
}
RUNBOOK_CLIS = (
	'proc/seis_ssl_cluster/inspect_f3_files.py',
	'proc/seis_ssl_cluster/prepare_f3_facies_volume.py',
	'proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py',
	'proc/seis_ssl_cluster/prepare_f3_lithology_voxel_section_layout_contract.py',
	'proc/seis_ssl_cluster/build_f3_lithology_voxel_section_layout_datasets.py',
	'proc/seis_ssl_cluster/check_f3_prepared_volume_parity.py',
	'proc/seis_ssl_cluster/extract_embeddings.py',
	'proc/seis_ssl_cluster/audit_f3_lithology_five_way_sources.py',
	'proc/seis_ssl_cluster/run_f3_lithology_five_way.py',
	'proc/seis_ssl_cluster/summarize_f3_lithology_five_way.py',
)
RUNBOOK_ORDER = (
	'## 環境',
	'## 1. v2 prepared volume',
	'## 2. v2 canonical voxel supervision',
	'## 3. candidate inspection',
	'## 4. v2 line selection',
	'## 5. v2 target calibration',
	'## 6. contract finalize',
	'## 7. 15 section-layout datasets',
	'## 8. checkpoint audit',
	'## 9. v1/v2 prepared volume parity gate',
	'## 10. v2 embedding extraction',
	'## 11. five-way source audit',
	'## 12. preflight',
	'## 13. full suite',
	'## 14. summary dry-run',
	'## 15. summary生成',
	'## 16. resume手順',
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', ARTIFACT_ROOT)
	monkeypatch.setenv('F3_ROOT', RAW_F3_ROOT)
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))


def _load(path: Path) -> dict[str, object]:
	return load_config(path)


def _strings(value: object) -> list[str]:
	if isinstance(value, str):
		return [value]
	if isinstance(value, dict):
		return [item for child in value.values() for item in _strings(child)]
	if isinstance(value, list):
		return [item for child in value for item in _strings(child)]
	return []


def _v2_yaml_files() -> list[Path]:
	return sorted(V2_ROOT.rglob('*.yaml'))


def _resolve_barlow_branch_configs(root: Path, covered: set[Path]) -> None:
	for stage_dir in ('10_stage1', '15_stage2'):
		for path in sorted((root / stage_dir).rglob('*.yaml')):
			resolve_barlow_twins_training_config(_load(path))
			covered.add(path)
	for path in sorted((root / '20_embeddings').glob('*.yaml')):
		resolve_embedding_extraction_config(_load(path))
		covered.add(path)


def test_every_v2_config_resolves_with_its_owning_resolver() -> None:
	covered: set[Path] = set()
	for name, stage in INSPECTION_STAGES.items():
		path = INSPECTION_ROOT / name
		resolved = resolve_f3_facies_inspection_config(_load(path), stage=stage)
		assert resolved['dataset']['version'] == V2_VERSION
		covered.add(path)
	prepare = _load(PREPARE_ROOT / '01_prepare_f3_volume.yaml')
	assert prepare['normalization'] == PREPROCESSING_CONTRACT
	assert f3_prepare_volume_config_from_mapping(prepare).dataset.version == V2_VERSION
	covered.add(PREPARE_ROOT / '01_prepare_f3_volume.yaml')
	reference = PREPARE_ROOT / '02_extract_reference_valid_tokens.yaml'
	resolve_embedding_extraction_config(_load(reference))
	covered.add(reference)
	canonical = f3_lithology_voxel_dataset_config_from_mapping(
		_load(PREPARE_ROOT / '03_build_voxel_supervision.yaml')
	)
	assert dict(canonical.dataset) == V2_DATASET
	covered.add(PREPARE_ROOT / '03_build_voxel_supervision.yaml')
	f3_section_layout_calibration_config_from_mapping(
		_load(LAYOUT_ROOT / '01_prepare_section_layout_contract.yaml')
	)
	covered.add(LAYOUT_ROOT / '01_prepare_section_layout_contract.yaml')
	assert set(_load(LAYOUT_ROOT / '02_layout_lines.yaml')) == {'layouts'}
	covered.add(LAYOUT_ROOT / '02_layout_lines.yaml')
	f3_lithology_voxel_section_layout_dataset_config_from_mapping(
		_load(LAYOUT_ROOT / '03_build_section_layout_datasets.yaml')
	)
	covered.add(LAYOUT_ROOT / '03_build_section_layout_datasets.yaml')
	f3_section_layout_calibration_config_from_mapping(
		_load(LAYOUT_V3_ROOT / '01_prepare_section_layout_contract.yaml')
	)
	covered.add(LAYOUT_V3_ROOT / '01_prepare_section_layout_contract.yaml')
	assert set(_load(LAYOUT_V3_ROOT / '02_layout_lines.yaml')) == {'layouts'}
	covered.add(LAYOUT_V3_ROOT / '02_layout_lines.yaml')
	f3_lithology_voxel_section_layout_dataset_config_from_mapping(
		_load(LAYOUT_V3_ROOT / '03_build_section_layout_datasets.yaml')
	)
	covered.add(LAYOUT_V3_ROOT / '03_build_section_layout_datasets.yaml')
	for name in EXTRACTION_CONFIGS.values():
		resolve_embedding_extraction_config(
			_load(FIVE_WAY_ROOT / '50_embeddings' / name)
		)
		covered.add(FIVE_WAY_ROOT / '50_embeddings' / name)
	five_way = f3_lithology_five_way_config_from_mapping(
		_load(FIVE_WAY_ROOT / '60_five_way.yaml')
	)
	assert dict(five_way.dataset) == V2_DATASET
	covered.add(FIVE_WAY_ROOT / '60_five_way.yaml')
	five_way_v3 = f3_lithology_five_way_config_from_mapping(
		_load(FIVE_WAY_V3_ROOT / '60_five_way.yaml')
	)
	assert dict(five_way_v3.dataset) == V2_DATASET
	covered.add(FIVE_WAY_V3_ROOT / '60_five_way.yaml')
	_resolve_barlow_branch_configs(GAUSSIAN_VIEW_ROOT, covered)
	base5_root = GAUSSIAN_VIEW_ROOT / '40_base5ep'
	_resolve_barlow_branch_configs(base5_root, covered)
	base1_root = GAUSSIAN_VIEW_ROOT / '50_base1ep'
	_resolve_barlow_branch_configs(base1_root, covered)
	_resolve_barlow_branch_configs(TRACE_DROP_VIEW_ROOT, covered)
	_resolve_barlow_branch_configs(TRACE_DROP_P002_VIEW_ROOT, covered)
	_resolve_barlow_branch_configs(ZERO_PHASE_Z_FILTER_VIEW_ROOT, covered)
	assert covered == set(_v2_yaml_files())


def test_v2_paths_stay_in_v2_namespace_except_explicit_v1_lineage() -> None:
	seen_allowed: set[str] = set()
	for path in _v2_yaml_files():
		for value in _strings(_load(path)):
			if not (
				path.is_relative_to(TRACE_DROP_VIEW_ROOT)
				or path.is_relative_to(TRACE_DROP_P002_VIEW_ROOT)
				or path.is_relative_to(ZERO_PHASE_Z_FILTER_VIEW_ROOT)
			):
				assert 'trace_drop' not in value, (path, value)
			assert 'datasets_v1' not in value, (path, value)
			if 'facies_benchmark_v1' not in value:
				continue
			matches = [
				reference
				for reference in ALLOWED_V1_REFERENCES
				if value == f'{ARTIFACT_ROOT}/{reference}'
			]
			assert matches, f'{path} references a v1 artifact outside the allowlist'
			seen_allowed.update(matches)
	assert seen_allowed == set(ALLOWED_V1_REFERENCES)


def test_v2_artifact_roots_chain_through_prepare_layout_and_five_way() -> None:
	reference = _load(PREPARE_ROOT / '02_extract_reference_valid_tokens.yaml')
	canonical = _load(PREPARE_ROOT / '03_build_voxel_supervision.yaml')
	calibration = _load(LAYOUT_ROOT / '01_prepare_section_layout_contract.yaml')
	builder = _load(LAYOUT_ROOT / '03_build_section_layout_datasets.yaml')
	five_way = _load(FIVE_WAY_ROOT / '60_five_way.yaml')
	inventory = str(Path.cwd() / PREPARE_ROOT / 'section_inventory_v2.csv')

	reference_dir = f'{ARTIFACT_ROOT}/{REFERENCE_GEOMETRY_DIR}'
	assert reference['embeddings']['output_dir'] == reference_dir
	assert reference['embeddings']['checkpoint'] == (
		f'{ARTIFACT_ROOT}/{REFERENCE_GEOMETRY_CHECKPOINT}'
	)
	assert reference['manifests']['input'] == (
		f'{ARTIFACT_ROOT}/registry/manifests/f3/{V2_VERSION}/f3_amplitude_manifest.json'
	)
	assert canonical['reference_embedding'] == {
		'metadata_json': f'{reference_dir}/f3_facies_benchmark.embedding_metadata.json',
		'valid_tokens': f'{reference_dir}/f3_facies_benchmark.valid_tokens.npy',
	}
	assert canonical['labels']['png_label_inventory'] == inventory
	canonical_dir = canonical['voxel_dataset']['output_dir']
	assert canonical_dir.startswith(
		f'{ARTIFACT_ROOT}/lithology/f3/{V2_VERSION}/voxel_supervision/'
	)

	assert calibration['inputs']['canonical_split_grid'] == (
		f'{canonical_dir}/supervision_split_grid.npy'
	)
	assert calibration['inputs']['line_inventory'] == inventory
	assert calibration['inputs']['layout_lines'] == (
		f'{Path.cwd()}/{LAYOUT_ROOT}/02_layout_lines.yaml'
	)
	dataset_root = builder['outputs']['output_root']
	assert dataset_root == (
		f'{ARTIFACT_ROOT}/lithology/f3/{V2_VERSION}/voxel_section_layout_v2'
	)
	assert dataset_root == five_way['section_layout']['dataset_root']
	for output in calibration['outputs'].values():
		assert not output.startswith(f'{dataset_root}/'), output
	assert (
		builder['inputs']['section_layout_contract']
		== (calibration['outputs']['canonical_contract'])
	)
	assert builder['inputs']['canonical_voxel_dataset'] == canonical_dir
	assert builder['inputs']['png_label_inventory'] == inventory
	assert (
		builder['inputs']['reference_valid_tokens']
		== (canonical['reference_embedding']['valid_tokens'])
	)
	assert five_way['labels']['png_label_inventory'] == inventory
	assert five_way['labels']['source_label_volume'] == (
		f'{ARTIFACT_ROOT}/registry/volumes/f3/{V2_VERSION}/f3_facies_labels.npy'
	)


def test_five_way_matrix_models_seed_and_namespaces() -> None:
	config = f3_lithology_five_way_config_from_mapping(
		_load(FIVE_WAY_ROOT / '60_five_way.yaml')
	)
	assert config.model_ids == MODEL_IDS
	jobs = plan_f3_lithology_five_way_jobs(config)
	assert len(jobs) == JOB_COUNT == len(MODEL_IDS) * len(LAYOUT_IDS) * len(SIZES)
	assert {job[1] for job in jobs} == set(LAYOUT_IDS)
	assert {job[2] for job in jobs} == set(SIZES)
	assert config.model_by_id('random').expected['random_seed'] == RANDOM_SEED
	for model in config.models:
		assert str(model.checkpoint) == (
			f'{ARTIFACT_ROOT}/{UPSTREAM_CHECKPOINTS[model.model_id]}'
		)
		assert str(model.embeddings_dir) == (
			f'{ARTIFACT_ROOT}/embeddings/f3/{V2_VERSION}/mae_local_bt_five_way_v2/'
			f'{model.model_id}/overlap_x64'
		)
		assert 'trace_drop' not in str(model.checkpoint)
	assert str(config.runs_root) == (
		f'{ARTIFACT_ROOT}/f3_lithology_benchmark/mae_local_bt_five_way_v2/runs'
	)
	assert str(config.summary_root) == (
		f'{ARTIFACT_ROOT}/f3_lithology_benchmark/mae_local_bt_five_way_v2/summary'
	)
	assert str(config.section_layout_dataset_root) == (
		f'{ARTIFACT_ROOT}/lithology/f3/{V2_VERSION}/voxel_section_layout_v2'
	)


def test_extraction_configs_match_five_way_sources_and_contract() -> None:
	five_way = _load(FIVE_WAY_ROOT / '60_five_way.yaml')
	by_id = {model['model_id']: model for model in five_way['models']}
	manifest = (
		f'{ARTIFACT_ROOT}/registry/manifests/f3/{V2_VERSION}/f3_amplitude_manifest.json'
	)
	configs = [
		*(
			FIVE_WAY_ROOT / '50_embeddings' / name
			for name in EXTRACTION_CONFIGS.values()
		),
		PREPARE_ROOT / '02_extract_reference_valid_tokens.yaml',
	]
	for path in configs:
		extraction = _load(path)
		assert extraction['manifests']['input'] == manifest
		for key, expected in EXTRACTION_CONTRACT.items():
			assert extraction['embedding'][key] == expected, (path, key)
		assert extraction['embeddings']['output_dir'].startswith(
			f'{ARTIFACT_ROOT}/embeddings/f3/{V2_VERSION}/'
		)
	for model_id, name in EXTRACTION_CONFIGS.items():
		extraction = _load(FIVE_WAY_ROOT / '50_embeddings' / name)
		assert extraction['embeddings']['checkpoint'] == by_id[model_id]['checkpoint']
		assert (
			extraction['embeddings']['output_dir'] == by_id[model_id]['embeddings_dir']
		)


def test_section_inventory_is_regular_guarded_and_keeps_benchmark_validation() -> None:
	records = load_f3_slice_split_records(PREPARE_ROOT / 'section_inventory_v2.csv')
	train = {
		slice_type: tuple(
			sorted(
				record.slice_index
				for record in records
				if record.split == 'train' and record.slice_type == slice_type
			)
		)
		for slice_type in ('inline', 'crossline')
	}
	validation = {
		slice_type: tuple(
			sorted(
				record.slice_index
				for record in records
				if record.split == 'validation' and record.slice_type == slice_type
			)
		)
		for slice_type in ('inline', 'crossline')
	}
	assert train == {'inline': TRAIN_INLINES, 'crossline': TRAIN_CROSSLINES}
	assert validation == VALIDATION_LINES
	assert len(records) == 43
	for slice_type, bounds in (
		('inline', INLINE_BOUNDS),
		('crossline', CROSSLINE_BOUNDS),
	):
		for line in train[slice_type]:
			assert bounds[0] < line < bounds[1]
			assert all(
				abs(line - held_out) >= VALIDATION_GUARD
				for held_out in validation[slice_type]
			)


def _expected_layout(k: int, candidates: tuple[int, ...]) -> list[int]:
	quartet = [candidates[k + 5 * j] for j in range(4)]
	start = (k + 1) % 4
	return [quartet[(start + i) % 4] for i in range(4)]


def test_layout_lines_follow_the_documented_rule_and_nest_strictly() -> None:
	layouts = _load(LAYOUT_ROOT / '02_layout_lines.yaml')['layouts']
	assert [layout['layout_id'] for layout in layouts] == list(LAYOUT_IDS)
	used = {'inline': [], 'crossline': []}
	small_pairs = set()
	for k, layout in enumerate(layouts):
		inlines = layout['ordered_inlines']
		crosslines = layout['ordered_crosslines']
		assert inlines == _expected_layout(k, TRAIN_INLINES)
		assert crosslines == _expected_layout(k, TRAIN_CROSSLINES)
		for lines, candidates, slice_type in (
			(inlines, TRAIN_INLINES, 'inline'),
			(crosslines, TRAIN_CROSSLINES, 'crossline'),
		):
			assert len(lines) == 4 == len(set(lines))
			assert set(lines) <= set(candidates)
			assert not set(lines) & set(VALIDATION_LINES[slice_type])
			prefixes = [set(lines[: PREFIX_COUNTS[size]]) for size in SIZES]
			assert prefixes[0] < prefixes[1] < prefixes[2]
			used[slice_type].extend(lines)
		small_pairs.add((inlines[0], crosslines[0]))
	for slice_type in ('inline', 'crossline'):
		assert len(used[slice_type]) == 20 == len(set(used[slice_type]))
	assert len(small_pairs) == 5


def test_calibration_config_uses_literal_v1_median_targets() -> None:
	raw = _load(LAYOUT_ROOT / '01_prepare_section_layout_contract.yaml')
	assert set(raw) == {'inputs', 'selection', 'targets', 'outputs'}
	assert 'legacy_budget_manifest' not in raw['inputs']
	assert raw['targets'] == {
		'rule': TARGET_RULE,
		'train_voxel_counts': TARGET_TRAIN_VOXEL_COUNTS,
	}
	assert raw['selection'] == {
		'semantics': SELECTION_SEMANTICS,
		'patch_size_xyz': [8, 8, 8],
		'allowed_relative_error': ALLOWED_RELATIVE_ERROR,
	}
	config = f3_section_layout_calibration_config_from_mapping(raw)
	assert config.target_rule == TARGET_RULE
	assert dict(config.target_train_voxel_counts or {}) == TARGET_TRAIN_VOXEL_COUNTS
	for path in (
		config.canonical_split_grid,
		config.label_volume,
		config.canonical_contract,
	):
		assert f'/{V2_VERSION}/' in str(path)


def test_v3_calibration_config_uses_literal_class_balanced_contract() -> None:
	raw = _load(LAYOUT_V3_ROOT / '01_prepare_section_layout_contract.yaml')
	assert set(raw) == {'inputs', 'selection', 'targets', 'outputs'}
	assert raw['selection'] == {
		'semantics': V3_SELECTION_SEMANTICS,
		'patch_size_xyz': [8, 8, 8],
		'allowed_relative_error': 0.05,
		'tokenization_policy': V3_TOKENIZATION_POLICY,
		'layout_subsample_seeds': V3_LAYOUT_SUBSAMPLE_SEEDS,
	}
	assert raw['targets'] == {
		'rule': V3_TARGET_RULE,
		'per_class_token_row_caps': V3_PER_CLASS_TOKEN_ROW_CAPS,
		'nominal_train_voxel_counts': V3_NOMINAL_TRAIN_VOXEL_COUNTS,
	}
	calibration_root = (
		f'{ARTIFACT_ROOT}/lithology/f3/facies_benchmark_v2/'
		'voxel_section_layout_v3_calibration'
	)
	assert raw['outputs'] == {
		'candidate_statistics_csv': f'{calibration_root}/section_candidates.csv',
		'candidate_statistics_json': f'{calibration_root}/section_candidates.json',
		'canonical_contract': (
			f'{calibration_root}/f3_voxel_section_layout_contract.json'
		),
	}
	assert raw['inputs']['layout_lines'] == (
		f'{Path.cwd()}/{LAYOUT_V3_ROOT}/02_layout_lines.yaml'
	)
	assert raw['inputs']['reference_valid_tokens'] == (
		f'{ARTIFACT_ROOT}/{REFERENCE_GEOMETRY_DIR}/f3_facies_benchmark.valid_tokens.npy'
	)
	config = f3_section_layout_calibration_config_from_mapping(raw)
	assert config.selection_semantics == V3_SELECTION_SEMANTICS
	assert config.target_rule == V3_TARGET_RULE
	assert dict(config.per_class_token_row_caps or {}) == V3_PER_CLASS_TOKEN_ROW_CAPS
	assert dict(config.target_train_voxel_counts or {}) == (
		V3_NOMINAL_TRAIN_VOXEL_COUNTS
	)
	assert dict(config.layout_subsample_seeds or {}) == V3_LAYOUT_SUBSAMPLE_SEEDS
	assert dict(config.tokenization_policy or {}) == V3_TOKENIZATION_POLICY


def test_v3_layout_lines_use_literal_layout_002_order() -> None:
	v2_layouts = _load(LAYOUT_ROOT / '02_layout_lines.yaml')['layouts']
	v3_layouts = _load(LAYOUT_V3_ROOT / '02_layout_lines.yaml')['layouts']
	assert [layout['layout_id'] for layout in v3_layouts] == [
		'layout_000',
		'layout_001',
		'layout_002',
		'layout_003',
		'layout_004',
	]
	assert v3_layouts[2]['ordered_inlines'] == V3_LAYOUT_002_INLINE_ORDER
	assert v3_layouts[2]['ordered_crosslines'] == [1075, 500, 625, 950]
	for index, (v2_layout, v3_layout) in enumerate(
		zip(v2_layouts, v3_layouts, strict=True)
	):
		assert v3_layout['layout_id'] == v2_layout['layout_id']
		if index == 2:
			assert set(v3_layout['ordered_inlines']) == set(
				v2_layout['ordered_inlines']
			)
		else:
			assert v3_layout['ordered_inlines'] == v2_layout['ordered_inlines']
		assert v3_layout['ordered_crosslines'] == v2_layout['ordered_crosslines']


@pytest.mark.parametrize(
	'path',
	[
		LAYOUT_V3_ROOT / 'README.md',
		FIVE_WAY_V3_ROOT / 'README.md',
	],
)
def test_v3_runbook_shell_blocks_are_fail_fast_and_valid_bash(
	path: Path, tmp_path: Path
) -> None:
	blocks = re.findall(
		r'```bash\n(.*?)```', path.read_text(encoding='utf-8'), flags=re.DOTALL
	)
	assert blocks
	for index, block in enumerate(blocks):
		assert 'set -euo pipefail' in block
		script = tmp_path / f'{path.parent.name}_{index}.sh'
		script.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(script)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)


def test_v3_five_way_runbook_uses_size_layout_model_loop_order() -> None:
	text = (FIVE_WAY_V3_ROOT / 'README.md').read_text(encoding='utf-8')
	full_suite = text[text.index('## full suite') : text.index('## summary')]
	assert full_suite.index('for size in small medium large') < full_suite.index(
		'for layout in layout_000'
	)
	assert full_suite.index('for layout in layout_000') < full_suite.index(
		'for model in'
	)


def test_v3_builder_and_five_way_use_versioned_outputs_and_reused_sources() -> None:
	calibration = _load(LAYOUT_V3_ROOT / '01_prepare_section_layout_contract.yaml')
	builder = _load(LAYOUT_V3_ROOT / '03_build_section_layout_datasets.yaml')
	five_way_raw = _load(FIVE_WAY_V3_ROOT / '60_five_way.yaml')
	dataset_root = (
		f'{ARTIFACT_ROOT}/lithology/f3/facies_benchmark_v2/voxel_section_layout_v3'
	)
	assert (
		builder['inputs']['section_layout_contract']
		== (calibration['outputs']['canonical_contract'])
	)
	assert builder['outputs'] == {'output_root': dataset_root}
	assert five_way_raw['dataset'] == {
		'name': 'f3_facies_benchmark',
		'version': 'facies_benchmark_v2',
	}
	assert five_way_raw['section_layout'] == {'dataset_root': dataset_root}
	assert five_way_raw['outputs'] == {
		'runs_root': (
			f'{ARTIFACT_ROOT}/f3_lithology_benchmark/mae_local_bt_five_way_v3/runs'
		),
		'summary_root': (
			f'{ARTIFACT_ROOT}/f3_lithology_benchmark/mae_local_bt_five_way_v3/summary'
		),
		'summary_name': V3_SUMMARY_NAME,
	}
	v2_models = {
		model['model_id']: model
		for model in _load(FIVE_WAY_ROOT / '60_five_way.yaml')['models']
	}
	assert [model['model_id'] for model in five_way_raw['models']] == list(MODEL_IDS)
	for model in five_way_raw['models']:
		model_id = model['model_id']
		assert model['checkpoint'] == (
			f'{ARTIFACT_ROOT}/{UPSTREAM_CHECKPOINTS[model_id]}'
		)
		assert model['embeddings_dir'] == (
			f'{ARTIFACT_ROOT}/embeddings/f3/facies_benchmark_v2/'
			f'mae_local_bt_five_way_v2/{model_id}/overlap_x64'
		)
		assert model['expected'] == v2_models[model_id]['expected']
	config = f3_lithology_five_way_config_from_mapping(five_way_raw)
	assert dict(config.dataset) == {
		'name': 'f3_facies_benchmark',
		'version': 'facies_benchmark_v2',
	}
	assert config.summary_name == V3_SUMMARY_NAME
	assert len(plan_f3_lithology_five_way_jobs(config)) == 75


def test_prepare_and_inspection_resolvers_accept_v2_only() -> None:
	prepare = _load(PREPARE_ROOT / '01_prepare_f3_volume.yaml')
	assert f3_prepare_volume_config_from_mapping(prepare).dataset.version == V2_VERSION
	unknown = deepcopy(prepare)
	unknown['dataset']['version'] = 'facies_benchmark_v3'
	with pytest.raises(ValueError, match=r'dataset\.version'):
		f3_prepare_volume_config_from_mapping(unknown)
	inspection = _load(INSPECTION_ROOT / '01_inspect_files.yaml')
	resolve_f3_facies_inspection_config(inspection, stage=STAGE_F3_INSPECT_FILES)
	unknown = deepcopy(inspection)
	unknown['dataset']['version'] = 'facies_benchmark_v3'
	with pytest.raises(ValueError, match=r'dataset\.version'):
		resolve_f3_facies_inspection_config(unknown, stage=STAGE_F3_INSPECT_FILES)


def _readme_text() -> str:
	return README.read_text(encoding='utf-8')


def _readme_variables(text: str) -> dict[str, str]:
	variables: dict[str, str] = {}
	for name, value in re.findall(r'^export (\w+)=(\S+)$', text, flags=re.MULTILINE):
		variables[name] = value.strip('"').replace(
			'$EXP/', variables.get('EXP', '') + '/'
		)
	return variables


def test_runbook_references_existing_clis_configs_and_tests() -> None:
	text = _readme_text()
	for cli in RUNBOOK_CLIS:
		assert cli in text
	for cli in set(re.findall(r'proc/seis_ssl_cluster/\w+\.py', text)):
		assert Path(cli).is_file(), cli
	variables = _readme_variables(text)
	references = [
		(variable, relative)
		for variable, relative in re.findall(
			r'"\$(\w+)/([^"\s]+\.(?:yaml|sh|csv))"', text
		)
		if '${' not in relative
	]
	assert references
	for variable, relative in references:
		assert variable in variables, variable
		assert (Path(variables[variable]) / relative).is_file(), (variable, relative)
	loop_extractions = re.findall(
		r'^  (0\d_extract_\w+)(?: \\)?$', text, flags=re.MULTILINE
	)
	assert loop_extractions == [
		name.removesuffix('.yaml') for name in EXTRACTION_CONFIGS.values()
	]
	for test_path in re.findall(r'tests/seis_ssl_cluster/\S+\.py', text):
		assert Path(test_path).is_file(), test_path
	assert '60_five_way.yaml' in text
	assert 'trace_drop' not in text


def test_runbook_shell_blocks_are_valid_bash(tmp_path: Path) -> None:
	blocks = re.findall(r'```bash\n(.*?)```', _readme_text(), flags=re.DOTALL)
	assert blocks
	for index, block in enumerate(blocks):
		script = tmp_path / f'block_{index}.sh'
		script.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(script)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)


def test_runbook_order_dry_runs_and_matrix() -> None:
	text = _readme_text()
	positions = [text.index(heading) for heading in RUNBOOK_ORDER]
	assert positions == sorted(positions)
	for cli in (
		'build_f3_lithology_voxel_dataset.py',
		'prepare_f3_lithology_voxel_section_layout_contract.py',
		'build_f3_lithology_voxel_section_layout_datasets.py',
		'check_f3_prepared_volume_parity.py',
		'audit_f3_lithology_five_way_sources.py',
		'summarize_f3_lithology_five_way.py',
	):
		lines = [
			index
			for index, line in enumerate(text.splitlines())
			if cli in line and 'python' in line
		]
		text_lines = text.splitlines()
		dry = [index for index in lines if '--dry-run' in text_lines[index]]
		live = [index for index in lines if '--dry-run' not in text_lines[index]]
		assert dry, cli
		assert live, cli
		assert dry[0] < live[0], cli
	loop_models = [
		name
		for name in re.findall(r'^\s+(\w+)(?: \\)?$', text, flags=re.MULTILINE)
		if name in MODEL_IDS
	]
	assert loop_models == list(MODEL_IDS) * 2
	for layout_id in LAYOUT_IDS:
		assert layout_id in text
	for size in SIZES:
		assert size in text
	assert 'complete_jobs: 75' in text
	assert 'macro_f1' in text
	assert 'seed 42' in text
	assert not re.search(r'(?m)^\s*(?:rm\s+-rf|cp\s|rsync\s|ln\s+-s)', text)
	assert text.index('--dry-run\ndone') < text.index('for layout in layout_000')
	full_suite = text[
		text.index('## 13. full suite') : text.index('## 14. summary dry-run')
	]
	assert full_suite.index('for size in small medium large') < full_suite.index(
		'for layout in layout_000'
	)
	assert full_suite.index('for layout in layout_000') < full_suite.index(
		'for model in'
	)


def test_layout_yaml_documents_the_rule_and_inventory() -> None:
	text = (LAYOUT_ROOT / '02_layout_lines.yaml').read_text(encoding='utf-8')
	assert 'j0 = (k + 1) mod 4' in text
	assert 'section_inventory_v2.csv' in text
	readme = (LAYOUT_ROOT / 'README.md').read_text(encoding='utf-8')
	assert TARGET_RULE in readme
	assert 'cap25' in readme
	data = yaml.safe_load(
		(LAYOUT_ROOT / '02_layout_lines.yaml').read_text(encoding='utf-8')
	)
	assert len(data['layouts']) == 5
