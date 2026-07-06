from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_inspection import resolve_f3_facies_inspection_config
from seis_ssl_cluster.config.schema import (
	STAGE_F3_INSPECT_FILES,
	STAGE_F3_INSPECTION_REPORT,
	STAGE_F3_LABEL_CONSISTENCY,
	STAGE_F3_PNG_LABELS,
	STAGE_F3_QUICKLOOK,
	STAGE_F3_SEGY_GEOMETRY,
	STAGE_F3_TOKENIZATION_PREVIEW,
)
from seis_ssl_cluster.f3 import check_f3_label_consistency

F3_INSPECTION_CONFIG_DIR = Path(
	'experiments/f3/facies_benchmark_v1/00_inspection',
)
F3_INSPECTION_CONFIGS = (
	(F3_INSPECTION_CONFIG_DIR / '01_inspect_files.yaml', STAGE_F3_INSPECT_FILES),
	(
		F3_INSPECTION_CONFIG_DIR / '02_inspect_segy_geometry.yaml',
		STAGE_F3_SEGY_GEOMETRY,
	),
	(F3_INSPECTION_CONFIG_DIR / '03_inspect_png_labels.yaml', STAGE_F3_PNG_LABELS),
	(
		F3_INSPECTION_CONFIG_DIR / '04_make_quicklook_figures.yaml',
		STAGE_F3_QUICKLOOK,
	),
	(
		F3_INSPECTION_CONFIG_DIR / '05_check_label_consistency.yaml',
		STAGE_F3_LABEL_CONSISTENCY,
	),
	(
		F3_INSPECTION_CONFIG_DIR / '06_make_tokenization_preview.yaml',
		STAGE_F3_TOKENIZATION_PREVIEW,
	),
	(
		F3_INSPECTION_CONFIG_DIR / '07_build_inspection_report.yaml',
		STAGE_F3_INSPECTION_REPORT,
	),
)


@pytest.mark.parametrize(('config_path', 'stage'), F3_INSPECTION_CONFIGS)
def test_f3_inspection_stage_module_resolves_active_configs(
	config_path: Path,
	stage: str,
) -> None:
	raw = load_config(config_path)
	original = deepcopy(raw)

	resolved = resolve_f3_facies_inspection_config(raw, stage=stage)

	assert raw == original
	assert resolved['stage'] == stage
	assert '/runs/' not in raw['outputs']['inspection_dir']


def test_f3_inspection_stage_module_rejects_runs_output() -> None:
	raw = load_config(F3_INSPECTION_CONFIG_DIR / '01_inspect_files.yaml')
	raw['outputs']['inspection_dir'] = (
		'/workspace/artifacts/seis_ssl_cluster/runs/f3/facies_benchmark_v1'
	)

	with pytest.raises(ValueError, match=r'outputs\.inspection_dir.*runs/ paths'):
		resolve_f3_facies_inspection_config(raw, stage=STAGE_F3_INSPECT_FILES)


def test_ignore_border_samples_z_runtime_validation_remains_active() -> None:
	with pytest.raises(
		ValueError,
		match='ignore_border_samples_z must be a nonnegative integer',
	):
		check_f3_label_consistency(
			segy=None,  # type: ignore[arg-type]
			png_labels=None,  # type: ignore[arg-type]
			ignore_border_samples_z='1',  # type: ignore[arg-type]
		)
