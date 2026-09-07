from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_five_way import (
	FIVE_WAY_MODEL_IDS,
	F3FiveWayConfig,
	F3FiveWayModelSource,
)
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	FIVE_WAY_SUMMARY_OUTPUT_NAMES,
	F3LithologyCandidateConfig,
	candidate_five_way_comparisons,
	f3_lithology_candidate_config_from_mapping,
	inspect_f3_lithology_candidate_five_way,
	load_f3_lithology_candidate_canonical_config,
	summarize_f3_lithology_candidate_five_way,
)
from seis_ssl_cluster.f3.lithology.five_way_results import (
	PAIRED_FIELDNAMES,
	SUMMARY_METRICS,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
	write_condition,
)
from tests.seis_ssl_cluster.test_f3_lithology_candidate_benchmark import (
	_files_snapshot,
	_write_completed_job,
)

CANDIDATE_ID = 'random_init_hmm_k6_example'
CANDIDATE_LAYOUT_OFFSETS = (-0.02, 0.0, 0.01, 0.02, 0.03)
EXPECTED_COMPARISON_IDS = [
	'candidate_minus_random',
	'candidate_minus_mae',
	'candidate_minus_mae_hmm_k6',
	'candidate_minus_local_bt',
	'candidate_minus_local_bt_hmm_k6',
]


@pytest.fixture
def five_way_candidate_universe(tmp_path: Path) -> dict[str, object]:
	root = tmp_path / 'synthetic'
	canonical_mapping = build_five_way_universe(root)
	source_model = canonical_mapping['models'][2]
	candidate_checkpoint = root / 'pretraining' / CANDIDATE_ID / 'latest.pt'
	candidate_checkpoint.parent.mkdir(parents=True)
	candidate_checkpoint.write_bytes(
		Path(source_model['checkpoint']).read_bytes() + b'\ncandidate-stage2'
	)
	candidate_embeddings = (
		root / 'embeddings/f3/facies_benchmark_v1' / CANDIDATE_ID / 'overlap_x64'
	)
	source_embeddings = Path(source_model['embeddings_dir'])
	candidate_embeddings.mkdir(parents=True)
	for name in (
		f'{SURVEY_ID}.embeddings.npy',
		f'{SURVEY_ID}.valid_tokens.npy',
	):
		shutil.copy2(source_embeddings / name, candidate_embeddings / name)
	metadata_path = source_embeddings / f'{SURVEY_ID}.embedding_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_path'] = str(candidate_checkpoint)
	metadata['checkpoint_sha256'] = file_sha256(candidate_checkpoint)
	(candidate_embeddings / f'{SURVEY_ID}.embedding_metadata.json').write_text(
		json.dumps(metadata), encoding='utf-8'
	)
	canonical_path = tmp_path / 'canonical.yaml'
	canonical_path.write_text(yaml.safe_dump(canonical_mapping), encoding='utf-8')
	candidate_mapping = {
		'benchmark': {'canonical_config': str(canonical_path)},
		'candidate': {
			'id': CANDIDATE_ID,
			'checkpoint': str(candidate_checkpoint),
			'embeddings_dir': str(candidate_embeddings),
		},
		'outputs': {
			'runs_root': str(root / 'candidate_runs'),
			'summary_root': str(root / 'candidate_summary'),
			'five_way_summary_root': str(root / 'candidate_summary_five_way'),
		},
	}
	return {
		'canonical_mapping': canonical_mapping,
		'candidate_mapping': candidate_mapping,
		'canonical_path': canonical_path,
		'root': root,
	}


def _resolved(
	universe: dict[str, object],
) -> tuple[F3LithologyCandidateConfig, F3FiveWayConfig]:
	config = f3_lithology_candidate_config_from_mapping(universe['candidate_mapping'])
	canonical = load_f3_lithology_candidate_canonical_config(config)
	return config, canonical


def _metric_value(model_id: str, layout_id: str, data_size: str, metric: str) -> float:
	layout_index = LAYOUT_IDS.index(layout_id)
	size_index = DATA_SIZES.index(data_size)
	metric_index = SUMMARY_METRICS.index(metric)
	random_value = 0.45 + 0.01 * size_index + 0.0005 * metric_index
	if model_id == 'random':
		return round(random_value, 6)
	if model_id == CANDIDATE_ID:
		return round(random_value + CANDIDATE_LAYOUT_OFFSETS[layout_index], 6)
	model_index = FIVE_WAY_MODEL_IDS.index(model_id)
	return round(
		0.5
		+ 0.02 * model_index
		+ 0.003 * layout_index
		+ 0.01 * size_index
		+ 0.0005 * metric_index,
		6,
	)


def _write_completed_job_with_all_metrics(
	canonical: F3FiveWayConfig,
	*,
	model: F3FiveWayModelSource,
	job_dir: Path,
	layout_id: str,
	data_size: str,
) -> None:
	_write_completed_job(
		canonical,
		model=model,
		job_dir=job_dir,
		layout_id=layout_id,
		data_size=data_size,
		value=_metric_value(model.model_id, layout_id, data_size, 'macro_f1'),
	)
	metrics_path = job_dir / 'evaluation' / 'metrics.json'
	metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
	metrics.update(
		{
			metric: _metric_value(model.model_id, layout_id, data_size, metric)
			for metric in SUMMARY_METRICS
		}
	)
	metrics_path.write_text(json.dumps(metrics), encoding='utf-8')


def _write_complete_runs(
	universe: dict[str, object],
	config: F3LithologyCandidateConfig,
	canonical: F3FiveWayConfig,
) -> None:
	for data_size in DATA_SIZES:
		for layout_id in LAYOUT_IDS:
			write_condition(universe['canonical_mapping'], layout_id, data_size)
	sources = [
		(
			F3FiveWayModelSource(
				model_id=config.candidate_id,
				checkpoint=config.checkpoint,
				embeddings_dir=config.embeddings_dir,
				expected={},
			),
			config.runs_root,
		),
		*((model, canonical.runs_root) for model in canonical.models),
	]
	for model, runs_root in sources:
		for data_size in DATA_SIZES:
			for layout_id in LAYOUT_IDS:
				_write_completed_job_with_all_metrics(
					canonical,
					model=model,
					job_dir=(
						runs_root
						/ f'model={model.model_id}'
						/ f'layout={layout_id}'
						/ f'size={data_size}'
					),
					layout_id=layout_id,
					data_size=data_size,
				)


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		return list(csv.DictReader(handle))


def test_comparisons_pair_the_candidate_with_every_canonical_model() -> None:
	comparisons = candidate_five_way_comparisons('example')

	assert [comparison_id for comparison_id, _, _ in comparisons] == (
		EXPECTED_COMPARISON_IDS
	)
	assert {left for _, left, _ in comparisons} == {'example'}
	assert [right for _, _, right in comparisons] == [
		'random',
		'mae',
		'mae_hmm_k6',
		'local_barlow_twins',
		'local_barlow_twins_hmm_k6',
	]


def test_config_without_five_way_summary_root_resolves_as_before(
	five_way_candidate_universe: dict[str, object],
) -> None:
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	del mapping['outputs']['five_way_summary_root']

	config = f3_lithology_candidate_config_from_mapping(mapping)
	canonical = load_f3_lithology_candidate_canonical_config(config)

	assert config.five_way_summary_root is None
	assert config.candidate_id == CANDIDATE_ID
	assert config.runs_root == Path(mapping['outputs']['runs_root'])
	assert config.summary_root == Path(mapping['outputs']['summary_root'])
	assert canonical.model_ids == FIVE_WAY_MODEL_IDS
	with pytest.raises(ValueError, match='five_way_summary_root is required'):
		inspect_f3_lithology_candidate_five_way(config, canonical)


def test_config_resolves_five_way_summary_root(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, _ = _resolved(five_way_candidate_universe)

	assert config.five_way_summary_root == Path(
		five_way_candidate_universe['candidate_mapping']['outputs'][
			'five_way_summary_root'
		]
	)


@pytest.mark.parametrize(
	('same_as', 'message'),
	[
		('runs_root', 'five_way_summary_root must differ'),
		('summary_root', 'five_way_summary_root must differ'),
	],
)
def test_config_rejects_five_way_summary_root_equal_to_other_outputs(
	five_way_candidate_universe: dict[str, object],
	same_as: str,
	message: str,
) -> None:
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	mapping['outputs']['five_way_summary_root'] = mapping['outputs'][same_as]

	with pytest.raises(ValueError, match=message):
		f3_lithology_candidate_config_from_mapping(mapping)


def test_config_rejects_five_way_summary_root_nested_in_other_outputs(
	five_way_candidate_universe: dict[str, object],
) -> None:
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	mapping['outputs']['five_way_summary_root'] = str(
		Path(mapping['outputs']['summary_root']) / 'five_way'
	)

	with pytest.raises(ValueError, match='five_way_summary_root must differ'):
		f3_lithology_candidate_config_from_mapping(mapping)


def test_config_rejects_relative_five_way_summary_root(
	five_way_candidate_universe: dict[str, object],
) -> None:
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	mapping['outputs']['five_way_summary_root'] = 'relative/summary_five_way'

	with pytest.raises(
		ValueError, match='five_way_summary_root must be an absolute path'
	):
		f3_lithology_candidate_config_from_mapping(mapping)


def test_config_still_rejects_unexpected_output_keys(
	five_way_candidate_universe: dict[str, object],
) -> None:
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	mapping['outputs']['unexpected_root'] = '/unexpected'

	with pytest.raises(ValueError, match='outputs keys must be exactly'):
		f3_lithology_candidate_config_from_mapping(mapping)


@pytest.mark.parametrize(
	('canonical_output_name', 'suffix'),
	[
		('runs_root', None),
		('summary_root', 'candidate_five_way'),
	],
)
def test_config_rejects_five_way_summary_root_overlapping_canonical_roots(
	five_way_candidate_universe: dict[str, object],
	canonical_output_name: str,
	suffix: str | None,
) -> None:
	canonical_root = Path(
		five_way_candidate_universe['canonical_mapping']['outputs'][
			canonical_output_name
		]
	)
	mapping = deepcopy(five_way_candidate_universe['candidate_mapping'])
	mapping['outputs']['five_way_summary_root'] = str(
		canonical_root if suffix is None else canonical_root / suffix
	)
	config = f3_lithology_candidate_config_from_mapping(mapping)

	with pytest.raises(
		ValueError, match=r'outputs\.five_way_summary_root overlaps canonical'
	):
		load_f3_lithology_candidate_canonical_config(config)


def test_summary_writes_five_files_and_aggregates_all_comparisons(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	summary_root = config.five_way_summary_root

	result = summarize_f3_lithology_candidate_five_way(config, canonical)

	assert result['candidate_id'] == CANDIDATE_ID
	assert result['complete_jobs'] == 90
	assert result['summary_root'] == str(summary_root)
	assert sorted(path.name for path in summary_root.iterdir()) == sorted(
		FIVE_WAY_SUMMARY_OUTPUT_NAMES
	)
	assert result['outputs'] == [
		str(summary_root / name) for name in FIVE_WAY_SUMMARY_OUTPUT_NAMES
	]
	comparison_rows = _read_csv(summary_root / 'comparison.csv')
	assert len(comparison_rows) == 90
	assert {row['model_id'] for row in comparison_rows} == {
		CANDIDATE_ID,
		*FIVE_WAY_MODEL_IDS,
	}
	assert comparison_rows[0]['model_id'] == CANDIDATE_ID
	assert comparison_rows[0]['encoder_checkpoint_sha256'] == file_sha256(
		config.checkpoint
	)
	paired_rows = _read_csv(summary_root / 'paired_deltas.csv')
	assert list(paired_rows[0]) == list(PAIRED_FIELDNAMES)
	assert len(paired_rows) == 5 * 15 * 4
	assert {row['comparison_id'] for row in paired_rows} == set(EXPECTED_COMPARISON_IDS)
	assert {row['left_model'] for row in paired_rows} == {CANDIDATE_ID}
	by_size_rows = _read_csv(summary_root / 'summary_by_size.csv')
	assert len(by_size_rows) == 3 * 5 * 4

	payload = json.loads((summary_root / 'summary.json').read_text(encoding='utf-8'))
	assert payload['schema_version'] == 1
	assert payload['candidate_id'] == CANDIDATE_ID
	assert payload['primary_metric'] == 'macro_f1'
	assert payload['statistical_unit'] == 'layout_id'
	assert payload['job_count'] == 90
	assert payload['models'] == [CANDIDATE_ID, *FIVE_WAY_MODEL_IDS]
	assert [item['comparison_id'] for item in payload['comparisons']] == (
		EXPECTED_COMPARISON_IDS
	)
	medium = payload['by_size']['medium']['candidate_minus_random']['macro_f1']
	assert medium['n_layouts'] == 5
	assert medium['mean'] == pytest.approx(0.008)
	assert medium['positive_count'] == 3
	assert medium['zero_count'] == 1
	assert medium['negative_count'] == 1
	assert medium['reference_mean'] == pytest.approx(0.46)
	assert medium['candidate_mean'] == pytest.approx(0.468)
	assert payload['provenance']['candidate']['checkpoint_sha256'] == file_sha256(
		config.checkpoint
	)
	assert set(payload['provenance']['canonical']) == set(FIVE_WAY_MODEL_IDS)
	mae_provenance = payload['provenance']['canonical']['mae']
	assert mae_provenance['checkpoint_sha256'] == file_sha256(
		canonical.model_by_id('mae').checkpoint
	)
	assert mae_provenance['embeddings_sha256'] == file_sha256(
		canonical.model_by_id('mae').embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
	)

	markdown = (summary_root / 'summary.md').read_text(encoding='utf-8')
	assert markdown.startswith('# F3 lithology candidate versus five-way summary\n')
	assert f'Candidate `{CANDIDATE_ID}`' in markdown
	assert (
		'| size | comparison | candidate mean | reference mean | delta mean '
		'| median | sample std | +/0/- |'
	) in markdown
	assert '| medium | candidate_minus_random |' in markdown


def test_dry_run_inspect_reads_90_jobs_and_writes_nothing(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	before = _files_snapshot(five_way_candidate_universe['root'])

	report = inspect_f3_lithology_candidate_five_way(config, canonical)

	assert report['candidate_id'] == CANDIDATE_ID
	assert report['complete_jobs'] == 90
	assert report['models'] == [CANDIDATE_ID, *FIVE_WAY_MODEL_IDS]
	assert report['comparisons'] == EXPECTED_COMPARISON_IDS
	assert _files_snapshot(five_way_candidate_universe['root']) == before
	assert not config.five_way_summary_root.exists()


def test_missing_canonical_cell_rejects_summary(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	missing = (
		canonical.runs_root
		/ 'model=mae_hmm_k6/layout=layout_002/size=medium/evaluation/metrics.json'
	)
	missing.unlink()

	with pytest.raises(
		FileNotFoundError,
		match=rf'missing 1 of 90 candidate/five-way.*{re.escape(str(missing))}',
	):
		summarize_f3_lithology_candidate_five_way(config, canonical)
	assert not config.five_way_summary_root.exists()


def test_canonical_checkpoint_drift_rejects_summary(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	canonical.model_by_id('mae').checkpoint.write_bytes(b'regenerated checkpoint')

	with pytest.raises(
		ValueError,
		match=(
			r'mae/layout_000/small completed job encoder_checkpoint_sha256 '
			r'does not match the current source'
		),
	):
		summarize_f3_lithology_candidate_five_way(config, canonical)
	assert not config.five_way_summary_root.exists()


def test_inspect_drops_the_private_tile_manifest_key_from_rows(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)

	report = inspect_f3_lithology_candidate_five_way(config, canonical)

	assert len(report['rows']) == 90
	assert all('_validation_tile_manifest_sha256' not in row for row in report['rows'])
	assert len({row['validation_identity'] for row in report['rows']}) == 1


def test_validation_mask_drift_in_one_condition_rejects_summary(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	metadata_path = (
		canonical.section_layout_dataset_root
		/ 'datasets/layout=layout_001/size=small/voxel_supervision'
		/ 'section_layout_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['identity']['validation_mask_sha256'] = 'f' * 64
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='validation mask identity must be shared'):
		summarize_f3_lithology_candidate_five_way(config, canonical)
	assert not config.five_way_summary_root.exists()


def test_validation_voxel_count_drift_in_one_condition_rejects_summary(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	condition = 'layout=layout_003/size=large'
	metadata_path = (
		canonical.section_layout_dataset_root
		/ 'datasets'
		/ condition
		/ 'voxel_supervision/section_layout_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	drifted = int(metadata['identity']['validation_voxel_count']) + 1
	metadata['identity']['validation_voxel_count'] = drifted
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	for model_id, runs_root in (
		(config.candidate_id, config.runs_root),
		*((model_id, canonical.runs_root) for model_id in FIVE_WAY_MODEL_IDS),
	):
		metrics_path = (
			runs_root / f'model={model_id}' / condition / 'evaluation/metrics.json'
		)
		metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
		metrics['evaluation_voxel_count'] = drifted
		metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

	with pytest.raises(ValueError, match='validation voxel count must be shared'):
		summarize_f3_lithology_candidate_five_way(config, canonical)
	assert not config.five_way_summary_root.exists()


def test_cli_dry_run_reports_completeness_and_writes_nothing(
	five_way_candidate_universe: dict[str, object], tmp_path: Path
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	candidate_path = tmp_path / 'candidate.yaml'
	candidate_path.write_text(
		yaml.safe_dump(five_way_candidate_universe['candidate_mapping']),
		encoding='utf-8',
	)
	before = _files_snapshot(five_way_candidate_universe['root'])

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/summarize_f3_lithology_candidate_five_way.py',
			'--config',
			str(candidate_path),
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert f'candidate_id: {CANDIDATE_ID}' in result.stdout
	assert 'complete_jobs: 90' in result.stdout
	assert f'models: {", ".join((CANDIDATE_ID, *FIVE_WAY_MODEL_IDS))}' in (
		result.stdout
	)
	assert f'comparisons: {", ".join(EXPECTED_COMPARISON_IDS)}' in result.stdout
	assert 'execution: dry-run; summary files skipped' in result.stdout
	assert _files_snapshot(five_way_candidate_universe['root']) == before
	assert not config.five_way_summary_root.exists()


def test_summary_refuses_to_overwrite_existing_five_way_summary_root(
	five_way_candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(five_way_candidate_universe)
	_write_complete_runs(five_way_candidate_universe, config, canonical)
	config.five_way_summary_root.mkdir(parents=True)
	sentinel = config.five_way_summary_root / 'keep.txt'
	sentinel.write_text('existing', encoding='utf-8')

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		summarize_f3_lithology_candidate_five_way(config, canonical)
	assert sorted(path.name for path in config.five_way_summary_root.iterdir()) == [
		'keep.txt'
	]
	assert sentinel.read_text(encoding='utf-8') == 'existing'
