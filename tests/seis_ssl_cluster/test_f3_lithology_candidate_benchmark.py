from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.config.f3_lithology_five_way import F3FiveWayModelSource
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	audit_f3_lithology_candidate_source,
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	summarize_f3_lithology_candidate,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	FIVE_WAY_EVALUATION_POLICY,
)
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
	write_condition,
)


@pytest.fixture
def candidate_universe(tmp_path: Path) -> dict[str, object]:
	root = tmp_path / 'synthetic'
	canonical_mapping = build_five_way_universe(root)
	candidate_id = 'local_barlow_twins_example'
	candidate_checkpoint = root / 'pretraining' / candidate_id / 'latest.pt'
	candidate_checkpoint.parent.mkdir(parents=True)
	shutil.copy2(canonical_mapping['models'][2]['checkpoint'], candidate_checkpoint)
	candidate_embeddings = (
		root / 'embeddings/f3/facies_benchmark_v1' / candidate_id / 'overlap_x64'
	)
	source_embeddings = Path(canonical_mapping['models'][2]['embeddings_dir'])
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
			'id': candidate_id,
			'checkpoint': str(candidate_checkpoint),
			'embeddings_dir': str(candidate_embeddings),
		},
		'outputs': {
			'runs_root': str(root / 'candidate_runs'),
			'summary_root': str(root / 'candidate_summary'),
		},
	}
	return {
		'canonical_mapping': canonical_mapping,
		'candidate_mapping': candidate_mapping,
		'canonical_path': canonical_path,
		'candidate_embeddings': candidate_embeddings,
		'root': root,
	}


def _resolved(
	universe: dict[str, object],
) -> tuple[object, object]:
	config = f3_lithology_candidate_config_from_mapping(universe['candidate_mapping'])
	canonical = load_f3_lithology_candidate_canonical_config(config)
	return config, canonical


def _files_snapshot(root: Path) -> dict[str, str]:
	return {
		str(path): file_sha256(path)
		for path in sorted(root.rglob('*'))
		if path.is_file()
	}


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_completed_job(  # noqa: PLR0913
	canonical: object,
	*,
	model: F3FiveWayModelSource,
	job_dir: Path,
	layout_id: str,
	data_size: str,
	value: float,
) -> None:
	condition_dir = (
		canonical.section_layout_dataset_root
		/ 'datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	condition_metadata = json.loads(
		(condition_dir / 'section_layout_metadata.json').read_text(encoding='utf-8')
	)
	decoder = job_dir / 'decoder'
	prediction = job_dir / 'prediction'
	evaluation = job_dir / 'evaluation'
	for directory in (decoder, prediction, evaluation):
		directory.mkdir(parents=True, exist_ok=True)
	(decoder / 'best.pt').write_text(
		f'{model.model_id}/{layout_id}/{data_size}', encoding='utf-8'
	)
	(decoder / 'resolved_config.json').write_text(
		json.dumps(
			{
				'embeddings': {
					'checkpoint_path': str(model.checkpoint),
					'input_dir': str(model.embeddings_dir),
				}
			}
		),
		encoding='utf-8',
	)
	(decoder / 'run_metadata.json').write_text(
		json.dumps(
			{
				'voxel_dataset_metadata': str(
					condition_dir / 'voxel_dataset_metadata.json'
				),
				'train_tile_manifest_sha256': f'{layout_id}-{data_size}-train',
				'validation_tile_manifest_sha256': (
					f'{layout_id}-{data_size}-validation'
				),
			}
		),
		encoding='utf-8',
	)
	embeddings = model.embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
	embedding_metadata = model.embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json'
	valid_tokens = model.embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
	prediction_metadata = prediction / 'prediction_metadata.json'
	prediction_metadata.write_text(
		json.dumps(
			{
				'model_tag': model.model_id,
				'source_identity': {
					'decoder_checkpoint': _identity(decoder / 'best.pt'),
					'artifact_identities': {
						'embeddings': _identity(embeddings),
						'embedding_metadata': _identity(embedding_metadata),
						'valid_tokens': _identity(valid_tokens),
					},
				},
			}
		),
		encoding='utf-8',
	)
	(evaluation / 'metrics.json').write_text(
		json.dumps(
			{
				'macro_f1': value,
				'aggregation_unit': 'unique_validation_voxel',
				'evaluation_voxel_count': condition_metadata['identity'][
					'validation_voxel_count'
				],
			}
		),
		encoding='utf-8',
	)
	(evaluation / 'evaluation_metadata.json').write_text(
		json.dumps(
			{
				'dataset': dict(canonical.dataset),
				'model_tag': model.model_id,
				'policy': {
					key: list(expected) if isinstance(expected, tuple) else expected
					for key, expected in FIVE_WAY_EVALUATION_POLICY.items()
				},
				'inputs': {
					'prediction_metadata': _identity(prediction_metadata),
					'voxel_dataset_metadata': _identity(
						condition_dir / 'voxel_dataset_metadata.json'
					),
					'voxel_split_grid': _identity(
						condition_dir / 'supervision_split_grid.npy'
					),
				},
			}
		),
		encoding='utf-8',
	)


def _write_complete_runs(
	universe: dict[str, object], config: object, canonical: object
) -> None:
	for data_size in ('small', 'medium', 'large'):
		for layout_index in range(5):
			write_condition(
				universe['canonical_mapping'], f'layout_{layout_index:03d}', data_size
			)
	candidate_source = F3FiveWayModelSource(
		model_id=config.candidate_id,
		checkpoint=config.checkpoint,
		embeddings_dir=config.embeddings_dir,
		expected={},
	)
	random_source = canonical.model_by_id('random')
	for size_index, data_size in enumerate(('small', 'medium', 'large')):
		for layout_index in range(5):
			layout_id = f'layout_{layout_index:03d}'
			random_value = 0.45 + (0.01 * size_index)
			candidate_value = (
				random_value + (-0.02, 0.0, 0.01, 0.02, 0.03)[layout_index]
			)
			candidate_job = (
				config.runs_root
				/ f'model={config.candidate_id}'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
			)
			random_job = (
				canonical.runs_root
				/ 'model=random'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
			)
			_write_completed_job(
				canonical,
				model=candidate_source,
				job_dir=candidate_job,
				layout_id=layout_id,
				data_size=data_size,
				value=candidate_value,
			)
			_write_completed_job(
				canonical,
				model=random_source,
				job_dir=random_job,
				layout_id=layout_id,
				data_size=data_size,
				value=random_value,
			)


def test_candidate_config_requires_exact_minimal_keys() -> None:
	with pytest.raises(ValueError, match='keys must be exactly'):
		f3_lithology_candidate_config_from_mapping(
			{
				'benchmark': {'canonical_config': '/canonical.yaml'},
				'candidate': {
					'id': 'candidate',
					'checkpoint': '/checkpoint.pt',
					'embeddings_dir': '/embeddings',
					'unexpected': True,
				},
				'outputs': {
					'runs_root': '/runs',
					'summary_root': '/summary',
				},
			}
		)


def test_candidate_config_rejects_canonical_model_id(
	candidate_universe: dict[str, object],
) -> None:
	candidate_universe['candidate_mapping']['candidate']['id'] = 'random'
	config = f3_lithology_candidate_config_from_mapping(
		candidate_universe['candidate_mapping']
	)

	with pytest.raises(ValueError, match='conflicts with canonical model ID'):
		load_f3_lithology_candidate_canonical_config(config)


@pytest.mark.parametrize(
	('output_name', 'canonical_output_name', 'suffix'),
	[
		('runs_root', 'runs_root', None),
		('summary_root', 'summary_root', 'candidate'),
	],
)
def test_candidate_config_rejects_canonical_output_overlap(
	candidate_universe: dict[str, object],
	output_name: str,
	canonical_output_name: str,
	suffix: str | None,
) -> None:
	canonical_root = Path(
		candidate_universe['canonical_mapping']['outputs'][canonical_output_name]
	)
	candidate_universe['candidate_mapping']['outputs'][output_name] = str(
		canonical_root if suffix is None else canonical_root / suffix
	)
	config = f3_lithology_candidate_config_from_mapping(
		candidate_universe['candidate_mapping']
	)

	with pytest.raises(ValueError, match=f'outputs.{output_name} overlaps canonical'):
		load_f3_lithology_candidate_canonical_config(config)


def test_candidate_source_audit_records_actual_sha_provenance(
	candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(candidate_universe)

	provenance = audit_f3_lithology_candidate_source(config, canonical)

	assert provenance['candidate_id'] == config.candidate_id
	assert provenance['checkpoint_sha256'] == file_sha256(config.checkpoint)
	assert provenance['embeddings_sha256'] == file_sha256(
		config.embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
	)
	assert (
		provenance['valid_tokens_sha256']
		== provenance['canonical_random']['valid_tokens_sha256']
	)


def test_candidate_source_audit_rejects_checkpoint_metadata_sha_mismatch(
	candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(candidate_universe)
	metadata_path = config.embeddings_dir / f'{SURVEY_ID}.embedding_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['checkpoint_sha256'] = '0' * 64
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match='checkpoint_sha256'):
		audit_f3_lithology_candidate_source(config, canonical)


def test_candidate_source_audit_rejects_random_valid_token_mismatch(
	candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(candidate_universe)
	valid_path = config.embeddings_dir / f'{SURVEY_ID}.valid_tokens.npy'
	valid_tokens = np.load(valid_path, allow_pickle=False)
	valid_tokens[0, 0, 0] = False
	np.save(valid_path, valid_tokens, allow_pickle=False)

	with pytest.raises(ValueError, match='byte-identical'):
		audit_f3_lithology_candidate_source(config, canonical)


def test_candidate_cli_dry_run_audits_source_and_writes_nothing(
	candidate_universe: dict[str, object], tmp_path: Path
) -> None:
	config, _ = _resolved(candidate_universe)
	write_condition(candidate_universe['canonical_mapping'], 'layout_000', 'small')
	candidate_path = tmp_path / 'candidate.yaml'
	candidate_path.write_text(
		yaml.safe_dump(candidate_universe['candidate_mapping']), encoding='utf-8'
	)
	before = _files_snapshot(candidate_universe['root'])

	result = subprocess.run(  # noqa: S603
		[
			sys.executable,
			'proc/seis_ssl_cluster/run_f3_lithology_candidate.py',
			'--config',
			str(candidate_path),
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--dry-run',
		],
		check=True,
		capture_output=True,
		text=True,
	)

	assert f'candidate_id: {config.candidate_id}' in result.stdout
	assert 'checkpoint_sha256:' in result.stdout
	assert 'valid_tokens_sha256:' in result.stdout
	assert 'execution: dry-run; no files written' in result.stdout
	assert _files_snapshot(candidate_universe['root']) == before


def test_summary_writes_three_files_and_aggregates_paired_deltas(
	candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(candidate_universe)
	_write_complete_runs(candidate_universe, config, canonical)

	result = summarize_f3_lithology_candidate(config, canonical)
	payload = json.loads(
		(config.summary_root / 'summary.json').read_text(encoding='utf-8')
	)

	assert result['complete_jobs'] == 15
	assert sorted(path.name for path in config.summary_root.iterdir()) == [
		'comparison.csv',
		'summary.json',
		'summary.md',
	]
	assert payload['statistical_unit'] == 'layout_id'
	assert payload['by_size']['medium']['positive_count'] == 3
	assert payload['by_size']['medium']['zero_count'] == 1
	assert payload['by_size']['medium']['negative_count'] == 1
	assert payload['by_size']['medium']['mean'] == pytest.approx(0.008)
	assert len(payload['provenance']['metrics']) == 15


def test_summary_rejects_missing_candidate_or_random_cell(
	candidate_universe: dict[str, object],
) -> None:
	config, canonical = _resolved(candidate_universe)
	_write_complete_runs(candidate_universe, config, canonical)
	missing = (
		config.runs_root
		/ f'model={config.candidate_id}'
		/ 'layout=layout_004/size=large/evaluation/metrics.json'
	)
	missing.unlink()

	with pytest.raises(FileNotFoundError, match='missing 1 candidate/random metric'):
		summarize_f3_lithology_candidate(config, canonical)
	assert not config.summary_root.exists()


def test_summary_rejects_completed_runs_from_previous_candidate_source(
	candidate_universe: dict[str, object],
) -> None:
	config_c1, canonical = _resolved(candidate_universe)
	_write_complete_runs(candidate_universe, config_c1, canonical)
	candidate_mapping_c2 = deepcopy(candidate_universe['candidate_mapping'])
	checkpoint_c2 = candidate_universe['root'] / 'pretraining/candidate_c2/latest.pt'
	checkpoint_c2.parent.mkdir(parents=True)
	checkpoint_c2.write_text('different candidate checkpoint', encoding='utf-8')
	embeddings_c2 = candidate_universe['root'] / 'embeddings/candidate_c2/overlap_x64'
	embeddings_c2.mkdir(parents=True)
	for name in (
		f'{SURVEY_ID}.embeddings.npy',
		f'{SURVEY_ID}.valid_tokens.npy',
	):
		shutil.copy2(config_c1.embeddings_dir / name, embeddings_c2 / name)
	metadata_name = f'{SURVEY_ID}.embedding_metadata.json'
	metadata_c2 = json.loads(
		(config_c1.embeddings_dir / metadata_name).read_text(encoding='utf-8')
	)
	metadata_c2['checkpoint_path'] = str(checkpoint_c2)
	metadata_c2['checkpoint_sha256'] = file_sha256(checkpoint_c2)
	(embeddings_c2 / metadata_name).write_text(
		json.dumps(metadata_c2), encoding='utf-8'
	)
	candidate_mapping_c2['candidate']['checkpoint'] = str(checkpoint_c2)
	candidate_mapping_c2['candidate']['embeddings_dir'] = str(embeddings_c2)
	config_c2 = f3_lithology_candidate_config_from_mapping(candidate_mapping_c2)

	with pytest.raises(ValueError, match='does not match the configured source'):
		summarize_f3_lithology_candidate(config_c2, canonical)
	assert not config_c2.summary_root.exists()
