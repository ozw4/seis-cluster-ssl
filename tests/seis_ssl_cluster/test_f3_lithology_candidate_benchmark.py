from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	audit_f3_lithology_candidate_source,
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	summarize_f3_lithology_candidate,
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


def _write_metrics(path: Path, value: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				'macro_f1': value,
				'aggregation_unit': 'unique_validation_voxel',
				'evaluation_voxel_count': 128,
			}
		),
		encoding='utf-8',
	)


def _write_complete_metrics(config: object, canonical: object) -> None:
	for size_index, data_size in enumerate(('small', 'medium', 'large')):
		for layout_index in range(5):
			layout_id = f'layout_{layout_index:03d}'
			random_value = 0.45 + (0.01 * size_index)
			candidate_value = (
				random_value + (-0.02, 0.0, 0.01, 0.02, 0.03)[layout_index]
			)
			candidate_path = (
				config.runs_root
				/ f'model={config.candidate_id}'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
				/ 'evaluation/metrics.json'
			)
			random_path = (
				canonical.runs_root
				/ 'model=random'
				/ f'layout={layout_id}'
				/ f'size={data_size}'
				/ 'evaluation/metrics.json'
			)
			_write_metrics(candidate_path, candidate_value)
			_write_metrics(random_path, random_value)


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
	_write_complete_metrics(config, canonical)

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
	_write_complete_metrics(config, canonical)
	missing = (
		config.runs_root
		/ f'model={config.candidate_id}'
		/ 'layout=layout_004/size=large/evaluation/metrics.json'
	)
	missing.unlink()

	with pytest.raises(FileNotFoundError, match='missing 1 candidate/random metric'):
		summarize_f3_lithology_candidate(config, canonical)
	assert not config.summary_root.exists()
