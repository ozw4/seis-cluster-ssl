from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

from proc.seis_ssl_cluster import extract_embeddings
from seis_ssl_cluster.config import load_config, resolve_embedding_extraction_config
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
	resolve_f3_lithology_candidate_job,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	_decoder_config_mapping,
	_evaluation_config_mapping,
	_inference_config_mapping,
	resolve_f3_lithology_five_way_job,
)
from tests.seis_ssl_cluster.test_f3_hmm_v2_manifest_configs import (
	CANDIDATES,
	EXPERIMENT,
)

BASELINE_EMBEDDING = Path(
	'experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v2/'
	'50_embeddings/02_extract_mae_hmm_k6.yaml'
)
CANONICAL = Path(
	'experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml'
)


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_WORKSPACE', str(Path.cwd()))
	monkeypatch.setenv('F3_ROOT', str(tmp_path / 'f3'))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256', 'a' * 64)


@pytest.mark.parametrize('tag', CANDIDATES)
def test_embedding_reuses_exact_baseline_and_audited_latest(
	tag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	candidate = f'mae100_hmm_v2_mh_k{tag}_distill020'
	path = EXPERIMENT / '40_embeddings' / f'{candidate}.yaml'
	config = load_config(path)
	full = load_config(EXPERIMENT / '30_pretraining' / candidate / '02_full_25ep.yaml')
	expected = load_config(BASELINE_EMBEDDING)
	expected['embeddings'] = {
		'checkpoint': str(Path(full['paths']['output_root']) / 'latest.pt'),
		'output_dir': str(
			Path(config['paths']['artifact_root'])
			/ 'embeddings/f3/facies_benchmark_v2/hmm_v2_multi_head_screening_v1'
			/ candidate
			/ 'overlap_x64'
		),
	}
	assert config == expected
	assert resolve_embedding_extraction_config(config)['embedding']['amp'] is False

	def fail_extraction(*_args: object, **_kwargs: object) -> None:
		pytest.fail('dry-run must not extract embeddings')

	monkeypatch.setattr(extract_embeddings, 'run_embedding_extraction', fail_extraction)
	monkeypatch.setattr(sys, 'argv', ['extract', '--config', str(path), '--dry-run'])
	extract_embeddings.main()
	assert 'execution: dry-run; extraction skipped' in capsys.readouterr().out
	assert not Path(config['embeddings']['output_dir']).exists()


def test_sixty_cells_share_condition_2b_decoder_inference_and_evaluation() -> None:
	outputs = set()
	summaries = set()
	embedding_roots = set()
	for tag in CANDIDATES:
		candidate = f'mae100_hmm_v2_mh_k{tag}_distill020'
		mapping = load_config(EXPERIMENT / '50_downstream' / f'{candidate}.yaml')
		config = f3_lithology_candidate_config_from_mapping(mapping)
		canonical = load_f3_lithology_candidate_canonical_config(config)
		embedding = load_config(EXPERIMENT / '40_embeddings' / f'{candidate}.yaml')
		assert config.candidate_id == candidate
		assert config.canonical_config == CANONICAL.resolve()
		assert str(config.checkpoint) == embedding['embeddings']['checkpoint']
		assert str(config.embeddings_dir) == embedding['embeddings']['output_dir']
		assert (
			config.runs_root
			== canonical.artifact_root
			/ 'f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/runs'
		)
		assert config.summary_root == config.runs_root.parent / 'summary' / candidate
		summaries.add(config.summary_root)
		embedding_roots.add(config.embeddings_dir)
		for layout in LAYOUT_IDS:
			for size in DATA_SIZES:
				job = resolve_f3_lithology_candidate_job(
					config, canonical, layout=layout, size=size
				)
				baseline = resolve_f3_lithology_five_way_job(
					canonical, model='mae_hmm_k6', layout=layout, size=size
				)
				assert job.condition_dir == baseline.condition_dir
				assert job.model.expected == {}
				outputs.add(job.output_dir)
				for builder in (
					_decoder_config_mapping,
					_inference_config_mapping,
					_evaluation_config_mapping,
				):
					actual = builder(job)
					expected = builder(baseline)
					# Only source and destination references distinguish paired jobs.
					replacements = {
						candidate: 'mae_hmm_k6',
						str(config.checkpoint): str(baseline.model.checkpoint),
						str(config.embeddings_dir): str(baseline.model.embeddings_dir),
						str(job.output_dir): str(baseline.output_dir),
					}
					assert _replace_references(actual, replacements) == expected
	assert len(outputs) == 60
	assert len(summaries) == len(embedding_roots) == 4
	assert len(list((EXPERIMENT / '40_embeddings').glob('*.yaml'))) == 4
	assert len(list((EXPERIMENT / '50_downstream').glob('*.yaml'))) == 4
	assert all(not path.exists() for path in outputs)


def _replace_references(value: object, replacements: dict[str, str]) -> object:
	if isinstance(value, dict):
		return {
			key: _replace_references(child, replacements)
			for key, child in value.items()
		}
	if isinstance(value, list):
		return [_replace_references(child, replacements) for child in value]
	if isinstance(value, str):
		for source, target in sorted(
			replacements.items(), key=lambda item: len(item[0]), reverse=True
		):
			if value == source or value.startswith(source + '/'):
				return target + value[len(source) :]
	return deepcopy(value)
