"""Canonical F3 controls remain read-only and retain all completion audits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import paired_candidate_results
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)
from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from tests.seis_ssl_cluster.helpers_f3_five_way import (
	SURVEY_ID,
	build_five_way_universe,
)


@pytest.fixture
def control(tmp_path):
	canonical = build_five_way_universe(tmp_path / 'sources')
	path = tmp_path / 'canonical.json'
	path.write_text(json.dumps(canonical))
	model = next(m for m in canonical['models'] if m['model_id'] == 'mae_hmm_k6')
	return {
		'benchmark': {'canonical_config': str(path)},
		'candidate': {
			'id': model['model_id'],
			'checkpoint': model['checkpoint'],
			'embeddings_dir': model['embeddings_dir'],
		},
		'outputs': {
			key: canonical['outputs'][key] for key in ('runs_root', 'summary_root')
		},
	}


def test_canonical_control_reuses_source_and_completed_job_audits(control, monkeypatch):
	calls = []

	def read_job(  # noqa: PLR0913
		pair, canonical, model, provenance, *, layout_id, data_size
	):
		assert pair.runs_root == canonical.runs_root
		assert model.checkpoint == canonical.model_by_id('mae_hmm_k6').checkpoint
		assert provenance['checkpoint_sha256'] == file_sha256(model.checkpoint)
		assert provenance['embeddings_sha256'] == file_sha256(
			model.embeddings_dir / f'{SURVEY_ID}.embeddings.npy'
		)
		calls.append((model.model_id, layout_id, data_size))
		return {'macro_f1': 0.6}

	monkeypatch.setattr(
		paired_candidate_results, 'read_f3_paired_candidate_job', read_job
	)
	assert audit_completed_cell('f3', control, 'mae_hmm_k6', 'layout_000', 'small') == {
		'macro_f1': 0.6,
	}
	assert calls == [('mae_hmm_k6', 'layout_000', 'small')]
	config = f3_lithology_candidate_config_from_mapping(control)
	with pytest.raises(ValueError, match='conflicts with canonical model ID'):
		load_f3_lithology_candidate_canonical_config(config)


@pytest.mark.parametrize(
	'field', ['checkpoint', 'embeddings_dir', 'runs_root', 'summary_root']
)
def test_canonical_control_rejects_foreign_source_or_output_paths(control, field):
	section = 'candidate' if field in ('checkpoint', 'embeddings_dir') else 'outputs'
	control[section][field] += '.foreign'
	with pytest.raises(ValueError, match=f'{field} identity drift'):
		audit_completed_cell('f3', control, 'mae_hmm_k6', 'layout_000', 'small')


def test_completed_cell_rejects_a_different_selected_model(control):
	with pytest.raises(ValueError, match='candidate identity drift'):
		audit_completed_cell('f3', control, 'random', 'layout_000', 'small')


def test_canonical_control_still_rejects_tampered_embedding_provenance(control):
	path = (
		Path(control['candidate']['embeddings_dir'])
		/ f'{SURVEY_ID}.embedding_metadata.json'
	)
	metadata = json.loads(path.read_text())
	metadata['checkpoint_sha256'] = '0' * 64
	path.write_text(json.dumps(metadata))
	with pytest.raises(ValueError, match='checkpoint_sha256 does not match'):
		audit_completed_cell('f3', control, 'mae_hmm_k6', 'layout_000', 'small')


def test_canonical_control_still_rejects_missing_completed_job(control):
	with pytest.raises(FileNotFoundError):
		audit_completed_cell('f3', control, 'mae_hmm_k6', 'layout_000', 'small')
