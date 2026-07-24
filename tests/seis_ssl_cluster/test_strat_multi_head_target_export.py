"""Contracts for strict K=6/8/10 pseudo-target bundle export."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.stratigraphy import multi_head_export
from seis_ssl_cluster.stratigraphy.multi_head_export import (
	export_multi_head_pseudo_targets,
	resolve_multi_head_pseudo_target_export_config,
)
from seis_ssl_cluster.stratigraphy.targets import (
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_multi_head_export_is_schema_v1_resumable_and_quarantines_one_head(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	clustering = _clustering_root(tmp_path)
	config = resolve_multi_head_pseudo_target_export_config(
		_config(tmp_path, clustering),
	)

	dry_run = export_multi_head_pseudo_targets(config, dry_run=True)
	assert [plan.action for plan in dry_run] == ['NEW', 'NEW', 'NEW']
	assert not config.pseudo_target_root.exists()

	export_multi_head_pseudo_targets(config)
	for k in (6, 8, 10):
		inputs = discover_pseudo_target_inputs(config.pseudo_target_root, k=k)
		assert len(inputs) == 1
		assert inputs[0].boundary_weight_path is None
		assert not list((config.pseudo_target_root / f'k{k}').glob('*boundary*'))
		arrays = load_pseudo_target_arrays(inputs[0])
		assert arrays.labels.dtype == np.int32
		assert arrays.confidence.dtype == np.float32
		assert arrays.valid_tokens.dtype == np.bool_
		assert np.all(arrays.confidence[arrays.valid_tokens] == 1.0)
		assert np.all(arrays.confidence[~arrays.valid_tokens] == 0.0)
		assert np.all(arrays.labels[~arrays.valid_tokens] == -1)
		assert load_pseudo_target_metadata(inputs[0])['schema_version'] == 1

	monkeypatch.setattr(
		multi_head_export,
		'export_hmm_cluster_labels_as_pseudo_targets',
		lambda **_: (_ for _ in ()).throw(AssertionError('must reuse complete heads')),
	)
	assert [
		plan.action
		for plan in export_multi_head_pseudo_targets(config, only_missing=True)
	] == ['REUSE', 'REUSE', 'REUSE']

	monkeypatch.undo()
	metadata_path = (
		config.pseudo_target_root / 'k8' / 'survey.pseudo_target_metadata.json'
	)
	metadata_path.write_text(
		'{}',
		encoding='utf-8',
	)
	plans = export_multi_head_pseudo_targets(config, only_missing=True)
	assert [plan.action for plan in plans] == ['REUSE', 'QUARANTINE', 'REUSE']
	assert list(config.pseudo_target_root.glob('k8.quarantine.*'))
	assert load_pseudo_target_metadata(
		discover_pseudo_target_inputs(config.pseudo_target_root, k=8)[0],
	)['schema_version'] == 1
	handoff = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	assert handoff['completion_status'] == 'COMPLETE'
	assert set(handoff['common_target_valid_sha256']) == {'survey'}
	assert set(handoff['source_embedding']['valid_tokens_sha256']) == {'survey'}


def test_multi_head_export_rejects_historical_k6_output_path(tmp_path: Path) -> None:
	clustering = _clustering_root(tmp_path)
	config = _config(tmp_path, clustering)
	config['pseudo_target_root'] = config['historical_k6_root']
	with pytest.raises(ValueError, match='historical_k6_root'):
		resolve_multi_head_pseudo_target_export_config(config)


def test_multi_head_export_rejects_target_valid_outside_source_embedding(
	tmp_path: Path,
) -> None:
	clustering = _clustering_root(tmp_path)
	config = resolve_multi_head_pseudo_target_export_config(
		_config(tmp_path, clustering),
	)
	valid_path = config.source_embedding_dir / 'survey.valid_tokens.npy'
	valid = np.load(valid_path)
	valid[0, 0, 0] = False
	np.save(valid_path, valid)

	with pytest.raises(ValueError, match='valid-token mask is not a subset'):
		export_multi_head_pseudo_targets(config, dry_run=True)


def _config(tmp_path: Path, clustering: Path) -> dict[str, object]:
	return {
		'clustering_output_dir': str(clustering),
		'source_embedding_dir': str(tmp_path / 'embeddings'),
		'pseudo_target_root': str(tmp_path / 'replay'),
		'historical_k6_root': str(tmp_path / 'historical'),
		'ks': [6, 8, 10],
		'confidence': 1.0,
		'schema_version': 1,
		'write_boundary_weight': False,
		'outputs': {'overwrite': False},
	}


def _clustering_root(tmp_path: Path) -> Path:
	root = tmp_path / 'clustering'
	embeddings = tmp_path / 'embeddings'
	embeddings.mkdir()
	shape = (1, 1, 3)
	np.save(embeddings / 'survey.embeddings.npy', np.zeros((*shape, 2), np.float32))
	np.save(embeddings / 'survey.valid_tokens.npy', np.ones(shape, np.bool_))
	(embeddings / 'survey.embedding_metadata.json').write_text(
		json.dumps({'survey_id': 'survey'}),
		encoding='utf-8',
	)
	for k in (6, 8, 10):
		label_dir = root / 'labels' / f'k{k}'
		label_dir.mkdir(parents=True)
		labels = np.array([[[0, -1, min(k - 1, 2)]]], dtype=np.int32)
		np.save(label_dir / 'survey.cluster_labels_token.npy', labels)
	return root
