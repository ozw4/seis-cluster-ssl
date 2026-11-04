"""Fixed unordered k-means targets, separate from ordered HMM contracts."""

# ruff: noqa: PLC0415

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	file_sha256,
	load_valid_tokens,
)
from seis_ssl_cluster.stratigraphy.targets import (
	validate_pseudo_target_arrays,
	write_pseudo_target,
)

TARGET_REPRESENTATION = 'unordered_kmeans_labels_v1'
ARTIFACT_TYPE = 'unordered_kmeans_multi_head_target_manifest'
CONSISTENCY_POLICY = 'disabled_for_unordered_kmeans_v1'


def file_reference(path: Path) -> dict[str, str]:
	"""Bind a file to its absolute path and digest."""
	if 'reports' in path.resolve().parts:
		raise ValueError('reports cannot be pipeline inputs')
	return {'path': str(path.absolute()), 'sha256': file_sha256(path)}


def _verify(reference: dict[str, str]) -> Path:
	path = Path(reference['path'])
	if file_reference(path) != reference:
		raise ValueError(f'file hash mismatch: {path}')
	return path


def support_mask(valid: np.ndarray, margins: list[int]) -> np.ndarray:
	"""Exclude the same symmetric edge margins as the matched HMM arm."""
	if len(margins) != 3 or any(
		type(m) is not int or m < 0 or 2 * m >= n
		for m, n in zip(margins, valid.shape, strict=True)
	):
		raise ValueError('invalid edge margins for token grid')
	mask = np.zeros(valid.shape, dtype=np.bool_)
	mask[tuple(slice(m, n - m) for m, n in zip(margins, valid.shape, strict=True))] = (
		True
	)
	return mask & valid


def validate_unordered_target_reference(  # noqa: PLR0913
	*,
	k: int,
	survey_id: str,
	labels_path: str | Path,
	confidence_path: str | Path,
	valid_tokens_path: str | Path,
	metadata_path: str | Path,
	hashes: dict[str, str],
	validate_array_semantics: bool = True,
) -> None:
	"""Validate categorical arrays and immutable references without imposing order."""
	paths = {
		'labels': Path(labels_path),
		'confidence': Path(confidence_path),
		'valid_tokens': Path(valid_tokens_path),
		'metadata': Path(metadata_path),
	}
	if set(hashes) != set(paths):
		raise ValueError('unordered target hash fields mismatch')
	for name, path in paths.items():
		_verify({'path': str(path), 'sha256': hashes[name]})
	metadata = json.loads(paths['metadata'].read_text())
	if any(
		metadata.get(key) != value
		for key, value in {
			'artifact_type': 'strat_hmm_pseudo_target',
			'schema_version': 1,
			'k': k,
			'survey_id': survey_id,
		}.items()
	):
		raise ValueError('unordered target metadata mismatch')
	if metadata.get('source') != {
		'source_method': 'minibatch_kmeans',
		'target_representation': TARGET_REPRESENTATION,
	}:
		raise ValueError('unordered target source metadata mismatch')
	if validate_array_semantics:
		labels, confidence, valid = (
			np.load(paths[name], mmap_mode='r', allow_pickle=False)
			for name in ('labels', 'confidence', 'valid_tokens')
		)
		validate_pseudo_target_arrays(
			labels, confidence, valid, k=k, survey_id=survey_id
		)
		if not np.all(confidence[valid] == 1):
			raise ValueError('unordered confidence must equal one on valid tokens')
		if np.any(np.bincount(labels[valid], minlength=k) == 0):
			raise ValueError('unordered target contains an empty cluster')


def load_unordered_target_manifest(  # noqa: C901
	path: str | Path,
	*,
	validate_array_semantics: bool = True,
) -> dict[str, Any]:
	"""Load the dedicated manifest and verify source support and target identity."""
	payload = json.loads(Path(path).read_text())
	if any(
		payload.get(key) != value
		for key, value in {
			'schema_version': 1,
			'artifact_type': ARTIFACT_TYPE,
			'target_representation': TARGET_REPRESENTATION,
			'status': 'complete',
		}.items()
	):
		raise ValueError('unsupported unordered target manifest')
	ks = payload['head_ks']
	if (
		not isinstance(ks, list)
		or len(ks) < 2
		or any(type(k) is not int or k < 2 for k in ks)
		or ks != sorted(set(ks))
	):
		raise ValueError('unordered head_ks must be unique ascending integers >= 2')
	if set(payload['heads']) != {str(k) for k in ks}:
		raise ValueError('unordered manifest head set mismatch')
	common = payload['common']
	source = payload['source_embedding']
	_verify(payload['clustering_config'])
	inputs = discover_embedding_inputs(source['path'])
	ids = [item.survey_id for item in inputs]
	if common['survey_ids'] != ids or set(source['surveys']) != set(ids):
		raise ValueError('unordered source survey set mismatch')
	for item in inputs:
		sid = item.survey_id
		recorded = source['surveys'][sid]
		for name, actual in (
			('metadata', item.metadata_path),
			('valid_tokens', item.valid_tokens_path),
		):
			if _verify(recorded[name]).resolve() != actual.resolve():
				raise ValueError('source embedding identity mismatch')
		support = (
			support_mask(load_valid_tokens(item), payload['edge_margin_tokens'])
			if validate_array_semantics
			else None
		)
		for k in ks:
			if set(payload['heads'][str(k)]['surveys']) != set(ids):
				raise ValueError('unordered target survey set mismatch')
			entry = payload['heads'][str(k)]['surveys'][sid]
			validate_unordered_target_reference(
				k=k,
				survey_id=sid,
				**{
					f'{name}_path': entry[name]['path']
					for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
				},
				hashes={name: ref['sha256'] for name, ref in entry.items()},
				validate_array_semantics=validate_array_semantics,
			)
			if entry['valid_tokens']['sha256'] != common['valid_tokens_sha256'][sid]:
				raise ValueError('unordered head masks differ')
			if support is not None:
				valid = np.load(
					entry['valid_tokens']['path'], mmap_mode='r', allow_pickle=False
				)
				if list(valid.shape) != common['token_grid_shapes'][
					sid
				] or not np.array_equal(valid, support):
					raise ValueError('unordered source/target support mismatch')
	return payload


def load_hard_target_manifest(
	path: str | Path,
	*,
	validate_array_semantics: bool = True,
) -> dict[str, Any]:
	"""Dispatch hard targets while preserving the strict ordered HMM loader."""
	if json.loads(Path(path).read_text()).get('artifact_type') == ARTIFACT_TYPE:
		return load_unordered_target_manifest(
			path, validate_array_semantics=validate_array_semantics
		)
	from seis_ssl_cluster.stratigraphy.multi_head import (
		load_multi_head_target_manifest,
	)

	return load_multi_head_target_manifest(
		path, validate_array_semantics=validate_array_semantics
	)


def build_kmeans_targets(config: dict[str, Any], config_path: Path) -> Path:
	"""Fit fixed k-means on matched support and publish a complete hashed manifest."""
	from seis_ssl_cluster.clustering.kmeans import (
		run_minibatch_kmeans_clustering,
	)

	if (
		config['clustering']['method'] != 'minibatch_kmeans'
		or 'stratigraphic_hmm' in config['clustering']
	):
		raise ValueError('unordered targets require plain minibatch_kmeans')
	root = Path(config['output_root'])
	if not root.is_absolute() or 'reports' in root.resolve().parts:
		raise ValueError('target output requires an absolute artifact directory')
	manifest = root / 'multi_head_target_manifest.json'
	if manifest.exists():
		payload = load_unordered_target_manifest(manifest)
		if payload['clustering_config'] != file_reference(config_path):
			raise ValueError('existing targets belong to another configuration')
		return manifest
	if root.exists():
		raise FileExistsError(f'incomplete target output; archive before retry: {root}')
	inputs = discover_embedding_inputs(config['embeddings']['input_dir'])
	margins = config['edge_margin_tokens']
	# Only masks are materialized: arrays and metadata retain the original identities.
	staged = root / 'masked_embedding_inputs'
	staged.mkdir(parents=True)
	sources, shapes = {}, {}
	for item in inputs:
		for source_path in (item.embeddings_path, item.metadata_path):
			(staged / source_path.name).symlink_to(source_path.resolve())
		valid = support_mask(load_valid_tokens(item), margins)
		np.save(staged / item.valid_tokens_path.name, valid)
		sources[item.survey_id] = {
			'metadata': file_reference(item.metadata_path),
			'valid_tokens': file_reference(item.valid_tokens_path),
		}
		shapes[item.survey_id] = list(valid.shape)
	cluster_config = copy.deepcopy(config)
	cluster_config['embeddings']['input_dir'] = str(staged)
	cluster_config['clustering']['output_dir'] = str(root / 'clustering')
	run = run_minibatch_kmeans_clustering(cluster_config)
	heads = {}
	for result in run.results:
		surveys = {}
		for labels in result.label_results:
			array = np.load(labels.labels_path, mmap_mode='r', allow_pickle=False)
			valid = array >= 0
			paths = write_pseudo_target(
				root / 'targets',
				k=result.k,
				survey_id=labels.survey_id,
				labels=array,
				confidence=valid.astype(np.float32),
				valid_tokens=valid,
				metadata={
					'source_method': 'minibatch_kmeans',
					'target_representation': TARGET_REPRESENTATION,
				},
				schema_version=1,
				write_boundary_weight=False,
			)
			surveys[labels.survey_id] = {
				name: file_reference(getattr(paths, name))
				for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
			}
		heads[str(result.k)] = {
			'pseudo_target_root': str(root / 'targets'),
			'surveys': surveys,
		}
	payload = {
		'schema_version': 1,
		'artifact_type': ARTIFACT_TYPE,
		'target_representation': TARGET_REPRESENTATION,
		'status': 'complete',
		'head_ks': list(config['clustering']['k_values']),
		'heads': heads,
		'clustering_config': file_reference(config_path),
		'edge_margin_tokens': margins,
		'source_embedding': {
			'path': str(Path(config['embeddings']['input_dir']).absolute()),
			'surveys': sources,
		},
		'common': {
			'survey_ids': [item.survey_id for item in inputs],
			'token_grid_shapes': shapes,
			'valid_tokens_sha256': {
				sid: next(iter(heads.values()))['surveys'][sid]['valid_tokens'][
					'sha256'
				]
				for sid in sources
			},
		},
	}
	temporary = root / 'manifest.tmp.json'
	temporary.write_text(json.dumps(payload, indent=2) + '\n')
	load_unordered_target_manifest(temporary)
	temporary.replace(manifest)
	return manifest
