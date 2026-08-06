"""Publish lightweight review evidence for unanimous XY-neighbour pretraining."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import (
	_multi_head_target_hashes,
	_xy_neighbor_unanimous_smoothing_identity,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.xy_neighbor_unanimous_pretraining_validation import (
	_target_temporal_transition_counts,
	_validate_embedding_identity,
	load_f3_xy_neighbor_unanimous_pretraining_handoff,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	load_f3_xy_neighbor_unanimous_target_audit,
	replay_f3_xy_neighbor_unanimous_target_audit,
)
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_manifest_to_dict,
	publish_selected_results,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_unanimous_targets import (
	load_multi_head_xy_neighbor_unanimous_target_manifest,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'target_manifest',
		'target_audit',
		'pretraining_handoff',
		'output_dir',
	}
)
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
SUMMARY_JSON = 'xy_neighbor_unanimous_summary.json'
SUMMARY_MARKDOWN = 'xy_neighbor_unanimous_summary.md'


@dataclass(frozen=True)
class F3XYNeighborUnanimousReviewConfig:
	"""Closed source/output locations for unanimous pretraining review."""

	artifact_root: Path
	workspace_root: Path
	target_manifest: Path
	target_audit: Path
	pretraining_handoff: Path
	output_dir: Path


@dataclass(frozen=True)
class F3XYNeighborUnanimousReviewResult:
	"""Review output paths and optional lightweight publish manifest."""

	output_dir: Path
	summary_json: Path
	summary_markdown: Path
	publish_manifest: PublishManifest | None


def f3_xy_neighbor_unanimous_review_config_from_mapping(
	config: Mapping[str, object],
) -> F3XYNeighborUnanimousReviewConfig:
	"""Resolve the deliberately small unanimous review config schema."""
	if not isinstance(config, Mapping):
		raise TypeError('unanimous review config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown unanimous review keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing unanimous review keys: {sorted(missing)!r}')

	def path(name: str) -> Path:
		value = config[name]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{name} must be a non-empty path string')
		return Path(value).resolve()

	result = F3XYNeighborUnanimousReviewConfig(
		artifact_root=path('artifact_root'),
		workspace_root=path('workspace_root'),
		target_manifest=path('target_manifest'),
		target_audit=path('target_audit'),
		pretraining_handoff=path('pretraining_handoff'),
		output_dir=path('output_dir'),
	)
	if not result.artifact_root.is_dir() or not result.workspace_root.is_dir():
		raise FileNotFoundError('artifact_root and workspace_root must be directories')
	for name, value in (
		('target_manifest', result.target_manifest),
		('target_audit', result.target_audit),
		('pretraining_handoff', result.pretraining_handoff),
	):
		if not value.is_file():
			raise FileNotFoundError(f'{name} is missing: {value}')
	return result


def load_f3_xy_neighbor_unanimous_review_config(
	path: str | Path,
) -> F3XYNeighborUnanimousReviewConfig:
	"""Load the unanimous review YAML through the repository config loader."""
	return f3_xy_neighbor_unanimous_review_config_from_mapping(load_config(path))


def publish_f3_xy_neighbor_unanimous_review(
	config: F3XYNeighborUnanimousReviewConfig,
	*,
	dry_run: bool = False,
) -> F3XYNeighborUnanimousReviewResult:
	"""Verify schema-6 lineage then publish only small JSON/Markdown evidence."""
	target = load_multi_head_xy_neighbor_unanimous_target_manifest(
		config.target_manifest,
		validate_array_semantics=False,
	)
	audit = load_f3_xy_neighbor_unanimous_target_audit(config.target_audit)
	audit = replay_f3_xy_neighbor_unanimous_target_audit(
		config.target_audit,
		artifact_root=config.artifact_root,
	)
	handoff = load_f3_xy_neighbor_unanimous_pretraining_handoff(
		config.pretraining_handoff
	)
	_validate_lineage(config, target=target, audit=audit, handoff=handoff)
	evidence = _review_evidence(config, target=target, audit=audit, handoff=handoff)
	result = F3XYNeighborUnanimousReviewResult(
		output_dir=config.output_dir,
		summary_json=config.output_dir / SUMMARY_JSON,
		summary_markdown=config.output_dir / SUMMARY_MARKDOWN,
		publish_manifest=None,
	)
	if dry_run:
		return result
	portable = _portable(evidence, config=config)
	manifest = publish_selected_results(
		items=(
			PublishItem(
				source=config.target_manifest,
				relative_target=Path(SUMMARY_JSON),
				content_text=json.dumps(portable, indent=2, sort_keys=True) + '\n',
			),
			PublishItem(
				source=config.target_manifest,
				relative_target=Path(SUMMARY_MARKDOWN),
				content_text=render_f3_xy_neighbor_unanimous_review_markdown(portable),
			),
		),
		output_dir=config.output_dir,
	)
	_write_portable_publish_manifest(manifest, config=config)
	return replace(result, publish_manifest=manifest)


def render_f3_xy_neighbor_unanimous_review_markdown(
	evidence: Mapping[str, object],
) -> str:
	"""Render compact audit and source-label diagnostics without array values."""
	lines = [
		'# F3 unanimous XY-neighbour hard-label review',
		'',
		f'- Target semantics: `{evidence["target_semantics"]}`',
		f'- Target representation: `{evidence["target_representation"]}`',
		f'- Target audit status: `{evidence["target_audit_status"]}`',
		f'- Target manifest SHA-256: `{evidence["target_manifest_sha256"]}`',
		f'- Pretraining handoff SHA-256: `{evidence["pretraining_handoff_sha256"]}`',
		'',
		'| K | Valid tokens | Changed tokens | Changed fraction | '
		'Source transitions | Output transitions |',
		'| --- | ---: | ---: | ---: | ---: | ---: |',
	]
	for row in evidence['head_diagnostics']:
		item = _mapping(row, 'head diagnostics')
		lines.append(
			(
				'| {k} | {valid} | {changed} | {fraction:.6f} | {source} | {output} |'
			).format(
				k=item['k'],
				valid=item['valid_token_count'],
				changed=item['changed_token_count'],
				fraction=float(item['changed_fraction']),
				source=item['source_temporal_transition_count'],
				output=item['output_temporal_transition_count'],
			)
		)
	lines.extend(
		[
			'',
			'No posterior tensors, lateral smoothing, Viterbi re-decoding, or '
			'target-refresh evidence is part of this review.',
		]
	)
	return '\n'.join(lines) + '\n'


def _validate_lineage(  # noqa: C901, PLR0912, PLR0915
	config: F3XYNeighborUnanimousReviewConfig,
	*,
	target: Mapping[str, object],
	audit: Mapping[str, object],
	handoff: Mapping[str, object],
) -> None:
	if audit.get('status') != 'XYUNANIM_TARGET_GO' or audit.get(
		'xy_neighbor_unanimous_target_manifest'
	) != _identity(config.target_manifest):
		raise ValueError('review target audit does not bind the unanimous target')
	if audit.get('source_hard_manifest') != target.get('source_hard_manifest'):
		raise ValueError('review target audit source hard manifest mismatch')
	targets = _mapping(handoff['targets'], 'handoff targets')
	if targets.get('target_manifest') != _identity(config.target_manifest):
		raise ValueError('review target manifest does not match pretraining handoff')
	if targets.get('target_audit') != _identity(config.target_audit):
		raise ValueError('review target audit does not match pretraining handoff')
	if targets.get('target_representation') != target.get(
		'target_representation'
	) or targets.get('target_semantics') != target.get('target_semantics'):
		raise ValueError('review target identity does not match pretraining handoff')
	if targets.get(
		'xy_neighbor_unanimous_target_head_hashes'
	) != _multi_head_target_hashes(target):
		raise ValueError('review unanimous target head hashes mismatch')
	if targets.get('xy_neighbor_unanimous_smoothing') != (
		_xy_neighbor_unanimous_smoothing_identity(target)
	):
		raise ValueError('review unanimous smoothing policy mismatch')
	if targets.get('source_hard_manifest') != target.get('source_hard_manifest'):
		raise ValueError('review source hard manifest mismatch')
	if targets.get('temporal_transition_counts') != _target_temporal_transition_counts(
		target
	):
		raise ValueError('review transition diagnostics mismatch')
	checkpoint_evidence = _mapping(handoff['checkpoint'], 'handoff checkpoint')
	checkpoint = _artifact_file(
		checkpoint_evidence['path'],
		label='handoff checkpoint',
	)
	if file_sha256(checkpoint) != checkpoint_evidence['sha256']:
		raise ValueError('review checkpoint digest mismatch')
	payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError('review checkpoint payload must be a mapping')
	validate_stratigraphy_checkpoint_payload(payload)
	identity = _mapping(payload['stratigraphy_checkpoint'], 'checkpoint identity')
	if identity.get('schema_version') != 6:
		raise ValueError('review requires a schema-6 unanimous checkpoint')
	if identity.get('target_representation') != target.get(
		'target_representation'
	) or identity.get('target_semantics') != target.get('target_semantics'):
		raise ValueError('review checkpoint target identity mismatch')
	if identity.get('xy_neighbor_unanimous_target_manifest') != _identity(
		config.target_manifest
	):
		raise ValueError('review checkpoint target manifest mismatch')
	if identity.get(
		'per_head_xy_neighbor_unanimous_targets'
	) != _multi_head_target_hashes(target):
		raise ValueError('review checkpoint target head hashes mismatch')
	source_hard = _mapping(target['source_hard_manifest'], 'source hard manifest')
	if identity.get('source_hard_manifest_sha256') != source_hard.get('sha256'):
		raise ValueError('review checkpoint source hard manifest mismatch')
	if identity.get('xy_neighbor_unanimous_smoothing') != (
		_xy_neighbor_unanimous_smoothing_identity(target)
	):
		raise ValueError('review checkpoint unanimous smoothing mismatch')
	if identity.get('model_tag') != handoff.get('model_tag'):
		raise ValueError('review checkpoint model tag mismatch')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if identity.get(key) != targets.get(key):
			raise ValueError(f'review checkpoint {key} mismatch')
	metrics = _mapping(payload['metrics'], 'checkpoint metrics')
	training_state = _mapping(payload['training_state'], 'checkpoint training state')
	if (
		payload.get('epoch') != checkpoint_evidence.get('selected_epoch')
		or payload.get('global_step') != checkpoint_evidence.get('selected_global_step')
		or metrics.get('loss') != checkpoint_evidence.get('selected_loss')
		or training_state.get('checkpoint_kind')
		!= checkpoint_evidence.get('selected_checkpoint_kind')
	):
		raise ValueError('review selected checkpoint evidence mismatch')
	embedding = _mapping(handoff['embedding'], 'handoff embedding')
	root = Path(str(embedding['root'])).resolve()
	files = output_paths(root, 'f3_facies_benchmark')
	if Path(str(embedding['metadata_path'])).resolve() != files.metadata:
		raise ValueError('review embedding metadata path mismatch')
	for path, key in (
		(files.metadata, 'metadata_sha256'),
		(files.embeddings, 'embeddings_sha256'),
		(files.valid_tokens, 'valid_tokens_sha256'),
	):
		if not path.is_file() or file_sha256(path) != embedding[key]:
			raise ValueError(f'review embedding digest mismatch: {key}')
	metadata = _mapping(
		json.loads(files.metadata.read_text(encoding='utf-8')), 'metadata'
	)
	if (
		Path(str(metadata.get('checkpoint_path', ''))).resolve() != checkpoint
		or metadata.get('checkpoint_sha256') != checkpoint_evidence['sha256']
	):
		raise ValueError('review embedding checkpoint binding mismatch')
	valid = np.load(files.valid_tokens, mmap_mode='r', allow_pickle=False)
	if valid.dtype != np.bool_ or int(valid.sum()) != embedding['valid_token_count']:
		raise ValueError('review embedding valid-token count mismatch')
	_validate_embedding_identity(
		metadata,
		identity,
		training=_mapping(payload['stratigraphy_config'], 'checkpoint config'),
	)


def _review_evidence(
	config: F3XYNeighborUnanimousReviewConfig,
	*,
	target: Mapping[str, object],
	audit: Mapping[str, object],
	handoff: Mapping[str, object],
) -> dict[str, object]:
	targets = _mapping(handoff['targets'], 'handoff targets')
	transitions = _mapping(targets['temporal_transition_counts'], 'transitions')
	rows = []
	for k in (6, 8, 10):
		head = _mapping(_mapping(target['heads'], 'heads')[str(k)], f'head K={k}')
		aggregate = _mapping(
			_mapping(head['diagnostics'], 'diagnostics')['aggregate'], 'aggregate'
		)
		pair = _mapping(transitions[str(k)], f'transitions K={k}')
		rows.append(
			{
				'k': k,
				'valid_token_count': aggregate['valid_token_count'],
				'changed_token_count': aggregate['changed_token_count'],
				'changed_fraction': aggregate['changed_fraction'],
				'source_temporal_transition_count': pair['source'],
				'output_temporal_transition_count': pair['output'],
			}
		)
	return {
		'artifact_type': 'f3_xy_neighbor_unanimous_review',
		'schema_version': 1,
		'target_representation': target['target_representation'],
		'target_semantics': target['target_semantics'],
		'target_manifest_sha256': file_sha256(config.target_manifest),
		'target_audit_sha256': file_sha256(config.target_audit),
		'target_audit_status': audit['status'],
		'pretraining_handoff_sha256': file_sha256(config.pretraining_handoff),
		'source_hard_manifest': target['source_hard_manifest'],
		'head_diagnostics': rows,
	}


def _artifact_file(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} path must be a non-empty string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _portable(  # noqa: PLR0911
	value: object, *, config: F3XYNeighborUnanimousReviewConfig
) -> object:
	if isinstance(value, Mapping):
		return {str(key): _portable(item, config=config) for key, item in value.items()}
	if isinstance(value, list):
		return [_portable(item, config=config) for item in value]
	if isinstance(value, tuple):
		return [_portable(item, config=config) for item in value]
	if isinstance(value, Path):
		value = str(value)
	if isinstance(value, str):
		artifact_root = str(config.artifact_root.resolve())
		workspace_root = str(config.workspace_root.resolve())
		if value == artifact_root:
			return _ARTIFACT_ROOT_PLACEHOLDER
		if value.startswith(f'{artifact_root}/'):
			return f'{_ARTIFACT_ROOT_PLACEHOLDER}{value[len(artifact_root) :]}'
		if value == workspace_root:
			return '.'
		if value.startswith(f'{workspace_root}/'):
			return value[len(workspace_root) + 1 :]
		return value.replace(f'{artifact_root}/', f'{_ARTIFACT_ROOT_PLACEHOLDER}/')
	return value


def _write_portable_publish_manifest(
	manifest: PublishManifest,
	*,
	config: F3XYNeighborUnanimousReviewConfig,
) -> None:
	"""Replace machine-specific paths in the publisher's lightweight manifest."""
	payload = _portable(publish_manifest_to_dict(manifest), config=config)
	if not isinstance(payload, dict):
		raise TypeError('portable publish manifest must be a mapping')
	payload['source_artifact_root'] = _ARTIFACT_ROOT_PLACEHOLDER
	manifest.manifest_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


__all__ = [
	'F3XYNeighborUnanimousReviewConfig',
	'F3XYNeighborUnanimousReviewResult',
	'f3_xy_neighbor_unanimous_review_config_from_mapping',
	'load_f3_xy_neighbor_unanimous_review_config',
	'publish_f3_xy_neighbor_unanimous_review',
	'render_f3_xy_neighbor_unanimous_review_markdown',
]
