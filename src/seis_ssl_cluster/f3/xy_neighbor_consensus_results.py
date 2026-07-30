"""Publish lightweight review evidence for XY-neighbour consensus pretraining.

The review never interprets target or embedding arrays, posterior artifacts,
facies labels, or downstream metrics.  It does verify the referenced
checkpoint and embedding-file digests plus their recorded identities before
publishing a portable summary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import (
	_multi_head_target_hashes,
	_xy_neighbor_consensus_smoothing_identity,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.xy_neighbor_consensus_pretraining_validation import (
	_validate_embedding_stratigraphy_identity,
	load_f3_xy_neighbor_consensus_pretraining_handoff,
)
from seis_ssl_cluster.paths import ensure_under_root
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_manifest_to_dict,
	publish_selected_results,
)
from seis_ssl_cluster.stratigraphy.xy_neighbor_consensus_targets import (
	load_multi_head_xy_neighbor_consensus_target_manifest,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	validate_stratigraphy_checkpoint_payload,
)

_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'target_manifest',
		'pretraining_handoff',
		'output_dir',
	}
)
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
SUMMARY_JSON = 'xy_neighbor_consensus_summary.json'
SUMMARY_MARKDOWN = 'xy_neighbor_consensus_summary.md'


@dataclass(frozen=True)
class F3XYNeighborConsensusReviewConfig:
	"""Closed set of source and output paths for the successor review."""

	artifact_root: Path
	workspace_root: Path
	target_manifest: Path
	pretraining_handoff: Path
	output_dir: Path


@dataclass(frozen=True)
class F3XYNeighborConsensusReviewResult:
	"""Portable review output paths and optional publication manifest."""

	output_dir: Path
	summary_json: Path
	summary_markdown: Path
	publish_manifest: PublishManifest | None


def f3_xy_neighbor_consensus_review_config_from_mapping(
	config: Mapping[str, object],
) -> F3XYNeighborConsensusReviewConfig:
	"""Resolve the deliberately small and explicit review configuration."""
	if not isinstance(config, Mapping):
		raise TypeError('XY-neighbour consensus review config must be a mapping')
	unknown, missing = set(config) - _CONFIG_KEYS, _CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(
			f'unknown XY-neighbour consensus review config keys: {sorted(unknown)!r}'
		)
	if missing:
		raise ValueError(
			f'missing XY-neighbour consensus review config keys: {sorted(missing)!r}'
		)

	def path(key: str) -> Path:
		value = config[key]
		if not isinstance(value, str) or not value:
			raise TypeError(f'{key} must be a non-empty path string')
		return Path(value).resolve()

	result = F3XYNeighborConsensusReviewConfig(
		artifact_root=path('artifact_root'),
		workspace_root=path('workspace_root'),
		target_manifest=path('target_manifest'),
		pretraining_handoff=path('pretraining_handoff'),
		output_dir=path('output_dir'),
	)
	if (
		not result.artifact_root.is_absolute()
		or not result.workspace_root.is_absolute()
	):
		raise ValueError('artifact_root and workspace_root must be absolute')
	for key, value in (
		('target_manifest', result.target_manifest),
		('pretraining_handoff', result.pretraining_handoff),
	):
		ensure_under_root(value, root=result.artifact_root, label=key)
		if not value.is_file():
			raise FileNotFoundError(f'{key} is missing: {value}')
	ensure_under_root(result.output_dir, root=result.workspace_root, label='output_dir')
	return result


def load_f3_xy_neighbor_consensus_review_config(
	path: str | Path,
) -> F3XYNeighborConsensusReviewConfig:
	"""Load the review config through the repository YAML loader."""
	return f3_xy_neighbor_consensus_review_config_from_mapping(load_config(path))


def publish_f3_xy_neighbor_consensus_review(
	config: F3XYNeighborConsensusReviewConfig,
	*,
	dry_run: bool = False,
) -> F3XYNeighborConsensusReviewResult:
	"""Publish source-only target and PASS-handoff review summaries."""
	target = load_multi_head_xy_neighbor_consensus_target_manifest(
		config.target_manifest, validate_array_semantics=False
	)
	handoff = load_f3_xy_neighbor_consensus_pretraining_handoff(
		config.pretraining_handoff
	)
	_validate_lineage(config, target=target, handoff=handoff)
	evidence = _review_evidence(config, target=target, handoff=handoff)
	result = F3XYNeighborConsensusReviewResult(
		output_dir=config.output_dir,
		summary_json=config.output_dir / SUMMARY_JSON,
		summary_markdown=config.output_dir / SUMMARY_MARKDOWN,
		publish_manifest=None,
	)
	if dry_run:
		return result
	manifest = publish_selected_results(
		items=(
			PublishItem(
				source=config.target_manifest,
				relative_target=Path(SUMMARY_JSON),
				content_text=json.dumps(evidence, indent=2, sort_keys=True) + '\n',
			),
			PublishItem(
				source=config.target_manifest,
				relative_target=Path(SUMMARY_MARKDOWN),
				content_text=render_f3_xy_neighbor_consensus_review_markdown(evidence),
			),
		),
		output_dir=config.output_dir,
	)
	_write_portable_publish_manifest(manifest, config=config)
	return replace(result, publish_manifest=manifest)


def render_f3_xy_neighbor_consensus_review_markdown(
	evidence: Mapping[str, object],
) -> str:
	"""Render the compact, no-array review summary."""
	source = _mapping(evidence['source_hard_manifest'], 'source hard manifest')
	lines = [
		'# F3 XY-neighbour consensus hard-label review',
		'',
		f'- Target semantics: `{evidence["target_semantics"]}`',
		f'- Training representation: `{evidence["target_representation"]}`',
		f'- Source hard manifest SHA-256: `{source["sha256"]}`',
		f'- Consensus target manifest SHA-256: `{evidence["target_manifest_sha256"]}`',
		f'- Pretraining handoff SHA-256: `{evidence["pretraining_handoff_sha256"]}`',
		'',
		'## Fixed exclusions',
		'',
		(
			'No embeddings, posterior tensors, affinities, emissions, Viterbi '
			're-decoding, beta calibration, target refresh, or downstream '
			'labels/metrics enter this target contract.'
		),
		'',
		'## Head diagnostics',
		'',
		'| K | Valid tokens | Changed tokens | Changed fraction |',
		'| --- | ---: | ---: | ---: |',
	]
	for row in evidence['head_diagnostics']:
		item = _mapping(row, 'head diagnostic')
		lines.append(
			'| {k} | {valid} | {changed} | {fraction:.6f} |'.format(
				k=item['k'],
				valid=item['valid_token_count'],
				changed=item['changed_token_count'],
				fraction=float(item['changed_fraction']),
			)
		)
	return '\n'.join(lines) + '\n'


def _validate_lineage(
	config: F3XYNeighborConsensusReviewConfig,
	*,
	target: Mapping[str, object],
	handoff: Mapping[str, object],
) -> None:
	targets = _mapping(handoff.get('targets'), 'pretraining handoff targets')
	reference = _mapping(targets.get('target_manifest'), 'pretraining target manifest')
	if Path(
		str(reference.get('path', ''))
	).resolve() != config.target_manifest or reference.get('sha256') != file_sha256(
		config.target_manifest
	):
		raise ValueError('review target manifest does not match pretraining handoff')
	if targets.get('target_semantics') != target.get('target_semantics'):
		raise ValueError('review target semantics does not match pretraining handoff')
	if targets.get('target_representation') != target.get('target_representation'):
		raise ValueError(
			'review target representation does not match pretraining handoff'
		)
	head_hashes = _multi_head_target_hashes(target)
	if targets.get('xy_neighbor_consensus_target_head_hashes') != head_hashes:
		raise ValueError(
			'review consensus target head hashes do not match pretraining handoff'
		)
	if targets.get(
		'xy_neighbor_consensus_smoothing'
	) != _xy_neighbor_consensus_smoothing_identity(target):
		raise ValueError(
			'review consensus smoothing policy does not match pretraining handoff'
		)
	if targets.get('source_hard_manifest') != target.get('source_hard_manifest'):
		raise ValueError(
			'review source hard manifest does not match pretraining handoff'
		)
	_validate_handoff_artifact_lineage(config, target=target, handoff=handoff)


def _validate_handoff_artifact_lineage(  # noqa: C901, PLR0912
	config: F3XYNeighborConsensusReviewConfig,
	*,
	target: Mapping[str, object],
	handoff: Mapping[str, object],
) -> None:
	"""Bind a PASS handoff to its live v5 checkpoint and extraction outputs."""
	targets = _mapping(handoff.get('targets'), 'pretraining handoff targets')
	checkpoint = _mapping(handoff.get('checkpoint'), 'pretraining checkpoint')
	checkpoint_path = _artifact_file(
		checkpoint.get('path'),
		root=config.artifact_root,
		label='handoff checkpoint',
	)
	if file_sha256(checkpoint_path) != checkpoint.get('sha256'):
		raise ValueError('handoff checkpoint SHA-256 does not match its file')
	payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	if not isinstance(payload, Mapping):
		raise TypeError('handoff checkpoint payload must be a mapping')
	validate_stratigraphy_checkpoint_payload(payload)
	identity = _mapping(payload.get('stratigraphy_checkpoint'), 'checkpoint identity')
	if identity.get('schema_version') != 5:
		raise ValueError('handoff checkpoint must use XY-neighbour consensus schema 5')
	if identity.get('target_representation') != target.get('target_representation'):
		raise ValueError(
			'handoff checkpoint target representation does not match target'
		)
	if identity.get('target_semantics') != target.get('target_semantics'):
		raise ValueError('handoff checkpoint target semantics does not match target')
	manifest = _mapping(
		identity.get('xy_neighbor_consensus_target_manifest'),
		'checkpoint XY-neighbour consensus target manifest',
	)
	if Path(
		str(manifest.get('path', ''))
	).resolve() != config.target_manifest or manifest.get('sha256') != file_sha256(
		config.target_manifest
	):
		raise ValueError(
			'handoff checkpoint target manifest does not match review target'
		)
	head_hashes = _multi_head_target_hashes(target)
	if identity.get('per_head_xy_neighbor_consensus_targets') != head_hashes:
		raise ValueError('handoff checkpoint head hashes do not match review target')
	source_hard = _mapping(target.get('source_hard_manifest'), 'source hard manifest')
	if identity.get('source_hard_manifest_sha256') != source_hard.get('sha256'):
		raise ValueError(
			'handoff checkpoint source hard manifest does not match target'
		)
	if identity.get(
		'xy_neighbor_consensus_smoothing'
	) != _xy_neighbor_consensus_smoothing_identity(target):
		raise ValueError('handoff checkpoint smoothing policy does not match target')
	if identity.get('model_tag') != handoff.get('model_tag'):
		raise ValueError('handoff checkpoint model tag does not match handoff')
	for key in ('initial_student_state_sha256', 'initial_head_state_sha256'):
		if identity.get(key) != targets.get(key):
			raise ValueError(f'handoff checkpoint {key} does not match handoff')
	if (
		payload.get('epoch') != checkpoint.get('selected_epoch')
		or payload.get('global_step') != checkpoint.get('selected_global_step')
		or _mapping(payload.get('metrics'), 'checkpoint metrics').get('loss')
		!= checkpoint.get('selected_loss')
		or _mapping(payload.get('training_state'), 'checkpoint training state').get(
			'checkpoint_kind'
		)
		!= checkpoint.get('selected_checkpoint_kind')
	):
		raise ValueError('handoff selected checkpoint evidence does not match its file')

	embedding = _mapping(handoff.get('embedding'), 'pretraining embedding')
	embedding_root = Path(str(embedding.get('root', ''))).resolve()
	ensure_under_root(
		embedding_root,
		root=config.artifact_root,
		label='handoff embedding root',
	)
	files = output_paths(embedding_root, 'f3_facies_benchmark')
	if Path(str(embedding.get('metadata_path', ''))).resolve() != files.metadata:
		raise ValueError('handoff embedding metadata path does not match output root')
	for path, key in (
		(files.metadata, 'metadata_sha256'),
		(files.embeddings, 'embeddings_sha256'),
		(files.valid_tokens, 'valid_tokens_sha256'),
	):
		if not path.is_file() or file_sha256(path) != embedding.get(key):
			raise ValueError(f'handoff embedding {key} does not match its file')
	metadata_payload = json.loads(files.metadata.read_text(encoding='utf-8'))
	metadata = _mapping(metadata_payload, 'embedding metadata')
	if Path(
		str(metadata.get('checkpoint_path', ''))
	).resolve() != checkpoint_path or metadata.get(
		'checkpoint_sha256'
	) != checkpoint.get('sha256'):
		raise ValueError('handoff embedding checkpoint binding does not match handoff')
	_validate_embedding_stratigraphy_identity(metadata, identity)


def _artifact_file(value: object, *, root: Path, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} path must be a non-empty string')
	path = Path(value).resolve()
	ensure_under_root(path, root=root, label=label)
	if not path.is_file():
		raise FileNotFoundError(f'{label} is missing: {path}')
	return path


def _review_evidence(
	config: F3XYNeighborConsensusReviewConfig,
	*,
	target: Mapping[str, object],
	handoff: Mapping[str, object],
) -> dict[str, object]:
	head_diagnostics: list[dict[str, object]] = []
	for k in target['head_ks']:
		head = _mapping(
			_mapping(target['heads'], 'target heads')[str(k)], 'target head'
		)
		diagnostics = _mapping(head.get('diagnostics'), 'target diagnostics')
		aggregate = _mapping(
			diagnostics.get('aggregate'), 'target aggregate diagnostics'
		)
		head_diagnostics.append(
			{
				'k': k,
				'valid_token_count': _diagnostic_int(aggregate, 'valid_token_count'),
				'changed_token_count': _diagnostic_int(
					aggregate, 'changed_token_count'
				),
				'changed_fraction': _diagnostic_fraction(aggregate, 'changed_fraction'),
			}
		)
	return {
		'artifact_type': 'f3_xy_neighbor_consensus_review',
		'schema_version': 1,
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': target['target_semantics'],
		'target_manifest_sha256': file_sha256(config.target_manifest),
		'pretraining_handoff_sha256': file_sha256(config.pretraining_handoff),
		'source_hard_manifest': _portable_reference(
			_mapping(target['source_hard_manifest'], 'source hard manifest'),
			artifact_root=config.artifact_root,
			workspace_root=config.workspace_root,
		),
		'fixed_policy': _portable_value(
			target.get('smoothing'),
			artifact_root=config.artifact_root,
			workspace_root=config.workspace_root,
		),
		'head_diagnostics': head_diagnostics,
		'pretraining': {
			'model_tag': handoff['model_tag'],
			'variant': handoff['variant'],
			'checkpoint_sha256': _mapping(handoff['checkpoint'], 'handoff checkpoint')[
				'sha256'
			],
			'embedding_metadata_sha256': _mapping(
				handoff['embedding'], 'handoff embedding'
			)['metadata_sha256'],
		},
	}


def _portable_reference(
	reference: Mapping[str, object],
	*,
	artifact_root: Path,
	workspace_root: Path,
) -> dict[str, object]:
	return {
		'path': _portable_value(
			reference.get('path'),
			artifact_root=artifact_root,
			workspace_root=workspace_root,
		),
		'sha256': reference.get('sha256'),
	}


def _portable_value(
	value: object,
	*,
	artifact_root: Path,
	workspace_root: Path,
) -> object:
	if isinstance(value, Mapping):
		return {
			str(key): _portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
			)
			for key, item in value.items()
		}
	if isinstance(value, tuple | list):
		return [
			_portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
			)
			for item in value
		]
	if isinstance(value, Path):
		return _portable_path(
			str(value),
			artifact_root=artifact_root,
			workspace_root=workspace_root,
		)
	if isinstance(value, str):
		return _portable_path(
			value,
			artifact_root=artifact_root,
			workspace_root=workspace_root,
		)
	return value


def _portable_path(
	value: str,
	*,
	artifact_root: Path,
	workspace_root: Path,
) -> str:
	artifact = str(artifact_root.resolve())
	workspace = str(workspace_root.resolve())
	if value == artifact:
		return _ARTIFACT_ROOT_PLACEHOLDER
	if value.startswith(f'{artifact}/'):
		return f'{_ARTIFACT_ROOT_PLACEHOLDER}{value[len(artifact) :]}'
	if value == workspace:
		return '.'
	if value.startswith(f'{workspace}/'):
		return value[len(workspace) + 1 :]
	return value.replace(f'{artifact}/', f'{_ARTIFACT_ROOT_PLACEHOLDER}/').replace(
		f'{workspace}/', ''
	)


def _write_portable_publish_manifest(
	manifest: PublishManifest,
	*,
	config: F3XYNeighborConsensusReviewConfig,
) -> None:
	payload = _portable_value(
		publish_manifest_to_dict(manifest),
		artifact_root=config.artifact_root,
		workspace_root=config.workspace_root,
	)
	if not isinstance(payload, dict):
		raise TypeError('portable publish manifest must be a mapping')
	payload['source_artifact_root'] = _ARTIFACT_ROOT_PLACEHOLDER
	manifest.manifest_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _diagnostic_int(diagnostics: Mapping[str, object], key: str) -> int:
	value = diagnostics.get(key)
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise TypeError(f'target diagnostics.{key} must be a nonnegative integer')
	return value


def _diagnostic_fraction(diagnostics: Mapping[str, object], key: str) -> float:
	value = diagnostics.get(key)
	if (
		isinstance(value, bool)
		or not isinstance(value, int | float)
		or not 0.0 <= value <= 1.0
	):
		raise TypeError(f'target diagnostics.{key} must be a fraction')
	return float(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


__all__ = [
	'SUMMARY_JSON',
	'SUMMARY_MARKDOWN',
	'F3XYNeighborConsensusReviewConfig',
	'F3XYNeighborConsensusReviewResult',
	'f3_xy_neighbor_consensus_review_config_from_mapping',
	'load_f3_xy_neighbor_consensus_review_config',
	'publish_f3_xy_neighbor_consensus_review',
	'render_f3_xy_neighbor_consensus_review_markdown',
]
