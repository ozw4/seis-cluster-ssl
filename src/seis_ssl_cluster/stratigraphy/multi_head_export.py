"""Strict, resumable export of the K=6/8/10 multi-head target bundle."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	file_sha256,
)
from seis_ssl_cluster.stratigraphy.export import (
	export_hmm_cluster_labels_as_pseudo_targets,
	prepare_hmm_cluster_label_pseudo_target_exports,
)
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetInput,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


CANONICAL_KS = (6, 8, 10)
_LABEL_SUFFIX = '.cluster_labels_token.npy'
_Action = Literal['NEW', 'REUSE', 'QUARANTINE', 'ERROR']


@dataclass(frozen=True)
class MultiHeadPseudoTargetExportConfig:
	"""Resolved immutable policy for a multi-head pseudo-target export."""

	clustering_output_dir: Path
	clustering_config: Path
	source_embedding_dir: Path
	pseudo_target_root: Path
	ks: tuple[int, ...]
	confidence: float
	schema_version: int
	write_boundary_weight: bool
	overwrite: bool
	historical_k6_root: Path | None
	handoff_manifest: Path


@dataclass(frozen=True)
class MultiHeadPseudoTargetExportPlan:
	"""Validated action for one K head."""

	k: int
	action: _Action
	source_labels: tuple[Path, ...]
	reason: str | None = None


def resolve_multi_head_pseudo_target_export_config(  # noqa: C901
	config: Mapping[str, object],
) -> MultiHeadPseudoTargetExportConfig:
	"""Validate the deliberately narrow schema-v1 multi-head export config."""
	allowed = {
		'clustering_output_dir',
		'clustering_config',
		'source_embedding_dir',
		'pseudo_target_root',
		'ks',
		'confidence',
		'schema_version',
		'write_boundary_weight',
		'outputs',
		'historical_k6_root',
		'handoff_manifest',
	}
	unknown = set(config) - allowed
	if unknown:
		raise ValueError(
			f'unknown multi-head pseudo-target config keys: {sorted(unknown)}',
		)
	clustering_output_dir = _path(config, 'clustering_output_dir')
	clustering_config = _path(config, 'clustering_config')
	if not clustering_config.is_file():
		raise FileNotFoundError(
			f'clustering_config is missing: {clustering_config}'
		)
	source_embedding_dir = _path(config, 'source_embedding_dir')
	pseudo_target_root = _path(config, 'pseudo_target_root')
	ks_value = config.get('ks')
	if not isinstance(ks_value, list) or any(
		isinstance(k, bool) or not isinstance(k, int) for k in ks_value
	):
		raise TypeError('ks must be a list of integers')
	ks = tuple(ks_value)
	if ks != CANONICAL_KS:
		raise ValueError(f'ks must be exactly {list(CANONICAL_KS)} in canonical order')
	confidence = config.get('confidence')
	if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
		raise TypeError('confidence must be a number')
	confidence = float(confidence)
	if confidence != 1.0:
		raise ValueError('confidence must be the historical bootstrap constant 1.0')
	if config.get('schema_version') != 1:
		raise ValueError('schema_version must be explicitly set to 1')
	if config.get('write_boundary_weight') is not False:
		raise ValueError('write_boundary_weight must be explicitly set to false')
	outputs = config.get('outputs')
	if not isinstance(outputs, Mapping) or set(outputs) != {'overwrite'}:
		raise ValueError('outputs must contain exactly overwrite')
	if outputs['overwrite'] is not False:
		raise ValueError('outputs.overwrite must be false for immutable exports')
	historical_value = config.get('historical_k6_root')
	historical_k6_root = (
		Path(historical_value)
		if isinstance(historical_value, str) and historical_value
		else None
	)
	if historical_value is not None and historical_k6_root is None:
		raise TypeError('historical_k6_root must be a non-empty path when provided')
	if historical_k6_root and _same_path(historical_k6_root, pseudo_target_root):
		raise ValueError('pseudo_target_root must differ from historical_k6_root')
	handoff_value = config.get('handoff_manifest')
	handoff_manifest = (
		Path(handoff_value)
		if isinstance(handoff_value, str) and handoff_value
		else pseudo_target_root / 'multi_head_pseudo_target_export_handoff.json'
	)
	return MultiHeadPseudoTargetExportConfig(
		clustering_output_dir=clustering_output_dir,
		clustering_config=clustering_config,
		source_embedding_dir=source_embedding_dir,
		pseudo_target_root=pseudo_target_root,
		ks=ks,
		confidence=confidence,
		schema_version=1,
		write_boundary_weight=False,
		overwrite=False,
		historical_k6_root=historical_k6_root,
		handoff_manifest=handoff_manifest,
	)


def plan_multi_head_pseudo_target_exports(
	config: MultiHeadPseudoTargetExportConfig,
	*,
	only_missing: bool,
) -> list[MultiHeadPseudoTargetExportPlan]:
	"""Validate all inputs and classify each head without writing arrays."""
	# The handoff is a provenance record, not merely a list of output hashes.
	# Validate its required clustering inputs before either reusing a handoff or
	# creating new pseudo-target arrays.
	_clustering_config_sha256(config.clustering_config)
	_clustering_provenance(config.clustering_output_dir, ks=config.ks)
	if config.historical_k6_root and _same_path(
		config.historical_k6_root, config.pseudo_target_root
	):
		raise ValueError('refusing to write K=6 replay into historical K=6 root')
	source_labels_by_k = {
		k: _source_label_paths(config.clustering_output_dir, k=k)
		for k in config.ks
	}
	# The exported target-valid mask is derived solely from ``labels >= 0``.
	# Validate that contract across all heads before classifying or publishing
	# any individual output directory.
	_validate_source_label_embedding_alignment(config, source_labels_by_k)
	plans: list[MultiHeadPseudoTargetExportPlan] = []
	for k in config.ks:
		source_labels = source_labels_by_k[k]
		output_dir = config.pseudo_target_root / f'k{k}'
		if not output_dir.exists():
			plans.append(MultiHeadPseudoTargetExportPlan(k, 'NEW', source_labels))
			continue
		if k == 6:
			_reject_historical_k6_identity(config, output_dir)
		try:
			_validate_complete_export(config, k=k, source_labels=source_labels)
		except (OSError, TypeError, ValueError) as exc:
			action: _Action = 'QUARANTINE' if only_missing else 'ERROR'
			plans.append(
				MultiHeadPseudoTargetExportPlan(
					k, action, source_labels, str(exc)
				),
			)
		else:
			action = 'REUSE' if only_missing else 'ERROR'
			reason = (
				None if only_missing else 'complete output exists; use --only-missing'
			)
			plans.append(
				MultiHeadPseudoTargetExportPlan(k, action, source_labels, reason),
			)
	return plans


def export_multi_head_pseudo_targets(
	config: MultiHeadPseudoTargetExportConfig,
	*,
	dry_run: bool = False,
	only_missing: bool = False,
) -> list[MultiHeadPseudoTargetExportPlan]:
	"""Export the complete bundle, quarantining only invalid resumptions."""
	plans = plan_multi_head_pseudo_target_exports(config, only_missing=only_missing)
	if any(plan.action == 'ERROR' for plan in plans) and not dry_run:
		raise FileExistsError(_plan_error(plans))
	if dry_run:
		return plans
	for plan in plans:
		if plan.action == 'REUSE':
			continue
		if plan.action == 'QUARANTINE':
			_quarantine(config.pseudo_target_root / f'k{plan.k}')
		export_hmm_cluster_labels_as_pseudo_targets(
			clustering_output_dir=config.clustering_output_dir,
			pseudo_target_root=config.pseudo_target_root,
			k=plan.k,
			confidence=config.confidence,
			overwrite=False,
			schema_version=config.schema_version,
			write_boundary_weight=config.write_boundary_weight,
		)
		_validate_complete_export(
			config,
			k=plan.k,
			source_labels=plan.source_labels,
			verify_recorded_hashes=False,
		)
	_validate_common_target_masks(config)
	_write_handoff(config)
	return plans


def _source_label_paths(root: Path, *, k: int) -> tuple[Path, ...]:
	label_dir = root / 'labels' / f'k{k}'
	paths = tuple(sorted(label_dir.glob(f'*{_LABEL_SUFFIX}')))
	if not paths:
		raise ValueError(f'no clustering labels found for k={k}: {label_dir}')
	# Reuse the existing exporter preflight for source array semantics and policy.
	prepare_hmm_cluster_label_pseudo_target_exports(
		clustering_output_dir=root,
		pseudo_target_root=root / '.multi-head-export-preflight',
		k=k,
		confidence=1.0,
		schema_version=1,
		write_boundary_weight=False,
	)
	return paths


def _validate_complete_export(  # noqa: C901
	config: MultiHeadPseudoTargetExportConfig,
	*,
	k: int,
	source_labels: Sequence[Path],
	verify_recorded_hashes: bool = True,
) -> None:
	output_dir = config.pseudo_target_root / f'k{k}'
	if any(output_dir.glob('*.hmm_boundary_weight_token.npy')):
		raise ValueError(f'k={k} contains forbidden boundary-weight artifacts')
	inputs = discover_pseudo_target_inputs(config.pseudo_target_root, k=k)
	expected = {
		path.name.removesuffix(_LABEL_SUFFIX): file_sha256(path)
		for path in source_labels
	}
	source_label_paths = {
		path.name.removesuffix(_LABEL_SUFFIX): path for path in source_labels
	}
	if {item.survey_id for item in inputs} != set(expected):
		raise ValueError(f'k={k} source/output survey set mismatch')
	recorded_hashes = (
		_recorded_head_hashes(config, k=k) if verify_recorded_hashes else None
	)
	if verify_recorded_hashes and recorded_hashes is None:
		raise ValueError(f'k={k} has no complete export handoff hashes')
	for item in inputs:
		metadata = load_pseudo_target_metadata(item)
		if metadata.get('schema_version') != 1 or metadata.get('k') != k:
			raise ValueError(f'k={k} has incompatible pseudo-target metadata')
		source = metadata.get('source')
		if not isinstance(source, Mapping):
			raise TypeError(f'k={k} {item.survey_id} is missing source identity')
		if source.get('export_confidence') != config.confidence:
			raise ValueError(f'k={k} {item.survey_id} confidence policy mismatch')
		label_path = source.get('source_label_path')
		expected_label_path = (
			config.clustering_output_dir
			/ 'labels'
			/ f'k{k}'
			/ f'{item.survey_id}{_LABEL_SUFFIX}'
		)
		if not isinstance(label_path, str) or not _same_path(
			Path(label_path), expected_label_path
		):
			raise ValueError(f'k={k} {item.survey_id} source label path mismatch')
		if source.get('source_label_sha256') != expected[item.survey_id]:
			raise ValueError(f'k={k} {item.survey_id} source label hash mismatch')
		arrays = load_pseudo_target_arrays(item)
		source_labels_array = np.load(
			source_label_paths[item.survey_id],
			mmap_mode='r',
			allow_pickle=False,
		)
		if not np.array_equal(arrays.labels, source_labels_array):
			raise ValueError(f'k={k} {item.survey_id} target labels mismatch')
		if (
			recorded_hashes is not None
			and recorded_hashes.get(item.survey_id) != _item_hashes(item)
		):
			raise ValueError(f'k={k} {item.survey_id} output hash mismatch')


def _write_handoff(config: MultiHeadPseudoTargetExportConfig) -> None:
	config_sha256 = _clustering_config_sha256(config.clustering_config)
	metadata_hashes, prepared_feature_identity = _clustering_provenance(
		config.clustering_output_dir, ks=config.ks
	)
	payload: dict[str, object] = {
		'artifact_type': 'strat_hmm_multi_head_pseudo_target_export_handoff',
		'schema_version': 2,
		'completion_status': 'COMPLETE',
		# This is the parent of the immutable, per-K roots recorded below.  Keep
		# both levels explicit: consumers must be able to locate a complete head
		# without inferring a directory layout from a hash-only record.
		'pseudo_target_root': str(config.pseudo_target_root),
		'clustering': {
			'path': str(config.clustering_output_dir),
			'config_path': str(config.clustering_config),
			'config_sha256': config_sha256,
			'metadata_sha256': metadata_hashes,
			'labels': {
				str(k): {
					path.name: file_sha256(path)
					for path in _source_label_paths(config.clustering_output_dir, k=k)
				}
				for k in config.ks
			},
		},
		'prepared_feature_identity': prepared_feature_identity,
		'source_embedding': {
			'path': str(config.source_embedding_dir),
			'valid_tokens_sha256': _embedding_valid_token_hashes(
				config.source_embedding_dir
			),
		},
		'policy': {
			'ks': list(config.ks),
			'confidence': config.confidence,
			'schema_version': config.schema_version,
			'write_boundary_weight': config.write_boundary_weight,
		},
		'heads': {
			str(k): {
				'pseudo_target_root': str(config.pseudo_target_root / f'k{k}'),
				'hashes': _head_hashes(config.pseudo_target_root, k=k),
			}
			for k in config.ks
		},
		'common_target_valid_sha256': {
			item.survey_id: file_sha256(item.valid_tokens_path)
			for item in discover_pseudo_target_inputs(config.pseudo_target_root, k=6)
		},
	}
	config.handoff_manifest.parent.mkdir(parents=True, exist_ok=True)
	temporary = config.handoff_manifest.with_suffix(
		config.handoff_manifest.suffix + '.tmp'
	)
	temporary.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)
	temporary.replace(config.handoff_manifest)


def _head_hashes(root: Path, *, k: int) -> dict[str, dict[str, str]]:
	return {
		item.survey_id: _item_hashes(item)
		for item in discover_pseudo_target_inputs(root, k=k)
	}


def _item_hashes(item: StratPseudoTargetInput) -> dict[str, str]:
	"""Return file hashes for a pseudo-target input without loading arrays."""
	return {
		'labels': file_sha256(item.labels_path),
		'confidence': file_sha256(item.confidence_path),
		'valid_tokens': file_sha256(item.valid_tokens_path),
		'metadata': file_sha256(item.metadata_path),
	}


def _recorded_head_hashes(  # noqa: C901, PLR0912
	config: MultiHeadPseudoTargetExportConfig,
	*,
	k: int,
) -> Mapping[str, object] | None:
	if not config.handoff_manifest.is_file():
		return None
	try:
		payload = json.loads(config.handoff_manifest.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError('multi-head export handoff must be valid JSON') from exc
	if (
		not isinstance(payload, Mapping)
		or payload.get('schema_version') != 2
		or payload.get('completion_status') != 'COMPLETE'
	):
		raise ValueError('multi-head export handoff schema/status mismatch')
	if (
		not isinstance(payload.get('pseudo_target_root'), str)
		or not _same_path(
			Path(payload['pseudo_target_root']), config.pseudo_target_root
		)
	):
		raise ValueError('multi-head export handoff pseudo-target root mismatch')
	metadata_hashes, prepared_feature_identity = _clustering_provenance(
		config.clustering_output_dir, ks=config.ks
	)
	config_sha256 = _clustering_config_sha256(config.clustering_config)
	clustering = payload.get('clustering')
	if not isinstance(clustering, Mapping):
		raise TypeError('multi-head export handoff is missing clustering provenance')
	if (
		not isinstance(clustering.get('path'), str)
		or not _same_path(Path(clustering['path']), config.clustering_output_dir)
	):
		raise ValueError('multi-head export handoff clustering path mismatch')
	if (
		not isinstance(clustering.get('config_path'), str)
		or not _same_path(Path(clustering['config_path']), config.clustering_config)
		or clustering.get('config_sha256') != config_sha256
	):
		raise ValueError('multi-head export handoff clustering config mismatch')
	if clustering.get('metadata_sha256') != metadata_hashes:
		raise ValueError('multi-head export handoff clustering metadata mismatch')
	if payload.get('prepared_feature_identity') != prepared_feature_identity:
		raise ValueError(
			'multi-head export handoff prepared-feature identity mismatch'
		)
	_validate_recorded_source_embedding(config, payload)
	heads = payload.get('heads')
	if not isinstance(heads, Mapping):
		raise TypeError('multi-head export handoff is missing head hashes')
	head = heads.get(str(k))
	if not isinstance(head, Mapping):
		raise TypeError(f'multi-head export handoff is missing k={k} hashes')
	if (
		not isinstance(head.get('pseudo_target_root'), str)
		or not _same_path(
			Path(head['pseudo_target_root']), config.pseudo_target_root / f'k{k}'
		)
	):
		raise ValueError(f'multi-head export handoff k={k} root mismatch')
	hashes = head.get('hashes')
	if not isinstance(hashes, Mapping):
		raise TypeError(f'multi-head export handoff is missing k={k} hashes')
	return hashes


def _validate_recorded_source_embedding(
	config: MultiHeadPseudoTargetExportConfig,
	payload: Mapping[str, object],
) -> None:
	"""Bind reusable exports to the source embedding identity in the handoff."""
	source_embedding = payload.get('source_embedding')
	if not isinstance(source_embedding, Mapping):
		raise TypeError('multi-head export handoff is missing source embedding')
	path = source_embedding.get('path')
	if not isinstance(path, str) or not _same_path(
		Path(path), config.source_embedding_dir
	):
		raise ValueError('multi-head export handoff source embedding path mismatch')
	recorded_hashes = source_embedding.get('valid_tokens_sha256')
	if not isinstance(recorded_hashes, Mapping):
		raise TypeError(
			'multi-head export handoff is missing source embedding valid-mask hashes'
		)
	if recorded_hashes != _embedding_valid_token_hashes(config.source_embedding_dir):
		raise ValueError(
			'multi-head export handoff source embedding valid-mask hashes mismatch'
		)


def _embedding_valid_token_hashes(source_embedding_dir: Path) -> dict[str, str]:
	return {
		item.survey_id: file_sha256(item.valid_tokens_path)
		for item in discover_embedding_inputs(source_embedding_dir)
	}


def _validate_common_target_masks(config: MultiHeadPseudoTargetExportConfig) -> None:
	reference = {
		item.survey_id: file_sha256(item.valid_tokens_path)
		for item in discover_pseudo_target_inputs(config.pseudo_target_root, k=6)
	}
	for k in config.ks[1:]:
		current = {
			item.survey_id: file_sha256(item.valid_tokens_path)
			for item in discover_pseudo_target_inputs(config.pseudo_target_root, k=k)
		}
		if current != reference:
			raise ValueError(f'k={k} valid-token mask differs from K=6')


def _validate_source_label_embedding_alignment(
	config: MultiHeadPseudoTargetExportConfig,
	source_labels_by_k: Mapping[int, Sequence[Path]],
) -> None:
	"""Require common target masks to be source-valid subsets before export."""
	embeddings = {
		item.survey_id: item
		for item in discover_embedding_inputs(config.source_embedding_dir)
	}
	reference_masks: dict[str, np.ndarray] | None = None
	for k in config.ks:
		labels_by_survey = {
			path.name.removesuffix(_LABEL_SUFFIX): path
			for path in source_labels_by_k[k]
		}
		if set(labels_by_survey) != set(embeddings):
			raise ValueError(
				f'k={k} clustering-label survey set does not match source embeddings',
			)
		current_masks: dict[str, np.ndarray] = {}
		for survey_id, label_path in labels_by_survey.items():
			labels = np.load(label_path, mmap_mode='r', allow_pickle=False)
			embedding_valid = np.load(
				embeddings[survey_id].valid_tokens_path,
				mmap_mode='r',
				allow_pickle=False,
			)
			if labels.shape != embedding_valid.shape:
				raise ValueError(
					f'k={k} {survey_id} token grid does not match source embedding',
				)
			target_valid = labels >= 0
			if np.any(target_valid & ~embedding_valid):
				raise ValueError(
					f'k={k} {survey_id} valid-token mask is not a subset '
					'of source embedding',
				)
			current_masks[survey_id] = target_valid
		if reference_masks is None:
			reference_masks = current_masks
			continue
		for survey_id, target_valid in current_masks.items():
			if not np.array_equal(target_valid, reference_masks[survey_id]):
				raise ValueError(f'k={k} valid-token mask differs from K=6')


def _reject_historical_k6_identity(
	config: MultiHeadPseudoTargetExportConfig,
	replay_dir: Path,
) -> None:
	"""Reject a replay target that aliases immutable historical K=6 files."""
	if config.historical_k6_root is None:
		return
	historical_dir = config.historical_k6_root / 'k6'
	if not historical_dir.is_dir():
		return
	for replay_path in replay_dir.iterdir():
		historical_path = historical_dir / replay_path.name
		if not historical_path.exists() or not replay_path.is_file():
			continue
		replay_stat = replay_path.stat()
		historical_stat = historical_path.stat()
		if (
			replay_stat.st_dev == historical_stat.st_dev
			and replay_stat.st_ino == historical_stat.st_ino
		):
			raise ValueError(
				'K=6 replay artifact must not hardlink immutable historical target: '
				f'{replay_path}',
			)


def _clustering_provenance(
	root: Path,
	*,
	ks: Sequence[int],
) -> tuple[dict[str, str], Mapping[str, object]]:
	"""Return complete, cross-K clustering provenance for a publishable handoff."""
	metadata_hashes = _clustering_metadata_hashes(root, ks=ks)
	prepared_feature_identity: Mapping[str, object] | None = None
	for k in ks:
		current = _prepared_feature_identity(root, k=k)
		if prepared_feature_identity is None:
			prepared_feature_identity = current
		elif current != prepared_feature_identity:
			raise ValueError(
				'clustering prepared-feature identity differs across K values'
			)
	if prepared_feature_identity is None:
		raise AssertionError('at least one K is required for clustering provenance')
	return metadata_hashes, prepared_feature_identity


def _clustering_config_sha256(path: Path) -> str:
	if not path.is_file():
		raise FileNotFoundError(f'clustering config is missing: {path}')
	return file_sha256(path)


def _prepared_feature_identity(root: Path, *, k: int) -> Mapping[str, object]:
	metadata = _clustering_metadata(root, k=k)
	hmm = metadata.get('stratigraphic_hmm')
	if not isinstance(hmm, Mapping):
		raise TypeError(f'k={k} clustering metadata is missing stratigraphic_hmm')
	prepared = hmm.get('prepared_feature_cache')
	if not isinstance(prepared, Mapping) or not prepared:
		raise ValueError(
			f'k={k} clustering metadata is missing prepared-feature identity'
		)
	return prepared


def _clustering_metadata_hashes(
	root: Path,
	*,
	ks: Sequence[int],
) -> dict[str, str]:
	result: dict[str, str] = {}
	for k in ks:
		metadata_path = root / 'models' / f'k{k}' / 'clustering_metadata.json'
		# Parse before hashing so a handoff cannot bind an opaque or wrong-K file.
		_clustering_metadata(root, k=k)
		result[str(k)] = file_sha256(metadata_path)
	return result


def _clustering_metadata(root: Path, *, k: int) -> Mapping[str, object]:
	metadata_path = root / 'models' / f'k{k}' / 'clustering_metadata.json'
	if not metadata_path.is_file():
		raise FileNotFoundError(
			f'k={k} clustering metadata is missing: {metadata_path}'
		)
	try:
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(
			f'k={k} clustering metadata must be valid JSON: {metadata_path}'
		) from exc
	if not isinstance(metadata, Mapping):
		raise TypeError(f'k={k} clustering metadata must be a JSON object')
	if metadata.get('k') != k:
		raise ValueError(f'k={k} clustering metadata K mismatch')
	if metadata.get('method') != 'stratigraphic_hmm_kmeans':
		raise ValueError(f'k={k} clustering metadata method mismatch')
	return metadata


def _quarantine(path: Path) -> Path:
	timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
	target = path.with_name(f'{path.name}.quarantine.{timestamp}')
	if target.exists():
		raise FileExistsError(f'quarantine path already exists: {target}')
	shutil.move(str(path), str(target))
	return target


def _path(config: Mapping[str, object], name: str) -> Path:
	value = config.get(name)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{name} must be a non-empty path')
	return Path(value)


def _same_path(left: Path, right: Path) -> bool:
	return left.resolve() == right.resolve()


def _plan_error(plans: Sequence[MultiHeadPseudoTargetExportPlan]) -> str:
	return '; '.join(
		f'k={plan.k}: {plan.reason or "output collision"}'
		for plan in plans
		if plan.action == 'ERROR'
	)


__all__ = [
	'CANONICAL_KS',
	'MultiHeadPseudoTargetExportConfig',
	'MultiHeadPseudoTargetExportPlan',
	'export_multi_head_pseudo_targets',
	'plan_multi_head_pseudo_target_exports',
	'resolve_multi_head_pseudo_target_export_config',
]
