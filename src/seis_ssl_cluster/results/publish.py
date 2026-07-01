"""Publish lightweight artifacts from local artifact storage into results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence

DEFAULT_ALLOWED_SUFFIXES = frozenset(
	{'.md', '.json', '.csv', '.txt', '.png', '.pdf', '.svg'}
)
FORBIDDEN_SUFFIXES = frozenset(
	{'.pt', '.pth', '.npy', '.npz', '.joblib', '.pkl', '.sgy', '.segy'}
)
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
PUBLISH_MANIFEST_NAME = 'publish_manifest.json'


@dataclass(frozen=True)
class PublishItem:
	"""One source artifact and its relative target under a results directory."""

	source: Path
	relative_target: Path
	required: bool = True
	text_replacements: tuple[tuple[str, str], ...] = ()
	content_text: str | None = None


@dataclass(frozen=True)
class PublishedItem:
	"""Manifest record for one published file."""

	source: Path
	target: Path
	size_bytes: int
	sha256: str


@dataclass(frozen=True)
class SkippedOptionalItem:
	"""Manifest record for an optional item that was not present."""

	source: Path
	target: Path
	reason: str


@dataclass(frozen=True)
class PublishManifest:
	"""Record of files published into a results directory."""

	created_at_utc: str
	source_artifact_root: Path | None
	output_dir: Path
	items: list[PublishedItem]
	skipped_optional_items: list[SkippedOptionalItem]
	warnings: list[str]
	manifest_path: Path


@dataclass(frozen=True)
class _PublishCopy:
	source: Path
	target: Path
	size_bytes: int
	content_bytes: bytes | None


@dataclass(frozen=True)
class _PublishPlan:
	copy_plan: list[_PublishCopy]
	skipped_optional_items: list[SkippedOptionalItem]
	warnings: list[str]


@dataclass(frozen=True)
class _PublishConstraints:
	output_root: Path
	manifest_path: Path
	allowed_suffixes: frozenset[str]
	max_file_size_bytes: int
	overwrite: bool


def publish_selected_results(
	*,
	items: Sequence[PublishItem],
	output_dir: Path,
	allowed_suffixes: set[str] | frozenset[str] = DEFAULT_ALLOWED_SUFFIXES,
	max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
	overwrite: bool = True,
) -> PublishManifest:
	"""Copy selected lightweight files into ``output_dir`` and write a manifest."""
	_validate_max_file_size_bytes(max_file_size_bytes)
	allowed = _normalize_suffixes(allowed_suffixes, name='allowed_suffixes')
	forbidden_overlap = allowed & FORBIDDEN_SUFFIXES
	if forbidden_overlap:
		msg = (
			'allowed_suffixes must not include forbidden suffixes: '
			f'{sorted(forbidden_overlap)}'
		)
		raise ValueError(msg)

	output_root = Path(output_dir).resolve(strict=False)
	output_root.mkdir(parents=True, exist_ok=True)
	manifest_path = output_root / PUBLISH_MANIFEST_NAME

	if manifest_path.exists() and not overwrite:
		msg = f'publish manifest already exists: {manifest_path}'
		raise FileExistsError(msg)

	plan = _build_copy_plan(
		items=items,
		constraints=_PublishConstraints(
			output_root=output_root,
			manifest_path=manifest_path,
			allowed_suffixes=allowed,
			max_file_size_bytes=max_file_size_bytes,
			overwrite=overwrite,
		),
	)
	manifest = PublishManifest(
		created_at_utc=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
		source_artifact_root=_resolve_source_artifact_root(
			[copy.source for copy in plan.copy_plan]
		),
		output_dir=Path(output_dir),
		items=_copy_publish_items(plan.copy_plan),
		skipped_optional_items=plan.skipped_optional_items,
		warnings=plan.warnings,
		manifest_path=manifest_path,
	)
	manifest_path.write_text(
		json.dumps(publish_manifest_to_dict(manifest), indent=2, sort_keys=True)
		+ '\n',
		encoding='utf-8',
	)
	return manifest


def publish_manifest_to_dict(manifest: PublishManifest) -> dict[str, object]:
	"""Convert a publish manifest to a JSON-compatible dictionary."""
	manifest_dir = manifest.manifest_path.parent
	return {
		'created_at_utc': manifest.created_at_utc,
		'source_artifact_root': (
			None
			if manifest.source_artifact_root is None
			else str(manifest.source_artifact_root)
		),
		'output_dir': str(manifest.output_dir),
		'items': [
			{
				'source': str(item.source),
				'target': _manifest_relative_target(
					target=item.target,
					manifest_dir=manifest_dir,
				),
				'size_bytes': item.size_bytes,
				'sha256': item.sha256,
			}
			for item in manifest.items
		],
		'skipped_optional_items': [
			{
				'source': str(item.source),
				'target': _manifest_relative_target(
					target=item.target,
					manifest_dir=manifest_dir,
				),
				'reason': item.reason,
			}
			for item in manifest.skipped_optional_items
		],
		'warnings': list(manifest.warnings),
	}


def _manifest_relative_target(*, target: Path, manifest_dir: Path) -> str:
	relative = target.resolve(strict=False).relative_to(
		manifest_dir.resolve(strict=False)
	)
	return relative.as_posix()


def _validate_max_file_size_bytes(max_file_size_bytes: int) -> None:
	if max_file_size_bytes <= 0:
		msg = (
			'max_file_size_bytes must be positive; '
			f'got {max_file_size_bytes!r}'
		)
		raise ValueError(msg)


def _build_copy_plan(
	*,
	items: Sequence[PublishItem],
	constraints: _PublishConstraints,
) -> _PublishPlan:
	copy_plan: list[_PublishCopy] = []
	skipped: list[SkippedOptionalItem] = []
	warnings: list[str] = []
	planned_targets: set[Path] = set()

	for item in items:
		source = Path(item.source)
		target = _resolve_target(
			output_root=constraints.output_root,
			relative_target=Path(item.relative_target),
		)
		copy_item = _plan_copy_for_item(
			item=item,
			source=source,
			target=target,
			constraints=constraints,
		)
		if copy_item is None:
			warning = f'optional publish source does not exist: {source}'
			warnings.append(warning)
			skipped.append(
				SkippedOptionalItem(
					source=source,
					target=target,
					reason='source_missing',
				)
			)
			continue

		_validate_publish_target(
			target=target,
			constraints=constraints,
			planned_targets=planned_targets,
		)
		copy_plan.append(copy_item)
		planned_targets.add(target)

	return _PublishPlan(
		copy_plan=copy_plan,
		skipped_optional_items=skipped,
		warnings=warnings,
	)


def _plan_copy_for_item(
	*,
	item: PublishItem,
	source: Path,
	target: Path,
	constraints: _PublishConstraints,
) -> _PublishCopy | None:
	_validate_suffixes(
		source=source,
		target=target,
		allowed_suffixes=constraints.allowed_suffixes,
	)
	if not _source_exists_for_publish(source=source, required=item.required):
		return None
	content_bytes = _publish_content_bytes(source=source, item=item)
	return _PublishCopy(
		source=source,
		target=target,
		size_bytes=_validated_source_size(
			source=source,
			size_bytes=(
				len(content_bytes)
				if content_bytes is not None
				else source.stat().st_size
			),
			max_file_size_bytes=constraints.max_file_size_bytes,
		),
		content_bytes=content_bytes,
	)


def _source_exists_for_publish(*, source: Path, required: bool) -> bool:
	if source.is_symlink():
		msg = f'publish source must not be a symlink: {source}'
		raise ValueError(msg)
	if not source.exists():
		if required:
			msg = f'required publish source does not exist: {source}'
			raise FileNotFoundError(msg)
		return False
	if not source.is_file():
		msg = f'publish source must be a file: {source}'
		raise ValueError(msg)
	return True


def _validated_source_size(
	*,
	source: Path,
	size_bytes: int,
	max_file_size_bytes: int,
) -> int:
	if size_bytes > max_file_size_bytes:
		msg = (
			'publish source exceeds max_file_size_bytes: '
			f'{source} has {size_bytes} bytes; limit is '
			f'{max_file_size_bytes} bytes'
		)
		raise ValueError(msg)
	return size_bytes


def _publish_content_bytes(*, source: Path, item: PublishItem) -> bytes | None:
	if item.content_text is None and not item.text_replacements:
		return None
	text = (
		source.read_text(encoding='utf-8')
		if item.content_text is None
		else item.content_text
	)
	for old, new in item.text_replacements:
		if not old:
			msg = f'text replacement source must be non-empty for {source}'
			raise ValueError(msg)
		text = text.replace(old, new)
	return text.encode('utf-8')


def _validate_publish_target(
	*,
	target: Path,
	constraints: _PublishConstraints,
	planned_targets: set[Path],
) -> None:
	if target == constraints.manifest_path:
		msg = f'relative_target is reserved for the publish manifest: {target}'
		raise ValueError(msg)
	if target in planned_targets:
		msg = f'duplicate publish target: {target}'
		raise ValueError(msg)
	if target.is_symlink():
		msg = f'publish target must not be a symlink: {target}'
		raise ValueError(msg)
	if target.is_dir():
		msg = f'publish target must not be a directory: {target}'
		raise IsADirectoryError(msg)
	if target.exists() and not constraints.overwrite:
		msg = f'publish target already exists: {target}'
		raise FileExistsError(msg)


def _copy_publish_items(copy_plan: Sequence[_PublishCopy]) -> list[PublishedItem]:
	published: list[PublishedItem] = []
	for plan_item in copy_plan:
		source = plan_item.source
		target = plan_item.target
		target.parent.mkdir(parents=True, exist_ok=True)
		if plan_item.content_bytes is None:
			shutil.copy2(source, target)
			sha256 = _sha256(source)
		else:
			target.write_bytes(plan_item.content_bytes)
			sha256 = _sha256_bytes(plan_item.content_bytes)
		published.append(
			PublishedItem(
				source=source,
				target=target,
				size_bytes=plan_item.size_bytes,
				sha256=sha256,
			)
		)
	return published


def _normalize_suffixes(
	suffixes: set[str] | frozenset[str],
	*,
	name: str,
) -> frozenset[str]:
	normalized = set()
	for suffix in suffixes:
		if not suffix.startswith('.'):
			msg = f'{name} entries must start with ".": {suffix!r}'
			raise ValueError(msg)
		normalized.add(suffix.lower())
	return frozenset(normalized)


def _validate_suffixes(
	*,
	source: Path,
	target: Path,
	allowed_suffixes: frozenset[str],
) -> None:
	source_suffix = source.suffix.lower()
	target_suffix = target.suffix.lower()
	for label, suffix, path in (
		('source', source_suffix, source),
		('target', target_suffix, target),
	):
		if suffix in FORBIDDEN_SUFFIXES:
			msg = f'publish {label} has forbidden suffix {suffix!r}: {path}'
			raise ValueError(msg)
		if suffix not in allowed_suffixes:
			msg = f'publish {label} suffix is not allowed: {path}'
			raise ValueError(msg)


def _resolve_target(*, output_root: Path, relative_target: Path) -> Path:
	if relative_target.is_absolute():
		msg = f'relative_target must be relative: {relative_target}'
		raise ValueError(msg)
	target = (output_root / relative_target).resolve(strict=False)
	try:
		target.relative_to(output_root)
	except ValueError as exc:
		msg = (
			'relative_target must stay within output_dir: '
			f'{relative_target}'
		)
		raise ValueError(msg) from exc
	return target


def _resolve_source_artifact_root(existing_sources: Sequence[Path]) -> Path | None:
	if not existing_sources:
		return None
	inferred_roots = {
		root
		for source in existing_sources
		if (root := _infer_seis_ssl_artifact_root(source)) is not None
	}
	if len(inferred_roots) == 1:
		return next(iter(inferred_roots))
	common_path = os.path.commonpath(
		[str(source.resolve(strict=False).parent) for source in existing_sources]
	)
	return Path(common_path)


def _infer_seis_ssl_artifact_root(source: Path) -> Path | None:
	parts = source.resolve(strict=False).parts
	for index in range(len(parts) - 1):
		if parts[index] == 'artifacts' and parts[index + 1] == 'seis_ssl_cluster':
			return Path(*parts[: index + 2])
	return None


def _sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
			hasher.update(chunk)
	return hasher.hexdigest()


def _sha256_bytes(content: bytes) -> str:
	return hashlib.sha256(content).hexdigest()
