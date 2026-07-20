"""Validate and publish a K=6/8/10 HMM multi-head target manifest."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.stratigraphy import (
	build_multi_head_target_manifest,
	load_multi_head_target_manifest,
	validate_multi_head_target_publication_preflight,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone multi-head manifest CLI parser."""
	parser = argparse.ArgumentParser(
		description='Publish validated schema-v1 K=6/8/10 HMM target references.',
	)
	parser.add_argument('--source-embedding-dir', type=Path, required=True)
	parser.add_argument('--head-root', action='append', required=True, metavar='K=PATH')
	parser.add_argument('--manifest', type=Path, required=True)
	parser.add_argument('--replay-k6-root', type=Path, required=True)
	parser.add_argument('--migration-decision', type=Path, required=True)
	parser.add_argument('--control-summary', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Publish only after all referenced arrays and hashes validate."""
	args = build_parser().parse_args(argv)
	validate_multi_head_target_publication_preflight(
		migration_decision=args.migration_decision,
		control_summary=args.control_summary,
	)
	heads = _head_roots(args.head_root)
	if args.only_missing and args.manifest.exists():
		reused = False
		try:
			payload = load_multi_head_target_manifest(args.manifest)
			reused = _matches_requested_inputs(
				payload,
				source_embedding_dir=args.source_embedding_dir,
				head_roots=heads,
				replay_k6_root=args.replay_k6_root,
			)
		except (OSError, TypeError, ValueError) as exc:
			if not args.quarantine_invalid:
				raise ValueError(
					'existing manifest is invalid; pass --quarantine-invalid'
				) from exc
			quarantine = args.manifest.with_name(f'{args.manifest.name}.quarantine')
			if quarantine.exists():
				raise FileExistsError(f'quarantine path exists: {quarantine}') from exc
			if args.dry_run:
				print(f'would quarantine: {quarantine}')
			else:
				args.manifest.replace(quarantine)
				print(f'quarantined: {quarantine}')
		if reused:
			print(f'execution: reused complete manifest {args.manifest}')
			return 0
	if args.dry_run:
		with tempfile.TemporaryDirectory(
			prefix=f'{args.manifest.name}.dry-run.',
		) as temporary_directory:
			payload = build_multi_head_target_manifest(
				manifest_path=Path(temporary_directory) / args.manifest.name,
				source_embedding_dir=args.source_embedding_dir,
				head_roots=heads,
				replay_k6_root=args.replay_k6_root,
				migration_decision=args.migration_decision,
				control_summary=args.control_summary,
			)
		print(f'execution: dry-run; validated heads {payload["head_ks"]}')
		return 0
	payload = build_multi_head_target_manifest(
		manifest_path=args.manifest,
		source_embedding_dir=args.source_embedding_dir,
		head_roots=heads,
		replay_k6_root=args.replay_k6_root,
		migration_decision=args.migration_decision,
		control_summary=args.control_summary,
	)
	print(f'manifest: {args.manifest}')
	print(f'head_ks: {payload["head_ks"]}')
	return 0


def _head_roots(values: list[str]) -> dict[int, Path]:
	result: dict[int, Path] = {}
	for value in values:
		try:
			key, path = value.split('=', maxsplit=1)
			k = int(key)
		except ValueError as exc:
			raise ValueError(f'--head-root must be K=PATH; got {value!r}') from exc
		if k in result or not path:
			raise ValueError(f'duplicate or empty --head-root: {value!r}')
		result[k] = Path(path)
	return result


def _matches_requested_inputs(
	payload: dict[str, object],
	*,
	source_embedding_dir: Path,
	head_roots: dict[int, Path],
	replay_k6_root: Path,
) -> bool:
	"""Return whether a valid manifest was built from these exact inputs."""
	if set(head_roots) != {6, 8, 10}:
		return False
	source_embedding = _object(payload['source_embedding'], 'source_embedding')
	if not _same_path(source_embedding['input_dir'], source_embedding_dir):
		return False
	heads = _object(payload['heads'], 'heads')
	for k, root in head_roots.items():
		head = _object(heads.get(str(k)), f'head k={k}')
		if not _same_path(head['pseudo_target_root'], root):
			return False
	parity = payload.get('k6_replay_parity')
	if parity is None:
		return False
	return _same_path(
		_object(parity, 'k6_replay_parity')['replay_root'],
		replay_k6_root,
	)


def _object(value: object, name: str) -> dict[str, object]:
	if not isinstance(value, dict):
		raise TypeError(f'{name} must be an object')
	return value


def _same_path(recorded: object, requested: Path) -> bool:
	return isinstance(recorded, str) and Path(recorded).resolve() == requested.resolve()


if __name__ == '__main__':
	raise SystemExit(main())
