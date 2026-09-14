"""Validate and publish a K=6/8/10 HMM multi-head target manifest."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.stratigraphy import (
	build_frozen_k6_reference_receipt,
	build_multi_head_target_manifest,
	compare_k6_replay,
	load_multi_head_target_manifest,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the standalone multi-head manifest CLI parser."""
	parser = argparse.ArgumentParser(
		description='Publish validated schema-v1 multi-head HMM target references.',
	)
	parser.add_argument('--source-embedding-dir', type=Path, required=True)
	parser.add_argument('--head-root', action='append', required=True, metavar='K=PATH')
	parser.add_argument('--manifest', type=Path)
	parser.add_argument('--freeze-k6-receipt', action='store_true')
	parser.add_argument('--reference-training-config', type=Path)
	evidence = parser.add_mutually_exclusive_group()
	evidence.add_argument('--frozen-k6-receipt', type=Path)
	evidence.add_argument(
		'--replay-k6-root',
		type=Path,
		help='Exact replay evidence, mutually exclusive with a frozen K=6 receipt.',
	)
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--only-missing', action='store_true')
	parser.add_argument('--quarantine-invalid', action='store_true')
	return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912
	"""Publish only after all referenced arrays and hashes validate."""
	args = build_parser().parse_args(argv)
	heads = _head_roots(args.head_root)
	if args.freeze_k6_receipt:
		return _freeze_receipt(args, heads)
	if args.manifest is None or args.reference_training_config is not None:
		raise ValueError(
			'manifest building requires --manifest; '
			'reference config belongs to receipt freezing'
		)
	if args.frozen_k6_receipt is not None and args.quarantine_invalid:
		raise ValueError(
			'frozen-reference manifests cannot be implicitly quarantined or replaced'
		)

	if args.manifest.exists():
		try:
			existing = json.loads(args.manifest.read_text())
		except json.JSONDecodeError:
			existing = {}
		if (
			isinstance(existing, dict)
			and 'k6_frozen_reference' in existing
			and (args.frozen_k6_receipt is None or args.quarantine_invalid)
		):
			raise ValueError('frozen-reference manifests cannot change evidence mode')
	if args.only_missing and args.manifest.exists():
		reused = False
		try:
			payload = load_multi_head_target_manifest(args.manifest)
			reused = _matches_requested_inputs(
				payload,
				source_embedding_dir=args.source_embedding_dir,
				head_roots=heads,
				replay_k6_root=args.replay_k6_root,
				frozen_k6_receipt=args.frozen_k6_receipt,
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
		if args.frozen_k6_receipt is not None and not reused:
			raise ValueError('frozen-reference manifest requested inputs drift')
	if args.dry_run:
		with tempfile.TemporaryDirectory(
			prefix=f'{args.manifest.name}.dry-run.',
		) as temporary_directory:
			payload = build_multi_head_target_manifest(
				manifest_path=Path(temporary_directory) / args.manifest.name,
				source_embedding_dir=args.source_embedding_dir,
				head_roots=heads,
				replay_k6_root=args.replay_k6_root,
				frozen_k6_receipt=args.frozen_k6_receipt,
			)
		print(f'execution: dry-run; validated heads {payload["head_ks"]}')
		return 0
	payload = build_multi_head_target_manifest(
		manifest_path=args.manifest,
		source_embedding_dir=args.source_embedding_dir,
		head_roots=heads,
		replay_k6_root=args.replay_k6_root,
		frozen_k6_receipt=args.frozen_k6_receipt,
	)
	print(f'manifest: {args.manifest}')
	print(f'head_ks: {payload["head_ks"]}')
	return 0


def _freeze_receipt(args: argparse.Namespace, heads: dict[int, Path]) -> int:
	"""Validate and explicitly publish the immutable historical-reference receipt."""
	if (
		set(heads) != {6}
		or args.frozen_k6_receipt is None
		or args.reference_training_config is None
		or args.manifest is not None
		or args.quarantine_invalid
	):
		raise ValueError(
			'receipt freezing requires only head 6, frozen receipt '
			'and reference training config'
		)
	if args.frozen_k6_receipt.exists() and not args.only_missing:
		raise FileExistsError(
			f'frozen K=6 receipt already exists: {args.frozen_k6_receipt}'
		)
	build_frozen_k6_reference_receipt(
		receipt_path=args.frozen_k6_receipt,
		reference_training_config=args.reference_training_config,
		historical_root=heads[6],
		source_embedding_dir=args.source_embedding_dir,
		dry_run=args.dry_run,
	)
	print(
		'execution: validated frozen K=6 inputs'
		if args.dry_run
		else f'frozen K=6 receipt: {args.frozen_k6_receipt}'
	)
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


def _matches_requested_inputs(  # noqa: PLR0911
	payload: dict[str, object],
	*,
	source_embedding_dir: Path,
	head_roots: dict[int, Path],
	replay_k6_root: Path | None,
	frozen_k6_receipt: Path | None = None,
) -> bool:
	"""Return whether a valid manifest was built from these exact inputs."""
	manifest_ks = payload.get('head_ks')
	if not isinstance(manifest_ks, list) or manifest_ks != sorted(head_roots):
		return False
	source_embedding = _object(payload['source_embedding'], 'source_embedding')
	if not _same_path(source_embedding['input_dir'], source_embedding_dir):
		return False
	heads = _object(payload['heads'], 'heads')
	for k, root in head_roots.items():
		head = _object(heads.get(str(k)), f'head k={k}')
		if not _same_path(head['pseudo_target_root'], root):
			return False
	if 6 not in head_roots:
		return (
			replay_k6_root is None
			and frozen_k6_receipt is None
			and not ({'k6_replay_parity', 'k6_frozen_reference'} & set(payload))
		)
	if frozen_k6_receipt is not None:
		frozen = _object(payload.get('k6_frozen_reference'), 'K=6 frozen reference')
		return _same_path(
			_object(frozen['receipt'], 'K=6 receipt')['path'], frozen_k6_receipt
		)
	if replay_k6_root is None:
		return False
	parity = payload.get('k6_replay_parity')
	if parity is None:
		return False
	if not _same_path(
		_object(parity, 'k6_replay_parity')['replay_root'],
		replay_k6_root,
	):
		return False
	return bool(
		compare_k6_replay(
			historical_root=head_roots[6],
			replay_root=replay_k6_root,
		)['exact']
	)


def _object(value: object, name: str) -> dict[str, object]:
	if not isinstance(value, dict):
		raise TypeError(f'{name} must be an object')
	return value


def _same_path(recorded: object, requested: Path) -> bool:
	return isinstance(recorded, str) and Path(recorded).resolve() == requested.resolve()


if __name__ == '__main__':
	raise SystemExit(main())
