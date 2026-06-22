"""Check that seis_ssl_cluster does not import the legacy package namespace."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / 'src' / 'seis_ssl_cluster'
LEGACY_PACKAGE = 'seis_attr_ssl'


def main() -> None:
	"""Fail if the standalone package imports the legacy namespace."""
	violations: list[str] = []
	for path in sorted(PACKAGE_ROOT.rglob('*.py')):
		tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				imported_names = (alias.name for alias in node.names)
			elif isinstance(node, ast.ImportFrom):
				imported_names = (node.module or '',)
			else:
				continue

			violations.extend(
				f'{path}: imports {name}'
				for name in imported_names
				if name == LEGACY_PACKAGE or name.startswith(f'{LEGACY_PACKAGE}.')
			)

	if violations:
		for violation in violations:
			print(violation)
		raise SystemExit(1)

	print('seis_ssl_cluster isolation check passed')


if __name__ == '__main__':
	main()
