from __future__ import annotations

import ast
import importlib
from pathlib import Path

NEW_PACKAGE_MODULES = (
	'seis_ssl_cluster',
	'seis_ssl_cluster.config',
	'seis_ssl_cluster.data',
	'seis_ssl_cluster.masking',
	'seis_ssl_cluster.models',
	'seis_ssl_cluster.models.common',
	'seis_ssl_cluster.models.mae',
	'seis_ssl_cluster.losses',
	'seis_ssl_cluster.training',
	'seis_ssl_cluster.embedding',
	'seis_ssl_cluster.clustering',
	'seis_ssl_cluster.visualization',
	'seis_ssl_cluster.utils',
)


def test_new_package_imports() -> None:
	seis_ssl_cluster = importlib.import_module('seis_ssl_cluster')

	assert seis_ssl_cluster is not None


def test_new_package_modules_import() -> None:
	for module_name in NEW_PACKAGE_MODULES:
		importlib.import_module(module_name)


def test_seis_ssl_cluster_has_no_legacy_package_imports() -> None:
	package_root = Path(__file__).parents[2] / 'src' / 'seis_ssl_cluster'

	for path in package_root.rglob('*.py'):
		tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				imported_names = (alias.name for alias in node.names)
			elif isinstance(node, ast.ImportFrom):
				imported_names = (node.module or '',)
			else:
				continue

			for name in imported_names:
				assert name != 'seis_attr_ssl'
				assert not name.startswith('seis_attr_ssl.')
