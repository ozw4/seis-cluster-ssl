"""Artifact and result path contracts for seismic SSL experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path('/workspace/artifacts/seis_ssl_cluster')
DEFAULT_RESULTS_ROOT = Path('results')
_SAFE_SLUG_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')


@dataclass(frozen=True)
class ArtifactRoot:
	"""Root directory for local, complete artifacts."""

	path: Path

	def __post_init__(self) -> None:
		"""Normalize and validate the root path."""
		path = Path(self.path)
		reject_runs_path(path, label='artifact root')
		object.__setattr__(self, 'path', path)


@dataclass(frozen=True)
class ExperimentKey:
	"""Names that identify one artifact path under the repository contract."""

	dataset: str
	version: str
	model_tag: str | None = None
	subset: str | None = None
	embed_spec: str | None = None
	cluster_spec: str | None = None
	viz_spec: str | None = None
	label_set: str | None = None
	probe_spec: str | None = None
	baseline_tag: str | None = None
	run_spec: str | None = None


class ArtifactPaths:
	"""Build local artifact paths without touching the filesystem."""

	def __init__(self, root: Path | ArtifactRoot = DEFAULT_ARTIFACT_ROOT) -> None:
		"""Initialize the builder with an artifact root."""
		root_path = root.path if isinstance(root, ArtifactRoot) else Path(root)
		reject_runs_path(root_path, label='artifact root')
		self.root = root_path

	def pretraining(self, key: ExperimentKey) -> Path:
		"""Return ``pretraining/<dataset>/<version>/<model>/<run>``."""
		return self._build(
			'pretraining',
			'pretraining',
			self._component(key, 'dataset', 'pretraining'),
			self._component(key, 'version', 'pretraining'),
			self._component(key, 'model_tag', 'pretraining'),
			self._component(key, 'run_spec', 'pretraining'),
		)

	def embeddings(self, key: ExperimentKey) -> Path:
		"""Return an embedding path, omitting subset only when it is absent."""
		components = [
			'embeddings',
			self._component(key, 'dataset', 'embeddings'),
			self._component(key, 'version', 'embeddings'),
			self._component(key, 'model_tag', 'embeddings'),
		]
		if key.subset is not None:
			components.append(self._component(key, 'subset', 'embeddings'))
		components.append(self._component(key, 'embed_spec', 'embeddings'))
		return self._build('embeddings', *components)

	def clustering(self, key: ExperimentKey) -> Path:
		"""Return ``clustering/.../<subset>/<embed_spec>/<cluster_spec>``."""
		return self._build(
			'clustering',
			'clustering',
			self._component(key, 'dataset', 'clustering'),
			self._component(key, 'version', 'clustering'),
			self._component(key, 'model_tag', 'clustering'),
			self._component(key, 'subset', 'clustering'),
			self._component(key, 'embed_spec', 'clustering'),
			self._component(key, 'cluster_spec', 'clustering'),
		)

	def cluster_visualization(self, key: ExperimentKey) -> Path:
		"""Return ``visualizations/clusters/.../<cluster_spec>/<viz_spec>``."""
		return self._build(
			'cluster visualization',
			'visualizations',
			'clusters',
			self._component(key, 'dataset', 'cluster visualization'),
			self._component(key, 'version', 'cluster visualization'),
			self._component(key, 'model_tag', 'cluster visualization'),
			self._component(key, 'subset', 'cluster visualization'),
			self._component(key, 'embed_spec', 'cluster visualization'),
			self._component(key, 'cluster_spec', 'cluster visualization'),
			self._component(key, 'viz_spec', 'cluster visualization'),
		)

	def lithology_dataset(self, key: ExperimentKey) -> Path:
		"""Return ``lithology/<dataset>/<version>``."""
		return self._build(
			'lithology dataset',
			'lithology',
			self._component(key, 'dataset', 'lithology dataset'),
			self._component(key, 'version', 'lithology dataset'),
		)

	def lithology_token_dataset(self, key: ExperimentKey) -> Path:
		"""Return a lithology token dataset directory."""
		return self._build(
			'lithology token dataset',
			*self._lithology_components(key, 'lithology token dataset'),
			'token_dataset',
		)

	def lithology_probe(self, key: ExperimentKey) -> Path:
		"""Return a lithology probe directory."""
		return self._build(
			'lithology probe',
			*self._lithology_components(key, 'lithology probe'),
			'probes',
			self._component(key, 'probe_spec', 'lithology probe'),
		)

	def lithology_predictions(self, key: ExperimentKey) -> Path:
		"""Return a lithology prediction directory."""
		return self._build(
			'lithology predictions',
			*self._lithology_components(key, 'lithology predictions'),
			'predictions',
			self._component(key, 'probe_spec', 'lithology predictions'),
		)

	def lithology_visualizations(self, key: ExperimentKey) -> Path:
		"""Return a lithology visualization directory."""
		return self._build(
			'lithology visualizations',
			*self._lithology_components(key, 'lithology visualizations'),
			'visualizations',
			self._component(key, 'probe_spec', 'lithology visualizations'),
		)

	def lithology_report(self, key: ExperimentKey) -> Path:
		"""Return a lithology report directory."""
		return self._build(
			'lithology report',
			*self._lithology_components(key, 'lithology report'),
			'reports',
			self._component(key, 'probe_spec', 'lithology report'),
		)

	def baseline_token_dataset(self, key: ExperimentKey) -> Path:
		"""Return a baseline token dataset directory."""
		return self._build(
			'baseline token dataset',
			'lithology',
			self._component(key, 'dataset', 'baseline token dataset'),
			self._component(key, 'version', 'baseline token dataset'),
			'baselines',
			self._component(key, 'baseline_tag', 'baseline token dataset'),
			self._component(key, 'label_set', 'baseline token dataset'),
			'token_dataset',
		)

	def baseline_probe(self, key: ExperimentKey) -> Path:
		"""Return a baseline probe directory."""
		return self._build(
			'baseline probe',
			'lithology',
			self._component(key, 'dataset', 'baseline probe'),
			self._component(key, 'version', 'baseline probe'),
			'baselines',
			self._component(key, 'baseline_tag', 'baseline probe'),
			self._component(key, 'label_set', 'baseline probe'),
			'probes',
			self._component(key, 'probe_spec', 'baseline probe'),
		)

	def baseline_comparison_report(self, key: ExperimentKey) -> Path:
		"""Return the baseline comparison report artifact directory."""
		return self._build(
			'baseline comparison report',
			'lithology',
			self._component(key, 'dataset', 'baseline comparison report'),
			self._component(key, 'version', 'baseline comparison report'),
			'reports',
			'baseline_comparison',
		)

	def _lithology_components(
		self,
		key: ExperimentKey,
		stage: str,
	) -> tuple[str, ...]:
		return (
			'lithology',
			self._component(key, 'dataset', stage),
			self._component(key, 'version', stage),
			self._component(key, 'model_tag', stage),
			self._component(key, 'embed_spec', stage),
			self._component(key, 'label_set', stage),
		)

	def _component(self, key: ExperimentKey, field: str, stage: str) -> str:
		value = getattr(key, field)
		if value is None:
			msg = f'{stage} path requires {field}'
			raise ValueError(msg)
		return safe_slug(value, label=f'{stage} {field}')

	def _build(self, label: str, *components: str) -> Path:
		path = self.root.joinpath(*components)
		reject_runs_path(path, label=label)
		ensure_under_root(path, root=self.root, label=label)
		return path


class ResultsPaths:
	"""Build repository-managed lightweight result paths."""

	def __init__(self, root: Path = DEFAULT_RESULTS_ROOT) -> None:
		"""Initialize the builder with a results root."""
		root_path = Path(root)
		reject_runs_path(root_path, label='results root')
		self.root = root_path

	def inspection(self, key: ExperimentKey) -> Path:
		"""Return ``results/<dataset>/<version>/inspection``."""
		return self._build(
			'inspection results',
			self._component(key, 'dataset', 'inspection results'),
			self._component(key, 'version', 'inspection results'),
			'inspection',
		)

	def lithology_probe(self, key: ExperimentKey) -> Path:
		"""Return ``results/<dataset>/<version>/lithology_probe/...``."""
		return self._build(
			'lithology probe results',
			self._component(key, 'dataset', 'lithology probe results'),
			self._component(key, 'version', 'lithology probe results'),
			'lithology_probe',
			self._component(key, 'model_tag', 'lithology probe results'),
			self._component(key, 'embed_spec', 'lithology probe results'),
			self._component(key, 'label_set', 'lithology probe results'),
			self._component(key, 'probe_spec', 'lithology probe results'),
		)

	def baseline_comparison(self, key: ExperimentKey) -> Path:
		"""Return ``results/<dataset>/<version>/baseline_comparison``."""
		return self._build(
			'baseline comparison results',
			self._component(key, 'dataset', 'baseline comparison results'),
			self._component(key, 'version', 'baseline comparison results'),
			'baseline_comparison',
		)

	def _component(self, key: ExperimentKey, field: str, stage: str) -> str:
		value = getattr(key, field)
		if value is None:
			msg = f'{stage} path requires {field}'
			raise ValueError(msg)
		return safe_slug(value, label=f'{stage} {field}')

	def _build(self, label: str, *components: str) -> Path:
		path = self.root.joinpath(*components)
		reject_runs_path(path, label=label)
		ensure_under_root(path, root=self.root, label=label)
		return path


def reject_runs_path(path: Path, *, label: str) -> None:
	"""Reject paths that use a ``runs/`` component."""
	candidate = Path(path)
	if 'runs' in candidate.parts:
		msg = f'{label} must not use runs/ paths; got {candidate}'
		raise ValueError(msg)


def ensure_under_root(path: Path, *, root: Path, label: str) -> None:
	"""Reject ``path`` when it resolves outside ``root``."""
	candidate = Path(path).resolve(strict=False)
	resolved_root = Path(root).resolve(strict=False)
	try:
		candidate.relative_to(resolved_root)
	except ValueError as exc:
		msg = f'{label} must be under root ({resolved_root}); got {candidate}'
		raise ValueError(msg) from exc


def safe_slug(value: str, *, label: str) -> str:
	"""Validate one path component and return it unchanged."""
	if not isinstance(value, str):
		msg = f'{label} must be a string; got {type(value).__name__}'
		raise TypeError(msg)
	if value in {'', '.'} or '..' in value:
		msg = f'{label} must be a safe slug; got {value!r}'
		raise ValueError(msg)
	if _SAFE_SLUG_PATTERN.fullmatch(value) is None:
		msg = (
			f'{label} must contain only A-Z, a-z, 0-9, "_", ".", or "-"; '
			f'got {value!r}'
		)
		raise ValueError(msg)
	return value


__all__ = [
	'DEFAULT_ARTIFACT_ROOT',
	'DEFAULT_RESULTS_ROOT',
	'ArtifactPaths',
	'ArtifactRoot',
	'ExperimentKey',
	'ResultsPaths',
	'ensure_under_root',
	'reject_runs_path',
	'safe_slug',
]
