"""Closed model roster for the F3 section-layout voxel benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_str,
	_validate_allowed_keys,
)

ROSTER_SCHEMA_VERSION = 'f3_voxel_section_layout_model_roster_v1'
EMBEDDING_ROOT_PREFIX = PurePosixPath(
	'embeddings/f3/facies_benchmark_v1'
)

# model_id: (model_tag, parent_model_id, selection_role)
EXPECTED_MODEL_ROSTER: Mapping[str, tuple[str, str | None, str]] = {
	'mae': (
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		None,
		'baseline',
	),
	'm1_k6': (
		'strat_hmm_pretext_m1_k6_topblock1_distill',
		'mae',
		'candidate',
	),
	'm1_distill_only': (
		'strat_hmm_m1_guardrail_distill_only',
		'mae',
		'diagnostic',
	),
	'm1_shuffled_hmm': (
		'strat_hmm_m1_guardrail_shuffled_hmm_seed42',
		'm1_k6',
		'diagnostic',
	),
	'm2a_boundary': (
		'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill',
		'm1_k6',
		'candidate',
	),
	'm1_current_k6': (
		'strat_hmm_pretext_m1_current_k6_topblock1_distill_v1',
		'm1_k6',
		'candidate',
	),
	'mh_nocons': (
		'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1',
		'm1_current_k6',
		'candidate',
	),
	'mh_cons010': (
		'strat_hmm_pretext_mh_k6810_cons010_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_soft_posterior': (
		'strat_hmm_pretext_mh_k6810_soft_nocons_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_lateral_smoothing': (
		'strat_hmm_pretext_mh_k6810_latmf1_nocons_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_xy_consensus': (
		'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_xy_unanimous': (
		'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_ctmask010_nocons': (
		'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1',
		'mh_nocons',
		'candidate',
	),
	'mh_ctmask010_refresh3ep_hmm2_nocons': (
		'strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1',
		'mh_ctmask010_nocons',
		'candidate',
	),
}
EXPECTED_MODEL_IDS = tuple(EXPECTED_MODEL_ROSTER)
SELECTION_ROLES = frozenset({'baseline', 'candidate', 'diagnostic'})


@dataclass(frozen=True)
class F3SectionLayoutModel:
	"""One explicitly named frozen embedding source."""

	model_id: str
	model_tag: str
	embedding_root: Path
	parent_model_id: str | None
	selection_role: str

	@property
	def selection_eligible(self) -> bool:
		"""Return whether formal adoption may select this model."""
		return self.selection_role == 'candidate'


@dataclass(frozen=True)
class F3SectionLayoutModelRoster:
	"""The complete immutable set of benchmark models."""

	artifact_root: Path
	models: tuple[F3SectionLayoutModel, ...]

	@property
	def model_by_id(self) -> Mapping[str, F3SectionLayoutModel]:
		"""Index the closed roster by model ID."""
		return {model.model_id: model for model in self.models}


def f3_lithology_voxel_section_layout_model_roster_from_mapping(
	config: Mapping[str, object],
) -> F3SectionLayoutModelRoster:
	"""Resolve the exact 14-model roster without filesystem discovery."""
	_validate_allowed_keys(
		config,
		frozenset({'schema_version', 'artifact_root', 'models'}),
		prefix='config',
	)
	if config.get('schema_version') != ROSTER_SCHEMA_VERSION:
		raise ValueError(
			f'schema_version must be exactly {ROSTER_SCHEMA_VERSION!r}'
		)
	artifact_root = _required_absolute_path(
		config, 'artifact_root', prefix='config'
	)
	raw_models = config.get('models')
	if not isinstance(raw_models, Sequence) or isinstance(raw_models, str | bytes):
		raise TypeError('models must be a list of model mappings')
	if len(raw_models) != len(EXPECTED_MODEL_IDS):
		raise ValueError(
		f'models must contain exactly {len(EXPECTED_MODEL_IDS)} entries'
		)

	models = tuple(
		_resolve_model(item, artifact_root=artifact_root, index=index)
		for index, item in enumerate(raw_models)
	)
	_validate_roster_graph(models)
	_validate_exact_roster(models, artifact_root=artifact_root)
	by_id = {model.model_id: model for model in models}
	return F3SectionLayoutModelRoster(
		artifact_root=artifact_root,
		models=tuple(by_id[model_id] for model_id in EXPECTED_MODEL_IDS),
	)


resolve_f3_lithology_voxel_section_layout_model_roster = (
	f3_lithology_voxel_section_layout_model_roster_from_mapping
)
f3_section_layout_model_roster_from_mapping = (
	f3_lithology_voxel_section_layout_model_roster_from_mapping
)


def _resolve_model(
	value: object,
	*,
	artifact_root: Path,
	index: int,
) -> F3SectionLayoutModel:
	if not isinstance(value, Mapping):
		raise TypeError(f'models[{index}] must be a mapping; got {value!r}')
	prefix = f'models[{index}]'
	_validate_allowed_keys(
		value,
		frozenset(
			{
				'model_id',
				'model_tag',
				'embedding_root',
				'parent_model_id',
				'selection_role',
			}
		),
		prefix=prefix,
	)
	model_id = _required_str(value, 'model_id', prefix=prefix)
	model_tag = _required_str(value, 'model_tag', prefix=prefix)
	selection_role = _required_str(value, 'selection_role', prefix=prefix)
	if selection_role not in SELECTION_ROLES:
		raise ValueError(
			f'{prefix}.selection_role must be one of {sorted(SELECTION_ROLES)!r}'
		)
	parent = value.get('parent_model_id')
	if parent is not None and (not isinstance(parent, str) or not parent):
		raise TypeError(
			f'{prefix}.parent_model_id must be a non-empty string or null'
		)
	relative_root = _safe_relative_path(
		_required_str(value, 'embedding_root', prefix=prefix),
		label=f'{prefix}.embedding_root',
	)
	return F3SectionLayoutModel(
		model_id=model_id,
		model_tag=model_tag,
		embedding_root=artifact_root.joinpath(*relative_root.parts),
		parent_model_id=parent,
		selection_role=selection_role,
	)


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
	if '\\' in value:
		raise ValueError(f'{label} must use POSIX path separators')
	path = PurePosixPath(value)
	if path.is_absolute() or '..' in path.parts or path == PurePosixPath('.'):
		raise ValueError(
			f'{label} must be a non-empty relative path without parent traversal'
		)
	return path


def _validate_roster_graph(models: tuple[F3SectionLayoutModel, ...]) -> None:
	ids = tuple(model.model_id for model in models)
	tags = tuple(model.model_tag for model in models)
	roots = tuple(model.embedding_root for model in models)
	for label, values in (
		('model IDs', ids),
		('model tags', tags),
		('embedding roots', roots),
	):
		if len(set(values)) != len(values):
			raise ValueError(f'{label} must be unique')
	known_ids = set(ids)
	for model in models:
		if model.parent_model_id is not None and model.parent_model_id not in known_ids:
			raise ValueError(
				f'parent model {model.parent_model_id!r} for {model.model_id!r} '
				'does not exist'
			)
	parents = {model.model_id: model.parent_model_id for model in models}
	for model_id in ids:
		seen: set[str] = set()
		cursor: str | None = model_id
		while cursor is not None:
			if cursor in seen:
				raise ValueError(f'parent cycle detected at model {cursor!r}')
			seen.add(cursor)
			cursor = parents[cursor]


def _validate_exact_roster(
	models: tuple[F3SectionLayoutModel, ...], *, artifact_root: Path
) -> None:
	by_id = {model.model_id: model for model in models}
	if set(by_id) != set(EXPECTED_MODEL_IDS):
		raise ValueError(
		'model IDs must be exactly the closed F3 section-layout roster'
		)
	for model_id, (tag, parent, role) in EXPECTED_MODEL_ROSTER.items():
		model = by_id[model_id]
		if (
			model.model_tag != tag
			or model.parent_model_id != parent
			or model.selection_role != role
		):
			raise ValueError(
				f'model {model_id!r} identity, parent, and selection role must '
				'match the closed roster'
			)
		expected_root = artifact_root.joinpath(
			*EMBEDDING_ROOT_PREFIX.parts, tag, 'overlap_x16'
		)
		if model.embedding_root != expected_root:
			raise ValueError(
				f'model {model_id!r} embedding_root must be {expected_root}'
			)
	if by_id['mae'].parent_model_id is not None:
		raise ValueError('baseline model mae must not have a parent')


__all__ = [
	'EMBEDDING_ROOT_PREFIX',
	'EXPECTED_MODEL_IDS',
	'EXPECTED_MODEL_ROSTER',
	'ROSTER_SCHEMA_VERSION',
	'F3SectionLayoutModel',
	'F3SectionLayoutModelRoster',
	'f3_lithology_voxel_section_layout_model_roster_from_mapping',
	'f3_section_layout_model_roster_from_mapping',
	'resolve_f3_lithology_voxel_section_layout_model_roster',
]
