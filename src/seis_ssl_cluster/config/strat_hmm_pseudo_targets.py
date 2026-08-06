"""Validation for strat HMM pseudo-target refresh configs."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from seis_ssl_cluster.config.base import _resolve_base
from seis_ssl_cluster.config.clustering import (
	_validate_stratigraphic_hmm_edge_margin_tokens,
	_validate_stratigraphic_hmm_path_prior,
	_validate_stratigraphic_hmm_transition,
)
from seis_ssl_cluster.config.common import (
	_is_int,
	_required_mapping,
	_validate_allowed_keys,
	_validate_bool,
	_validate_fraction,
	_validate_non_empty_path,
	_validate_nonnegative_int_triplet,
	_validate_output_path,
	_validate_path,
	_validate_positive_finite_number,
	_validate_positive_int,
	_validate_positive_int_triplet,
)
from seis_ssl_cluster.config.schema import STAGE_STRAT_HMM_PSEUDO_TARGETS

if TYPE_CHECKING:
	from pathlib import Path

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])

_MANIFESTS_KEYS = frozenset({'train'})
_CHECKPOINT_KEYS = frozenset({'path'})
_MODEL_KEYS = frozenset({'patch_size'})
_INFERENCE_KEYS = frozenset(
	{
		'window_size',
		'overlap',
		'batch_size',
		'output_dtype',
		'min_token_valid_fraction',
		'device',
	},
)
_HMM_KEYS = frozenset(
	{'k', 'edge_margin_tokens', 'transition', 'path_prior', 'boundary_weighting'},
)
_HMM_REQUIRED_KEYS = frozenset({'k', 'transition'})
_BOUNDARY_WEIGHTING_KEYS = frozenset({'alpha', 'tau'})
_OUTPUTS_KEYS = frozenset({'pseudo_target_root', 'overwrite', 'skip_existing'})


def resolve_strat_hmm_pseudo_target_config(config: _T) -> Config:
	"""Validate and resolve raw config for strat HMM pseudo-target refresh."""
	resolved, paths = _resolve_base(
		config,
		STAGE_STRAT_HMM_PSEUDO_TARGETS,
		require_nopims_root=False,
	)
	_validate_sections(resolved)

	manifests = _required_mapping(resolved, 'manifests')
	checkpoint = _required_mapping(resolved, 'checkpoint')
	model = _required_mapping(resolved, 'model')
	inference = _required_mapping(resolved, 'inference')
	hmm = _required_mapping(resolved, 'hmm')
	outputs = _required_mapping(resolved, 'outputs')
	if not isinstance(hmm, dict):
		msg = 'hmm must be a mapping'
		raise TypeError(msg)
	if 'boundary_weighting' not in hmm:
		hmm['boundary_weighting'] = {'alpha': 0.0, 'tau': 1.0}
	else:
		boundary_weighting = hmm.get('boundary_weighting')
		if not isinstance(boundary_weighting, dict):
			msg = 'hmm.boundary_weighting must be a mapping'
			raise TypeError(msg)
		boundary_weighting.setdefault('alpha', 0.0)
		boundary_weighting.setdefault('tau', 1.0)
	if 'skip_existing' not in outputs:
		outputs['skip_existing'] = False

	_validate_non_empty_path(manifests, 'train', prefix='manifests')
	checkpoint_path = _validate_non_empty_path(
		checkpoint,
		'path',
		prefix='checkpoint',
	)
	if not checkpoint_path.is_file():
		msg = f'checkpoint.path must exist and be a file: {checkpoint_path}'
		raise FileNotFoundError(msg)

	patch_size = _validate_positive_int_triplet(
		model,
		'patch_size',
		prefix='model',
	)
	window_size = _validate_positive_int_triplet(
		inference,
		'window_size',
		prefix='inference',
	)
	_validate_window_patch_divisibility(window_size, patch_size)
	overlap = _validate_nonnegative_int_triplet(
		inference,
		'overlap',
		prefix='inference',
	)
	_validate_overlap_less_than_window(overlap, window_size)
	_validate_positive_int(inference, 'batch_size', prefix='inference')
	_validate_output_dtype(inference)
	_validate_fraction(
		inference,
		'min_token_valid_fraction',
		prefix='inference',
	)
	_validate_device(inference)

	_validate_hmm(hmm, checkpoint_path=checkpoint_path)
	_validate_outputs(outputs, input_root=paths.nopims_root)
	return resolved


def _validate_sections(config: Mapping[str, object]) -> None:
	for section, allowed in (
		('manifests', _MANIFESTS_KEYS),
		('checkpoint', _CHECKPOINT_KEYS),
		('model', _MODEL_KEYS),
		('inference', _INFERENCE_KEYS),
		('hmm', _HMM_KEYS),
		('outputs', _OUTPUTS_KEYS),
	):
		value = config.get(section)
		if isinstance(value, Mapping):
			_validate_allowed_keys(value, allowed, prefix=section)


def _validate_window_patch_divisibility(
	window_size: Sequence[int],
	patch_size: Sequence[int],
) -> None:
	if any(
		window_axis % patch_axis != 0
		for window_axis, patch_axis in zip(window_size, patch_size, strict=True)
	):
		msg = (
			'inference.window_size values must be divisible by model.patch_size '
			f'values; got window_size={list(window_size)!r}, '
			f'patch_size={list(patch_size)!r}'
		)
		raise ValueError(msg)


def _validate_overlap_less_than_window(
	overlap: Sequence[int],
	window_size: Sequence[int],
) -> None:
	if any(
		overlap_axis >= window_axis
		for overlap_axis, window_axis in zip(overlap, window_size, strict=True)
	):
		msg = (
			'inference.overlap values must be less than inference.window_size '
			f'values; got overlap={list(overlap)!r}, '
			f'window_size={list(window_size)!r}'
		)
		raise ValueError(msg)


def _validate_output_dtype(inference: Mapping[str, object]) -> None:
	value = inference.get('output_dtype')
	if not isinstance(value, str) or not value:
		msg = f'inference.output_dtype must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	if value != 'float32':
		msg = 'inference.output_dtype must be "float32" for prototype logits'
		raise ValueError(msg)


def _validate_device(inference: Mapping[str, object]) -> None:
	value = inference.get('device')
	if value not in {'auto', 'cpu', 'cuda'}:
		msg = 'inference.device must be "auto", "cpu", or "cuda"'
		raise ValueError(msg)


def _validate_hmm(
	hmm: Mapping[str, object],
	*,
	checkpoint_path: Path,
) -> None:
	_validate_allowed_keys(hmm, _HMM_KEYS, prefix='hmm')
	for key in sorted(_HMM_REQUIRED_KEYS):
		if key not in hmm:
			msg = f'hmm.{key} is required'
			raise ValueError(msg)
	_validate_positive_int(hmm, 'k', prefix='hmm')
	_validate_stratigraphic_hmm_edge_margin_tokens(hmm, prefix='hmm')
	_validate_stratigraphic_hmm_transition(hmm, prefix='hmm')
	_validate_stratigraphic_hmm_path_prior(hmm, prefix='hmm')
	_validate_boundary_weighting(hmm)
	_validate_checkpoint_prototype_count(checkpoint_path, k=int(hmm['k']))


def _validate_boundary_weighting(hmm: Mapping[str, object]) -> None:
	value = hmm.get('boundary_weighting')
	if not isinstance(value, Mapping):
		msg = 'hmm.boundary_weighting must be a mapping'
		raise TypeError(msg)
	_validate_allowed_keys(
		value,
		_BOUNDARY_WEIGHTING_KEYS,
		prefix='hmm.boundary_weighting',
	)
	_validate_fraction(value, 'alpha', prefix='hmm.boundary_weighting')
	_validate_positive_finite_number(
		value,
		'tau',
		prefix='hmm.boundary_weighting',
	)


def _validate_checkpoint_prototype_count(checkpoint_path: Path, *, k: int) -> None:
	num_prototypes = _inspect_checkpoint_num_prototypes(checkpoint_path)
	if num_prototypes is None:
		return
	if num_prototypes != k:
		msg = (
			'hmm.k must match checkpoint head num_prototypes; '
			f'got hmm.k={k!r}, checkpoint={num_prototypes!r}'
		)
		raise ValueError(msg)


def _inspect_checkpoint_num_prototypes(checkpoint_path: Path) -> int | None:
	try:
		# Keep config validation to safe metadata inspection; runtime loading
		# performs the full checkpoint compatibility checks.
		with FakeTensorMode():
			payload = torch.load(
				checkpoint_path,
				map_location='cpu',
				weights_only=True,
			)
	except (
		EOFError,
		OSError,
		RuntimeError,
		TypeError,
		ValueError,
		pickle.UnpicklingError,
	):
		return None
	if not isinstance(payload, Mapping):
		return None
	if 'stratigraphy_checkpoint' in payload:
		raise ValueError(
			'multi-head checkpoint requires a future explicit '
			'head-selection/multi-output contract'
		)

	config_value = payload.get('stratigraphy_config')
	if isinstance(config_value, Mapping):
		head = config_value.get('head')
		if isinstance(head, Mapping):
			value = head.get('num_prototypes')
			if _is_int(value) and int(value) > 0:
				return int(value)

	state = payload.get('stratigraphy_state_dict')
	if isinstance(state, Mapping):
		prototypes = state.get('prototypes')
		shape = getattr(prototypes, 'shape', None)
		if (
			isinstance(shape, Sequence)
			and len(shape) >= 1
			and _is_int(shape[0])
			and int(shape[0]) > 0
		):
			return int(shape[0])
	return None


def _validate_outputs(
	outputs: Mapping[str, object], *, input_root: Path | None
) -> None:
	pseudo_target_root = _validate_path(
		outputs,
		'pseudo_target_root',
		prefix='outputs',
	)
	_validate_output_path(
		pseudo_target_root,
		'outputs.pseudo_target_root',
		input_root=input_root,
		input_root_label='paths.nopims_root',
	)
	_validate_bool(outputs, 'overwrite', prefix='outputs')
	_validate_bool(outputs, 'skip_existing', prefix='outputs')
	if outputs['overwrite'] and outputs['skip_existing']:
		msg = 'outputs.overwrite and outputs.skip_existing cannot both be true'
		raise ValueError(msg)


__all__ = ['resolve_strat_hmm_pseudo_target_config']
