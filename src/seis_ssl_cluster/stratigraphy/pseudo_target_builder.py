"""Build refreshed strat HMM pseudo-target artifacts from prototype logits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from seis_ssl_cluster.clustering.stratigraphic_hmm import (
	HMMAnchorPriorSettings,
	HMMExpectedBoundariesSettings,
	HMMPathPriorSettings,
	HMMTransitionSettings,
)
from seis_ssl_cluster.data.normalization import (
	AmplitudeAgcConfig,
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_ssl_cluster.data.schema import CropRequest, SurveyManifest, read_manifest_json
from seis_ssl_cluster.data.volume_store import NpyMemmapVolumeStore
from seis_ssl_cluster.data.window_preprocessing import (
	AmplitudePreprocessSettings,
	read_amplitude_crop,
	resolve_manifest_path,
)
from seis_ssl_cluster.data.zero_mask import ZeroMaskConfig
from seis_ssl_cluster.embedding.extractor import build_model_from_config
from seis_ssl_cluster.embedding.sliding_window import (
	SlidingWindow,
	iter_sliding_windows,
	token_grid_shape_xyz,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.stratigraphy.hmm_decode import decode_ordered_logits_survey
from seis_ssl_cluster.stratigraphy.prototypes import OrderedPrototypeHead
from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetPaths,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	write_pseudo_target,
)
from seis_ssl_cluster.training.checkpoint import load_checkpoint

XYZ = tuple[int, int, int]
BUILDER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BuiltPseudoTargetResult:
	"""Pseudo-target files built for one survey."""

	survey_id: str
	labels_path: Path
	confidence_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	valid_token_count: int
	skipped: bool = False


@dataclass(frozen=True)
class _BuilderSettings:
	checkpoint_path: Path
	pseudo_target_root: Path
	window_size_xyz: XYZ
	overlap_xyz: XYZ
	batch_size: int
	min_token_valid_fraction: float
	overwrite: bool
	skip_existing: bool
	zero_mask: ZeroMaskConfig
	normalized_clip_abs: float | None
	amplitude_agc: AmplitudeAgcConfig
	hmm_k: int
	edge_margin_tokens: XYZ
	transition: HMMTransitionSettings
	path_prior: HMMPathPriorSettings | None
	expected_boundaries: HMMExpectedBoundariesSettings | None


def build_strat_hmm_pseudo_targets(
	config: Mapping[str, object],
	*,
	device: str | None = None,
	overwrite: bool | None = None,
	skip_existing: bool | None = None,
) -> list[Path]:
	"""Refresh strat HMM pseudo-targets from a trained prototype head."""
	results = build_strat_hmm_pseudo_target_results(
		config,
		device=device,
		overwrite=overwrite,
		skip_existing=skip_existing,
	)
	return [result.metadata_path for result in results]


def build_strat_hmm_pseudo_target_results(
	config: Mapping[str, object],
	*,
	device: str | None = None,
	overwrite: bool | None = None,
	skip_existing: bool | None = None,
) -> list[BuiltPseudoTargetResult]:
	"""Refresh strat HMM pseudo-targets and return per-survey output details."""
	checkpoint_path = _checkpoint_path(config)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	mae_config = _checkpoint_config(payload)
	stratigraphy_config = _stratigraphy_config(payload)
	settings = _settings_from_config(
		config,
		checkpoint_path=checkpoint_path,
		mae_config=mae_config,
		overwrite=overwrite,
		skip_existing=skip_existing,
	)
	model, head = _load_checkpoint_models(
		payload,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		hmm_k=settings.hmm_k,
		configured_patch_size=_xyz_from_config_section(config, 'model', 'patch_size'),
	)
	resolved_device = _resolve_device(device, config)
	model_dtype = _checkpoint_floating_dtype(_model_state_dict(payload))
	model.to(device=resolved_device, dtype=model_dtype)
	head.to(device=resolved_device, dtype=model_dtype)
	model.eval()
	head.eval()

	manifests = read_manifest_json(_manifest_path(config))
	if not manifests:
		msg = 'strat HMM pseudo-target manifest is empty'
		raise ValueError(msg)
	checkpoint_sha256 = file_sha256(settings.checkpoint_path)
	store = NpyMemmapVolumeStore()
	return [
		_build_survey_pseudo_targets(
			manifest,
			model=model,
			head=head,
			store=store,
			settings=settings,
			device=resolved_device,
			mae_config=mae_config,
			stratigraphy_config=stratigraphy_config,
			checkpoint_sha256=checkpoint_sha256,
			checkpoint_payload=payload,
		)
		for manifest in manifests
	]


def _load_checkpoint_models(
	payload: Mapping[str, object],
	*,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	hmm_k: int,
	configured_patch_size: XYZ,
) -> tuple[torch.nn.Module, OrderedPrototypeHead]:
	model = build_model_from_config(mae_config)
	if tuple(model.patch_size_xyz) != configured_patch_size:
		msg = (
			'model.patch_size must match checkpoint model geometry; '
			f'got config={configured_patch_size!r}, '
			f'checkpoint={tuple(model.patch_size_xyz)!r}'
		)
		raise ValueError(msg)
	model.load_state_dict(_model_state_dict(payload))

	head_config = _head_config(stratigraphy_config)
	head = OrderedPrototypeHead(
		feature_dim=model.encoder_dim,
		num_prototypes=_positive_int(
			head_config.get('num_prototypes'),
			'head.num_prototypes',
		),
		projection_dim=_optional_positive_int(
			head_config.get('projection_dim'),
			'head.projection_dim',
		),
		temperature=_positive_float(
			head_config.get('temperature', 0.1),
			'head.temperature',
		),
		normalize=_bool_value(head_config.get('normalize', True), 'head.normalize'),
	)
	if head.num_prototypes != hmm_k:
		msg = (
			'hmm.k must match checkpoint head num_prototypes; '
			f'got hmm.k={hmm_k!r}, checkpoint={head.num_prototypes!r}'
		)
		raise ValueError(msg)
	head.load_state_dict(_stratigraphy_state_dict(payload))
	return model, head


def _build_survey_pseudo_targets(  # noqa: PLR0913
	manifest: SurveyManifest,
	*,
	model: torch.nn.Module,
	head: OrderedPrototypeHead,
	store: NpyMemmapVolumeStore,
	settings: _BuilderSettings,
	device: torch.device,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	checkpoint_sha256: str,
	checkpoint_payload: Mapping[str, object],
) -> BuiltPseudoTargetResult:
	manifest.validate()
	paths = pseudo_target_paths(
		settings.pseudo_target_root,
		k=settings.hmm_k,
		survey_id=manifest.survey_id,
	)
	amplitude_path = resolve_manifest_path(manifest, manifest.amplitude.path)
	stats_path = resolve_manifest_path(
		manifest,
		manifest.amplitude.normalization_stats_path,
	)
	token_grid = token_grid_shape_xyz(
		manifest.amplitude.shape_xyz,
		cast('AmplitudeMAE3DProtocol', model).patch_size_xyz,
	)
	request_metadata = _pseudo_target_request_metadata(
		manifest=manifest,
		amplitude_path=amplitude_path,
		stats_path=stats_path,
		settings=settings,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		checkpoint_sha256=checkpoint_sha256,
		checkpoint_payload=checkpoint_payload,
		token_grid=token_grid,
	)
	if _prepare_output_paths(
		paths,
		request_metadata=request_metadata,
		settings=settings,
	):
		return _skipped_result(
			manifest.survey_id,
			paths=paths,
			metadata=load_pseudo_target_metadata(paths),
		)

	stats = load_normalization_stats(stats_path)
	logits, valid_tokens, count_array = _survey_logits(
		manifest,
		amplitude_path=amplitude_path,
		stats=stats,
		store=store,
		model=model,
		head=head,
		settings=settings,
		device=device,
		token_grid=token_grid,
	)
	decoded = decode_ordered_logits_survey(
		logits,
		valid_tokens,
		transition=settings.transition,
		path_prior=settings.path_prior,
		edge_margin_tokens=settings.edge_margin_tokens,
		expected_boundaries=settings.expected_boundaries,
	)
	metadata = _pseudo_target_source_metadata(
		request_metadata=request_metadata,
		count_array=count_array,
		decode_metadata=decoded.metadata,
	)
	written = write_pseudo_target(
		settings.pseudo_target_root,
		k=settings.hmm_k,
		survey_id=manifest.survey_id,
		labels=decoded.labels,
		confidence=decoded.confidence,
		valid_tokens=decoded.valid_tokens,
		metadata=metadata,
	)
	return BuiltPseudoTargetResult(
		survey_id=manifest.survey_id,
		labels_path=written.labels,
		confidence_path=written.confidence,
		valid_tokens_path=written.valid_tokens,
		metadata_path=written.metadata,
		valid_token_count=int(np.count_nonzero(decoded.valid_tokens)),
	)


class AmplitudeMAE3DProtocol(torch.nn.Module):
	"""Typing-only protocol substitute for the MAE attributes used here."""

	patch_size_xyz: XYZ
	encoder_dim: int

	def encode_tokens(
		self,
		x: torch.Tensor,
		*,
		valid_mask: torch.Tensor | None = None,
	) -> Mapping[str, object]:
		"""Return MAE token embeddings."""
		raise NotImplementedError


def _survey_logits(  # noqa: PLR0913
	manifest: SurveyManifest,
	*,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	model: torch.nn.Module,
	head: OrderedPrototypeHead,
	settings: _BuilderSettings,
	device: torch.device,
	token_grid: XYZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	mae = cast('AmplitudeMAE3DProtocol', model)
	logit_sum = np.zeros((*token_grid, settings.hmm_k), dtype=np.float32)
	counts = np.zeros(token_grid, dtype=np.uint32)
	windows = list(
		iter_sliding_windows(
			manifest.amplitude.shape_xyz,
			window_size_xyz=settings.window_size_xyz,
			overlap_xyz=settings.overlap_xyz,
			patch_size_xyz=mae.patch_size_xyz,
		),
	)
	for batch_start in range(0, len(windows), settings.batch_size):
		_process_window_batch(
			windows[batch_start : batch_start + settings.batch_size],
			manifest=manifest,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			model=mae,
			head=head,
			settings=settings,
			device=device,
			logit_sum=logit_sum,
			counts=counts,
		)
	valid_tokens = counts > 0
	if not np.any(valid_tokens):
		msg = f'survey {manifest.survey_id!r} produced no valid prototype logits'
		raise ValueError(msg)
	averaged = np.zeros_like(logit_sum, dtype=np.float32)
	averaged[valid_tokens] = logit_sum[valid_tokens] / counts[valid_tokens, None]
	return averaged, valid_tokens, counts


def _process_window_batch(  # noqa: PLR0913
	windows: Sequence[SlidingWindow],
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	model: AmplitudeMAE3DProtocol,
	head: OrderedPrototypeHead,
	settings: _BuilderSettings,
	device: torch.device,
	logit_sum: np.ndarray,
	counts: np.ndarray,
) -> None:
	prepared = [
		_read_window(
			window,
			manifest=manifest,
			amplitude_path=amplitude_path,
			stats=stats,
			store=store,
			settings=settings,
			patch_size_xyz=model.patch_size_xyz,
		)
		for window in windows
	]
	usable = [item for item in prepared if item[2].any()]
	if not usable:
		return
	x = torch.from_numpy(np.stack([item[1] for item in usable], axis=0)).to(
		device=device,
		dtype=_model_floating_dtype(model),
	)
	token_masks = torch.from_numpy(
		np.stack([item[2] for item in usable], axis=0),
	).to(device)
	with torch.no_grad():
		encoded = model.encode_tokens(x, valid_mask=token_masks)
		tokens = cast('torch.Tensor', encoded['tokens'])
		prototype_output = head(tokens)
	logits = prototype_output.logits.detach().to(dtype=torch.float32).cpu().numpy()
	window_token_shape = cast('tuple[int, int, int]', encoded['token_grid_shape'])
	for index, (window, _x, token_valid) in enumerate(usable):
		_add_window_logits(
			window,
			patch_size_xyz=model.patch_size_xyz,
			window_logits=logits[index].reshape(
				*window_token_shape,
				head.num_prototypes,
			),
			token_valid_mask=token_valid,
			logit_sum=logit_sum,
			counts=counts,
		)


def _read_window(  # noqa: PLR0913
	window: SlidingWindow,
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats: SurveyNormalizationStats,
	store: NpyMemmapVolumeStore,
	settings: _BuilderSettings,
	patch_size_xyz: XYZ,
) -> tuple[SlidingWindow, np.ndarray, np.ndarray]:
	request = CropRequest(
		survey_id=manifest.survey_id,
		start_xyz=window.start_xyz,
		size_xyz=window.size_xyz,
	)
	prepared = read_amplitude_crop(
		request=request,
		amplitude_path=amplitude_path,
		stats=stats,
		store=store,
		patch_size_xyz=patch_size_xyz,
		settings=AmplitudePreprocessSettings(
			zero_mask=settings.zero_mask,
			normalized_clip_abs=settings.normalized_clip_abs,
			amplitude_agc=settings.amplitude_agc,
			min_token_valid_fraction=settings.min_token_valid_fraction,
		),
	)
	return window, prepared.x, prepared.token_valid_mask


def _add_window_logits(  # noqa: PLR0913
	window: SlidingWindow,
	*,
	patch_size_xyz: XYZ,
	window_logits: np.ndarray,
	token_valid_mask: np.ndarray,
	logit_sum: np.ndarray,
	counts: np.ndarray,
) -> None:
	token_start = tuple(
		start // patch
		for start, patch in zip(window.start_xyz, patch_size_xyz, strict=True)
	)
	token_slices = tuple(
		slice(start, start + size)
		for start, size in zip(token_start, token_valid_mask.shape, strict=True)
	)
	region_sum = logit_sum[token_slices]
	region_counts = counts[token_slices]
	region_sum[token_valid_mask] += window_logits[token_valid_mask].astype(
		np.float32,
		copy=False,
	)
	region_counts[token_valid_mask] += np.uint32(1)


def _prepare_output_paths(
	paths: StratPseudoTargetPaths,
	*,
	request_metadata: Mapping[str, object],
	settings: _BuilderSettings,
) -> bool:
	"""Prepare final outputs and return true when a matching artifact is skipped."""
	paths.labels.parent.mkdir(parents=True, exist_ok=True)
	existing = [
		path
		for path in (paths.labels, paths.confidence, paths.valid_tokens, paths.metadata)
		if path.exists()
	]
	if not existing:
		return False
	complete_outputs = (
		paths.labels.exists()
		and paths.confidence.exists()
		and paths.valid_tokens.exists()
		and paths.metadata.exists()
	)
	if not complete_outputs:
		if not settings.overwrite:
			msg = (
				'incomplete existing pseudo-target output requires overwrite=True: '
				f'{paths.metadata}'
			)
			raise ValueError(msg)
		for path in existing:
			path.unlink()
		return False

	if complete_outputs and _metadata_request_matches(
		paths,
		request_metadata,
	):
		if settings.skip_existing:
			return True
		if not settings.overwrite:
			msg = (
				'existing pseudo-target output requires overwrite=True '
				f'or skip_existing=True: {paths.metadata}'
			)
			raise ValueError(msg)
	elif complete_outputs and not settings.overwrite:
		msg = (
			'existing pseudo-target output metadata does not match current settings; '
			'pass overwrite=True to replace: '
			f'{paths.metadata}'
		)
		raise ValueError(msg)

	for path in existing:
		path.unlink()
	return False


def _metadata_request_matches(
	paths: StratPseudoTargetPaths,
	request_metadata: Mapping[str, object],
) -> bool:
	try:
		metadata = load_pseudo_target_metadata(paths)
	except (OSError, TypeError, ValueError):
		return False
	source = metadata.get('source')
	if not isinstance(source, Mapping):
		return False
	return source.get('request') == dict(request_metadata)


def _skipped_result(
	survey_id: str,
	*,
	paths: StratPseudoTargetPaths,
	metadata: Mapping[str, object],
) -> BuiltPseudoTargetResult:
	valid_token_count = metadata.get('valid_token_count')
	if isinstance(valid_token_count, bool) or not isinstance(valid_token_count, int):
		valid_token_count = 0
	return BuiltPseudoTargetResult(
		survey_id=survey_id,
		labels_path=paths.labels,
		confidence_path=paths.confidence,
		valid_tokens_path=paths.valid_tokens,
		metadata_path=paths.metadata,
		valid_token_count=int(valid_token_count),
		skipped=True,
	)


def _pseudo_target_request_metadata(  # noqa: PLR0913
	*,
	manifest: SurveyManifest,
	amplitude_path: Path,
	stats_path: Path,
	settings: _BuilderSettings,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	checkpoint_sha256: str,
	checkpoint_payload: Mapping[str, object],
	token_grid: XYZ,
) -> dict[str, object]:
	return {
		'builder_schema_version': BUILDER_SCHEMA_VERSION,
		'checkpoint_path': str(settings.checkpoint_path),
		'checkpoint_sha256': checkpoint_sha256,
		'checkpoint_training_stage': _checkpoint_training_stage(checkpoint_payload),
		'command': 'build_strat_hmm_pseudo_targets',
		'head_config': dict(_head_config(stratigraphy_config)),
		'hmm': _hmm_metadata(settings),
		'inference': {
			'batch_size': settings.batch_size,
			'min_token_valid_fraction': settings.min_token_valid_fraction,
			'overlap': list(settings.overlap_xyz),
			'output_dtype': 'float32',
			'window_size': list(settings.window_size_xyz),
		},
		'mae_model': dict(cast('Mapping[str, object]', mae_config['model'])),
		'normalization_stats_path': str(stats_path),
		'preprocessing': {
			'amplitude_agc': settings.amplitude_agc.to_dict(),
			'normalized_clip_abs': settings.normalized_clip_abs,
			'zero_mask': _zero_mask_metadata(settings.zero_mask),
		},
		'source_amplitude_path': str(amplitude_path),
		'survey_id': manifest.survey_id,
		'token_grid_shape': list(token_grid),
		'volume_shape_xyz': list(manifest.amplitude.shape_xyz),
	}


def _pseudo_target_source_metadata(
	*,
	request_metadata: Mapping[str, object],
	count_array: np.ndarray,
	decode_metadata: Mapping[str, object],
) -> dict[str, object]:
	counted = count_array[count_array > 0]
	decode_valid = int(decode_metadata['effective_valid_token_count'])
	metadata = dict(request_metadata)
	metadata.update(
		{
			'confidence_summary': decode_metadata['confidence_summary'],
			'count_summary': {
				'max': int(counted.max()) if counted.size else 0,
				'mean': float(counted.mean()) if counted.size else 0.0,
				'min': int(counted.min()) if counted.size else 0,
			},
			'decode': dict(decode_metadata),
			'request': dict(request_metadata),
			'valid_summary': {
				'decoded_valid_token_count': decode_valid,
				'edge_margin_excluded_valid_token_count': int(
					decode_metadata['edge_margin_excluded_valid_token_count'],
				),
				'logit_valid_token_count': int(np.count_nonzero(count_array > 0)),
				'total_token_count': int(count_array.size),
			},
		},
	)
	return metadata


def _hmm_metadata(settings: _BuilderSettings) -> dict[str, object]:
	return {
		'edge_margin_tokens': list(settings.edge_margin_tokens),
		'expected_boundaries': (
			None
			if settings.expected_boundaries is None
			else asdict(settings.expected_boundaries)
		),
		'k': settings.hmm_k,
		'path_prior': (
			None if settings.path_prior is None else asdict(settings.path_prior)
		),
		'transition': asdict(settings.transition),
	}


def _zero_mask_metadata(config: ZeroMaskConfig) -> dict[str, object]:
	return {
		'enabled': config.enabled,
		'xy_trace_influence_radius': config.xy_trace_influence_radius,
		'z_sample_influence_radius': config.z_sample_influence_radius,
		'zero_atol': config.zero_atol,
	}


def _settings_from_config(
	config: Mapping[str, object],
	*,
	checkpoint_path: Path,
	mae_config: Mapping[str, object],
	overwrite: bool | None,
	skip_existing: bool | None,
) -> _BuilderSettings:
	inference = _mapping(config, 'inference')
	hmm = _mapping(config, 'hmm')
	outputs = _mapping(config, 'outputs')
	resolved_overwrite = (
		bool(outputs['overwrite']) if overwrite is None else bool(overwrite)
	)
	resolved_skip_existing = (
		bool(outputs.get('skip_existing', False))
		if skip_existing is None
		else bool(skip_existing)
	)
	if resolved_overwrite and resolved_skip_existing:
		msg = 'overwrite and skip_existing cannot both be true'
		raise ValueError(msg)
	path_prior = _path_prior_from_config(hmm)
	expected_boundaries = (
		None
		if path_prior is None or not path_prior.enabled
		else path_prior.expected_boundaries
	)
	return _BuilderSettings(
		checkpoint_path=checkpoint_path,
		pseudo_target_root=Path(cast('str', outputs['pseudo_target_root'])),
		window_size_xyz=_xyz_from_mapping(inference, 'window_size', 'inference'),
		overlap_xyz=_xyz_from_mapping(inference, 'overlap', 'inference'),
		batch_size=_positive_int(inference.get('batch_size'), 'inference.batch_size'),
		min_token_valid_fraction=float(inference['min_token_valid_fraction']),
		overwrite=resolved_overwrite,
		skip_existing=resolved_skip_existing,
		zero_mask=_zero_mask_from_config(mae_config),
		normalized_clip_abs=_normalized_clip_abs(mae_config),
		amplitude_agc=_amplitude_agc_from_config(mae_config),
		hmm_k=_positive_int(hmm.get('k'), 'hmm.k'),
		edge_margin_tokens=_xyz_from_mapping(
			hmm,
			'edge_margin_tokens',
			'hmm',
			default=(0, 0, 0),
		),
		transition=_transition_from_config(_mapping(hmm, 'transition')),
		path_prior=path_prior,
		expected_boundaries=expected_boundaries,
	)


def _transition_from_config(config: Mapping[str, object]) -> HMMTransitionSettings:
	return HMMTransitionSettings(
		same_cost=float(config['same_cost']),
		advance_cost=float(config['advance_cost']),
		jump_cost=float(config['jump_cost']),
		reverse_cost=float(config['reverse_cost']),
		forbid_reverse=bool(config['forbid_reverse']),
		max_jump=None if config['max_jump'] is None else int(config['max_jump']),
	)


def _path_prior_from_config(hmm: Mapping[str, object]) -> HMMPathPriorSettings | None:
	if 'path_prior' not in hmm:
		return None
	path_prior = _mapping(hmm, 'path_prior')
	if not bool(path_prior['enabled']):
		return None
	return HMMPathPriorSettings(
		enabled=True,
		initial_state=_anchor_prior(path_prior, 'initial_state'),
		terminal_state=_anchor_prior(path_prior, 'terminal_state'),
		expected_boundaries=_expected_boundaries(path_prior),
	)


def _anchor_prior(
	path_prior: Mapping[str, object],
	key: str,
) -> HMMAnchorPriorSettings:
	if key not in path_prior:
		return HMMAnchorPriorSettings(mode='none', weight=0.0)
	config = _mapping(path_prior, key)
	return HMMAnchorPriorSettings(
		mode=str(config['mode']),
		weight=float(config['weight']),
	)


def _expected_boundaries(
	path_prior: Mapping[str, object],
) -> HMMExpectedBoundariesSettings:
	if 'expected_boundaries' not in path_prior:
		return HMMExpectedBoundariesSettings(
			enabled=False,
			target='auto_k_minus_1',
			weight=0.0,
		)
	config = _mapping(path_prior, 'expected_boundaries')
	return HMMExpectedBoundariesSettings(
		enabled=bool(config['enabled']),
		target=cast('str | int', config.get('target', 'auto_k_minus_1')),
		weight=float(config.get('weight', 0.0)),
	)


def _checkpoint_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	config = payload.get('config')
	if not isinstance(config, Mapping):
		msg = 'checkpoint is missing config'
		raise TypeError(msg)
	return cast('Mapping[str, object]', config)


def _stratigraphy_config(payload: Mapping[str, object]) -> Mapping[str, object]:
	config = payload.get('stratigraphy_config')
	if not isinstance(config, Mapping):
		msg = 'checkpoint is missing stratigraphy_config'
		raise TypeError(msg)
	return cast('Mapping[str, object]', config)


def _model_state_dict(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
	state = payload.get('model_state_dict')
	if not isinstance(state, Mapping):
		msg = 'checkpoint is missing model_state_dict'
		raise TypeError(msg)
	return cast('Mapping[str, torch.Tensor]', state)


def _stratigraphy_state_dict(
	payload: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
	state = payload.get('stratigraphy_state_dict')
	if not isinstance(state, Mapping):
		msg = 'checkpoint is missing stratigraphy_state_dict'
		raise TypeError(msg)
	return cast('Mapping[str, torch.Tensor]', state)


def _head_config(stratigraphy_config: Mapping[str, object]) -> Mapping[str, object]:
	head = stratigraphy_config.get('head')
	if not isinstance(head, Mapping):
		msg = 'checkpoint stratigraphy_config is missing head'
		raise TypeError(msg)
	return cast('Mapping[str, object]', head)


def _checkpoint_floating_dtype(
	state_dict: Mapping[str, torch.Tensor],
) -> torch.dtype:
	dtypes = {
		tensor.dtype
		for tensor in state_dict.values()
		if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
	}
	if not dtypes:
		msg = 'checkpoint model_state_dict does not contain floating point tensors'
		raise ValueError(msg)
	if len(dtypes) != 1:
		msg = f'checkpoint model_state_dict has multiple floating dtypes: {dtypes!r}'
		raise ValueError(msg)
	return next(iter(dtypes))


def _model_floating_dtype(model: torch.nn.Module) -> torch.dtype:
	for parameter in model.parameters():
		if parameter.is_floating_point():
			return parameter.dtype
	for buffer in model.buffers():
		if buffer.is_floating_point():
			return buffer.dtype
	msg = 'model does not contain floating point tensors'
	raise ValueError(msg)


def _zero_mask_from_config(config: Mapping[str, object]) -> ZeroMaskConfig:
	value = config.get('zero_mask')
	if not isinstance(value, Mapping):
		msg = 'checkpoint config is missing zero_mask'
		raise TypeError(msg)
	zero_mask = ZeroMaskConfig(**dict(value))
	zero_mask.validate()
	return zero_mask


def _amplitude_agc_from_config(config: Mapping[str, object]) -> AmplitudeAgcConfig:
	data = _mapping(config, 'data')
	value = data.get('amplitude_agc')
	return AmplitudeAgcConfig.from_mapping(cast('Mapping[str, object] | None', value))


def _normalized_clip_abs(config: Mapping[str, object]) -> float | None:
	data = _mapping(config, 'data')
	value = data.get('normalized_clip_abs')
	return None if value is None else float(value)


def _checkpoint_training_stage(payload: Mapping[str, object]) -> object:
	training_state = payload.get('training_state')
	if isinstance(training_state, Mapping) and 'stage' in training_state:
		return training_state['stage']
	stratigraphy_config = payload.get('stratigraphy_config')
	if isinstance(stratigraphy_config, Mapping):
		return stratigraphy_config.get('stage')
	return None


def _resolve_device(
	device: str | None,
	config: Mapping[str, object],
) -> torch.device:
	value = device
	if value is None:
		value = str(_mapping(config, 'inference')['device'])
	if value == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	return torch.device(str(value))


def _checkpoint_path(config: Mapping[str, object]) -> Path:
	return Path(cast('str', _mapping(config, 'checkpoint')['path']))


def _manifest_path(config: Mapping[str, object]) -> Path:
	return Path(cast('str', _mapping(config, 'manifests')['train']))


def _xyz_from_config_section(
	config: Mapping[str, object],
	section: str,
	key: str,
) -> XYZ:
	return _xyz_from_mapping(_mapping(config, section), key, section)


def _xyz_from_mapping(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
	*,
	default: Sequence[int] | None = None,
) -> XYZ:
	value = parent.get(key, default)
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
	):
		msg = f'{prefix}.{key} must be a length-3 integer sequence'
		raise TypeError(msg)
	return (int(value[0]), int(value[1]), int(value[2]))


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return int(value)


def _optional_positive_int(value: object, name: str) -> int | None:
	if value is None:
		return None
	return _positive_int(value, name)


def _positive_float(value: object, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	if not np.isfinite(float(value)) or float(value) <= 0.0:
		msg = f'{name} must be positive and finite; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _bool_value(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


__all__ = [
	'BuiltPseudoTargetResult',
	'build_strat_hmm_pseudo_target_results',
	'build_strat_hmm_pseudo_targets',
]
