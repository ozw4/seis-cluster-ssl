"""Deterministically shuffle valid strat-HMM pseudo-target assignments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seis_ssl_cluster.stratigraphy.targets import (
	StratPseudoTargetArrays,
	StratPseudoTargetInput,
	StratPseudoTargetPaths,
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	pseudo_target_paths,
	validate_pseudo_target_arrays,
	write_pseudo_target,
)

GLOBAL_VALID_TOKENS = 'global_valid_tokens'


@dataclass(frozen=True)
class ShuffledPseudoTargetResult:
	"""Files written for one shuffled survey pseudo-target."""

	survey_id: str
	paths: StratPseudoTargetPaths
	valid_token_count: int


def shuffle_pseudo_target_arrays(  # noqa: PLR0913
	labels: np.ndarray,
	confidence: np.ndarray,
	valid_tokens: np.ndarray,
	*,
	k: int,
	seed: int,
	survey_id: str | None = None,
) -> StratPseudoTargetArrays:
	"""Shuffle label/confidence pairs over valid positions in one survey."""
	_validate_seed(seed)
	validate_pseudo_target_arrays(
		labels,
		confidence,
		valid_tokens,
		k=k,
		survey_id=survey_id,
	)
	labels_array = np.asarray(labels)
	confidence_array = np.asarray(confidence)
	valid_array = np.asarray(valid_tokens)
	shuffled_labels = labels_array.copy(order='K')
	shuffled_confidence = confidence_array.copy(order='K')
	valid_flat_indices = np.flatnonzero(valid_array)
	permutation = np.random.default_rng(seed).permutation(valid_flat_indices.size)

	source_labels = labels_array.flat[valid_flat_indices].copy()
	source_confidence = confidence_array.flat[valid_flat_indices].copy()
	shuffled_labels.flat[valid_flat_indices] = source_labels[permutation]
	shuffled_confidence.flat[valid_flat_indices] = source_confidence[permutation]

	result = StratPseudoTargetArrays(
		labels=shuffled_labels,
		confidence=shuffled_confidence,
		valid_tokens=valid_array.copy(),
		boundary_weight=valid_array.astype(np.float32),
	)
	_assert_preserved(
		labels_array,
		valid_array,
		result,
		k=k,
		survey_id=survey_id,
	)
	return result


def plan_shuffled_hmm_pseudo_targets(
	source_root: str | Path,
	output_root: str | Path,
	*,
	k: int,
	overwrite: bool = False,
) -> list[StratPseudoTargetPaths]:
	"""Validate shuffle inputs and return the output files that would be written."""
	inputs, resolved_output = _prepare_roots(source_root, output_root, k=k)
	outputs = [
		pseudo_target_paths(resolved_output, k=k, survey_id=item.survey_id)
		for item in inputs
	]
	_prepare_outputs(outputs, overwrite=overwrite)
	for item in inputs:
		load_pseudo_target_arrays(item, mmap_mode='r')
		load_pseudo_target_metadata(item)
	return outputs


def shuffle_strat_hmm_pseudo_targets(  # noqa: PLR0913
	source_root: str | Path,
	output_root: str | Path,
	*,
	k: int,
	seed: int,
	mode: str = GLOBAL_VALID_TOKENS,
	overwrite: bool = False,
	preserve_label_confidence_pairs: bool = True,
) -> list[ShuffledPseudoTargetResult]:
	"""Build shuffled pseudo-target artifacts for every survey under one root."""
	_validate_seed(seed)
	if mode != GLOBAL_VALID_TOKENS:
		raise ValueError(
			f'shuffle mode must be {GLOBAL_VALID_TOKENS!r}; got {mode!r}',
		)
	if not preserve_label_confidence_pairs:
		raise ValueError('label-confidence pairs must be preserved')
	inputs, resolved_output = _prepare_roots(source_root, output_root, k=k)
	output_paths = [
		pseudo_target_paths(resolved_output, k=k, survey_id=item.survey_id)
		for item in inputs
	]
	_prepare_outputs(output_paths, overwrite=overwrite)
	resolved_source = Path(source_root).resolve(strict=True)

	return [
		_shuffle_input(
			item,
			output_root=resolved_output,
			seed=seed,
			mode=mode,
			source_root=resolved_source,
		)
		for item in inputs
	]


def _shuffle_input(
	item: StratPseudoTargetInput,
	*,
	output_root: Path,
	seed: int,
	mode: str,
	source_root: Path,
) -> ShuffledPseudoTargetResult:
	source_arrays = load_pseudo_target_arrays(item)
	shuffled = shuffle_pseudo_target_arrays(
		source_arrays.labels,
		source_arrays.confidence,
		source_arrays.valid_tokens,
		k=item.k,
		seed=seed,
		survey_id=item.survey_id,
	)
	source_metadata = load_pseudo_target_metadata(item)
	paths = write_pseudo_target(
		output_root,
		k=item.k,
		survey_id=item.survey_id,
		labels=shuffled.labels,
		confidence=shuffled.confidence,
		valid_tokens=shuffled.valid_tokens,
		boundary_weight=shuffled.boundary_weight,
		metadata=source_metadata,
	)
	_add_shuffle_metadata(
		paths.metadata,
		mode=mode,
		seed=seed,
		source_root=source_root,
	)
	return ShuffledPseudoTargetResult(
		survey_id=item.survey_id,
		paths=paths,
		valid_token_count=int(np.count_nonzero(shuffled.valid_tokens)),
	)


def _prepare_roots(
	source_root: str | Path,
	output_root: str | Path,
	*,
	k: int,
) -> tuple[list[StratPseudoTargetInput], Path]:
	resolved_source = Path(source_root).resolve(strict=True)
	if not resolved_source.is_dir():
		raise NotADirectoryError(
			f'pseudo-target source root is not a directory: {resolved_source}',
		)
	resolved_output = Path(output_root).resolve(strict=False)
	if resolved_output == resolved_source:
		raise ValueError('shuffled pseudo-target output must not overwrite its source')
	if resolved_output.exists() and not resolved_output.is_dir():
		raise NotADirectoryError(
			f'pseudo-target output root is not a directory: {resolved_output}',
		)
	return discover_pseudo_target_inputs(resolved_source, k=k), resolved_output


def _prepare_outputs(
	outputs: list[StratPseudoTargetPaths],
	*,
	overwrite: bool,
) -> None:
	expected = {
		path
		for output in outputs
		for path in (
			output.labels,
			output.confidence,
			output.valid_tokens,
			output.boundary_weight,
			output.metadata,
		)
	}
	output_dir = outputs[0].labels.parent
	artifact_patterns = (
		'*.hmm_labels_token.npy',
		'*.hmm_confidence_token.npy',
		'*.valid_tokens.npy',
		'*.hmm_boundary_weight_token.npy',
		'*.pseudo_target_metadata.json',
	)
	unplanned = sorted(
		{
			path
			for pattern in artifact_patterns
			for path in output_dir.glob(pattern)
			if path not in expected
		},
	)
	if unplanned:
		raise FileExistsError(
			'shuffled pseudo-target output contains unplanned survey artifacts: '
			+ ', '.join(str(path) for path in unplanned),
		)
	existing = [
		path for path in sorted(expected) if path.exists()
	]
	if existing and not overwrite:
		raise FileExistsError(
			'shuffled pseudo-target outputs already exist; use overwrite: '
			+ ', '.join(str(path) for path in existing),
		)


def _assert_preserved(
	source_labels: np.ndarray,
	source_valid: np.ndarray,
	shuffled: StratPseudoTargetArrays,
	*,
	k: int,
	survey_id: str | None,
) -> None:
	if not np.array_equal(source_valid, shuffled.valid_tokens):
		raise AssertionError('valid-token mask changed during pseudo-target shuffle')
	source_counts = np.bincount(source_labels[source_valid], minlength=k)
	shuffled_counts = np.bincount(
		shuffled.labels[shuffled.valid_tokens],
		minlength=k,
	)
	if not np.array_equal(source_counts, shuffled_counts):
		raise AssertionError('label counts changed during pseudo-target shuffle')
	if np.any(shuffled.confidence[~shuffled.valid_tokens] != 0.0):
		raise AssertionError('invalid-token confidence changed during shuffle')
	validate_pseudo_target_arrays(
		shuffled.labels,
		shuffled.confidence,
		shuffled.valid_tokens,
		boundary_weight=shuffled.boundary_weight,
		k=k,
		survey_id=survey_id,
	)


def _add_shuffle_metadata(
	path: Path,
	*,
	mode: str,
	seed: int,
	source_root: Path,
) -> None:
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['shuffle'] = {
		'enabled': True,
		'label_counts_preserved': True,
		'mode': mode,
		'preserve_label_confidence_pairs': True,
		'seed': seed,
		'source_pseudo_target_root': str(source_root),
		'valid_mask_preserved': True,
	}
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _validate_seed(seed: int) -> None:
	if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
		raise ValueError(f'shuffle seed must be a nonnegative integer; got {seed!r}')


__all__ = [
	'GLOBAL_VALID_TOKENS',
	'ShuffledPseudoTargetResult',
	'plan_shuffled_hmm_pseudo_targets',
	'shuffle_pseudo_target_arrays',
	'shuffle_strat_hmm_pseudo_targets',
]
