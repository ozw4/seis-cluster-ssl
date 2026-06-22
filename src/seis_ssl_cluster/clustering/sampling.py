"""Deterministic sampling of valid embedding tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	count_valid_tokens,
	extract_token_features,
	valid_flat_indices,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


@dataclass(frozen=True)
class SampledTokens:
	"""Sampled training features and their source token indices."""

	features: np.ndarray
	per_survey_token_indices: dict[str, np.ndarray]
	requested_count: int
	total_valid_count: int
	sample_count: int


def sample_valid_embedding_tokens(
	embedding_inputs: Sequence[EmbeddingInput],
	*,
	sample_tokens: int,
	seed: int,
) -> SampledTokens:
	"""Sample valid tokens deterministically across one or more surveys."""
	if sample_tokens <= 0:
		msg = f'sample_tokens must be positive; got {sample_tokens!r}'
		raise ValueError(msg)
	if not embedding_inputs:
		msg = 'at least one embedding input is required'
		raise ValueError(msg)

	valid_counts = [count_valid_tokens(item) for item in embedding_inputs]
	total_valid = int(sum(valid_counts))
	if total_valid == 0:
		msg = 'cannot cluster embeddings because no valid tokens were found'
		raise ValueError(msg)

	sample_count = min(int(sample_tokens), total_valid)
	rng = np.random.default_rng(seed)
	selected_global = np.sort(
		rng.choice(total_valid, size=sample_count, replace=False),
	)
	per_survey_indices: dict[str, np.ndarray] = {}
	feature_blocks: list[np.ndarray] = []
	offset = 0
	for item, count in zip(embedding_inputs, valid_counts, strict=True):
		stop = offset + count
		mask = (selected_global >= offset) & (selected_global < stop)
		local_valid_ordinals = selected_global[mask] - offset
		if local_valid_ordinals.size:
			all_valid_indices = valid_flat_indices(item)
			token_indices = all_valid_indices[local_valid_ordinals]
			per_survey_indices[item.survey_id] = token_indices
			feature_blocks.append(extract_token_features(item, token_indices))
		else:
			per_survey_indices[item.survey_id] = np.empty(0, dtype=np.int64)
		offset = stop

	features = np.concatenate(feature_blocks, axis=0)
	return SampledTokens(
		features=np.asarray(features, dtype=np.float32),
		per_survey_token_indices=per_survey_indices,
		requested_count=int(sample_tokens),
		total_valid_count=total_valid,
		sample_count=sample_count,
	)


__all__ = ['SampledTokens', 'sample_valid_embedding_tokens']
