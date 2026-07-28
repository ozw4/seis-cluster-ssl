"""Batch loss orchestration for stratigraphic HMM pretext training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from typing import NamedTuple

import torch

from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
	expected_normalized_order_coordinate,
	feature_distillation_loss,
	soft_categorical_cross_entropy,
	structured_hmm_prototype_loss,
	usage_entropy_floor_loss,
)
from seis_ssl_cluster.training.strat_hmm.runtime import (
	_float_config,
	_required_tensor,
)


def compute_strat_hmm_pretext_losses(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	head: OrderedPrototypeHead,
	encoded: Mapping[str, object],
	teacher_encoded: Mapping[str, object] | None,
	batch: Mapping[str, object],
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
	"""Compute strat HMM losses and valid-supervised-token weight metrics.

	The three boundary/effective-weight metrics use the supervised ``valid_mask``
	after student-token and minimum-confidence filtering as their denominator.
	They are zero when that mask is empty.
	"""
	tokens = _encoded_tokens(encoded)
	if not bool(torch.isfinite(tokens).all().item()):
		raise FloatingPointError('non-finite student encoded tokens')
	if teacher_encoded is not None and not bool(
		torch.isfinite(_encoded_tokens(teacher_encoded)).all().item()
	):
		raise FloatingPointError('non-finite teacher encoded tokens')
	prototype_weight = _float_config(loss_config, 'prototype_weight', 1.0)
	usage_weight = _float_config(loss_config, 'usage_weight', 0.0)
	logits = (
		head(tokens).logits if prototype_weight > 0.0 or usage_weight > 0.0 else None
	)
	if logits is not None and not bool(torch.isfinite(logits).all().item()):
		raise FloatingPointError('non-finite ordered prototype logits')
	reference = logits if logits is not None else tokens
	labels = _flatten_token_tensor(
		_required_tensor(batch, 'strat_labels'),
		reference,
		'strat_labels',
	).long()
	confidence = _flatten_token_tensor(
		_required_tensor(batch, 'strat_confidence'),
		reference,
		'strat_confidence',
	)
	boundary_weight = _flatten_token_tensor(
		_required_tensor(batch, 'strat_boundary_weight'),
		reference,
		'strat_boundary_weight',
	)
	_validate_weight_tensor_pair(confidence, boundary_weight, reference)
	confidence = confidence.to(dtype=reference.dtype)
	boundary_weight = boundary_weight.to(dtype=reference.dtype)
	valid_mask = _flatten_token_tensor(
		_required_tensor(batch, 'strat_valid_mask'),
		reference,
		'strat_valid_mask',
	).bool()
	distillation_valid_mask = valid_mask
	token_valid_mask = encoded.get('token_valid_mask')
	if token_valid_mask is not None:
		student_valid_mask = _encoded_token_valid_mask(token_valid_mask, reference)
		valid_mask = valid_mask & student_valid_mask
		distillation_valid_mask = distillation_valid_mask & student_valid_mask
	if teacher_encoded is not None:
		teacher_token_valid_mask = teacher_encoded.get('token_valid_mask')
		if teacher_token_valid_mask is not None:
			teacher_valid_mask = _encoded_token_valid_mask(
				teacher_token_valid_mask,
				reference,
			)
			distillation_valid_mask = distillation_valid_mask & teacher_valid_mask
	min_confidence = _float_config(pseudo_target_config, 'min_confidence', 0.0)
	if min_confidence > 0.0:
		valid_mask = valid_mask & confidence.ge(min_confidence)

	if prototype_weight > 0.0:
		if logits is None:  # pragma: no cover - guarded by logits construction
			raise AssertionError('prototype logits were not computed')
		prototype_loss = structured_hmm_prototype_loss(
			logits,
			labels,
			valid_mask=valid_mask,
			confidence=confidence,
			boundary_weight=boundary_weight,
		)
	else:
		prototype_loss = tokens.new_zeros(())
	if usage_weight > 0.0:
		if logits is None:  # pragma: no cover - guarded by logits construction
			raise AssertionError('prototype logits were not computed')
		probs = torch.nn.functional.softmax(logits, dim=-1)
		entropy_floor = loss_config.get('entropy_floor')
		if entropy_floor is None:
			# Prompt-07 default: a weak half-uniform floor if usage loss is enabled.
			entropy_floor_value = 0.5 * math.log(logits.shape[-1])
		else:
			entropy_floor_value = float(entropy_floor)
		usage_loss = usage_entropy_floor_loss(
			probs,
			valid_mask=valid_mask,
			entropy_floor=entropy_floor_value,
		)
	else:
		probs = (
			torch.nn.functional.softmax(logits, dim=-1) if logits is not None else None
		)
		usage_loss = tokens.new_zeros(())
	distillation_weight = _float_config(loss_config, 'distillation_weight', 0.0)
	if distillation_weight > 0.0:
		if teacher_encoded is None:
			msg = 'teacher encoded tokens are required for feature distillation'
			raise ValueError(msg)
		distillation_loss = feature_distillation_loss(
			tokens,
			_encoded_tokens(teacher_encoded),
			valid_mask=distillation_valid_mask,
		)
	else:
		distillation_loss = tokens.new_zeros(())
	total_loss = (
		prototype_weight * prototype_loss
		+ usage_weight * usage_loss
		+ distillation_weight * distillation_loss
	)
	effective_weight = confidence * boundary_weight
	valid_weight_count = valid_mask.sum()
	valid_weight_denominator = valid_weight_count.clamp_min(1).to(
		dtype=reference.dtype,
	)
	return {
		'loss': total_loss,
		'loss_prototype': prototype_loss,
		'loss_usage': usage_loss,
		'loss_distillation': distillation_loss,
		'valid_supervised_token_fraction': valid_mask.float().mean(),
		'valid_distillation_token_fraction': distillation_valid_mask.float().mean(),
		'mean_boundary_weight_valid': (
			boundary_weight[valid_mask].sum() / valid_weight_denominator
		),
		'mean_effective_prototype_weight': (
			effective_weight[valid_mask].sum() / valid_weight_denominator
		),
		'positive_effective_weight_fraction': (
			(effective_weight[valid_mask] > 0).sum().to(dtype=reference.dtype)
			/ valid_weight_denominator
		),
		'target_usage_entropy': _target_usage_entropy(
			labels,
			valid_mask,
			num_prototypes=head.num_prototypes,
		),
		'prototype_usage_entropy': (
			_prototype_usage_entropy(probs, valid_mask)
			if probs is not None
			else tokens.new_zeros(())
		),
	}


def compute_strat_hmm_multi_head_losses(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	heads: MultiResolutionOrderedPrototypeHeads,
	encoded: Mapping[str, object],
	teacher_encoded: Mapping[str, object] | None,
	batch: Mapping[str, object],
	loss_config: Mapping[str, object],
	pseudo_target_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
	"""Compute equally weighted ordered-prototype losses across resolutions."""
	tokens = _encoded_tokens(encoded)
	if not bool(torch.isfinite(tokens).all().item()):
		raise FloatingPointError('non-finite student encoded tokens')
	if teacher_encoded is not None and not bool(
		torch.isfinite(_encoded_tokens(teacher_encoded)).all().item()
	):
		raise FloatingPointError('non-finite teacher encoded tokens')
	prototype_weight = _float_config(loss_config, 'prototype_weight', 1.0)
	usage_weight = _float_config(loss_config, 'usage_weight', 0.0)
	consistency_weight = _float_config(loss_config, 'consistency_weight', 0.0)
	consistency_beta = _float_config(loss_config, 'consistency_beta', 0.1)
	if consistency_weight < 0.0:
		raise ValueError('consistency_weight must be nonnegative')
	if consistency_beta <= 0.0:
		raise ValueError('consistency_beta must be positive')

	outputs = heads(tokens)
	head_ks = heads.head_ks
	head_keys = tuple(_head_key(k) for k in head_ks)
	if outputs.head_ks != head_ks or tuple(outputs.outputs) != head_keys:
		raise ValueError('multi-head model output does not match configured heads')
	targets = _multi_head_targets(batch, head_keys)
	min_confidence = _float_config(pseudo_target_config, 'min_confidence', 0.0)

	reference = tokens
	student_valid_mask = _multi_head_student_valid_mask(encoded, reference)
	shared_pseudo_valid_mask: torch.Tensor | None = None
	shared_pseudo_valid_shape: tuple[int, ...] | None = None
	head_values: dict[str, _MultiHeadTargetValues] = {}
	for k, head_key in zip(head_ks, head_keys, strict=True):
		logits = outputs.outputs[head_key].logits
		if not bool(torch.isfinite(logits).all().item()):
			raise FloatingPointError(f'non-finite multi-head logits for {head_key}')
		if logits.shape[-1] != k:
			raise ValueError(
				f'multi-head {head_key!r} logits last dimension must equal {k}',
			)
		values = _multi_head_target_values(
			targets[head_key],
			reference=logits,
			head_key=head_key,
		)
		_validate_multi_head_labels(
			values.labels,
			valid_mask=values.valid_mask,
			num_prototypes=k,
			head_key=head_key,
		)
		if shared_pseudo_valid_mask is None:
			shared_pseudo_valid_mask = values.valid_mask
			shared_pseudo_valid_shape = tuple(values.valid_mask.shape)
		elif tuple(values.valid_mask.shape) != shared_pseudo_valid_shape:
			raise ValueError('all multi-head valid mask shapes must match')
		elif not torch.equal(values.valid_mask, shared_pseudo_valid_mask):
			raise ValueError('all multi-head valid masks must match')
		head_values[head_key] = values

	if shared_pseudo_valid_mask is None:  # pragma: no cover - model needs two heads
		raise AssertionError('multi-head targets were unexpectedly empty')
	distillation_valid_mask = shared_pseudo_valid_mask
	if student_valid_mask is not None:
		distillation_valid_mask = distillation_valid_mask & student_valid_mask
	if teacher_encoded is not None:
		teacher_valid_mask = _multi_head_student_valid_mask(
			teacher_encoded,
			reference,
		)
		if teacher_valid_mask is not None:
			distillation_valid_mask = distillation_valid_mask & teacher_valid_mask

	entropy_floor = loss_config.get('entropy_floor')
	prototype_losses: list[torch.Tensor] = []
	usage_losses: list[torch.Tensor] = []
	supervised_valid_fractions: list[torch.Tensor] = []
	result: dict[str, torch.Tensor] = {}
	for k, head_key in zip(head_ks, head_keys, strict=True):
		logits = outputs.outputs[head_key].logits
		values = head_values[head_key]
		valid_mask = values.valid_mask
		if student_valid_mask is not None:
			valid_mask = valid_mask & student_valid_mask
		if min_confidence > 0.0:
			valid_mask = valid_mask & values.confidence.ge(min_confidence)
		probs = torch.nn.functional.softmax(logits, dim=-1)
		if prototype_weight > 0.0 and bool(valid_mask.any().item()):
			prototype_loss = structured_hmm_prototype_loss(
				logits,
				values.labels,
				valid_mask=valid_mask,
				confidence=values.confidence,
				boundary_weight=values.boundary_weight,
			)
		else:
			prototype_loss = _graph_zero(logits)
		if usage_weight > 0.0 and bool(valid_mask.any().item()):
			entropy_floor_value = (
				0.5 * math.log(k) if entropy_floor is None else float(entropy_floor)
			)
			usage_loss = usage_entropy_floor_loss(
				probs,
				valid_mask=valid_mask,
				entropy_floor=entropy_floor_value,
			)
		else:
			usage_loss = _graph_zero(logits)
		prototype_losses.append(prototype_loss)
		usage_losses.append(usage_loss)
		supervised_valid_fractions.append(
			_safe_fraction(valid_mask, dtype=tokens.dtype),
		)
		result[f'loss_prototype_{head_key}'] = prototype_loss
		result[f'loss_usage_{head_key}'] = usage_loss
		result[f'target_usage_entropy_{head_key}'] = _target_usage_entropy(
			values.labels,
			valid_mask,
			num_prototypes=k,
		)
		result[f'prototype_usage_entropy_{head_key}'] = _prototype_usage_entropy(
			probs,
			valid_mask,
		)
		result[f'mean_confidence_valid_{head_key}'] = _masked_mean(
			values.confidence,
			valid_mask,
		)
		head_values[head_key] = values._replace(valid_mask=valid_mask)

	consistency_losses: list[torch.Tensor] = []
	eligible_pair_count = 0
	for first_k, second_k in combinations(sorted(head_ks), 2):
		first_key = _head_key(first_k)
		second_key = _head_key(second_k)
		first = head_values[first_key]
		second = head_values[second_key]
		pair_valid = first.valid_mask & second.valid_mask
		# ``sqrt(a) * sqrt(b)`` is algebraically the requested geometric mean,
		# while ``sqrt(a * b)`` can overflow for otherwise valid float inputs.
		pair_weight = torch.sqrt(first.confidence) * torch.sqrt(second.confidence)
		pair_name = f'{first_key}_{second_key}'
		if bool(pair_weight[pair_valid].detach().gt(0.0).any().item()):
			first_coordinate = expected_normalized_order_coordinate(
				outputs.outputs[first_key].logits,
			)
			second_coordinate = expected_normalized_order_coordinate(
				outputs.outputs[second_key].logits,
			)
			error = torch.nn.functional.smooth_l1_loss(
				first_coordinate,
				second_coordinate,
				beta=consistency_beta,
				reduction='none',
			)
			pair_loss = _stable_weighted_mean(error, pair_weight, pair_valid)
			eligible_pair_count += 1
		else:
			pair_loss = _graph_zero(outputs.outputs[first_key].logits) + _graph_zero(
				outputs.outputs[second_key].logits,
			)
		consistency_losses.append(pair_loss)
		result[f'loss_consistency_{pair_name}'] = pair_loss
		result[f'mean_consistency_weight_{pair_name}'] = _masked_mean(
			pair_weight,
			pair_valid,
		)
		result[f'valid_consistency_token_fraction_{pair_name}'] = _safe_fraction(
			pair_valid,
			dtype=tokens.dtype,
		)

	prototype_loss = torch.stack(prototype_losses).mean()
	usage_loss = torch.stack(usage_losses).mean()
	consistency_loss = (
		torch.stack(consistency_losses).sum() / eligible_pair_count
		if eligible_pair_count > 0
		else torch.stack(consistency_losses).sum()
	)
	distillation_weight = _float_config(loss_config, 'distillation_weight', 0.0)
	if distillation_weight > 0.0:
		if teacher_encoded is None:
			raise ValueError(
				'teacher encoded tokens are required for feature distillation',
			)
		distillation_loss = (
			feature_distillation_loss(
				tokens,
				_encoded_tokens(teacher_encoded),
				valid_mask=distillation_valid_mask,
			)
			if bool(distillation_valid_mask.any().item())
			else _graph_zero(tokens)
		)
	else:
		distillation_loss = _graph_zero(tokens)
	result.update(
		{
			'loss': (
				prototype_weight * prototype_loss
				+ usage_weight * usage_loss
				+ consistency_weight * consistency_loss
				+ distillation_weight * distillation_loss
			),
			'loss_prototype': prototype_loss,
			'loss_usage': usage_loss,
			'loss_consistency': consistency_loss,
			'loss_distillation': distillation_loss,
			'valid_supervised_token_fraction': torch.stack(
				supervised_valid_fractions,
			).mean(),
			'valid_distillation_token_fraction': _safe_fraction(
				distillation_valid_mask,
				dtype=tokens.dtype,
			),
			'consistency_eligible_pair_count': tokens.new_tensor(
				eligible_pair_count,
			),
		},
	)
	return result


def compute_strat_hmm_multi_head_posterior_losses(  # noqa: C901, PLR0912, PLR0915
	*,
	heads: MultiResolutionOrderedPrototypeHeads,
	encoded: Mapping[str, object],
	teacher_encoded: Mapping[str, object] | None,
	batch: Mapping[str, object],
	loss_config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
	"""Compute M5-U's equal-head soft posterior objective.

	This deliberately has no confidence, boundary, temperature, or consistency
	path: the frozen posterior itself is the complete categorical target.
	"""
	tokens = _encoded_tokens(encoded)
	if not bool(torch.isfinite(tokens).all().item()):
		raise FloatingPointError('non-finite student encoded tokens')
	if teacher_encoded is not None and not bool(
		torch.isfinite(_encoded_tokens(teacher_encoded)).all().item()
	):
		raise FloatingPointError('non-finite teacher encoded tokens')
	if heads.head_ks != (6, 8, 10):
		raise ValueError('soft posterior training requires canonical heads (6, 8, 10)')
	consistency_weight = _float_config(loss_config, 'consistency_weight', 0.0)
	if consistency_weight != 0.0:
		raise ValueError('consistency_weight must be zero for soft posterior training')
	usage_weight = _float_config(loss_config, 'usage_weight', 0.0)
	distillation_weight = _float_config(loss_config, 'distillation_weight', 0.0)
	outputs = heads(tokens)
	head_keys = tuple(_head_key(k) for k in heads.head_ks)
	if outputs.head_ks != heads.head_ks or tuple(outputs.outputs) != head_keys:
		raise ValueError('multi-head model output does not match configured heads')
	targets = _multi_head_posteriors(batch, head_keys)
	student_valid_mask = _multi_head_student_valid_mask(encoded, tokens)
	shared_pseudo_valid: torch.Tensor | None = None
	posterior_values: dict[str, _MultiHeadPosteriorValues] = {}
	for k, head_key in zip(heads.head_ks, head_keys, strict=True):
		logits = outputs.outputs[head_key].logits
		if not bool(torch.isfinite(logits).all().item()):
			raise FloatingPointError(f'non-finite multi-head logits for {head_key}')
		if logits.shape[-1] != k:
			raise ValueError(
				f'multi-head {head_key!r} logits last dimension must equal {k}',
			)
		values = _multi_head_posterior_values(
			targets[head_key], reference=logits, head_key=head_key
		)
		if shared_pseudo_valid is None:
			shared_pseudo_valid = values.valid_mask
		elif not torch.equal(shared_pseudo_valid, values.valid_mask):
			raise ValueError('all multi-head posterior valid masks must match')
		posterior_values[head_key] = values
	if shared_pseudo_valid is None:  # pragma: no cover - canonical heads are nonempty
		raise AssertionError('multi-head posterior targets were unexpectedly empty')
	distillation_valid = shared_pseudo_valid
	if student_valid_mask is not None:
		distillation_valid = distillation_valid & student_valid_mask
	if teacher_encoded is not None:
		teacher_valid_mask = _multi_head_student_valid_mask(teacher_encoded, tokens)
		if teacher_valid_mask is not None:
			distillation_valid = distillation_valid & teacher_valid_mask

	entropy_floor = loss_config.get('entropy_floor')
	prototype_losses: list[torch.Tensor] = []
	usage_losses: list[torch.Tensor] = []
	result: dict[str, torch.Tensor] = {}
	valid_supervised = shared_pseudo_valid
	if student_valid_mask is not None:
		valid_supervised = valid_supervised & student_valid_mask
	for k, head_key in zip(heads.head_ks, head_keys, strict=True):
		logits = outputs.outputs[head_key].logits
		values = posterior_values[head_key]
		effective_posterior = values.posterior.masked_fill(
			~valid_supervised.unsqueeze(-1), 0.0
		)
		prototype_loss = soft_categorical_cross_entropy(
			logits, effective_posterior, valid_mask=valid_supervised
		)
		probs = torch.nn.functional.softmax(logits, dim=-1)
		if usage_weight > 0.0 and bool(valid_supervised.any().item()):
			entropy_floor_value = (
				0.5 * math.log(k) if entropy_floor is None else float(entropy_floor)
			)
			usage_loss = usage_entropy_floor_loss(
				probs, valid_mask=valid_supervised, entropy_floor=entropy_floor_value
			)
		else:
			usage_loss = _graph_zero(logits)
		target_entropy = _posterior_target_entropy(
			effective_posterior, valid_supervised
		)
		prototype_losses.append(prototype_loss)
		usage_losses.append(usage_loss)
		result[f'loss_prototype_{head_key}'] = prototype_loss
		result[f'loss_usage_{head_key}'] = usage_loss
		result[f'target_entropy_{head_key}'] = target_entropy
		result[f'prototype_kl_{head_key}'] = prototype_loss - target_entropy
		result[f'prototype_usage_entropy_{head_key}'] = _prototype_usage_entropy(
			probs, valid_supervised
		)
	prototype_loss = torch.stack(prototype_losses).mean()
	usage_loss = torch.stack(usage_losses).mean()
	if distillation_weight > 0.0:
		if teacher_encoded is None:
			raise ValueError(
				'teacher encoded tokens are required for feature distillation',
			)
		distillation_loss = (
			feature_distillation_loss(
				tokens,
				_encoded_tokens(teacher_encoded),
				valid_mask=distillation_valid,
			)
			if bool(distillation_valid.any().item())
			else _graph_zero(tokens)
		)
	else:
		distillation_loss = _graph_zero(tokens)
	result.update(
		{
			'loss': prototype_loss
			+ usage_weight * usage_loss
			+ distillation_weight * distillation_loss,
			'loss_prototype': prototype_loss,
			'loss_usage': usage_loss,
			'loss_distillation': distillation_loss,
			'valid_supervised_token_fraction': _safe_fraction(
				valid_supervised, dtype=tokens.dtype
			),
			'valid_distillation_token_fraction': _safe_fraction(
				distillation_valid, dtype=tokens.dtype
			),
		},
	)
	return result


class _MultiHeadTargetValues(NamedTuple):
	labels: torch.Tensor
	confidence: torch.Tensor
	boundary_weight: torch.Tensor
	valid_mask: torch.Tensor


class _MultiHeadPosteriorValues(NamedTuple):
	posterior: torch.Tensor
	valid_mask: torch.Tensor


def _head_key(k: int) -> str:
	return f'k{k}'


def _multi_head_targets(
	batch: Mapping[str, object],
	head_keys: tuple[str, ...],
) -> Mapping[str, object]:
	targets = batch.get('strat_multi_targets')
	if not isinstance(targets, Mapping):
		raise TypeError('strat_multi_targets must be a mapping')
	if set(targets) != set(head_keys):
		raise ValueError(
			'strat_multi_targets keys must exactly match multi-head model keys; '
			f'got {tuple(targets)!r}, expected {head_keys!r}',
		)
	return targets


def _multi_head_posteriors(
	batch: Mapping[str, object],
	head_keys: tuple[str, ...],
) -> Mapping[str, object]:
	posteriors = batch.get('strat_multi_posteriors')
	if not isinstance(posteriors, Mapping):
		raise TypeError('strat_multi_posteriors must be a mapping')
	if tuple(posteriors) != head_keys:
		raise ValueError(
			'strat_multi_posteriors keys must match multi-head model order; '
			f'got {tuple(posteriors)!r}, expected {head_keys!r}',
		)
	return posteriors


def _multi_head_target_values(
	target: object,
	*,
	reference: torch.Tensor,
	head_key: str,
) -> _MultiHeadTargetValues:
	if not isinstance(target, Mapping):
		raise TypeError(f'strat_multi_targets[{head_key!r}] must be a mapping')
	required_keys = {'labels', 'confidence', 'boundary_weight', 'valid_mask'}
	if set(target) != required_keys:
		raise ValueError(
			f'strat_multi_targets[{head_key!r}] must contain exactly '
			"'labels', 'confidence', 'boundary_weight', and 'valid_mask'",
		)
	raw_labels = _required_tensor(target, 'labels')
	raw_confidence = _required_tensor(target, 'confidence')
	raw_boundary_weight = _required_tensor(target, 'boundary_weight')
	raw_valid_mask = _required_tensor(target, 'valid_mask')
	if (
		tuple(raw_labels.shape) != tuple(raw_confidence.shape)
		or tuple(raw_confidence.shape) != tuple(raw_boundary_weight.shape)
		or tuple(raw_confidence.shape) != tuple(raw_valid_mask.shape)
	):
		raise ValueError(
			f'multi-head {head_key!r} labels, confidence, boundary weight, and '
			'valid mask shapes must match',
		)
	labels = _flatten_token_tensor(
		raw_labels,
		reference,
		f'strat_multi_targets[{head_key!r}].labels',
	).long()
	confidence = _flatten_token_tensor(
		raw_confidence,
		reference,
		f'strat_multi_targets[{head_key!r}].confidence',
	)
	boundary_weight = _flatten_token_tensor(
		raw_boundary_weight,
		reference,
		f'strat_multi_targets[{head_key!r}].boundary_weight',
	)
	valid_mask = _flatten_token_tensor(
		raw_valid_mask,
		reference,
		f'strat_multi_targets[{head_key!r}].valid_mask',
	).bool()
	_validate_weight_tensor_pair(confidence, boundary_weight, reference)
	if not bool(torch.isfinite(confidence).all().item()):
		raise ValueError(f'multi-head {head_key!r} confidence must be finite')
	if bool(confidence.lt(0.0).any().item()):
		raise ValueError(f'multi-head {head_key!r} confidence must be nonnegative')
	if not bool(torch.isfinite(boundary_weight).all().item()):
		raise ValueError(f'multi-head {head_key!r} boundary weight must be finite')
	if bool(boundary_weight.lt(0.0).any().item()):
		raise ValueError(
			f'multi-head {head_key!r} boundary weight must be nonnegative',
		)
	if not bool(torch.all(boundary_weight[valid_mask] == 1.0).item()):
		raise ValueError('multi-head boundary weight must be one for every valid token')
	return _MultiHeadTargetValues(
		labels=labels,
		confidence=confidence.to(dtype=reference.dtype).detach(),
		boundary_weight=boundary_weight.to(dtype=reference.dtype),
		valid_mask=valid_mask,
	)


def _multi_head_posterior_values(
	target: object,
	*,
	reference: torch.Tensor,
	head_key: str,
) -> _MultiHeadPosteriorValues:
	if not isinstance(target, Mapping):
		raise TypeError(f'strat_multi_posteriors[{head_key!r}] must be a mapping')
	if tuple(target) != ('posterior', 'valid_mask'):
		raise ValueError(
			f'strat_multi_posteriors[{head_key!r}] must contain ordered '
			"'posterior' and 'valid_mask' fields",
		)
	posterior = _required_tensor(target, 'posterior')
	valid_mask = _required_tensor(target, 'valid_mask')
	if not torch.is_floating_point(posterior):
		raise TypeError(
			f'multi-head {head_key!r} posterior must be floating point',
		)
	if posterior.device != reference.device:
		raise ValueError(
			f'multi-head {head_key!r} posterior must be on logits device',
		)
	if (
		posterior.shape[0] != reference.shape[0]
		or posterior.shape[-1] != reference.shape[-1]
	):
		raise ValueError(
			f'multi-head {head_key!r} posterior batch/K dimensions must match logits',
		)
	flattened = posterior.reshape(reference.shape[0], -1, posterior.shape[-1])
	if tuple(flattened.shape) != tuple(reference.shape):
		raise ValueError(
			f'multi-head {head_key!r} posterior token count must match logits',
		)
	if valid_mask.dtype != torch.bool:
		raise TypeError(
			f'multi-head {head_key!r} valid_mask must have dtype torch.bool',
		)
	if valid_mask.device != reference.device:
		raise ValueError(
			f'multi-head {head_key!r} valid_mask must be on logits device',
		)
	flattened_valid = _flatten_token_tensor(
		valid_mask, reference, f'strat_multi_posteriors[{head_key!r}].valid_mask'
	).bool()
	return _MultiHeadPosteriorValues(
		posterior=flattened.to(dtype=reference.dtype).detach(),
		valid_mask=flattened_valid,
	)


def _multi_head_student_valid_mask(
	encoded: Mapping[str, object],
	reference: torch.Tensor,
) -> torch.Tensor | None:
	value = encoded.get('token_valid_mask')
	return _encoded_token_valid_mask(value, reference) if value is not None else None


def _validate_multi_head_labels(
	labels: torch.Tensor,
	*,
	valid_mask: torch.Tensor,
	num_prototypes: int,
	head_key: str,
) -> None:
	"""Reject invalid pseudo-target labels before loss weights are considered."""
	selected_labels = labels[valid_mask]
	if bool(selected_labels.lt(0).any().item()) or bool(
		selected_labels.ge(num_prototypes).any().item(),
	):
		raise ValueError(
			f'multi-head {head_key!r} valid labels must be in prototype range '
			f'[0, {num_prototypes})',
		)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	if not bool(mask.any().item()):
		return values.new_zeros(())
	selected = values[mask]
	scale = selected.abs().max()
	if bool(scale.eq(0.0).item()):
		return scale
	return scale * (selected / scale).mean()


def _stable_weighted_mean(
	values: torch.Tensor,
	weights: torch.Tensor,
	mask: torch.Tensor,
) -> torch.Tensor:
	"""Return a finite weighted mean without summing unbounded weights."""
	selected_values = values[mask]
	selected_weights = weights[mask]
	scale = selected_weights.max()
	if bool(scale.le(0.0).item()):
		return _graph_zero(values)
	normalized_weights = selected_weights / scale
	return (normalized_weights * selected_values).sum() / normalized_weights.sum()


def _graph_zero(reference: torch.Tensor) -> torch.Tensor:
	"""Return an exact zero that retains ``reference``'s autograd path."""
	return reference.sum() * 0.0


def _safe_fraction(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
	if mask.numel() == 0:
		return torch.zeros((), device=mask.device, dtype=dtype)
	return mask.to(dtype=dtype).mean()


def _encoded_tokens(encoded: Mapping[str, object]) -> torch.Tensor:
	value = encoded.get('tokens')
	if not isinstance(value, torch.Tensor):
		msg = 'encoded output is missing tensor key "tokens"'
		raise TypeError(msg)
	return value


def _encoded_token_valid_mask(value: object, logits: torch.Tensor) -> torch.Tensor:
	if not isinstance(value, torch.Tensor):
		msg = 'encoded token_valid_mask must be a tensor or None'
		raise TypeError(msg)
	mask = value.bool()
	if tuple(mask.shape) != tuple(logits.shape[:-1]):
		msg = (
			'encoded token_valid_mask shape must match token logits prefix; '
			f'got {tuple(mask.shape)!r}, expected {tuple(logits.shape[:-1])!r}'
		)
		raise ValueError(msg)
	return mask


def _flatten_token_tensor(
	tensor: torch.Tensor,
	logits: torch.Tensor,
	name: str,
) -> torch.Tensor:
	if tensor.shape[0] != logits.shape[0]:
		msg = f'{name} batch dimension must match logits'
		raise ValueError(msg)
	flattened = tensor.reshape(logits.shape[0], -1)
	if flattened.shape[1] != logits.shape[1]:
		msg = (
			f'{name} flattened token count must match logits token dimension; '
			f'got {flattened.shape[1]}, expected {logits.shape[1]}'
		)
		raise ValueError(msg)
	return flattened


def _validate_weight_tensor_pair(
	confidence: torch.Tensor,
	boundary_weight: torch.Tensor,
	reference: torch.Tensor,
) -> None:
	for name, tensor in (
		('strat_confidence', confidence),
		('strat_boundary_weight', boundary_weight),
	):
		if not torch.is_floating_point(tensor):
			msg = f'{name} must be floating point; got {tensor.dtype}'
			raise TypeError(msg)
		if tensor.device != reference.device:
			msg = (
				f'{name} must be on the encoded-token device; '
				f'got {tensor.device}, expected {reference.device}'
			)
			raise ValueError(msg)
	if boundary_weight.dtype != confidence.dtype:
		msg = (
			'strat_boundary_weight dtype must match strat_confidence dtype; '
			f'got {boundary_weight.dtype}, expected {confidence.dtype}'
		)
		raise TypeError(msg)


def _target_usage_entropy(
	labels: torch.Tensor,
	valid_mask: torch.Tensor,
	*,
	num_prototypes: int,
) -> torch.Tensor:
	selected = labels[valid_mask & labels.ge(0) & labels.lt(num_prototypes)]
	if selected.numel() == 0:
		return labels.new_tensor(0.0, dtype=torch.float32)
	counts = torch.bincount(
		selected.clamp_min(0),
		minlength=num_prototypes,
	).to(dtype=torch.float32, device=labels.device)
	probs = counts / counts.sum().clamp_min(1.0)
	return -(probs * (probs + 1.0e-8).log()).sum()


def _prototype_usage_entropy(
	probs: torch.Tensor,
	valid_mask: torch.Tensor,
) -> torch.Tensor:
	selected = probs.reshape(-1, probs.shape[-1])[valid_mask.reshape(-1)]
	if selected.numel() == 0:
		return probs.new_zeros(())
	q_bar = selected.mean(dim=0)
	return -(q_bar * (q_bar + 1.0e-8).log()).sum()


def _posterior_target_entropy(
	posterior: torch.Tensor,
	valid_mask: torch.Tensor,
) -> torch.Tensor:
	"""Return the arithmetic mean entropy of detached valid posterior rows."""
	if not bool(valid_mask.any().item()):
		return posterior.sum() * 0.0
	selected = posterior.detach()[valid_mask]
	return -(selected * selected.clamp_min(1.0e-12).log()).sum(dim=-1).mean()


_strat_head_losses = compute_strat_hmm_pretext_losses


__all__ = [
	'compute_strat_hmm_multi_head_losses',
	'compute_strat_hmm_multi_head_posterior_losses',
	'compute_strat_hmm_pretext_losses',
]
