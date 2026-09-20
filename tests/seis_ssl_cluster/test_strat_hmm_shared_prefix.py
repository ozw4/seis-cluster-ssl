"""Exact encoder/optimizer parity and conservative shared-prefix fallbacks."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
import torch

from seis_ssl_cluster.models.common.transformer import (
	TransformerBlock,
	TransformerStack,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.stratigraphy import (
	MultiResolutionOrderedPrototypeHeads,
	OrderedPrototypeHead,
)
from seis_ssl_cluster.training.encoder_trainability import (
	freeze_all_and_unfreeze_top_encoder_blocks,
)
from seis_ssl_cluster.training.strat_hmm import epoch
from seis_ssl_cluster.training.strat_hmm.encoding import UnmaskedEncoderPair
from tests.seis_ssl_cluster.test_strat_center_trace_masked import _runtime_batch
from tests.seis_ssl_cluster.test_strat_hmm_synchronization import (
	_assert_same_run,
)
from tests.seis_ssl_cluster.test_strat_hmm_synchronization import (
	_run_epoch as _run_toy_epoch,
)

_CUDA = pytest.param(
	'cuda',
	marks=[
		pytest.mark.requires_cuda,
		pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable'),
	],
)


def _models(device, top=1, *, mode='once', encoder_dim=24, patch_size=(1, 1, 1)):
	torch.manual_seed(101)
	student = AmplitudeMAE3D(
		patch_size_xyz=patch_size,
		encoder_dim=encoder_dim,
		encoder_depth=8,
		encoder_heads=4,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=2,
		runtime_check_mode=mode,
	).to(device)
	teacher = deepcopy(student).requires_grad_(requires_grad=False).eval()
	freeze_all_and_unfreeze_top_encoder_blocks(student, unfreeze_top_blocks=top)
	epoch._set_student_training_mode(student)  # noqa: SLF001
	return student, teacher


def _separate(student, teacher, x, mask):
	with torch.set_grad_enabled(any(p.requires_grad for p in student.parameters())):
		encoded = student.encode_tokens(x, valid_mask=mask)
	teacher.eval()
	with torch.no_grad():
		teacher_encoded = teacher.encode_tokens(x, valid_mask=mask)
	return encoded, teacher_encoded


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('top', [0, 1, 3, 8])
@pytest.mark.parametrize('mode', ['strict', 'once', 'minimal'])
def test_shared_forward_gradients_updates_and_rng_match_separate(
	device, top, mode, monkeypatch
):
	student, teacher = _models(device, top, mode=mode)
	baseline_student, baseline_teacher = deepcopy(student), deepcopy(teacher)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	x = torch.randn(2, 1, 2, 2, 2, device=device)
	mask = torch.ones(2, 2, 2, 2, dtype=torch.bool, device=device)
	mask[0, 0, 0, 0] = False
	optimizers = [
		torch.optim.AdamW(model.parameters(), lr=1e-3)
		for model in (student, baseline_student)
	]
	calls = []
	forward = TransformerBlock._forward_unchecked  # noqa: SLF001

	def counted(block, *args):
		calls.append(id(block))
		return forward(block, *args)

	monkeypatch.setattr(TransformerBlock, '_forward_unchecked', counted)
	for _step in range(3):
		calls.clear()
		rng = torch.get_rng_state()
		cuda_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
		actual = pair.encode(x, valid_mask=mask)
		assert len(calls) == 8 + top
		assert pair.prefix_layers == 8 - top
		assert torch.equal(torch.get_rng_state(), rng)
		if cuda_rng is not None:
			assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
		expected = _separate(baseline_student, baseline_teacher, x, mask)
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
		assert not actual[1]['tokens'].requires_grad
		if top:
			for outputs, optimizer in zip((actual, expected), optimizers, strict=True):
				optimizer.zero_grad(set_to_none=True)
				features, targets = (result['tokens'] for result in outputs)
				loss = (
					features.square().mean()
					+ 0.2 * (features - targets).square().mean()
				)
				loss.backward()
				optimizer.step()
			for left, right in zip(
				student.parameters(), baseline_student.parameters(), strict=True
			):
				torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
				if not left.requires_grad:
					assert left.grad is None
			torch.testing.assert_close(
				optimizers[0].state_dict(), optimizers[1].state_dict(), rtol=0, atol=0
			)
		torch.testing.assert_close(
			student.state_dict(), baseline_student.state_dict(), rtol=0, atol=0
		)
		torch.testing.assert_close(
			teacher.state_dict(), baseline_teacher.state_dict(), rtol=0, atol=0
		)


@pytest.mark.parametrize(
	'mutation',
	[
		'prefix_weight',
		'projection_weight',
		'requires_grad',
		'train_mode',
		'norm_eps',
		'hook',
		'replace_parameter',
		'replace_block',
		'position_cache',
	],
)
@pytest.mark.parametrize('when', ['before', 'after'])
def test_mismatched_or_changed_prefix_uses_original_path(mutation, when):  # noqa: C901
	student, teacher = _models('cpu')
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	if when == 'after' or mutation == 'position_cache':
		pair.encode(x, valid_mask=mask)
		assert pair.prefix_layers == 7
	with torch.no_grad():
		if mutation == 'prefix_weight':
			teacher.encoder.layers[0].norm1.weight.add_(0.1)
		elif mutation == 'projection_weight':
			teacher.patch_projection.weight.add_(0.1)
		elif mutation == 'requires_grad':
			student.patch_projection.weight.requires_grad_(requires_grad=True)
		elif mutation == 'train_mode':
			student.encoder.layers[0].train()
		elif mutation == 'norm_eps':
			teacher.encoder.layers[0].norm1.eps = 0.1
		elif mutation == 'hook':
			teacher.encoder.layers[0].register_forward_hook(
				lambda _module, _args, output: output + 0.1
			)
		elif mutation == 'replace_parameter':
			teacher.patch_projection.weight = torch.nn.Parameter(
				teacher.patch_projection.weight.clone(), requires_grad=False
			)
		elif mutation == 'replace_block':
			teacher.encoder.layers[0] = deepcopy(teacher.encoder.layers[0])
		elif mutation == 'position_cache':
			next(iter(teacher._position_embedding_cache.values())).add_(0.1)  # noqa: SLF001
	expected = _separate(deepcopy(student), deepcopy(teacher), x, mask)
	actual = pair.encode(x, valid_mask=mask)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	# Replacing an identical parameter/module before discovery remains eligible.
	if when == 'after' or mutation not in {'replace_parameter', 'replace_block'}:
		assert pair.prefix_layers == 0


def test_input_gradients_use_original_path():
	student, teacher = _models('cpu')
	x = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	baseline_x = x.detach().clone().requires_grad_(requires_grad=True)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	actual = pair.encode(x, valid_mask=mask)
	expected = _separate(deepcopy(student), deepcopy(teacher), baseline_x, mask)
	actual[0]['tokens'].square().mean().backward()
	expected[0]['tokens'].square().mean().backward()
	torch.testing.assert_close(x.grad, baseline_x.grad, rtol=0, atol=0)
	assert pair.prefix_layers == 0


@pytest.mark.parametrize('cache', ['matching', 'different', 'only_student'])
def test_prepopulated_position_cache_is_verified(cache):
	student, teacher = _models('cpu')
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	student.encode_tokens(x, valid_mask=mask)
	if cache != 'only_student':
		teacher.encode_tokens(x, valid_mask=mask)
	if cache == 'different':
		next(iter(teacher._position_embedding_cache.values())).add_(0.1)  # noqa: SLF001
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	actual = pair.encode(x, valid_mask=mask)
	expected = _separate(deepcopy(student), deepcopy(teacher), x, mask)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	assert pair.prefix_layers == (7 if cache == 'matching' else 0)


def test_global_hooks_keep_original_module_calls():
	student, teacher = _models('cpu')
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	calls = []
	handle = torch.nn.modules.module.register_module_forward_hook(
		lambda module, _args, _output: calls.append(type(module))
	)
	try:
		actual = pair.encode(x, valid_mask=mask)
		actual_calls = calls.copy()
		calls.clear()
		expected = _separate(student, teacher, x, mask)
	finally:
		handle.remove()
	assert actual_calls == calls
	assert pair.prefix_layers == 0
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_custom_encoder_without_layers_uses_original_path():
	class CustomEncoder(torch.nn.Module):
		def forward(self, tokens, _mask=None):
			return tokens * 2

	student, teacher = _models('cpu')
	student.encoder = CustomEncoder()
	teacher.encoder = CustomEncoder()
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	actual = pair.encode(x, valid_mask=mask)
	expected = _separate(student, teacher, x, mask)
	torch.testing.assert_close(actual, expected, rtol=0, atol=0)
	assert pair.prefix_layers == 0


@pytest.mark.parametrize('use_teacher', [False, True])
def test_absent_teacher_retains_optional_and_required_behavior(use_teacher):
	student, _teacher = _models('cpu')
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	pair = UnmaskedEncoderPair(student, None, use_teacher=use_teacher)
	if use_teacher:
		with pytest.raises(ValueError, match='teacher is required'):
			pair.encode(x, valid_mask=mask)
	else:
		encoded, teacher_encoded = pair.encode(x, valid_mask=mask)
		assert teacher_encoded is None
		torch.testing.assert_close(
			encoded, student.encode_tokens(x, valid_mask=mask), rtol=0, atol=0
		)


@pytest.mark.parametrize('bad_mask', ['dtype', 'shape', 'empty'])
def test_invalid_masks_preserve_exception_type_and_message(bad_mask):
	student, teacher = _models('cpu', mode='strict')
	x = torch.randn(1, 1, 2, 2, 2)
	mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
	if bad_mask == 'dtype':
		mask = mask.float()
	elif bad_mask == 'shape':
		mask = mask[:, :1]
	else:
		mask.zero_()
	with pytest.raises((ValueError, TypeError)) as expected:
		_separate(student, teacher, x, mask)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	with pytest.raises(type(expected.value)) as actual:
		pair.encode(x, valid_mask=mask)
	assert str(actual.value) == str(expected.value)


def test_center_trace_masked_route_never_uses_shared_prefix(monkeypatch):
	def reject(*_args, **_kwargs):
		pytest.fail('replacement-masked inputs must not use the shared encoder pair')

	monkeypatch.setattr(UnmaskedEncoderPair, 'encode', reject)
	_run_toy_epoch('center_trace', 'cpu', grad_clip_norm=0.1)


@pytest.mark.parametrize('device', ['cpu', _CUDA])
@pytest.mark.parametrize('route', ['head_only', 'multi_head', 'posterior'])
def test_real_encoder_epoch_metrics_and_state_match_original(
	monkeypatch, device, route
):
	actual = _run_epoch(device, route)
	monkeypatch.setattr(UnmaskedEncoderPair, '_check_prefix', lambda _self: None)
	expected = _run_epoch(device, route)
	_assert_same_run(actual, expected)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('route', ['head_only', 'multi_head'])
def test_amp_epoch_matches_original(monkeypatch, route):
	actual = _run_epoch('cuda', route, amp=True)
	monkeypatch.setattr(UnmaskedEncoderPair, '_check_prefix', lambda _self: None)
	expected = _run_epoch('cuda', route, amp=True)
	_assert_same_run(actual, expected)


@pytest.mark.slow
@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.skipif(
	os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'},
	reason='deterministic CUDA comparison requires CUBLAS_WORKSPACE_CONFIG',
)
def test_4096_token_cuda_updates_match_with_deterministic_algorithms():
	previous = torch.are_deterministic_algorithms_enabled()
	warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
	torch.use_deterministic_algorithms(mode=True)
	try:
		_compare_large_cuda_updates()
	finally:
		torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _compare_large_cuda_updates():
	student, teacher = _models('cuda', encoder_dim=384, patch_size=(8, 8, 8))
	baseline_student, baseline_teacher = deepcopy(student), deepcopy(teacher)
	pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
	x = torch.randn(1, 1, 128, 128, 128, device='cuda')
	mask = torch.ones(1, 128, 128, 128, dtype=torch.bool, device='cuda')
	mask[:, :8, :8, :8] = False
	optimizers = [
		torch.optim.AdamW(model.parameters(), lr=1e-5)
		for model in (student, baseline_student)
	]
	for _step in range(2):
		actual = pair.encode(x, valid_mask=mask)
		expected = _separate(baseline_student, baseline_teacher, x, mask)
		torch.testing.assert_close(actual, expected, rtol=0, atol=0)
		for outputs, optimizer in zip((actual, expected), optimizers, strict=True):
			optimizer.zero_grad(set_to_none=True)
			features, targets = (result['tokens'] for result in outputs)
			loss = features.square().mean() + 0.2 * (features - targets).square().mean()
			loss.backward()
			optimizer.step()
		for left, right in zip(
			student.parameters(), baseline_student.parameters(), strict=True
		):
			torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
		torch.testing.assert_close(
			student.state_dict(), baseline_student.state_dict(), rtol=0, atol=0
		)
		torch.testing.assert_close(
			optimizers[0].state_dict(), optimizers[1].state_dict(), rtol=0, atol=0
		)
	assert pair.prefix_layers == 7


def _run_epoch(device, route, *, amp=False):
	student, teacher = _models(device)
	head_type = (
		OrderedPrototypeHead
		if route == 'head_only'
		else MultiResolutionOrderedPrototypeHeads
	)
	head_options = {'num_prototypes': 6} if route == 'head_only' else {'ks': (6, 8, 10)}
	head = head_type(
		feature_dim=24,
		projection_dim=12,
		temperature=0.5,
		normalize=True,
		**head_options,
	).to(device)
	optimizer = torch.optim.AdamW(
		[p for model in (student, head) for p in model.parameters() if p.requires_grad],
		lr=1e-3,
	)
	batches = [_runtime_batch() for _ in range(3)]
	for batch in batches:
		if route == 'head_only':
			target = batch.pop('strat_multi_targets')['k6']
			batch.update({f'strat_{key}': value for key, value in target.items()})
		elif route == 'posterior':
			batch['strat_multi_posteriors'] = {
				key: {
					'posterior': torch.nn.functional.one_hot(
						target['labels'], int(key[1:])
					).float(),
					'valid_mask': target['valid_mask'],
				}
				for key, target in batch.pop('strat_multi_targets').items()
			}
	callbacks = []
	scaler = torch.amp.GradScaler('cuda', init_scale=16) if amp else None
	kwargs = {'head': head} if route == 'head_only' else {'heads': head}
	if route == 'posterior':
		kwargs['target_representation'] = 'ordered_path_state_posterior_v1'
	train = (
		epoch.train_strat_hmm_head_only_one_epoch
		if route == 'head_only'
		else epoch.train_strat_hmm_multi_head_one_epoch
	)
	state = train(
		student=student,
		teacher=teacher,
		dataloader=batches,
		optimizer=optimizer,
		device=torch.device(device),
		epoch=1,
		loss_config={
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'distillation_weight': 0.2,
		},
		pseudo_target_config={},
		grad_clip_norm=0.1,
		amp_enabled=amp,
		scaler=scaler,
		step_callback=callbacks.append,
		**kwargs,
	)
	return (
		(state, callbacks),
		deepcopy([model.state_dict() for model in (student, head, teacher)]),
		deepcopy(optimizer.state_dict()),
		None if scaler is None else scaler.state_dict(),
		torch.get_rng_state(),
		torch.cuda.get_rng_state() if device == 'cuda' else None,
	)


@pytest.mark.parametrize('bounds', [(-1, 2), (2, 1), (0, 9), (True, 1), (0, 1.5)])
def test_transformer_rejects_invalid_layer_range(bounds):
	stack = TransformerStack(embed_dim=12, num_heads=2, depth=8)
	with pytest.raises(ValueError, match='layer range'):
		stack(torch.ones(1, 2, 12), start_layer=bounds[0], stop_layer=bounds[1])


def test_transformer_split_and_empty_ranges_match_full_stack():
	stack = TransformerStack(embed_dim=12, num_heads=2, depth=8)
	x = torch.randn(1, 2, 12)
	torch.testing.assert_close(
		stack(stack(x, stop_layer=7), start_layer=7), stack(x), rtol=0, atol=0
	)
	assert stack(x, start_layer=8) is x
