from __future__ import annotations

import pytest
import torch

import seis_ssl_cluster.models.common.transformer as transformer_module
from seis_ssl_cluster.models.common import TransformerStack
from seis_ssl_cluster.runtime_checks import RuntimeChecks


@pytest.mark.parametrize(
	('mode', 'expected_calls', 'second_behavior'),
	[
		('strict', 2, 'raises'),
		('once', 1, 'skips'),
		('minimal', 0, 'skips'),
	],
)
def test_runtime_value_check_execution_policy(
	mode: str,
	expected_calls: int,
	second_behavior: str,
) -> None:
	checks = RuntimeChecks(mode)  # type: ignore[arg-type]
	calls = 0

	def condition(value: int) -> torch.Tensor:
		nonlocal calls
		calls += 1
		return torch.tensor(value, dtype=torch.bool)

	checks.check('value', lambda: condition(1), error=ValueError('invalid'))
	if second_behavior == 'raises':
		with pytest.raises(ValueError, match='invalid'):
			checks.check(
				'value',
				lambda: condition(0),
				error=ValueError('invalid'),
			)
	else:
		checks.check(
			'value',
			lambda: condition(0),
			error=ValueError('invalid'),
		)

	assert calls == expected_calls


@pytest.mark.parametrize('mode', ['strict', 'once'])
def test_runtime_value_check_failure_on_first_execution(mode: str) -> None:
	checks = RuntimeChecks(mode)  # type: ignore[arg-type]

	with pytest.raises(ValueError, match='invalid'):
		checks.check(
			'value',
			lambda: torch.zeros((), dtype=torch.bool),
			error=ValueError('invalid'),
		)


def test_transformer_stack_validates_once_at_public_boundary(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	stack = TransformerStack(embed_dim=8, num_heads=2, depth=3)
	original = transformer_module._validate_tokens  # noqa: SLF001
	calls = 0

	def counted(tokens: torch.Tensor, embed_dim: int) -> tuple[int, int]:
		nonlocal calls
		calls += 1
		return original(tokens, embed_dim)

	monkeypatch.setattr(transformer_module, '_validate_tokens', counted)
	stack(torch.zeros((1, 2, 8)))

	assert calls == 1


def test_transformer_stack_preserves_layer_module_call_semantics(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	stack = TransformerStack(embed_dim=8, num_heads=2, depth=1)
	layer = stack.layers[0]
	original_forward = layer.forward
	forward_calls = 0
	hook_calls = 0

	def overridden_forward(
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		nonlocal forward_calls
		forward_calls += 1
		return original_forward(tokens, key_padding_mask)

	def forward_hook(
		_module: torch.nn.Module,
		_inputs: tuple[object, ...],
		_output: torch.Tensor,
	) -> None:
		nonlocal hook_calls
		hook_calls += 1

	monkeypatch.setattr(layer, 'forward', overridden_forward)
	layer.register_forward_hook(forward_hook)
	stack(torch.zeros((1, 2, 8)))

	assert forward_calls == 1
	assert hook_calls == 1


def test_minimal_mode_keeps_structural_transformer_checks() -> None:
	stack = TransformerStack(
		embed_dim=8,
		num_heads=2,
		depth=1,
		runtime_check_mode='minimal',
	)

	with pytest.raises(TypeError, match='dtype must be bool'):
		stack(
			torch.zeros((1, 2, 8)),
			torch.zeros((1, 2)),
		)
