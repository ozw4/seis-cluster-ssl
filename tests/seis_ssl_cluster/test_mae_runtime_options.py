from __future__ import annotations

# ruff: noqa: FBT001
import pytest
import torch

import seis_ssl_cluster.training.mae as mae_training
from seis_ssl_cluster.training.collate import move_batch_to_device
from seis_ssl_cluster.training.dataloaders import build_mae_dataloader


def test_mae_dataloader_disables_worker_only_options_with_no_workers() -> None:
	dataloader = build_mae_dataloader(
		[{'x': torch.zeros(1)}],
		batch_size=1,
		num_workers=0,
		device='cpu',
		prefetch_factor=4,
		persistent_workers=True,
	)

	assert dataloader.num_workers == 0
	assert dataloader.pin_memory is False
	assert dataloader.prefetch_factor is None
	assert dataloader.persistent_workers is False


def test_mae_dataloader_propagates_cuda_worker_options() -> None:
	dataloader = build_mae_dataloader(
		[{'x': torch.zeros(1)}],
		batch_size=1,
		num_workers=1,
		device='cuda',
		prefetch_factor=4,
		persistent_workers=True,
	)

	assert dataloader.num_workers == 1
	assert dataloader.pin_memory is True
	assert dataloader.prefetch_factor == 4
	assert dataloader.persistent_workers is True


@pytest.mark.parametrize(
	('device', 'pinned', 'expected'),
	[
		(torch.device('cpu'), True, False),
		(torch.device('cuda'), False, False),
		(torch.device('cuda'), True, True),
	],
)
def test_batch_transfer_uses_non_blocking_only_for_pinned_cuda_tensors(
	monkeypatch: pytest.MonkeyPatch,
	device: torch.device,
	pinned: bool,
	expected: bool,
) -> None:
	calls: list[bool] = []
	tensor = torch.zeros(1)
	monkeypatch.setattr(torch.Tensor, 'is_pinned', lambda _: pinned)
	monkeypatch.setattr(
		torch.Tensor,
		'to',
		lambda self, _device, *, non_blocking: (
			calls.append(non_blocking) or self
		),
	)

	result = move_batch_to_device(
		{'x': tensor, 'metadata': 'kept'},
		device,
		non_blocking=True,
	)

	assert result == {'x': tensor, 'metadata': 'kept'}
	assert calls == [expected]


@pytest.mark.parametrize('amp_dtype', ['auto', 'bfloat16', 'float16'])
def test_cpu_amp_requests_resolve_to_float32(amp_dtype: str) -> None:
	precision = mae_training._resolve_amp_precision(  # noqa: SLF001
		{'amp': True, 'amp_dtype': amp_dtype},
		device=torch.device('cpu'),
	)

	assert precision.resolved_dtype == 'float32'
	assert precision.amp_enabled is False
	assert precision.autocast_dtype is None
	assert precision.scaler_enabled is False


def test_disabled_amp_resolves_to_float32_without_querying_cuda(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def unexpected_bf16_query() -> bool:
		raise AssertionError

	monkeypatch.setattr(torch.cuda, 'is_bf16_supported', unexpected_bf16_query)
	precision = mae_training._resolve_amp_precision(  # noqa: SLF001
		{'amp': False, 'amp_dtype': 'auto'},
		device=torch.device('cuda'),
	)

	assert precision.resolved_dtype == 'float32'
	assert precision.amp_enabled is False
	assert precision.scaler_enabled is False


@pytest.mark.parametrize(
	('bf16_supported', 'expected_dtype', 'expected_scaler'),
	[(True, 'bfloat16', False), (False, 'float16', True)],
)
def test_auto_amp_selects_cuda_precision_without_allocating_cuda(
	monkeypatch: pytest.MonkeyPatch,
	bf16_supported: bool,
	expected_dtype: str,
	expected_scaler: bool,
) -> None:
	monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: bf16_supported)

	precision = mae_training._resolve_amp_precision(  # noqa: SLF001
		{'amp': True, 'amp_dtype': 'auto'},
		device=torch.device('cuda'),
	)

	assert precision.resolved_dtype == expected_dtype
	assert precision.scaler_enabled is expected_scaler


@pytest.mark.parametrize(
	('amp_dtype', 'expected_dtype', 'expected_scaler'),
	[('bfloat16', 'bfloat16', False), ('float16', 'float16', True)],
)
def test_explicit_cuda_amp_precision_and_scaler_selection(
	monkeypatch: pytest.MonkeyPatch,
	amp_dtype: str,
	expected_dtype: str,
	expected_scaler: bool,
) -> None:
	monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)

	precision = mae_training._resolve_amp_precision(  # noqa: SLF001
		{'amp': True, 'amp_dtype': amp_dtype},
		device=torch.device('cuda'),
	)

	assert precision.resolved_dtype == expected_dtype
	assert precision.scaler_enabled is expected_scaler


def test_explicit_unsupported_bfloat16_fails_early(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)

	with pytest.raises(ValueError, match='bfloat16 is not supported'):
		mae_training._resolve_amp_precision(  # noqa: SLF001
			{'amp': True, 'amp_dtype': 'bfloat16'},
			device=torch.device('cuda'),
		)


def test_runtime_metadata_records_resolved_transfer_and_precision() -> None:
	dataloader = build_mae_dataloader(
		[{'x': torch.zeros(1)}],
		batch_size=1,
		num_workers=0,
		device='cpu',
	)
	precision = mae_training._resolve_amp_precision(  # noqa: SLF001
		{'amp': True, 'amp_dtype': 'auto'},
		device=torch.device('cpu'),
	)

	metadata = mae_training._mae_runtime_metadata(  # noqa: SLF001
		precision=precision,
		dataloader=dataloader,
		device=torch.device('cpu'),
	)

	assert metadata['precision'] == {
		'amp_requested': True,
		'amp_dtype_requested': 'auto',
		'resolved_dtype': 'float32',
		'amp_enabled': False,
		'grad_scaler_enabled': False,
	}
	assert metadata['data_loading'] == {
		'num_workers': 0,
		'pin_memory': False,
		'pin_memory_device': '',
		'prefetch_factor': None,
		'persistent_workers': False,
		'non_blocking_h2d': False,
	}
