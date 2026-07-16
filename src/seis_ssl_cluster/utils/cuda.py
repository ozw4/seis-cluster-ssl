"""CUDA device capability helpers."""

from __future__ import annotations

import torch


def cuda_device_supports_bfloat16(device: torch.device) -> bool:
	"""Return BF16 support for the selected CUDA device."""
	if device.type != 'cuda':
		msg = f'BF16 CUDA capability requires a CUDA device; got {device}'
		raise ValueError(msg)
	if device.index is None:
		return torch.cuda.is_bf16_supported()
	with torch.cuda.device(device):
		return torch.cuda.is_bf16_supported()


__all__ = ['cuda_device_supports_bfloat16']
