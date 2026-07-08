"""Stratigraphic HMM pretext training entrypoint placeholder."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path


def run_strat_hmm_pretext_training(
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run strat HMM pretext training."""
	_ = (config, resume)
	raise NotImplementedError(
		'strat HMM pretext training loop is implemented in the next milestone task',
	)


__all__ = ['run_strat_hmm_pretext_training']
