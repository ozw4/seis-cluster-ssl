"""Named, fixed head selections for the multi-source comparison studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path


@dataclass(frozen=True)
class MultiSourceProtocol:
	"""Keep candidate, target and study identities tied to one head selection."""

	head_ks: tuple[int, ...]
	consistency_weight: float = 0.0

	@property
	def suffix(self) -> str:
		"""Keep the consistency ablation in a separate execution namespace."""
		return '_cons010' if self.consistency_weight == 0.1 else ''

	@property
	def namespace(self) -> str:
		"""Return the survey output namespace."""
		return f'hmm_v2_{self.tag}{self.suffix}_multi_source_v1'

	def reuses_arm(self, survey: str, family: str) -> bool:
		"""Only the original zero-consistency studies reuse completed F3 MAE."""
		return self.consistency_weight == 0.0 and (survey, family) == ('f3', 'mae')

	@property
	def tag(self) -> str:
		"""Return the portable K identifier."""
		return 'k' + ''.join(map(str, self.head_ks))

	@property
	def study(self) -> str:
		"""Return the cross-survey experiment directory name."""
		return f'{self.tag}{self.suffix}_multi_source_evaluation_v1'

	@property
	def new_ks(self) -> list[int]:
		"""Return generated heads; K6 always retains its historical identity."""
		return [k for k in self.head_ks if k != 6]

	@property
	def cluster_filename(self) -> str:
		"""Return the target-stage definition filename."""
		return '01_cluster_hmm_' + ''.join(f'k{k}' for k in self.new_ks) + '.yaml'

	@property
	def candidate_ids(self) -> dict[str, str]:
		"""Return the three fixed source-family candidate identities."""
		return {
			family: f'{source}_hmm_v2_mh_{self.tag}_distill020{self.suffix}'
			for family, source in {
				'mae': 'mae100',
				'local_bt': 'local_bt_rot90_asym_g060_3ep',
				'random': 'random_init',
			}.items()
		}


K6810 = MultiSourceProtocol((6, 8, 10))
K468 = MultiSourceProtocol((4, 6, 8))
K6810_CONS010 = MultiSourceProtocol((6, 8, 10), 0.1)
K468_CONS010 = MultiSourceProtocol((4, 6, 8), 0.1)
PROTOCOLS = (K6810, K468, K6810_CONS010, K468_CONS010)


def protocol_for_heads(
	heads: object, consistency_weight: object = 0.0
) -> MultiSourceProtocol:
	"""Accept only the versioned head/consistency combinations."""
	for protocol in PROTOCOLS:
		if (
			isinstance(heads, (list, tuple))
			and all(type(k) is int for k in heads)
			and tuple(heads) == protocol.head_ks
			and type(consistency_weight) in (int, float)
			and consistency_weight == protocol.consistency_weight
		):
			return protocol
	raise ValueError('unsupported fixed multi-source head contract')


def protocol_for_matrix(matrix: dict) -> MultiSourceProtocol:
	"""Bind both the heads and loss weight from a study matrix."""
	return protocol_for_heads(
		matrix.get('selected_head_ks'), matrix.get('consistency_weight')
	)


def experiment_protocol(root: Path) -> MultiSourceProtocol:
	"""Bind a survey experiment namespace to its fixed head selection."""
	for protocol in PROTOCOLS:
		if root.name.endswith(f'_{protocol.namespace}'):
			return protocol
	raise ValueError('unknown multi-source execution namespace')
