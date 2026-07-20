"""Compatibility wrapper for legacy strat HMM pretext training imports."""

from seis_ssl_cluster.training.strat_hmm.components import (
	build_strat_hmm_components,
	build_strat_hmm_head_only_components,
	build_strat_hmm_multi_head_components,
	configure_student_trainability,
)
from seis_ssl_cluster.training.strat_hmm.epoch import (
	train_strat_hmm_head_only_one_epoch,
	train_strat_hmm_multi_head_one_epoch,
)
from seis_ssl_cluster.training.strat_hmm.runner import (
	run_strat_hmm_pretext_training,
)
from seis_ssl_cluster.training.strat_hmm.state import (
	StratHmmHeadOnlyComponents,
	StratHmmMultiHeadComponents,
	StratHmmResumeState,
	StratHmmTrainingState,
	TrainabilitySummary,
)

__all__ = [
	'StratHmmHeadOnlyComponents',
	'StratHmmMultiHeadComponents',
	'StratHmmResumeState',
	'StratHmmTrainingState',
	'TrainabilitySummary',
	'build_strat_hmm_components',
	'build_strat_hmm_head_only_components',
	'build_strat_hmm_multi_head_components',
	'configure_student_trainability',
	'run_strat_hmm_pretext_training',
	'train_strat_hmm_head_only_one_epoch',
	'train_strat_hmm_multi_head_one_epoch',
]
