"""Public API for stratigraphic HMM pretext training."""

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
from seis_ssl_cluster.training.strat_hmm.losses import (
	compute_strat_hmm_multi_head_losses,
	compute_strat_hmm_multi_head_posterior_losses,
	compute_strat_hmm_pretext_losses,
)
from seis_ssl_cluster.training.strat_hmm.masking import (
	COMMON_HARD_TARGET_HEAD_KS,
	XYTokenColumnMaskPlan,
	plan_xy_token_column_mask,
	validate_common_hard_target_valid_masks,
)
from seis_ssl_cluster.training.strat_hmm.resume import (
	restore_strat_hmm_training_checkpoint,
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
	'COMMON_HARD_TARGET_HEAD_KS',
	'StratHmmHeadOnlyComponents',
	'StratHmmMultiHeadComponents',
	'StratHmmResumeState',
	'StratHmmTrainingState',
	'TrainabilitySummary',
	'XYTokenColumnMaskPlan',
	'build_strat_hmm_components',
	'build_strat_hmm_head_only_components',
	'build_strat_hmm_multi_head_components',
	'compute_strat_hmm_multi_head_losses',
	'compute_strat_hmm_multi_head_posterior_losses',
	'compute_strat_hmm_pretext_losses',
	'configure_student_trainability',
	'plan_xy_token_column_mask',
	'restore_strat_hmm_training_checkpoint',
	'run_strat_hmm_pretext_training',
	'train_strat_hmm_head_only_one_epoch',
	'train_strat_hmm_multi_head_one_epoch',
	'validate_common_hard_target_valid_masks',
]
