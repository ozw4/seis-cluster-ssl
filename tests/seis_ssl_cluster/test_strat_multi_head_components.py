"""Component construction coverage for multi-head strat HMM training."""

from __future__ import annotations

import torch

from seis_ssl_cluster.training.strat_hmm import components
from seis_ssl_cluster.training.strat_hmm.state import TrainabilitySummary


class _Student(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder_dim = 4
		self.encoder = torch.nn.Module()
		self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


def test_multi_head_builder_groups_all_heads_once_and_is_consistency_invariant(
	monkeypatch,
	tmp_path,
) -> None:
	def build_student_teacher(*_args, **_kwargs):
		student = _Student()
		return (
			student,
			None,
			{'paths': {'output_root': 'unused'}, 'data': {}, 'zero_mask': {}},
			TrainabilitySummary(
				trainable_parameter_count=sum(
					parameter.numel() for parameter in student.parameters()
				),
				frozen_parameter_count=0,
				trainable_names=tuple(name for name, _ in student.named_parameters()),
			),
		)

	monkeypatch.setattr(components, '_build_student_teacher', build_student_teacher)
	config = _config(tmp_path)
	torch.manual_seed(273)
	without_consistency = components.build_strat_hmm_multi_head_components(
		config,
		device='cpu',
	)
	config['loss']['consistency_weight'] = 0.1
	torch.manual_seed(273)
	with_consistency = components.build_strat_hmm_multi_head_components(
		config,
		device='cpu',
	)

	assert without_consistency.head_spec == 'multi_resolution_ordered_prototypes_v1'
	assert without_consistency.head_ks == (6, 8, 10)
	assert [group['name'] for group in without_consistency.optimizer.param_groups] == [
		'head',
		'encoder',
	]
	assert [group['lr'] for group in without_consistency.optimizer.param_groups] == [
		3.0e-4,
		1.0e-5,
	]
	head_parameters = tuple(without_consistency.heads.parameters())
	optimizer_head_parameters = tuple(
		without_consistency.optimizer.param_groups[0]['params']
	)
	assert {id(parameter) for parameter in optimizer_head_parameters} == {
		id(parameter) for parameter in head_parameters
	}
	assert len({id(parameter) for parameter in optimizer_head_parameters}) == len(
		head_parameters
	)
	assert all(
		torch.equal(value, with_consistency.heads.state_dict()[key])
		for key, value in without_consistency.heads.state_dict().items()
	)
	assert all(
		torch.equal(value, with_consistency.student.state_dict()[key])
		for key, value in without_consistency.student.state_dict().items()
	)
	without_names = _optimizer_parameter_names(without_consistency)
	with_names = _optimizer_parameter_names(with_consistency)
	assert without_names == with_names


def _config(tmp_path) -> dict[str, object]:
	return {
		'paths': {'output_root': str(tmp_path)},
		'data': {},
		'zero_mask': {},
		'head': {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 3,
			'temperature': 0.1,
			'normalize': True,
		},
		'loss': {'consistency_weight': 0.0},
		'train': {'lr': 3.0e-4, 'encoder_lr': 1.0e-5, 'weight_decay': 0.05},
	}


def _optimizer_parameter_names(components_) -> list[str]:
	parameter_names = {
		id(parameter): name
		for module in (components_.student, components_.heads)
		for name, parameter in module.named_parameters()
	}
	return [
		parameter_names[id(parameter)]
		for group in components_.optimizer.param_groups
		for parameter in group['params']
	]
