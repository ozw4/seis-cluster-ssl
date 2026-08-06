from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.config.f3_baselines import (
	f3_lithology_comparison_report_config_from_mapping,
)

BASELINE_ROOT = (
	Path('experiments')
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology_baselines'
)
RANDOM_ENCODER_TAG = (
	'random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1'
)


def test_f3_lithology_baseline_configs_are_present_and_have_metadata() -> None:
	expected_files = (
		Path('README.md'),
		Path('05_build_baseline_comparison_report.yaml'),
		Path('z_only_v1/01_build_baseline_token_dataset.yaml'),
		Path('z_only_v1/02_train_linear_probe.yaml'),
		Path('z_only_v1/03_build_report.yaml'),
		Path('amplitude_stats_v1/01_build_baseline_token_dataset.yaml'),
		Path('amplitude_stats_v1/02_train_linear_probe.yaml'),
		Path('amplitude_stats_v1/03_build_report.yaml'),
		Path('xyz_coordinates_v1/01_build_baseline_token_dataset.yaml'),
		Path('xyz_coordinates_v1/02_train_linear_probe.yaml'),
		Path('xyz_coordinates_v1/03_build_report.yaml'),
		Path(RANDOM_ENCODER_TAG) / '01_create_random_checkpoint.yaml',
		Path(RANDOM_ENCODER_TAG) / '02_extract_embeddings.yaml',
		Path(RANDOM_ENCODER_TAG) / '03_build_token_dataset.yaml',
		Path(RANDOM_ENCODER_TAG) / '04_train_linear_probe.yaml',
		Path(RANDOM_ENCODER_TAG) / '05_build_report.yaml',
	)

	for relative in expected_files:
		path = BASELINE_ROOT / relative
		assert path.is_file(), relative

	for yaml_path in sorted(BASELINE_ROOT.glob('**/*.yaml')):
		payload = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
		assert isinstance(payload, dict), yaml_path
		assert 'stage' not in payload
		if yaml_path.name not in {
			'02_extract_embeddings.yaml',
			'05_build_baseline_comparison_report.yaml',
		}:
			assert _feature_sources(payload), yaml_path

		if yaml_path.name == '05_build_baseline_comparison_report.yaml':
			comparison = payload['comparison']
			resolved = f3_lithology_comparison_report_config_from_mapping(payload)
			assert str(resolved.search_root) == comparison['search_root']
			assert str(resolved.output_csv) == comparison['output_csv']
			assert str(resolved.output_markdown) == comparison['output_markdown']
			assert comparison['figure_dpi'] == 300
			assert payload['publish'] == {
				'enabled': True,
				'output_dir': (
					'results/f3/facies_benchmark_v1/baseline_comparison'
				),
				'include_figures': True,
				'max_file_size_mb': 10,
			}

		if yaml_path.name.endswith('train_linear_probe.yaml'):
			feature_source = payload['token_dataset']['feature_source']
			assert feature_source['kind'] in {
				'z_only',
				'xyz_coordinates',
				'amplitude_stats',
				'random_encoder',
			}
			assert feature_source['embedding_spec'] == 'overlap_x16'
			if feature_source['kind'] == 'random_encoder':
				assert payload['token_dataset']['input_dir'].endswith(
					f'/{RANDOM_ENCODER_TAG}/overlap_x16/png_slices_segy_labels_v1/token_dataset',
				)
				assert payload['probe']['output_dir'].endswith(
					f'/{RANDOM_ENCODER_TAG}/overlap_x16/png_slices_segy_labels_v1/probes/linear_balanced_v1',
				)
				assert '/baselines/' not in payload['token_dataset']['input_dir']
			else:
				assert 'baselines' in payload['probe']['output_dir']

		if yaml_path.name.endswith('build_report.yaml'):
			comparison = payload['comparison']
			assert comparison['search_root']
			assert comparison['output_csv']

		if yaml_path.name == '01_create_random_checkpoint.yaml':
			assert payload['reference_model']['checkpoint']

	readme = (BASELINE_ROOT / 'README.md').read_text(encoding='utf-8')
	assert 'FEATURE_SOURCE_KIND' in readme
	assert 'Z_BASELINE_TAG=z_only_v1' in readme
	assert 'AMP_BASELINE_TAG=amplitude_stats_v1' in readme
	assert f'RANDOM_ENCODER_TAG={RANDOM_ENCODER_TAG}' in readme


def _feature_sources(value: Any) -> list[dict[str, object]]:
	if isinstance(value, dict):
		found = []
		if isinstance(value.get('feature_source'), dict):
			found.append(value['feature_source'])
		for item in value.values():
			found.extend(_feature_sources(item))
		return found
	if isinstance(value, list):
		found = []
		for item in value:
			found.extend(_feature_sources(item))
		return found
	return []
