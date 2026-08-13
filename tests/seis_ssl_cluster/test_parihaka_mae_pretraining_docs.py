
from pathlib import Path

DOC_PATH = Path('docs/parihaka_mae_pretraining.md')
DOCS_INDEX_PATH = Path('docs/README.md')


def test_parihaka_mae_doc_fixes_provenance_and_volume_contract() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')

	for expected in (
		'Mendeley Data',
		'10.17632/gnvyh3msrj.1',
		'AIcrowd-derived redistribution',
		'Byte identity with the AIcrowd distribution',
		'`data_train.npz`',
		'`parihaka_data_train.npz`',
		'`data.npy`',
		'`[Z, X, Y]`',
		'`[1006, 782, 590]`',
		'`464148280`',
		'`0.6766075433795379`',
		'`390.30892519280377`',
		'`source.transpose(1, 2, 0)`',
		'`[X, Y, Z]`',
		'`[782, 590, 1006]`',
		"`numpy.load(..., mmap_mode='r')`",
	):
		assert expected in document


def test_parihaka_mae_doc_fixes_paths_training_and_claim_boundary() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')

	for expected in (
		'${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/data/parihaka/facies_benchmark_v1/',
		'parihaka_amplitude_manifest.json',
		'`SurveyManifest`',
		'`SurveyNormalizationStats`',
		'`grid_order` to `[x, y, z]`',
		'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
		'<MODEL_TAG>/smoke_2step',
		'<MODEL_TAG>/full_100ep',
		'local_crop_size: [128, 128, 128]',
		'spatial_mask_ratio: 0.75',
		'amp: true',
		'amp_dtype: auto',
		'Initialization is random initialization from seed 42.',
		'`latest.pt` is the checkpoint after completion of epoch 100.',
		'strictly-lower training-loss policy',
		'survey-specific transductive',
	):
		assert expected in document

	assert '`parihaka_labels_train.npz`' in document
	assert 'are not inputs to preparation, normalization, the MAE config' in document
	assert 'must not be opened or hashed' in document


def test_parihaka_mae_doc_excludes_indirect_contracts_and_is_indexed() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')
	index = DOCS_INDEX_PATH.read_text(encoding='utf-8')

	assert (
		'A registry hierarchy, `ArtifactPaths`, a common publisher, a publish '
		'manifest, a PASS handoff, or a workflow state machine. None is introduced '
		'or required.'
	) in document
	assert '(parihaka_mae_pretraining.md)' in index
