
from pathlib import Path

DOC_PATH = Path('docs/parihaka_mae_pretraining.md')
DOCS_INDEX_PATH = Path('docs/README.md')
CLAIM_BOUNDARY_PATH = Path('docs/same_survey_pretraining_claims.md')


def test_parihaka_mae_doc_records_non_derivable_provenance() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')

	for expected in (
		'Mendeley Data',
		'10.17632/gnvyh3msrj.1',
		'AIcrowd-derived redistribution',
		'AIcrowd distribution',
		'not been verified',
		'`data_train.npz`',
		'`parihaka_data_train.npz`',
		'`data.npy`',
		'`[Z, X, Y]`',
		'`[X, Y, Z]`',
		'`output[x, y, z] == source[z, x, y]`',
	):
		assert expected in document


def test_parihaka_mae_doc_records_label_and_claim_boundaries() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')
	claim_boundary = CLAIM_BOUNDARY_PATH.read_text(encoding='utf-8')

	assert '`parihaka_labels_train.npz`' in document
	assert 'are not inputs to amplitude preparation' in document
	assert 'opened or hashed' in document
	assert '(same_survey_pretraining_claims.md)' in document
	assert 'same-survey, transductive' in claim_boundary
	assert 'unseen survey' in claim_boundary


def test_parihaka_mae_doc_links_the_single_execution_runbook_and_is_indexed() -> None:
	document = DOC_PATH.read_text(encoding='utf-8')
	index = DOCS_INDEX_PATH.read_text(encoding='utf-8')

	assert '../experiments/parihaka/' in document
	assert '(parihaka_mae_pretraining.md)' in index
