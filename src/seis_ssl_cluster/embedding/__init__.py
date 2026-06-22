"""Embedding components for seismic SSL clustering."""

from seis_ssl_cluster.embedding.extractor import (
	EmbeddingExtractionSettings,
	SurveyEmbeddingResult,
	build_embedding_metadata,
	build_model_from_config,
	extract_survey_embeddings,
	extraction_settings_from_config,
	reduce_valid_mask_to_tokens,
	run_embedding_extraction,
)
from seis_ssl_cluster.embedding.merge import EmbeddingMerger
from seis_ssl_cluster.embedding.sliding_window import (
	SlidingWindow,
	compute_stride_xyz,
	iter_sliding_windows,
	padded_volume_shape_xyz,
	token_grid_shape_xyz,
)
from seis_ssl_cluster.embedding.writer import (
	EmbeddingOutputPaths,
	cleanup_temp_outputs,
	commit_staged_outputs,
	create_merge_memmaps,
	file_sha256,
	metadata_matches,
	output_paths,
	prepare_outputs,
	write_metadata,
)

__all__ = [
	'EmbeddingExtractionSettings',
	'EmbeddingMerger',
	'EmbeddingOutputPaths',
	'SlidingWindow',
	'SurveyEmbeddingResult',
	'build_embedding_metadata',
	'build_model_from_config',
	'cleanup_temp_outputs',
	'commit_staged_outputs',
	'compute_stride_xyz',
	'create_merge_memmaps',
	'extract_survey_embeddings',
	'extraction_settings_from_config',
	'file_sha256',
	'iter_sliding_windows',
	'metadata_matches',
	'output_paths',
	'padded_volume_shape_xyz',
	'prepare_outputs',
	'reduce_valid_mask_to_tokens',
	'run_embedding_extraction',
	'token_grid_shape_xyz',
	'write_metadata',
]
