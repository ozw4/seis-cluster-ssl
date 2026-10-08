"""Thin CLI for the controlled F3 mask-token learning-rate sweep."""

from seis_ssl_cluster.hmm.mask_token_lr import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
