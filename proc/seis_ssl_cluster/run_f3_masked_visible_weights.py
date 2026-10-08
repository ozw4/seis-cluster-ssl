"""Thin CLI for the F3 normalized masked/visible prototype-weight sweep."""

from seis_ssl_cluster.hmm.masked_visible_weights import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
