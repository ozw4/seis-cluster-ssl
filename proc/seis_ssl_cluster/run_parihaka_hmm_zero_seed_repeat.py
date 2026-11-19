"""Thin CLI for the thirty-cell Parihaka HMM zero/zero decoder-seed repeat."""

from seis_ssl_cluster.parihaka.hmm_zero_seed_repeat import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
