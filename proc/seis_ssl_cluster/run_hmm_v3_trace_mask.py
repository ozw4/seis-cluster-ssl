"""Thin CLI for the canonical-v02 trace-mask successor experiment."""

from seis_ssl_cluster.hmm.trace_mask_driver import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
