"""Thin CLI for the non-F3 3e-5, 50:50 trace-mask follow-up."""

from seis_ssl_cluster.hmm.trace_mask_lr3 import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
