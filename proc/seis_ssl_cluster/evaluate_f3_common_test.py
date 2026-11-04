"""Thin CLI for explicit F3 common-test evaluation of frozen predictions."""

from seis_ssl_cluster.f3.lithology.common_test import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
